import argparse
import datasets
import numpy as np
import evaluate
import sys
import torch
import warnings
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from scaled_rope.patch import *


class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description="",
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
        )

    def _compute(
        self, predictions, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
    ):
        if device is not None:
            assert device in ["gpu", "cpu",
                              "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens(
                {"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)
                             ), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat(
                    [bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(
                    encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels)
                 * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    perplexity = Perplexity()
    input_texts = datasets.load_dataset(args.dataset, args.subset, split=args.split)[
        args.feature][:args.samples]

    tokens = [x for x in range(
        args.min_tokens, args.max_tokens + 1, args.tokens_step)]

    results = []
    for model in tqdm(models, desc="Model", leave=False):
        torch.cuda.empty_cache()

        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        if "MPTForCausalLM" in config.architectures:
            config.max_seq_len = max(args.length, config.max_seq_len)

        loaded = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            config=config
        )

        if "GPTNeoXForCausalLM" in loaded.config.architectures:
            patch_gptneox_for_longer_sequences(loaded, args.length)
        if args.dynamic:
            suffix = "dynamic"
            if "GPTNeoXForCausalLM" in loaded.config.architectures:
                patch_gptneox_for_scaled_rotary_embeddings(loaded)
            elif "LlamaForCausalLM" in loaded.config.architectures:
                patch_llama_for_scaled_rotary_embeddings(loaded)
            else:
                raise RuntimeError(
                    f"Unknown architectures {loaded.config.architectures} to patch {suffix}")
        elif args.ntk:
            suffix = "ntk"
            if "GPTNeoXForCausalLM" in loaded.config.architectures:
                patch_gptneox_for_ntk_scaled_rotary_embeddings(
                    loaded, args.ntk)
            elif "LlamaForCausalLM" in loaded.config.architectures:
                patch_llama_for_ntk_scaled_rotary_embeddings(loaded, args.ntk)
            else:
                raise RuntimeError(
                    f"Unknown architectures {loaded.config.architectures} to patch {suffix}")
        else:
            suffix = None

        result = []
        for max_length in tokens:
            ppl = perplexity.compute(model=loaded, tokenizer=tokenizer, predictions=input_texts,
                                     batch_size=args.batch_size, add_start_token=tokenizer.bos_token is not None, max_length=max_length)["mean_perplexity"]
            print(f"{model}: {max_length}={ppl}")
            result.append(ppl)

        result.insert(0, f"{model}{'-' + suffix if suffix else ''}")
        results.append(result)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str, default="tau/scrolls")
    parser.add_argument("-s", "--subset", type=str, default="gov_report")
    parser.add_argument("-f", "--feature", type=str, default="input")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--tokens-step", type=int, default=200)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--ntk", type=int)
    parser.add_argument("--output-file", type=str)
    main(parser.parse_args())
