import argparse
import os
import sys
from pathlib import Path

import torch
from lora import peft_model
from scaled_rope.configuration_llama import LlamaConfig
from scaled_rope.modelling_llama import LlamaForCausalLM
from transformers import GenerationConfig, LlamaTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="checkpoint path or model name")
    parser.add_argument("--model_name", type=str, help="model name of base model")
    parser.add_argument(
        "--dtype", type=str, default="float16", help="float16 or bfloat16"
    )
    parser.add_argument(
        "--save_merged_model",
        type=bool,
        default=False,
        help="Whether to save merged model",
    )
    parser.add_argument("--hf_repo_name", type=str, help="Huggingface repository name")
    parser.add_argument("--auth_token", type=str, help="User access token")
    parser.add_argument("--output_dir", type=str, help="output folder path")
    parser.add_argument(
        "--max_position_embeddings",
        type=int,
        default=16384,
        help="Max Position Embeddings",
    )
    parser.add_argument(
        "--position_interpolation_scale",
        type=float,
        default=0.125,
        help="Position interpolation scale",
    )
    parser.add_argument("--max_shard_size", type=str, default="10GB")
    parser.add_argument("--cache_dir", type=str)
    return parser.parse_args()


device = "cuda"

generation_config = GenerationConfig(
    temperature=0,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)


def generate(
    model,
    tokenizer,
    prompt,
    generation_config=generation_config,
    max_new_tokens=100,
    device=device,
):
    model = model.half().to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            min_new_tokens=3,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print("Text generated:")
    print(output)
    return output


def get_model(
    model_name, max_position_embeddings, position_interpolation_scale, torch_dtype
):
    print("Loading Tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name, add_bos_token=False)
    print(f"{type(tokenizer).__name__} (vocab_size={len(tokenizer)})")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_position_embeddings

    config = LlamaConfig.from_pretrained(model_name)
    config.use_xpos = False
    config.max_position_embeddings = max_position_embeddings
    config.transformer_engine = False
    config.ntk_alpha = None
    config.position_interpolation_scale = position_interpolation_scale

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        config=config,
    )
    return model, tokenizer


def test_output(model, tokenizer):
    for p in [
        "<s>What is a meme, and what's the history behind this word?",
        "<s>Henry VIII (28 June 1491 – 28 January 1547) was King of England from 22 April 1509 until his death in 1547. Henry is best known for his six marriages, and his efforts to have his first marriage (to Catherine of Aragon) annulled. His disagreement with Pope Clement VII about such an annulment led Henry to initiate the English Reformation, separating the Church of England from papal authority. He appointed himself Supreme Head of the Church of England",
    ]:
        generate(
            model,
            tokenizer,
            p,
            generation_config=generation_config,
            max_new_tokens=100,
            device=device,
        )


def main():
    args = parse_args()
    print(args)

    if args.dtype in ("float16", "fp16"):
        torch_dtype = torch.float16
    elif args.dtype in ("float32", "fp32"):
        torch_dtype = torch.float32
    elif args.dtype in ("bfloat16", "bf16"):
        torch_dtype = torch.bfloat16
    else:
        print(f"Unsupported dtpye: {args.dtype}")
        sys.exit(1)

    if not args.hf_repo_name and not args.output_folder:
        print(
            "Please specify either `--hf_repo_name` to push to HF or `--output_folder` "
            "to export the model to a local folder."
        )
        sys.exit(1)

    print(f"Loading model '{args.model_name}' ({args.dtype}) ...")

    if (Path(args.ckpt_path) / "adapter_model.bin").exists():
        model, tokenizer = get_model(
            args.model_name,
            args.max_position_embeddings,
            args.position_interpolation_scale,
            torch_dtype,
        )

        # todo test
        print("testing original Model with interpolation")
        test_output(model, tokenizer)
        # finish

        # dummy PEFT MODEL
        model = peft_model(model, args.model_name)

        if (Path(args.ckpt_path) / "adapter_model.bin").exists():
            adapters_weights = torch.load(
                Path(args.ckpt_path).joinpath("adapter_model.bin"),
                map_location=model.device,
            )

            adapter_keys = set(list(adapters_weights.keys()))
            base_model_keys = set(list(model.state_dict().keys()))
            if len(base_model_keys.intersection(adapter_keys)) != len(adapter_keys):
                print("Default in keys")
                new_state_dict = {}
                for k, v in adapters_weights.items():
                    new_k = k[: -len(".weight")] + ".default.weight"
                    new_state_dict[new_k] = v
                adapter_keys = set(list(new_state_dict.keys()))
                if len(base_model_keys.intersection(adapter_keys)) == len(adapter_keys):
                    model.load_state_dict(new_state_dict, strict=False)
                else:
                    raise ValueError("Adapter keys do not match with base model keys")
        else:
            model.load_state_dict(
                torch.load(Path(args.ckpt_path) / "pytorch_model.bin")
            )
    print("testing Adapter Model pre merge...")
    test_output(model, tokenizer)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    adapter_save_path = Path(args.output_dir).joinpath("adapter")
    if not os.path.exists(adapter_save_path):
        os.makedirs(adapter_save_path)

    print("Saving Lora Weights...")
    model.save_pretrained(Path(adapter_save_path), torch_dtype=torch_dtype)
    print("Model architecture:")
    print(model)

    if args.save_merged_model:
        print("Merging model")
        model = model.merge_and_unload()
        model.to(dtype=torch_dtype)
        print("Saving merged Model...")
        print("testing Adapter Model post merge...")
        test_output(model, tokenizer)
        model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(args.output_dir)

    if args.hf_repo_name:
        print("Uploading model to HF...")
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        create_repo(args.hf_repo_name)

        api.upload_folder(
            folder_path=args.output_dir,
            path_in_repo=".",
            repo_id=args.hf_repo_name,
            repo_type="model",
        )


if __name__ == "__main__":
    main()