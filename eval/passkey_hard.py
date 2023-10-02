import argparse
import random
import re
import sys
import torch
import warnings
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from model_loader import *
from datasets import load_dataset
import random
#import pickle
import json

# from https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py


def order(i):
    if i % 10 == 1 and i % 10 != 11:
        return str(i) + "st"
    elif i % 10 == 2 and i % 10 != 12:
        return str(i) + "nd"
    elif i % 19 == 3 and i % 10 != 13:
        return str(i) + "rd"
    else:
        return str(i) + "th"
    
def generate_prompt(docs, num_keys=1):
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    pass_keys = [random.randint(1, 50000) for _ in range(num_keys)]
    start_pos = sorted([random.randint(1, len(docs)) for _ in range(num_keys)])
    information_lines = [f"The {order(i+1)} pass key is {pass_key}. Remember it. {pass_key} is the {order(i+1)} pass key." for i, pass_key in enumerate(pass_keys)]
    retrieve_number = random.randint(0, num_keys - 1)
    final_question = f"What is the {order(retrieve_number + 1)} pass key? The {order(retrieve_number + 1)} pass key is"
    lines = [task_description]
    prev = 0
    for line, pos in zip(information_lines, start_pos):
        lines.append("".join(docs[prev:pos]))
        lines.append(line)
        prev = pos
    lines.append("".join(docs[prev:]))
    lines.append(final_question)

    return "\n".join(lines), pass_keys, start_pos, retrieve_number


def test_model(pipe, prompt_text):
    response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[
        0]["generated_text"][len(prompt_text):]

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key, response

def construct_junk(data, length, tokenizer):
    token_count = 0
    docs = []
    length = length or 8192
    
    while token_count < length:
        sample = random.choice(data)["text"]
        toks = tokenizer(sample, return_offsets_mapping=True)
        offsets = [(i, j) for i, j in toks["offset_mapping"] if i < j]
        num_tok_to_add = min(length - token_count, len(offsets))
        pretokenized = [sample[i:j] for i, j in offsets[:num_tok_to_add]]
        docs.extend(pretokenized)
        token_count += num_tok_to_add

    return docs


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, padding_side="right", trust_remote_code=True)

    data = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")["train"]
    junks = construct_junk(data, args.fixed_length, tokenizer)

    # We restrict tokens to a small subset: digits, eos and continuous spaces/linebreaks
    # This is to prevent continuations like " is a special number" blah blah blah...
    if args.restrict_tokens:
        vocab = tokenizer.vocab 

        escape_char = "▁"  # for Llama family

        digit_tokens = [vocab[a] for a in vocab.keys() if a.lstrip(escape_char).isdigit()]
        # Add EOS
        digit_tokens.append(vocab[tokenizer.eos_token])
        # Add spaces/linebreaks
        extra = [vocab[a] for a in vocab.keys() if a.strip(" \n" + escape_char) == ""]
        digit_tokens.extend(extra)

        mask = torch.ones(tokenizer.vocab_size, dtype=torch.bool)
        mask[digit_tokens] = 0

        def filter_digits(module, input, output):
            output.logits[..., mask[:output.logits.size(-1)]] = -1e4
        
        print(f"Decoding restricted to {len(digit_tokens)} tokens.")


    results = []
    success_count = 0
    for model in tqdm(models, desc="Model", leave=False):
        torch.cuda.empty_cache()

        loaded = load_model_and_apply_patches(model, args)
        if args.restrict_tokens:
            loaded.register_forward_hook(filter_digits)

        pipe = pipeline("text-generation", model=loaded,
                        tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

        for _ in trange(0, args.iterations, desc="Iterations", leave=False):
            prompt_text, pass_keys, start_pos, target = generate_prompt(junks, args.num_keys)
            num_tokens = len(pipe.tokenizer.encode(prompt_text))
            answer, return_text = test_model(pipe, prompt_text)
            passed = str(answer).startswith(str(pass_keys[target]))
            result = {"prompt_text": prompt_text, "start_pos": start_pos, "pass_keys": pass_keys, "return_text": return_text, "passed": passed}
            success_count += passed
            results.append(result)

    results.append({"original_prompt": junks})
    print(f"Iteration: {args.iterations}")
    print(f"Successes: {success_count}")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("--fixed-length", type=int, default=8192)
    parser.add_argument("--restrict-tokens", type=bool, default=True)
    parser.add_argument("--num-keys", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-file", type=str)
    main(add_args(parser).parse_args())
