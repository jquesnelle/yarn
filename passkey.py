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

# from https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 2000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def test_model(pipe, prompt_text, pass_key):
    response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[
        0]["generated_text"][len(prompt_text):]
    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, padding_side="right", trust_remote_code=True)

    if args.fixed_length:
        lengths = [args.fixed_length]
        tokens = [len(tokenizer.encode(generate_prompt(args.fixed_length)[0]))]
    else:
        tokens = [x for x in range(
            args.min_tokens, args.max_tokens + 1, args.tokens_step)]
        lengths = []
        last_n = 0
        for target in tqdm(tokens, desc="Determining sequence lengths"):
            num_tokens = 0
            n = last_n
            while num_tokens < target:
                last_n = n
                n += args.length_step
                prompt = generate_prompt(n)[0]
                num_tokens = len(tokenizer.encode(prompt))
            lengths.append(last_n)

    results = []
    for model in tqdm(models, desc="Model", leave=False):
        torch.cuda.empty_cache()

        loaded = load_model(model, args.load_in_8bit,
                            args.load_in_4bit, args.max_tokens + args.tokens_step)
        apply_patches(loaded, args.max_tokens + args.tokens_step, args.dynamic_ntk,
                      args.dynamic_linear, args.ntk, args.linear)

        pipe = pipeline("text-generation", model=loaded,
                        tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

        result = [0] * len(lengths)
        for i, length in tenumerate(lengths, desc="Lengths", leave=False):
            for _ in trange(0, args.iterations, desc="Iterations", leave=False):
                prompt_text, pass_key = generate_prompt(length)
                num_tokens = len(pipe.tokenizer.encode(prompt_text))
                answer = test_model(pipe, prompt_text, pass_key)
                if answer == pass_key:
                    result[i] += 1
            result[i] /= args.iterations
            print(f"{model}: {tokens[i]}={int(result[i]*100)}%")

        result.insert(0, model)
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
    parser.add_argument("--fixed-length", type=int)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--tokens-step", type=int, default=200)
    parser.add_argument("--length-step", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    main(parser.parse_args())
