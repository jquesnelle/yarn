import argparse
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from model_loader import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model_and_apply_patches(args.model, args)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id,
                    temperature=args.temperature, repetition_penalty=args.repetition_penalty, do_sample=args.temperature is not None)

    while True:
        if args.input_file is None:
            prompt_text = input("> ")
        else:
            input(f"Press enter to read {args.input_file} ")
            prompt_text = open(args.input_file, encoding="utf=8").read()
        response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=args.max_new_tokens)[
            0]["generated_text"][len(prompt_text):]
        print(f"< {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repetition-penalty", type=float)

    args = add_args(parser).parse_args()
    main(args)
