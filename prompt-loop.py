import argparse
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from model_loader import load_model, apply_patches


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model, args.load_in_8bit,
                       args.load_in_4bit, args.max_new_tokens)
    apply_patches(model, args.max_new_tokens, args.dynamic_ntk,
                  args.dynamic_linear, args.dynamic_part_ntk, args.ntk, args.linear, args.part_ntk)

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
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--part-ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--dynamic-part-ntk", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repetition-penalty", type=float)

    args = parser.parse_args()
    main(args)
