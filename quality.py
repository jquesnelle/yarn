import argparse
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from tqdm import tqdm
from model_loader import load_model, apply_patches


ZERO_SCROLLS_QUALITY_PROMPT = "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).\n\nStory:\n{story}\n\nQuestion and Possible Answers:\n{question}\n (A) {a}\n (B) {b}\n (C) {c}\n (D) {d}"
CHOICES = ["A", "B", "C", "D"]

def get_prompt(instruct_type, sample):
    options = sample["options"]
    instruction = ZERO_SCROLLS_QUALITY_PROMPT.format(story=sample["article"], question=sample["question"], a=options[0], b=options[1], c=options[2], d=options[3])
    if instruct_type == "mpt":
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
        return template.format(instruction=instruction)
    else:
        return f"{instruction}\n\nAnswer: ("

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=sys.maxsize)
    tokenizer.pad_token = tokenizer.eos_token

    instruct_type = None
    if "mosaicml/mpt" in args.model:
        instruct_type = "mpt"

    dataset = load_dataset("emozilla/quality", split=args.split)
    dataset = dataset.map(lambda sample: {"prompt": get_prompt(instruct_type, sample)})
    dataset = dataset.filter(lambda sample: len(tokenizer(sample["prompt"]).input_ids) <= args.max_tokens - 1)
    
    model = load_model(args.model, args.load_in_8bit, args.load_in_4bit, args.max_tokens)
    apply_patches(model, args.max_tokens, args.dynamic_ntk,
                    args.dynamic_linear, args.ntk)
    
    choice_tokens = [x[0] for x in tokenizer(CHOICES, add_special_tokens=False).input_ids]
    decoded_choice = tokenizer.decode(choice_tokens, clean_up_tokenization_spaces=True)

    correct_answers = 0
    i = 0
    max = len(dataset) if args.limit is None else args.limit
    bar = tqdm(total=max)
    while i < max:
        sample = dataset[i]
        tokenized_prompt = tokenizer(sample["prompt"], return_tensors="pt").input_ids.to("cuda")

        output = model.generate(tokenized_prompt, max_new_tokens=1)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            selection = CHOICES.index(decoded_output[-1])
        except ValueError:
            selection = -1

        correct_answers += 1 if selection == sample["answer"] else 0
        print(f"Choice: {CHOICES[selection]} Correct: {CHOICES[sample['answer']]}")

        i += 1
        bar.update()

    percent = (correct_answers / len(dataset)) * 100.0
    print(f"Accuracy: {percent:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--split", type=str, default="validation")

    args = parser.parse_args()
    main(args)
