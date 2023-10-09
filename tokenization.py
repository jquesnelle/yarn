from itertools import chain
from functools import reduce
import multiprocessing
import argparse
from typing import List
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

def main(args):
    if args.dataset is None or len(args.dataset[0]) == 0:
        raise RuntimeError("No datasets provided")
    datasets = args.dataset[0]

    splits = [x.split(",")[1] if len(x.split(",")) == 2 else "" for x in datasets]
    datasets = [x.split(",")[0] for x in datasets]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.json:
        dataset = load_dataset("json", data_files=datasets)[args.split]
        if reduce(lambda x,y: x or len(y) > 0, splits, False):
            if len(datasets) > 1:
                raise RuntimeError("Can only use splitting on json datasets if there is exactly one input file")
            dataset = dataset.train_test_split(train_size=float(splits[0]), seed=args.seed)["train"]
    else:
        to_concatenate = []
        for i in range(0, len(datasets)):
            try:
                loaded = load_from_disk(datasets[i])
            except:
                loaded = load_dataset([i])[args.split]
            if len(splits[i]) > 0:
                loaded = loaded.train_test_split(train_size=float(splits[i]), seed=args.seed)["train"]
            to_concatenate.append(loaded)
        dataset = concatenate_datasets(to_concatenate)

    dataset = dataset.remove_columns([x for x in dataset.column_names if x not in [args.feature]])

    tokenized_dataset = dataset.map(
        lambda example: tokenizer(
            [t + tokenizer.eos_token for t in example[args.feature]]),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=[args.feature],
    )

    block_size = args.sequence_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    train_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=args.num_proc,
    )

    if args.output:
        train_dataset.save_to_disk(args.output)
    if args.push_to_hub:
        train_dataset.push_to_hub(args.push_to_hub, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", nargs="+")
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--feature", type=str, default="text")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str)
    parser.add_argument("--push-to-hub", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--num-proc", type=int,
                        default=multiprocessing.cpu_count())
    main(parser.parse_args())
