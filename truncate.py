import argparse
from datasets import load_dataset

def main(args):
    dataset = load_dataset(args.dataset, split="train")
    def truncate(sample):
        sample["input_ids"] = sample["input_ids"][0:args.truncate]
        sample["labels"] = sample["labels"][0:args.truncate]
        sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
        return sample
    dataset = dataset.map(truncate, desc="Truncating", num_proc=args.num_proc)
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("truncate", type=int)
    args.add_argument("output", type=str)
    args.add_argument("--num-proc", type=int, default=32)
    args.add_argument("--dataset", type=str,
                      default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    main(args.parse_args())