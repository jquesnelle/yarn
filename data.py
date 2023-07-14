from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTrainedTokenizerBase,
                                                  TruncationStrategy)


class DatasetEntryLm(NamedTuple):
    """Language modelling dataset entry"""

    text: Union[str, None] = None


class LMDataset(Dataset):
    name = "LMDataset"

    def __init__(self, dataset_name, char_max_len: str = 200000) -> None:
        super().__init__()
        self.char_max_len = char_max_len
        self.dataset = datasets.load_dataset(dataset_name)['train']

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> DatasetEntryLm:
        dialogue = DatasetEntryLm(text=self.dataset[index]["text"][: self.char_max_len])
        return dialogue


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    mix_length_threshold: Optional[int] = 256
    mix_probability: Optional[float] = 0.6
    pad_to_multiple_of: Optional[int] = None
    samples_mixing: Optional[bool] = False

    def __post_init__(self):
        assert self.tokenizer.eos_token

    def process_one(self, messages, return_length=False):
        truncation = TruncationStrategy.LONGEST_FIRST
        max_length = self.max_length

        messages = messages.text

        flatten_message = self.tokenizer(
            "".join(messages),
            max_length=max_length,
            truncation=truncation,
            padding=False,
            return_token_type_ids=False,
        )

        label_mask = np.ones(len(flatten_message.input_ids), dtype=bool)
        return flatten_message, label_mask, 0

    def __call__(self, features):
        flatten_messages = []
        label_masks = []
        total_short_context = 0
        for messages in features:
            flatten_message, label_mask, total_short_context_one = self.process_one(
                messages
            )
            flatten_messages.append(flatten_message)
            label_masks.append(label_mask)
            total_short_context += total_short_context_one

        batch = self.tokenizer.pad(
            flatten_messages,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()

        return batch


def train_val_dataset(dataset, val_split=0.2):
    if val_split == 0:
        return dataset, None

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split, random_state=666, shuffle=True
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_one_dataset(
    conf,
    val_split: float = 0.025,
    data_path: str = None,
    mode: str = "sft",
    max_val_set: Optional[int] = 50,
    **kwargs,
):
    data_path = data_path or conf.cache_dir
    # dataset_name = dataset_name.lower()
    train_datasets = []
    eval_datasets = []
    for data_file in conf.dataset_names:
        dataset = LMDataset(data_file)

        # if eval not already defined
        if not ("eval" in locals() and "train" in locals()):
            train, eval = train_val_dataset(dataset, val_split=val_split)

        if eval and max_val_set and len(eval) > max_val_set:
            subset_indices = np.random.choice(len(eval), max_val_set)
            eval = Subset(eval, subset_indices)
        train_datasets.append(train)
        eval_datasets.append(eval)

    train = ConcatDataset(train_datasets)
    eval = ConcatDataset(eval_datasets)
    return train, eval


