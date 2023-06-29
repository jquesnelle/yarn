import torch
import evaluate
import os
import sys
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from itertools import chain
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    LlamaTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from scaled_rope.modelling_llama import LlamaForCausalLM
from scaled_rope.configuration_llama import LlamaConfig


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="openlm-research/open_llama_3b",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    max_position_embeddings: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum sequence length of the model."
            )
        },
    )

    position_interpolation_scale: Optional[float] = field(default=1)
    use_xpos: Optional[bool] = field(default=False)
    fp8: Optional[bool] = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="togethercomputer/RedPajama-Data-1T-Sample", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    group_texts: Optional[bool] = field(default=True)
    streaming: Optional[bool] = field(default=False)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = get_last_checkpoint(
        training_args.output_dir) if os.path.exists(training_args.output_dir) else None
    set_seed(training_args.seed)

    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_args.model_name_or_path, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = model_args.max_position_embeddings

        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        config.use_xpos = model_args.use_xpos
        config.max_position_embeddings = model_args.max_position_embeddings
        config.transformer_engine = model_args.fp8
        config.position_interpolation_scale = model_args.position_interpolation_scale

        model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path, device_map={
                                                 "": f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"}, torch_dtype=torch.bfloat16, config=config)
    else:
        raise NotImplementedError

    # patching for the random contiguous tensors bug
    for p in model.parameters():
        p = p.contiguous()

    block_size = model_args.max_position_embeddings

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
        result["labels"] = result["input_ids"].copy()
        return result

    def copy_input_ids(examples):
        examples["labels"] = examples["input_ids"].copy()

    datasets = load_dataset(data_args.dataset_name,
                            streaming=data_args.streaming)
    # better heuristic? maybe 3xlength?
    datasets = datasets.filter(lambda x: len(
        x["text"]) >= model_args.max_position_embeddings)
    datasets = datasets.map(
        lambda examples: {"text": [tokenizer.bos_token + x for x in examples["text"]]},
        batched=True)
    tokenized_datasets = datasets.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True
    )
    lm_datasets = tokenized_datasets.map(
        group_texts if data_args.group_texts else copy_input_ids,
        batched=True,
    )
    lm_datasets = lm_datasets.filter(
        lambda x: len(x["input_ids"]) >= model_args.max_position_embeddings)

    if not data_args.streaming:
        lm_datasets = lm_datasets.train_test_split(test_size=0.1)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"] if not data_args.streaming else None

    it = iter(train_dataset)
    next(it)
    print(next(iter(it))["input_ids"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    trainer.save_state()


if __name__ == "__main__":
    main()
