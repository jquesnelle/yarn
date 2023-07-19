import os

import datasets
import torch
from lora import peft_model
from scaled_rope.configuration_llama import LlamaConfig
from scaled_rope.modelling_llama import LlamaForCausalLM
from transformers import (LlamaTokenizer, LlamaTokenizerFast, Trainer,
                          TrainingArguments, default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import OptimizerNames
from utilities.config import argument_parsing, rank_zero_info
from utilities.efficiency_utils import fuse_gelu

from data import DataCollator, get_one_dataset


def main():

    training_conf = argument_parsing()
    optimizer = (
        OptimizerNames.ADAMW_BNB
        if training_conf.quantization
        else OptimizerNames.ADAMW_HF
    )

    args = TrainingArguments(
        output_dir=training_conf.output_dir,
        num_train_epochs=training_conf.num_train_epochs,
        warmup_steps=training_conf.warmup_steps,
        learning_rate=float(training_conf.learning_rate),
        deepspeed=training_conf.deepspeed_config if training_conf.deepspeed else None,
        optim=optimizer,
        fp16=training_conf.dtype in ["fp16", "float16"],
        bf16=training_conf.dtype in ["bf16", "bfloat16"],
        local_rank=training_conf.local_rank,
        gradient_checkpointing=training_conf.gradient_checkpointing,
        gradient_accumulation_steps=training_conf.gradient_accumulation_steps,
        per_device_train_batch_size=training_conf.per_device_train_batch_size,
        per_device_eval_batch_size=training_conf.per_device_eval_batch_size,
        adam_beta1=training_conf.adam_beta1,
        adam_beta2=training_conf.adam_beta2,
        adam_epsilon=float(training_conf.adam_epsilon),
        weight_decay=training_conf.weight_decay,
        max_grad_norm=training_conf.max_grad_norm,
        logging_steps=training_conf.logging_steps,
        save_total_limit=training_conf.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=training_conf.eval_steps,
        save_strategy=training_conf.save_strategy,
        save_steps=training_conf.save_steps,
        eval_accumulation_steps=training_conf.eval_accumulation_steps,
        resume_from_checkpoint=training_conf.resume_from_checkpoint,
        report_to="wandb" if training_conf.log_wandb else None,
        ddp_find_unused_parameters=training_conf.ddp_find_unused_parameters,
    )
    if training_conf.multinode:
        device_count = torch.cuda.device_count()
        rank = args.local_rank
        device = rank % device_count
        torch.cuda.set_device(device)
        args.ddp_find_unused_parameters = False
    last_checkpoint = (
        get_last_checkpoint(training_conf.output_dir)
        if os.path.exists(training_conf.output_dir)
        else None
    )
    set_seed(training_conf.seed)

    if "llama" in training_conf.model_name_or_path:
        if training_conf.tokenizer_name is not None:
            tokenizer_name = training_conf.tokenizer_name
        else:
            tokenizer_name = training_conf.model_name_or_path
        tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, add_bos_token=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = training_conf.max_position_embeddings

        config = LlamaConfig.from_pretrained(training_conf.model_name_or_path)
        config.use_xpos = training_conf.use_xpos
        config.max_position_embeddings = training_conf.max_position_embeddings
        config.transformer_engine = training_conf.fp8
        config.ntk_alpha = training_conf.ntk_alpha
        config.part_ntk_scale = training_conf.part_ntk_scale

        if training_conf.max_length is None:
            training_conf.max_length = config.max_position_embeddings
        model = LlamaForCausalLM.from_pretrained(
            training_conf.model_name_or_path,
            torch_dtype=torch.bfloat16
            if training_conf.dtype == "bf16"
            else torch.float16,
            config=config,
        )
        model.max_sequence_length = training_conf.max_position_embeddings

    else:
        raise NotImplementedError

    # patching for the random contiguous tensors bug
    for p in model.parameters():
        p = p.contiguous()

    # Use Flash attn
    if training_conf.flash_patch:
        from scaled_rope.flash_patch import patch_model

        patch_model(
            model,
            resid_pdrop=training_conf.residual_dropout,
            flash_attention=training_conf.use_flash_attention,
            residual_dropout_lima=training_conf.residual_dropout_lima,
        )

    if training_conf.pretokenized is False:
        # "Loads training_conf.dataset_name Text datasets that have been packed with <s> ... </s> but not tokenized
        train_dataset, eval_dataset = get_one_dataset(
            training_conf, max_val_set=training_conf.max_val_set
        )
        collate_fn = DataCollator(
            tokenizer,
            max_length=training_conf.max_length,
            pad_to_multiple_of=16,
        )
    else:
        # Loads pre-tokenized datasets (
        train_dataset = datasets.load_dataset(training_conf.dataset_names[0])
        train_dataset["labels"] = train_dataset[
            "input_ids"
        ].clone()  # For CausalLM LM shifting is done in model forward.
        train_val_split = train_dataset["train"].train_test_split(
            test_size=training_conf.max_val_set, seed=42
        )
        eval_dataset = train_val_split["test"]
        train_dataset = train_val_split["train"]
        collate_fn = default_data_collator

    if training_conf.log_wandb and (
        not training_conf.deepspeed or training_conf.local_rank == 0
    ):
        import wandb

        wandb.init(
            project=training_conf.wandb_project,
            entity=training_conf.wandb_entity,
            resume=training_conf.resume_from_checkpoint,
            name=f"lora-rope-{training_conf.max_position_embeddings}-{training_conf.model_name_or_path.split('/')[-1]}",
            config=training_conf,
        )
        wandb.config["_max_length"] = training_conf.max_length



    if training_conf.fuse_gelu:
        model = fuse_gelu(model)

    if training_conf.lora:
        rank_zero_info("Using PEFT model")
        model = peft_model(
            model,
            model_name=training_conf.model_name_or_path,
            gradient_checkpointing=training_conf.gradient_checkpointing,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )
    if training_conf.local_rank == 0:
        #todo remove debug message
        print("Model....")
        print(model)
        b = next(iter(trainer.get_train_dataloader()))
        print("\nInput shape Check:", b["input_ids"].shape)
        print("\nDecoded batch element:", tokenizer.decode(b["input_ids"][0].tolist()))
        print("\ntokens", b["input_ids"][:5])
        print("tokenizer bos token",tokenizer.bos_token_id,tokenizer.bos_token)
        print("tokenizer eos token",tokenizer.eos_token_id,tokenizer.eos_token)


    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    trainer.save_state()


if __name__ == "__main__":
    main()