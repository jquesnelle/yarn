import argparse
import copy
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from tqdm import tqdm
from transformers import set_seed, default_data_collator, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout]
    )
    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "yarn",
    )
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    if args.architecture == "llama":
        from scaled_rope.modeling_llama_yarn import LlamaForCausalLM
        from scaled_rope.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
        original_max_position_embeddings = args.original_max_position_embeddings if args.original_max_position_embeddings else 4096
    elif args.architecture == "mistral":
        from scaled_rope.modeling_mistral_yarn import MistralForCausalLM
        from scaled_rope.configuration_mistral import MistralConfig
        config_cls = MistralConfig
        model_cls = MistralForCausalLM
        original_max_position_embeddings = args.original_max_position_embeddings if args.original_max_position_embeddings else 8192

    config = config_cls.from_pretrained(args.model)
    config.rope_scaling = {
        "type": args.scaling_type,
        "factor": args.scaling_factor,
        "original_max_position_embeddings": original_max_position_embeddings
    }
    config.rope_theta = args.rope_theta
    config.max_position_embeddings = int(args.scaling_factor * original_max_position_embeddings) \
        if not args.max_position_embeddings else args.max_position_embeddings

    sliding_window_attention_schedule = [int(x) for x in args.sliding_window_attention_schedule.split(",")] \
        if args.sliding_window_attention_schedule else None
    if sliding_window_attention_schedule is not None and len(sliding_window_attention_schedule) == 1:
        config.sliding_window = sliding_window_attention_schedule[0]
        accelerator.print(
            f"Sliding attention window set to {config.sliding_window}")

    model = model_cls.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        config=config,
        use_flash_attention_2=True
    )

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    if "labels" not in train_dataset.column_names:
        def add_labels(sample):
            sample["labels"] = copy.deepcopy(sample["input_ids"])
            return sample
        train_dataset = train_dataset.map(
            add_labels, desc="Adding labels", num_proc=args.num_proc)
    if "attention_mask" not in train_dataset.column_names:
        def add_attention_mask(sample):
            sample["attention_mask"] = torch.ones(
                len(sample["input_ids"]), dtype=torch.int8)
            return sample
        train_dataset = train_dataset.map(
            add_attention_mask, desc="Adding attention mask", num_proc=args.num_proc)

    if args.truncate:
        def truncate(sample):
            sample["input_ids"] = sample["input_ids"][0:args.truncate]
            sample["labels"] = sample["labels"][0:args.truncate]
            sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
            return sample
        train_dataset = train_dataset.map(
            truncate, desc="Truncating", num_proc=args.num_proc)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size
    )

    if args.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        target_modules = find_all_linear_names(model)
        accelerator.print(f"LoRA target modules: {target_modules}")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                                 r=16, lora_alpha=64, lora_dropout=0.05, target_modules=target_modules)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if args.deepspeed:
        optim = DummyOptim(model.parameters(), lr=args.learning_rate)
        scheduler = DummyScheduler(
            optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
        model, optim, train_loader, scheduler = accelerator.prepare(
            model, optim, train_loader, scheduler
        )
    else:
        model = accelerator.prepare(model)
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if args.lr_schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
        elif args.lr_schedule == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optim, num_warmup_steps=args.warmup_steps)
        optim, train_loader, scheduler = accelerator.prepare(
            optim, train_loader, scheduler)

    if not args.lora:
        model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    )

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    accelerator.print(f"Total batch size: {total_batch_size}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        resume_step = (
            int(training_difference.replace("step_", ""))
        )

    if args.resume_from_checkpoint and resume_step is not None:
        train_loader = accelerator.skip_first_batches(
            train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    loss_file = open(args.log_loss, "a" if args.resume_from_checkpoint else "w") if args.log_loss and accelerator.is_main_process else None

    if not args.save_only:
        model.train()
        for step, batch in enumerate(train_loader):
            if sliding_window_attention_schedule is not None:
                model.config.sliding_window = sliding_window_attention_schedule[completed_steps % len(
                    sliding_window_attention_schedule)]

            loss_log = None
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    loss_log = {"loss": loss.item()}
                    accelerator.log(loss_log, step=completed_steps)
                    if loss_file is not None:
                        loss_file.write(f"{loss_log['loss']},")
                        loss_file.flush()
                    if isinstance(args.grad_norm, float):
                        accelerator.clip_grad_norm_(
                            model.parameters(), args.grad_norm)

                optim.step()
                scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if loss_log is not None:
                    progress_bar.set_postfix(loss_log)
                completed_steps += 1

                if isinstance(args.checkpointing_steps, int) and completed_steps > 0:
                    if completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        accelerator.print(f"Training Finished")
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()

        if args.deepspeed:
            state_dict = accelerator.get_state_dict(model)
        else:
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = accelerator.get_state_dict(model, unwrap=False)

        accelerator.unwrap_model(model).save_pretrained(
            f"{args.output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--grad-norm", action="store_true")
    args.add_argument("--lora", action="store_true")
    args.add_argument("--model", type=str,
                      default="NousResearch/Llama-2-7b-hf")
    args.add_argument("--scaling-factor", type=float, default=16.0)
    args.add_argument("--scaling-type", type=str, default="yarn")
    args.add_argument("--rope-theta", type=float, default=10000.0)
    args.add_argument("--truncate", type=int)
    args.add_argument("--dataset", type=str,
                      default="emozilla/pg_books-tokenized-bos-eos-chunked-65536")
    args.add_argument("--deepspeed", action="store_true")
    args.add_argument("--num-proc", type=int, default=32)
    args.add_argument("--architecture", type=str,
                      choices=["llama", "mistral"], default="llama")
    args.add_argument("--max-position-embeddings", type=int)
    args.add_argument("--sliding-window-attention-schedule", type=str)
    args.add_argument("--lr-schedule", type=str,
                      choices=["linear", "constant"], default="linear")
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--original-max-position-embeddings", type=int)
    main(args.parse_args())
