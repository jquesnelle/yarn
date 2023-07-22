"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import importlib
import logging
import math
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import torch
import yaml

# add src to the pythonpath so we don't need to pip install this
from optimum.bettertransformer import BetterTransformer
from transformers import GenerationConfig, TextStreamer, BitsAndBytesConfig

from axolotl.logging_config import configure_logging
from axolotl.utils.data import load_prepare_datasets, load_pretraining_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_tokenizer, load_adapter
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.validation import validate_config
from axolotl.utils.wandb import setup_wandb_env_vars

from scaled_rope.modeling_llama import LlamaForCausalLM
from scaled_rope.configuration_llama import LlamaConfig
#from transformers import LlamaForCausalLM
#from transformers import LlamaConfig

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")


DEFAULT_DATASET_PREPARED_PATH = "last_run_prepared"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def choose_device(cfg):
    def get_device():
        try:
            if torch.cuda.is_available():
                return f"cuda:{cfg.local_rank}"

            if torch.backends.mps.is_available():
                return "mps"

            raise SystemError("No CUDA/mps device found")
        except Exception:  # pylint: disable=broad-exception-caught
            return "cpu"

    cfg.device = get_device()
    if cfg.device_map != "auto":
        if cfg.device.startswith("cuda"):
            cfg.device_map = {"": cfg.local_rank}
        else:
            cfg.device_map = {"": cfg.device}


def get_multi_line_input() -> Optional[str]:
    print("Give me an instruction (Ctrl + D to finish): ")
    instruction = ""
    for line in sys.stdin:
        instruction += line  # pylint: disable=consider-using-join
    # instruction = pathlib.Path("/proc/self/fd/0").read_text()
    return instruction


def do_inference(cfg, model, tokenizer, prompter: Optional[str]):
    default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

    for token, symbol in default_tokens.items():
        # If the token isn't already specified in the config, add it
        if not (cfg.special_tokens and token in cfg.special_tokens):
            tokenizer.add_special_tokens({token: symbol})

    prompter_module = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )

    if cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import set_model_mem_id

        set_model_mem_id(model, tokenizer)
        model.set_mem_cache_args(
            max_seq_len=255, mem_freq=50, top_k=5, max_cache_size=None
        )

    while True:
        print("=" * 80)
        # support for multiline inputs
        instruction = get_multi_line_input()
        if not instruction:
            return
        if prompter_module:
            prompt: str = next(
                prompter_module().build_prompt(instruction=instruction.strip("\n"))
            )
        else:
            prompt = instruction.strip()
        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

        print("=" * 40)
        model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=1024,
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            streamer = TextStreamer(tokenizer)
            generated = model.generate(
                inputs=batch["input_ids"].to(cfg.device),
                generation_config=generation_config,
                streamer=streamer,
            )
        print("=" * 40)
        print(tokenizer.decode(generated["sequences"].cpu().tolist()[0]))


def choose_config(path: Path):
    yaml_files = list(path.glob("*.yml"))

    if not yaml_files:
        raise ValueError(
            "No YAML config files found in the specified directory. Are you using a .yml extension?"
        )

    print("Choose a YAML file:")
    for idx, file in enumerate(yaml_files):
        print(f"{idx + 1}. {file}")

    chosen_file = None
    while chosen_file is None:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(yaml_files):
                chosen_file = yaml_files[choice - 1]
            else:
                print("Invalid choice. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return chosen_file


def check_not_in(list1: List[str], list2: Union[Dict[str, Any], List[str]]) -> bool:
    return not any(el in list2 for el in list1)


def load_model(
    base_model, base_model_config, model_type, tokenizer, cfg, adapter="lora"
):
    # type: (str, str, str, PreTrainedTokenizerBase, DictDefault, Optional[str]) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    """
    Load a model from a base model and a model type.
    """

    assert "llama" in model_type.lower()

    # TODO refactor as a kwarg
    load_in_8bit = cfg.load_in_8bit

    if cfg.bf16 or cfg.bfloat16:
        torch_dtype = torch.bfloat16
    elif cfg.load_in_8bit or cfg.fp16 or cfg.float16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    try:
        from peft import prepare_model_for_kbit_training
    except ImportError:
        # For backward compatibility
        from peft import (
            prepare_model_for_int8_training as prepare_model_for_kbit_training,
        )

    model_kwargs = {}
    if cfg.model_revision:
        model_kwargs["revision"] = cfg.model_revision
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    config = LlamaConfig.from_pretrained(
        base_model,
        trust_remote_code=cfg.trust_remote_code or False,
    )
    # Shouldn't be a problem most of the time. will obviously error if the model doesn't support this
    # when training starts
    if (
        hasattr(config, "max_seq_len")
        and config.max_seq_len
        and cfg.sequence_len > config.max_seq_len
    ):
        config.max_seq_len = cfg.sequence_len
        LOG.warning(f"increasing context length to {cfg.sequence_len}")
    elif (
        hasattr(config, "max_sequence_length")
        and config.max_sequence_length
        and cfg.sequence_len > config.max_sequence_length
    ):
        config.max_sequence_length = cfg.sequence_len
        LOG.warning(f"increasing context length to {cfg.sequence_len}")
    if hasattr(cfg, "use_flash_attention"):
        config.use_flash_attention = cfg.use_flash_attention
    if hasattr(cfg, "rope_scaling"):
        config.rope_scaling = cfg.rope_scaling
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        config=config,
        load_in_8bit=cfg.load_in_8bit and cfg.adapter is not None,
        load_in_4bit=cfg.load_in_4bit and cfg.adapter is not None,
        torch_dtype=torch_dtype,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code or False,
        **model_kwargs,
    )

    embeddings_len = math.ceil(len(tokenizer) / 32) * 32
    model.resize_token_embeddings(embeddings_len)

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings
        and cfg.sequence_len >= model.config.max_position_embeddings
    ):
        LOG.warning(
            f"increasing model.config.max_position_embeddings to {cfg.sequence_len}"
        )
        model.config.max_position_embeddings = cfg.sequence_len

    if not cfg.gptq and (
        (cfg.adapter == "lora" and load_in_8bit)
        or (cfg.adapter == "qlora" and cfg.load_in_4bit)
    ):
        LOG.info("converting PEFT model w/ prepare_model_for_kbit_training")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

    model, lora_config = load_adapter(model, cfg, adapter)

    if cfg.ddp and not load_in_8bit:
        model.to(f"cuda:{cfg.local_rank}")

    if (
        torch.cuda.device_count() > 1
        and int(os.getenv("WORLD_SIZE", "1")) > 1
        and (cfg.gptq or cfg.load_in_4bit)
    ):
        # llama is PROBABLY model parallelizable, but the default isn't that it is
        # so let's only set it for the 4bit, see
        # https://github.com/johnsmith0031/alpaca_lora_4bit/blob/08b3fca4a4a9e0d3945be1bab4529f100a428636/finetune.py#L130-L133
        setattr(model, "is_parallelizable", True)
        setattr(model, "model_parallel", True)

    requires_grad = []
    for name, param in model.named_parameters(recurse=True):
        if param.requires_grad:
            requires_grad.append(f"{name}: {param.requires_grad}")
    if len(requires_grad) == 0:
        LOG.warning("there are no parameters that require gradient updates")
    model.config.use_cache = False

    if cfg.flash_optimum:
        model = BetterTransformer.transform(model)

    # TODO resume_from_checkpoint handling
    return model, lora_config

def train(
    config: Path = Path("configs/"),
    prepare_ds_only: bool = False,
    **kwargs,
):
    if Path(config).is_dir():
        config = choose_config(config)

    # load the config from the yaml file
    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    validate_config(cfg)

    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps or (
        cfg.batch_size // cfg.micro_batch_size
    )
    cfg.batch_size = (
        cfg.batch_size or cfg.micro_batch_size * cfg.gradient_accumulation_steps
    )
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    choose_device(cfg)
    cfg.ddp = cfg.ddp if cfg.ddp is not None else cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.batch_size = cfg.batch_size * cfg.world_size

    setup_wandb_env_vars(cfg)
    if cfg.device == "mps":
        cfg.load_in_8bit = False
        cfg.tf32 = False
        if cfg.bf16:
            cfg.fp16 = True
        cfg.bf16 = False

    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # load the tokenizer first
    tokenizer_config = cfg.tokenizer_config or cfg.base_model_config
    LOG.info(f"loading tokenizer... {tokenizer_config}")
    tokenizer = load_tokenizer(tokenizer_config, cfg.tokenizer_type, cfg)

    if (
        check_not_in(["shard", "merge_lora"], kwargs) and not cfg.inference
    ):  # don't need to load dataset for these
        if not cfg.pretraining_dataset:
            train_dataset, eval_dataset = load_prepare_datasets(
                tokenizer, cfg, DEFAULT_DATASET_PREPARED_PATH
            )
        else:
            train_dataset = load_pretraining_dataset(
                cfg.pretraining_dataset,
                tokenizer,
                max_tokens=cfg.sequence_len,
                seed=cfg.seed,
            )
            # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
            train_dataset = train_dataset.with_format("torch")
            eval_dataset = None

    if cfg.debug or "debug" in kwargs:
        LOG.info("check_dataset_labels...")
        check_dataset_labels(
            train_dataset.select(
                [random.randrange(0, len(train_dataset) - 1) for _ in range(5)]  # nosec
            ),
            tokenizer,
        )

    if prepare_ds_only:
        LOG.info("Finished preparing dataset. Exiting...")
        return

    # Load the model and tokenizer
    LOG.info("loading model and peft_config...")
    model, peft_config = load_model(
        cfg.base_model,
        cfg.base_model_config,
        cfg.model_type,
        tokenizer,
        cfg,
        adapter=cfg.adapter,
    )

    if "merge_lora" in kwargs and cfg.adapter is not None:
        LOG.info("running merge of LoRA with base model")
        model = model.merge_and_unload()
        model.to(dtype=torch.float16)

        if cfg.local_rank == 0:
            LOG.info("saving merged model")
            model.save_pretrained(str(Path(cfg.output_dir) / "merged"))
        return

    if cfg.inference:
        LOG.info("calling do_inference function")
        prompter: Optional[str] = "AlpacaPrompter"
        if "prompter" in kwargs:
            if kwargs["prompter"] == "None":
                prompter = None
            else:
                prompter = kwargs["prompter"]
        do_inference(cfg, model, tokenizer, prompter=prompter)
        return

    if "shard" in kwargs:
        model.save_pretrained(cfg.output_dir)
        return

    trainer = setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer)

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        LOG.info("Compiling torch model")
        model = torch.compile(model)

    # go ahead and presave, so we have the adapter config available to inspect
    if peft_config:
        LOG.info(f"Pre-saving adapter config to {cfg.output_dir}")
        peft_config.save_pretrained(cfg.output_dir)

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:

        def terminate_handler(_, __, model):
            if cfg.flash_optimum:
                model = BetterTransformer.reverse(model)
            model.save_pretrained(cfg.output_dir)
            sys.exit(0)

        signal.signal(
            signal.SIGINT, lambda signum, frame: terminate_handler(signum, frame, model)
        )

    LOG.info("Starting trainer...")
    if cfg.group_by_length:
        LOG.info("hang tight... sorting dataset for group_by_length")
    resume_from_checkpoint = cfg.resume_from_checkpoint
    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        possible_checkpoints = [
            str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")
        ]
        if len(possible_checkpoints) > 0:
            sorted_paths = sorted(
                possible_checkpoints,
                key=lambda path: int(path.split("-")[-1]),
            )
            resume_from_checkpoint = sorted_paths[-1]
            LOG.info(
                f"Using Auto-resume functionality to start with checkpoint at {resume_from_checkpoint}"
            )

    if not Path(cfg.output_dir).is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    LOG.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.local_rank == 0:
        if cfg.flash_optimum:
            model = BetterTransformer.reverse(model)
        model.save_pretrained(cfg.output_dir)

    # trainer.save_model(cfg.output_dir)  # TODO this may be needed for deepspeed to work? need to review another time


if __name__ == "__main__":
    fire.Fire(train)