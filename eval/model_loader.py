from argparse import ArgumentParser
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from scaled_rope.patch import *


def load_model(model, args):
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)

    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = None
    else:
        quantization_config = None
        torch_dtype = torch.bfloat16

    loaded = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        config=config,
        quantization_config=quantization_config
    )

    return loaded


def add_args(parser: ArgumentParser):
    parser.add_argument("--dynamic-linear", action="store_true")
    parser.add_argument("--dynamic-ntk", type=float)
    parser.add_argument("--dynamic-part-ntk", action="store_true")
    parser.add_argument("--ntk", type=float)
    parser.add_argument("--part-ntk", type=float)
    parser.add_argument("--linear", type=float)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--gpt-neox-max-length", type=int)
    return parser


def apply_patches(model, args):
    if "GPTNeoXForCausalLM" in model.config.architectures:
        assert args.gpt_neox_max_length is not None
        patch_gptneox_for_longer_sequences(model, args.gpt_neox_max_length)
    if args.dynamic_linear:
        if "GPTNeoXForCausalLM" in model.config.architectures:
            patch_gptneox_for_scaled_rotary_embeddings(model)
        elif "LlamaForCausalLM" in model.config.architectures:
            patch_llama_for_dynamic_scaled_rotary_embeddings(model)
        else:
            raise RuntimeError(
                f"Unsupported architecture {model.config.architectures} for dyanmic linear")
    elif args.dynamic_ntk:
        if "LlamaForCausalLM" in model.config.architectures:
            patch_llama_for_dynamic_scaled_rotary_embeddings(
                model, ntk=args.dynamic_ntk)
        else:
            raise RuntimeError(
                f"Unsupported architecture {model.config.architectures} for dyanmic ntk")
    elif args.dynamic_part_ntk:
        if "LlamaForCausalLM" in model.config.architectures:
            patch_llama_for_dynamic_part_ntk_rotary_embeddings(
                model, args.finetuned)
        elif "RWForCausalLM" in model.config.architectures:
            patch_falcon_for_dynamic_part_ntk_rotary_embeddings(model)
        else:
            raise RuntimeError(
                f"Unsupported architecture {model.config.architectures} for dyanmic part ntk")
    elif args.ntk:
        if "GPTNeoXForCausalLM" in model.config.architectures:
            patch_gptneox_for_ntk_scaled_rotary_embeddings(
                model, args.ntk)
        elif "LlamaForCausalLM" in model.config.architectures:
            patch_llama_for_ntk_scaled_rotary_embeddings(model, args.ntk)
        else:
            raise RuntimeError(
                f"Unsupported architecture {model.config.architectures} for ntk")
    elif args.linear:
        if "LlamaForCausalLM" in model.config.architectures:
            patch_llama_for_linear_scaled_rotary_embeddings(
                model, scale=args.linear)
        else:
            raise RuntimeError(
                f"Unsupported architecture {model.config.architectures} for linear")
    elif args.part_ntk:
        if "LlamaForCausalLM" in model.config.architectures:
            patch_llama_for_part_ntk_scaled_rotary_embeddings(
                model, scale=args.part_ntk)
        else:
            raise RuntimeError(
                f"Unsupported architecture {model.config.architectures} for part ntk")
    return model


def load_model_and_apply_patches(model, args):
    return apply_patches(load_model(model, args), args)
