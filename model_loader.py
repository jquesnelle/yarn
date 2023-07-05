from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from scaled_rope.patch import *

def load_model(model, load_in_8bit, load_in_4bit, length):
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if length and "MPTForCausalLM" in config.architectures:
        config.max_seq_len = max(length, config.max_seq_len)

    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
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

def apply_patches(loaded, length, dynamic_ntk, dynamic_linear, ntk, linear):
    if "GPTNeoXForCausalLM" in loaded.config.architectures:
        patch_gptneox_for_longer_sequences(loaded, length)
    if dynamic_linear:
        if "GPTNeoXForCausalLM" in loaded.config.architectures:
            patch_gptneox_for_scaled_rotary_embeddings(loaded)
        elif "LlamaForCausalLM" in loaded.config.architectures:
            patch_llama_for_dynamic_scaled_rotary_embeddings(loaded)
        else:
            raise RuntimeError(
                f"Unsupported architecture {loaded.config.architectures} for dyanmic linear")
    elif dynamic_ntk:
        if "LlamaForCausalLM" in loaded.config.architectures:
            patch_llama_for_dynamic_scaled_rotary_embeddings(loaded, ntk=dynamic_ntk)
        else:
            raise RuntimeError(
                f"Unsupported architecture {loaded.config.architectures} for dyanmic ntk")
    elif ntk:
        if "GPTNeoXForCausalLM" in loaded.config.architectures:
            patch_gptneox_for_ntk_scaled_rotary_embeddings(
                loaded, ntk)
        elif "LlamaForCausalLM" in loaded.config.architectures:
            patch_llama_for_ntk_scaled_rotary_embeddings(loaded, ntk)
        else:
            raise RuntimeError(
                f"Unsupported architecture {loaded.config.architectures} for ntk")
    elif linear:
        if "LlamaForCausalLM" in loaded.config.architectures:
            patch_llama_for_linear_scaled_rotary_embeddings(loaded, scale=linear)
        else:
            raise RuntimeError(
                f"Unsupported architecture {loaded.config.architectures} for linear")
