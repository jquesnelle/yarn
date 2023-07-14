from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


def prepare_model_for_gradient_checkpointing(model):
    r"""
    Prepares the model for gradient checkpointing if necessary
    """
    if not getattr(model, "is_loaded_in_8bit", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model


def peft_model(model, model_name, int8_training=False, gradient_checkpointing=False):

    if "falcon" in model_name:
        target_modules = ["dense_4h_to_h", "dense", "query_key_value", "dense_h_to_4h"]
        r = 64
    elif "llama" in model_name:
        target_modules = [
            "down_proj",
            "k_proj",
            "q_proj",
            "gate_proj",
            "o_proj",
            "up_proj",
            "v_proj",
        ]
        if "65" in model_name:
            r = 16
        else:
            r = 64
    else:
        raise ValueError(
            f"Invalid model name '{model_name}'. The model name should contain 'falcon' or 'llama'"
        )
    config = LoraConfig(
        r=r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    if int8_training:
        model = prepare_model_for_int8_training(model)

    if gradient_checkpointing:
        model = prepare_model_for_gradient_checkpointing(model)
    model.print_trainable_parameters()
    return model


def load_peft_finetuned_model(model, peft_model_path):
    adapters_weights = torch.load(
        Path(peft_model_path).joinpath("adapter_model.bin"), map_location=model.device
    )
    model.load_state_dict(adapters_weights, strict=False)
    return model