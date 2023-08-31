import torch


def patch_llama_for_dynamic_scaled_rotary_embeddings(model, ntk):
    from .LlamaDynamicScaledRotaryEmbedding import LlamaDynamicScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaDynamicScaledRotaryEmbedding(
            each.self_attn.head_dim, device=each.self_attn.rotary_emb.inv_freq.device, ntk=ntk)
        
def patch_llama_for_dynamic_part_ntk_rotary_embeddings(model, finetuned):
    from .LlamaDynamicPartNTKScaledRotaryEmbedding import LlamaDynamicPartNTKScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaDynamicPartNTKScaledRotaryEmbedding(
            each.self_attn.head_dim, finetuned=finetuned, device=each.self_attn.rotary_emb.inv_freq.device)
        
def patch_llama_for_dynamic_yarn_rotary_embeddings(model, original_max_position_embeddings, finetuned):
    from .LlamaDynamicYaRNScaledRotaryEmbedding import LlamaDynamicYaRNScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(
            each.self_attn.head_dim, finetuned=finetuned, original_max_position_embeddings=original_max_position_embeddings, device=each.self_attn.rotary_emb.inv_freq.device)

def patch_falcon_for_dynamic_part_ntk_rotary_embeddings(model):
    from .FalconDynamicPartNTKScaledRotaryEmbedding import FalconDynamicPartNTKScaledRotaryEmbedding
    for each in model.transformer.h:
        each.self_attention.maybe_rotary = FalconDynamicPartNTKScaledRotaryEmbedding(each.self_attention.head_dim)

def patch_llama_for_ntk_scaled_rotary_embeddings(model, alpha):
    from .LlamaNTKScaledRotaryEmbedding import LlamaNTKScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaNTKScaledRotaryEmbedding(
            each.self_attn.head_dim, alpha=alpha, device=each.self_attn.rotary_emb.inv_freq.device)


def patch_llama_for_linear_scaled_rotary_embeddings(model, scale):
    from .LlamaLinearScaledRotaryEmbedding import LlamaLinearScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaLinearScaledRotaryEmbedding(
            each.self_attn.head_dim, scale=scale, device=each.self_attn.rotary_emb.inv_freq.device)


def patch_llama_for_part_ntk_scaled_rotary_embeddings(model, scale):
    from .LlamaPartNTKScaledRotaryEmbedding import LlamaPartNTKScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaPartNTKScaledRotaryEmbedding(
            each.self_attn.head_dim, scale=scale, device=each.self_attn.rotary_emb.inv_freq.device)
        
def patch_llama_for_yarn_scaled_rotary_embeddings(model, scale, original_max_position_embeddings):
    from .LlamaYaRNScaledRotaryEmbedding import LlamaYaRNScaledRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
            each.self_attn.head_dim, scale=scale, original_max_position_embeddings=original_max_position_embeddings, device=each.self_attn.rotary_emb.inv_freq.device)

def patch_gptneox_for_scaled_rotary_embeddings(model):
    from .GPTNeoXDynamicScaledRotaryEmbedding import GPTNeoXDynamicScaledRotaryEmbedding
    for each in model.gpt_neox.layers:
        each.attention.rotary_emb = GPTNeoXDynamicScaledRotaryEmbedding(
            each.attention.rotary_ndims, model.config.max_position_embeddings, device=each.attention.rotary_emb.inv_freq.device)


def patch_gptneox_for_ntk_scaled_rotary_embeddings(model, alpha):
    from .GPTNeoXNTKScaledRotaryEmbedding import GPTNeoXNTKScaledRotaryEmbedding
    for each in model.gpt_neox.layers:
        each.attention.rotary_emb = GPTNeoXNTKScaledRotaryEmbedding(
            each.attention.rotary_ndims, model.config.max_position_embeddings, alpha=alpha, device=each.attention.rotary_emb.inv_freq.device)


def patch_gptneox_for_longer_sequences(model, max_positions):
    for each in model.gpt_neox.layers:
        each.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=each.attention.bias.dtype, device=each.attention.bias.device)).view(
            1, 1, max_positions, max_positions
        )

def patch_llama_for_rerope(model, training_length, window):
    from .LlamaReRoPE import forward_with_rerope
    for each in model.model.layers:
        def forward(*args, **kwargs):
            return forward_with_rerope(each.self_attn, *args, **kwargs)

        each.self_attn.training_length = int(training_length)
        each.self_attn.window = int(window)
        # each.self_attn.forward = forward