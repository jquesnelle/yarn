import torch


def patch_llama_for_scaled_rotary_embeddings(model):
    from .LlamaScaledRotaryEmbedding import LlamaRotaryEmbedding
    for each in model.model.layers:
        each.self_attn.rotary_emb = LlamaRotaryEmbedding(
            each.self_attn.head_dim, device=each.self_attn.rotary_emb.inv_freq.device)


def patch_gptneox_for_scaled_rotary_embeddings(model):
    from .GPTNeoXScaledRotaryEmbedding import GPTNeoXScaledRotaryEmbedding
    for each in model.gpt_neox.layers:
        each.attention.rotary_emb = GPTNeoXScaledRotaryEmbedding(
            each.attention.rotary_ndims, model.config.max_position_embeddings, device=each.attention.rotary_emb.inv_freq.device)


def patch_gptneox_for_longer_sequences(model, max_positions):
    for each in model.gpt_neox.layers:
        each.attention.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=each.attention.bias.dtype, device=each.attention.bias.device)).view(
            1, 1, max_positions, max_positions
        )
