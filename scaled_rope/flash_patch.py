import math
import warnings
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scaled_rope.modelling_llama import (LlamaAttention, LlamaForCausalLM,
                                         LlamaModel)


def compute_flash_attention(flash_attn, q, k, v, attention_mask=None, head_mask=None):
    # q, k, v: [bs, seq_len, num_attention_heads, attn_head_size]
    # attention_mask (float): [bs, seq_len]
    batch_size, max_len = q.size(0), q.size(1)

    qkv = torch.stack([q, k, v], dim=2).to(
        torch.float16
    )  # need to truncate in case input is fp32
    cu_seqlens, max_seqlen = None, None

    if attention_mask is None:
        return flash_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    else:
        # Limitation: non-contiguous attention mask will not be handled correctly
        # model will be able to pay attention between the first and last non-masked token, i.e. left- and right-side padding is supported.
        csums = (attention_mask >= 0).cumsum(dim=1)
        ends = csums.argmax(dim=1) + 1
        starts = ends - csums.max(dim=1).values
        seqlens = ends - starts

        qkv = torch.cat([qkv[i, starts[i] : ends[i]] for i in range(batch_size)], dim=0)
        zero = torch.zeros_like(
            seqlens[:1]
        )  # torch.tensor([0]) with correct dtype and device
        cu_seqlens = torch.cat([zero, seqlens.cumsum(dim=0)], dim=0).to(torch.int32)
        max_seqlen = seqlens.max().item()

        out = flash_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        # out: [num_unmasked_tokens, num_attention_heads, attn_head_size]

        seqs = [out[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        # stack and pad sequences together
        padded_seqs = [
            F.pad(
                seqs[i],
                (0, 0) * (seqs[i].dim() - 1) + (starts[i], max_len - ends[i]),
                value=0.0,
            )
            for i in range(batch_size)
        ]
        out = torch.stack(padded_seqs)
        return out


def llama_forward_with_flash_attn(
    self: LlamaAttention,
    flash_attn: nn.Module,  # flash_attn.modules.mha.FlashSelfAttention
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # self: transformers.models.llama.modeling_llama.LlamaAttention
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    if (
        query_states.shape == key_states.shape
    ):  # and (attention_mask is None or attention_mask[:, 0, -1, 0].min() >= 0):
        if attention_mask is not None:
            attention_mask = attention_mask[:, 0, -1]

        flash_attn.train(self.training)
        out_dtype = value_states.dtype
        q, k, v = (
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
        )
        attn_output = compute_flash_attention(flash_attn, q, k, v, attention_mask)
        attn_output = attn_output.transpose(1, 2).to(out_dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
    else:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def add_dropout(module: nn.Module, patched_fwd: Callable, p_dropout: float = 0.1):
    dropout = nn.Dropout(p=p_dropout)
    module.old_forward = module.forward
    module.forward = partial(patched_fwd, dropout, module)


def add_flash_attn(module: nn.Module, causal: bool = True):
    """
    Replaces the standard attention implementation with Flash Attention [1].
    Limitations:
      - Only works for fp16 or bf16 inputs
      - Requires inputs to be on CUDA
      - `output_attentions=True` does not work after patching, attention weights will be None
      - Non-contiguous attention masks are not supported (e.g. [1, 1, 0, 1, 1, 0, 0] will just become [1, 1, 1, 1, 1, 0, 0]).

    [1] https://github.com/HazyResearch/flash-attention
    """

    flash_attn = FlashSelfAttention(causal=causal)
    if isinstance(module, LlamaAttention):
        module.old_forward = module.forward
        module.forward = partial(llama_forward_with_flash_attn, module, flash_attn)


def _patched_mlp_forward(post_module: nn.Module, module: nn.Module, *args, **kwargs):
    post_module.train(module.training)
    out = module.old_forward(*args, **kwargs)
    out = post_module(out)
    return out


def _patched_attn_forward(post_module: nn.Module, module: nn.Module, *args, **kwargs):
    post_module.train(module.training)
    out = module.old_forward(*args, **kwargs)
    hiddens = post_module(out[0])
    return (hiddens,) + out[1:]


def patch_model(
    model: nn.Module,
    resid_pdrop: Optional[float] = 0.1,
    flash_attention: bool = True,
    patch_unsupported: bool = False,
    residual_dropout_lima: bool = False,
):
    """
    Helper function for patching HF language models.
    Currently supports: GPTNeoX-based models

    Limitations:
      - Flash attention requires CUDA and fp16/bf16 training. It also requires contiguous attention masks.
      - Residual dropout does not support multi-GPU training without DeepDpeed.
    """
    global FlashSelfAttention
    if flash_attention:
        try:
            from flash_attn.modules.mha import \
                FlashSelfAttention  # pyright: reportMissingImports=false
        except ModuleNotFoundError:
            warnings.warn(
                """\nmodule flash_attn not found - either install:
  pip3 install flash_attn
or run with:
  --use_flash_attention=false """
            )
            exit(1)

    if isinstance(model, LlamaForCausalLM):
        model = model.model

    if model.__class__.__name__ == "RWForCausalLM":
        model = model.base_model

    attention_key_lookup = {
        LlamaModel: "self_attn",
    }
    mlp_key_lookup = {
        LlamaModel: "mlp",
    }
    if model.__class__.__name__ == "RWModel":
        layers = model.h
        attention_key = "self_attention"
        mlp_key = "mlp"
    else:
        layers = model.layers
        attention_key = attention_key_lookup.get(model.__class__, "attention")
        mlp_key = mlp_key_lookup.get(model.__class__, "mlp")
    num_layers = len(layers)
    resid_pdrop_last_layer = resid_pdrop
    for i, layer in enumerate(layers):
        if flash_attention:
            add_flash_attn(getattr(layer, attention_key), causal=True)
        if residual_dropout_lima:
            resid_pdrop = i / (num_layers - 1) * resid_pdrop_last_layer
        if resid_pdrop is not None and resid_pdrop > 0:
            add_dropout(
                getattr(layer, attention_key), _patched_attn_forward, resid_pdrop
            )
            add_dropout(getattr(layer, mlp_key), _patched_mlp_forward, resid_pdrop)