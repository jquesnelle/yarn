import torch
import math


def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
    # Inverse dim formula to find number of rotations
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_factor(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_factor(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base=10000, scale=1):
    return base * scale ** (dim / (dim-2))


class LlamaDynamicPartNTKScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, original_max_position_embeddings=2048, base=10000, ntk_factor=1, extrapolation_factor=1, finetuned=False, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.ntk_factor = ntk_factor
        self.extrapolation_factor = extrapolation_factor
        self.max_position_embeddings = max_position_embeddings
        if finetuned:
            self.ntk(self.max_position_embeddings / original_max_position_embeddings, device)
        else:
            inv_freq = 1.0 / \
                (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
            self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached,
                         device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[
                             None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[
                             None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            self.ntk(seq_len / self.max_position_embeddings, x.device)

            t = torch.arange(self.max_seq_len_cached,
                             device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[
                                 None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[
                                 None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def ntk(self, scale, device):

        # Interpolation constants found experimentally for LLaMA (might not be totally optimal though)
        # Do not change unless there is a good reason for doing so!
        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        # Three RoPE extrapolation/interpolation methods
        inv_freq_base = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        inv_freq_linear = 1.0 / \
            (scale * (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)))
        inv_freq_ntk = 1.0 / (find_newbase_ntk(self.dim, self.base, scale)
                              ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        current_dtype = inv_freq_ntk.dtype
        current_device = inv_freq_ntk.device

        # Combine NTK and Linear
        low, high = find_correction_range(
            beta_0, beta_1, self.dim, self.base, self.max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim //
                         2).type(current_dtype).to(current_device)) * self.ntk_factor
        inv_freq = inv_freq_linear * \
            (1 - inv_freq_mask) + inv_freq_ntk * inv_freq_mask

        # Combine Extrapolation and NTK and Linear
        low, high = find_correction_range(
            gamma_0, gamma_1, self.dim, self.base, self.max_position_embeddings)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).type(
            current_dtype).to(current_device)) * self.extrapolation_factor
        inv_freq = inv_freq * (1 - inv_freq_mask) + \
            inv_freq_base * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)
