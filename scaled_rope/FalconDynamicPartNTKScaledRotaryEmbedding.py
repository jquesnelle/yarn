import torch
import math

# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0

def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base)) #Inverse dim formula to find number of rotations

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_factor(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_factor(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1) #Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001 #Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def find_newbase_ntk(dim, base=10000, scale=1):
    return base * scale ** (dim / (dim-2))

class FalconDynamicPartNTKScaledRotaryEmbedding(torch.nn.Module):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """

    def __init__(
        self,
        head_dim: int,
        base=10000,
        max_position_embeddings=2048,
        ntk_factor=1,
        extrapolation_factor=1,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None
        self.base = base
        self.ntk_factor = ntk_factor
        self.extrapolation_factor = extrapolation_factor
        self.max_position_embeddings = max_position_embeddings

    def cos_sin(
        self,
        seq_len: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len

            if seq_len >= self.max_position_embeddings:
                #Interpolation constants found experimentally for LLaMA (might not be totally optimal though)
                #Do not change unless there is a good reason for doing so!
                beta_0 = 1.25
                beta_1 = 0.75
                gamma_0 = 16
                gamma_1 = 2

                # the "dynamic" part
                scale = seq_len / self.max_position_embeddings

                #Three RoPE extrapolation/interpolation methods
                inv_freq_base = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
                inv_freq_linear = 1.0 / (scale * (self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim)))
                inv_freq_ntk = 1.0 / (find_newbase_ntk(self.head_dim, self.base, scale) ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
                
                #Combine NTK and Linear
                low, high = find_correction_range(beta_0, beta_1, self.head_dim, self.base, self.max_position_embeddings)
                inv_freq_mask = (1 - linear_ramp_mask(low, high, self.head_dim // 2).type(dtype).to(device)) * self.ntk_factor
                inv_freq = inv_freq_linear * (1 - inv_freq_mask) + inv_freq_ntk * inv_freq_mask
            
                #Combine Extrapolation and NTK and Linear
                low, high = find_correction_range(gamma_0, gamma_1, self.head_dim, self.base, self.max_position_embeddings)
                inv_freq_mask = (1 - linear_ramp_mask(low, high, self.head_dim // 2).type(dtype).to(device)) * self.extrapolation_factor
                inv_freq = inv_freq * (1 - inv_freq_mask) + inv_freq_base * inv_freq_mask

                self.register_buffer("inv_freq", inv_freq, persistent=False)

            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)