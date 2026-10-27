import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    """Embeds flow-matching time t in [0,1] into a vector of dim `dim`."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        freqs = sinusoidal_embedding(t * 1000.0, self.freq_dim)
        return self.mlp(freqs)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
        return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    # q, k: (B, n_heads, T, head_dim); cos, sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class AdaLNZero(nn.Module):
    """Zero-initialized conditioning projection: cond -> n_params modulation tensors."""

    def __init__(self, cond_dim: int, d_model: int, n_params: int):
        super().__init__()
        self.n_params = n_params
        self.proj = nn.Linear(cond_dim, n_params * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: Tensor) -> Tuple[Tensor, ...]:
        return self.proj(cond).chunk(self.n_params, dim=-1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE. Bidirectional by default (offline,
    whole-segment synthesis); pass causal=True for autoregressive decoding, where
    position t must not attend to t+1.

    An explicit `attn_mask` can be passed to forward() for masks that aren't plain
    lower-triangular causality -- e.g. a local attention window, or a per-string
    block-diagonal causal mask (dataset/note_score_dataset.py's joint "recording" mode
    interleaves multiple strings' notes into one time-sorted sequence, so a note's AR
    history must skip OTHER strings' notes, not just future ones -- see
    model/note_ar_decoder.py). When given, it overrides `causal` for that call; boolean,
    True = attend-allowed, shape broadcastable to (B, n_heads, T, T)."""

    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, T, head_dim)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rotary(q, k, cos, sin)
        if attn_mask is not None:
            if attn_mask.dim() == 3:  # (B,T,T) -> (B,1,T,T), broadcasts over heads
                attn_mask = attn_mask.unsqueeze(1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Pre-norm transformer block modulated per-frame by AdaLN-Zero from a shared conditioning tensor."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, cond_dim: int, rope: RotaryEmbedding, causal: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = SelfAttention(d_model, n_heads, rope, causal=causal)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = MLP(d_model, mlp_ratio)
        self.adaLN = AdaLNZero(cond_dim, d_model, n_params=6)

    def forward(self, x: Tensor, cond: Tensor, attn_mask: Tensor = None) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(cond)
        x = x + gate1 * self.attn(modulate(self.norm1(x), shift1, scale1), attn_mask=attn_mask)
        x = x + gate2 * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, cond_dim: int, d_model: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN = AdaLNZero(cond_dim, d_model, n_params=2)
        self.proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.adaLN(cond)
        return self.proj(modulate(self.norm(x), shift, scale))
