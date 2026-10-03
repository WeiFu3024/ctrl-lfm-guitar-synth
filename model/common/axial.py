# Axial cross-string attention: the machinery that lets six strings interact, shared by the
# synth (model/synth/joint_synth.py) and the performance predictor (model/perf/perf_predictor.py).
#
# It lived in joint_synth.py until model/ was split by stage, at which point stage 2 importing
# a private name out of stage 1 stopped being defensible. Sharing the blocks is deliberate and
# load-bearing -- the two stages must not drift apart in how they treat the string axis -- so
# the fix is a shared home, not a copy.
#
# Alternate a within-string temporal block (RoPE over T) with a cross-string block over the
# 6-string axis (no RoPE: string index is identity, not position). O(S*T^2 + T*S^2) instead of
# O((S*T)^2) -- about 6x cheaper at S=6, T=256. Validated in
# experiments/frame_rate_perf_decoder/perf_decoder_joint.py.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.common.dit import MLP, RotaryEmbedding, SelfAttention

N_STRINGS = 6


class _CrossStringAttention(nn.Module):
    """Bidirectional attention across the 6-string axis at a fixed timestep. No RoPE:
    string index is not a sequential quantity, and identity arrives as a learned
    embedding at the input (same reasoning as perf_decoder_joint.py's version)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:  # (N, S, D)
        N, S, D = x.shape
        qkv = self.qkv(x).reshape(N, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.out_proj(out.transpose(1, 2).reshape(N, S, D))


class _GatedCrossStringBlock(nn.Module):
    """Cross-string attention + MLP behind a single zero-initialised scalar gate, so at
    init the block is exactly the identity and the model must choose to open it.

    The gate is not decoration. The earlier frame-rate joint decoder mixed across strings
    unconditionally at every layer and the result was lopsided: envelope correlation
    improved (0.564 -> 0.606) while pitch collapsed (0.700 -> 0.317), diagnosed as mixing
    with no way to switch itself off. Zero-init makes "off" the default and leaves the
    learned gate directly inspectable afterwards -- same device as
    model/note_ar_decoder.py's _GatedCrossStringBlock and model/common/dit.py's AdaLNZero.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _CrossStringAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:  # (N, S, D)
        x = x + self.gate * self.attn(self.norm1(x))
        x = x + self.gate * self.mlp(self.norm2(x))
        return x


class _WithinStringBlock(nn.Module):
    """Plain pre-norm temporal self-attention (RoPE) + MLP, for the conditioning encoder
    where there is nothing external to modulate on (cf. model/synth/signal.py's _EncoderBlock)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, rope: RotaryEmbedding):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)

    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def _to_strings(x: Tensor, S: int) -> Tensor:
    """(B, S*T, D) -> (B, S, T, D)."""
    B, L, D = x.shape
    return x.reshape(B, S, L // S, D)


def _flatten_strings(x: Tensor) -> Tensor:
    """(B, S, T, D) -> (B, S*T, D)."""
    B, S, T, D = x.shape
    return x.reshape(B, S * T, D)
