# Deterministic envelope regressor: score (+ GT pitch) -> log-RMS envelope, all six strings.
#
# Same axial structure as model/synth/joint_synth.py -- within-string temporal attention alternating
# with GATED cross-string attention -- and the blocks are IMPORTED from model/common/axial.py rather than
# copied, so the two arms cannot drift apart. What is removed is the diffusion machinery: no
# noisy input, no timestep, no CFG. This is the plain regressor the loss study needs, and it
# is held FIXED across every loss variant so the sweep measures losses and not architecture.
#
# The cross-string gates stay zero-initialised for the same measured reason as in joint_synth:
# an earlier decoder that mixed across strings unconditionally improved envelope while
# collapsing pitch, with no way to switch the mixing off.
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from model.common.dit import RotaryEmbedding
from model.common.axial import _GatedCrossStringBlock, _WithinStringBlock, _to_strings


class EnvelopeRegressor(nn.Module):
    def __init__(self, input_names: List[str], input_channels: Dict[str, int],
                 n_strings: int = 6, d_model: int = 256, depth: int = 6,
                 n_heads: int = 8, mlp_ratio: float = 4.0, rope_theta: float = 10000.0,
                 out_dim: int = 1, n_players: Optional[int] = None,
                 detail_dim: Optional[int] = None, cross_string_attention: bool = True):
        super().__init__()
        self.input_names = list(input_names)
        self.n_strings = n_strings

        total_c = sum(input_channels[n] for n in self.input_names)
        self.in_proj = nn.Linear(total_c, d_model)
        self.detail_proj = nn.Linear(detail_dim, d_model) if detail_dim else None
        if self.detail_proj is not None:
            nn.init.zeros_(self.detail_proj.weight)
            nn.init.zeros_(self.detail_proj.bias)
        self.string_embed = nn.Embedding(n_strings, d_model)
        # Optional performer identity conditioning (dataset/perf/perf_curve_dataset.py's
        # N_PLAYERS/_player_id): GuitarSet's 6 players all appear in train AND test, so this
        # is identity conditioning, not held-out-speaker generalisation. None (the default)
        # omits the embedding entirely, keeping every existing checkpoint's state dict and
        # behaviour byte-identical.
        self.player_embed = nn.Embedding(n_players, d_model) if n_players else None
        rope = RotaryEmbedding(d_model // n_heads, theta=rope_theta)
        self.within = nn.ModuleList(
            [_WithinStringBlock(d_model, n_heads, mlp_ratio, rope) for _ in range(depth)])
        # Cross-string ablation (mirrors model/synth/joint_synth.py's decoder_cross_string_attention):
        # False omits these blocks and their parameters entirely rather than zero-gating, so
        # the ablated checkpoint genuinely has no cross-string communication anywhere.
        self.cross_string_attention = cross_string_attention
        self.cross = nn.ModuleList(
            [_GatedCrossStringBlock(d_model, n_heads, mlp_ratio) for _ in range(depth)]
        ) if cross_string_attention else None
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = out_dim
        self.head = nn.Linear(d_model, out_dim)

    def forward(
        self,
        inputs: Dict[str, Tensor],
        player_id: Optional[Tensor] = None,
        detail_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict normalised curves from frame-aligned inputs and optional oracle detail.

        ``inputs`` values and ``detail_features`` use shape ``(B, S*T, C)``. Detail is
        required exactly when ``detail_dim`` was configured.
        """
        stacked = torch.cat([inputs[n] for n in self.input_names], dim=-1)
        projected = self.in_proj(stacked)
        if self.detail_proj is not None:
            if detail_features is None:
                raise ValueError("model has detail_proj but no detail_features were given")
            projected = projected + self.detail_proj(detail_features)
        elif detail_features is not None and detail_features.shape[-1]:
            raise ValueError("model has no detail_proj but non-empty detail_features were given")
        x = _to_strings(projected, self.n_strings)                        # (B,S,T,D)
        B, S, T, D = x.shape
        x = x + self.string_embed(torch.arange(S, device=x.device)).view(1, S, 1, D)
        if self.player_embed is not None:
            if player_id is None:
                raise ValueError("model has a player embedding but no player_id was given")
            x = x + self.player_embed(player_id).view(B, 1, 1, D)

        for i, within in enumerate(self.within):
            x = within(x.reshape(B * S, T, D)).reshape(B, S, T, D)
            if self.cross is not None:
                cross = self.cross[i]
                x = cross(x.permute(0, 2, 1, 3).reshape(B * T, S, D)).reshape(B, T, S, D).permute(0, 2, 1, 3)

        return self.head(self.norm(x)).reshape(B, S * T, self.out_dim)

    def cross_string_gates(self) -> List[float]:
        """The learned gate per layer -- whether the model actually opened cross-string
        mixing, inspectable the same way scripts/synth/train_joint_synth.py reports it.
        Empty when cross_string_attention=False: there is no gate to inspect."""
        return [float(b.gate.detach().cpu()) for b in self.cross] if self.cross is not None else []
