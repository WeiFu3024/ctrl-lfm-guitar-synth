# Joint 6-string audio synth: one model that sees all six strings' controls, lets them
# interact, and predicts all six latents at once -- or, in "mix" mode, the single mono
# mic recording. Every synth before this treated a string in isolation
# (dataset/synth/dataset.py emits six independent samples), so nothing could represent a strum
# hitting six strings together, sympathetic ringing, or which strings are silent.
#
# SHAPE CONTRACT: the outside world sees rank-3 (B, S*T, D), and this module unflattens
# to (B,S,T,D) internally. That is what lets model/common/flow_matching.py stay untouched -- it
# assumes rank-3 (t.view(B,1,1), mask (B,L,1), B,_,_ = shape) and none of that cares
# whether L means frames or strings x frames. It also means the flow-matching timestep is
# drawn once per RECORDING, which is correct: all six strings are denoised together.
#
# ATTENTION: axial, the pattern validated in experiments/frame_rate_perf_decoder/
# perf_decoder_joint.py -- alternate a within-string temporal block (RoPE) with a
# cross-string block over the 6-string axis (no RoPE; string identity is an embedding,
# not a position). O(S*T^2 + T*S^2) instead of O((S*T)^2), ~6x cheaper here at S=6, T=256.
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.common.axial import (N_STRINGS, _GatedCrossStringBlock, _WithinStringBlock,
                                _flatten_strings, _to_strings)
from model.common.dit import DiTBlock, FinalLayer, RotaryEmbedding, TimestepEmbedding


@dataclass
class JointSynthConfig:
    output_mode: str = "per_string"  # or "mix"
    n_strings: int = N_STRINGS
    # Names (a subset of dataset.control_signals, e.g. ["onset", "voiced"]) fed as an
    # ADDITIVE learned embedding instead of being concatenated through control_proj: at a
    # frame where the raw value is 1, that channel's own nn.Parameter(embed) vector is
    # added to the conditioning; at 0, nothing is added. A binary event ("a note started
    # here") is a categorical fact, not a continuous quantity a single shared linear layer
    # should learn to scale -- concatenation forces onset/voiced through the exact same
    # projection as pitch/envelope, which can only ever scale them by one learned weight
    # each. Empty by default, which reproduces every existing config's behavior (plain
    # concat -> control_proj) exactly.
    event_channels: List[str] = field(default_factory=list)
    # Keep the conditioning encoder's cross-string path independent from this decoder/DiT
    # ablation. False omits decoder cross-string blocks and their parameters entirely.
    decoder_cross_string_attention: bool = True
    # Same ablation, applied to the conditioning encoder's cross-string blocks (enc_cross)
    # instead. False omits them entirely. Combine both flags False for a model with no
    # cross-string communication anywhere -- six fully independent per-string paths.
    encoder_cross_string_attention: bool = True


class JointSynthDecoder(nn.Module):
    """controls {name: (B, S*T, C)} + noisy latent -> velocity, jointly over six strings.

    per_string: noisy (B, S*T, latent_dim) -> (B, S*T, latent_dim)
    mix:        noisy (B, T,   latent_dim) -> (B, T,   latent_dim), conditioned on all six
    """

    def __init__(self, model_cfg, control_names: List[str], control_channels: Dict[str, int],
                 joint_cfg: Optional[JointSynthConfig] = None):
        super().__init__()
        # Fall back to the typed field on model_cfg (config/schema.py's ModelConfig.joint)
        # rather than to per_string: model_cfg already carries the setting, so a caller
        # that forgets the explicit joint_cfg would otherwise silently train the wrong
        # output_mode for a full run and only discover it from the output shapes.
        jc = joint_cfg or getattr(model_cfg, "joint", None) or JointSynthConfig()
        self.output_mode = jc.output_mode
        self.n_strings = jc.n_strings
        self.control_names = control_names
        self.event_channels = list(jc.event_channels)
        unknown_event = [n for n in self.event_channels if n not in control_names]
        if unknown_event:
            raise ValueError(f"event_channels {unknown_event} not in control_names {control_names}")
        self.cont_names = [n for n in control_names if n not in self.event_channels]
        if self.output_mode not in ("per_string", "mix"):
            raise ValueError(f"unknown output_mode {self.output_mode!r}")

        d_model = model_cfg.dit.d_model
        embed = model_cfg.control_encoder.embed_dim
        n_heads_e = model_cfg.control_encoder.n_heads
        n_heads_d = model_cfg.dit.n_heads
        mlp_ratio = model_cfg.dit.mlp_ratio

        # ---- conditioning encoder: per-string temporal + gated cross-string ----
        total_c = sum(control_channels[n] for n in self.cont_names)
        self.control_proj = nn.Linear(total_c, embed)
        if self.event_channels:
            # one learned vector per event channel; 0.02-scaled init matches this repo's
            # other near-zero-init additive terms (e.g. the cross-string gate's residual),
            # so the encoder starts close to "ignore this event" and has to learn to use it
            self.event_embed = nn.ParameterDict(
                {name: nn.Parameter(torch.randn(embed) * 0.02) for name in self.event_channels})
        self.string_embed = nn.Embedding(self.n_strings, embed)
        rope_e = RotaryEmbedding(embed // n_heads_e, theta=model_cfg.control_encoder.rope_theta)
        depth_e = model_cfg.control_encoder.depth
        self.enc_within = nn.ModuleList(
            [_WithinStringBlock(embed, n_heads_e, model_cfg.control_encoder.mlp_ratio, rope_e) for _ in range(depth_e)])
        encoder_cross = jc.encoder_cross_string_attention
        self.enc_cross = nn.ModuleList(
            [_GatedCrossStringBlock(embed, n_heads_e, model_cfg.control_encoder.mlp_ratio) for _ in range(depth_e)]
        ) if encoder_cross else None
        self.enc_norm = nn.LayerNorm(embed)
        self.null_embedding = nn.Parameter(torch.zeros(embed))  # CFG, as ControlSignalEncoder

        # ---- denoiser: AdaLN DiT within-string, optional gated cross-string between ----
        self.input_proj = nn.Linear(model_cfg.latent_dim, d_model)
        self.t_embed = TimestepEmbedding(dim=embed)
        rope_d = RotaryEmbedding(d_model // n_heads_d, theta=model_cfg.dit.rope_theta)
        depth_d = model_cfg.dit.depth
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads_d, mlp_ratio, cond_dim=embed, rope=rope_d) for _ in range(depth_d)])
        # This switch is deliberately decoder-only: the encoder still gets the full control
        # context across strings, while the ablation tests whether the latent denoiser itself
        # needs cross-string communication. Omitting the blocks is stronger and cleaner than
        decoder_cross = jc.decoder_cross_string_attention
        self.dec_cross = nn.ModuleList(
            [_GatedCrossStringBlock(d_model, n_heads_d, mlp_ratio) for _ in range(depth_d)]
        ) if self.output_mode == "per_string" and decoder_cross else None
        self.final = FinalLayer(embed, d_model, model_cfg.latent_dim)

    def encode_controls(self, controls: Dict[str, Tensor], cfg_drop_mask: Optional[Tensor]) -> Tensor:
        """{name: (B, S*T, C)} -> (B, S, T, embed)."""
        stacked = torch.cat([controls[n] for n in self.cont_names], dim=-1)  # (B,S*T,C)
        x = _to_strings(self.control_proj(stacked), self.n_strings)          # (B,S,T,E)
        B, S, T, E = x.shape
        for name in self.event_channels:
            flag = _to_strings(controls[name], self.n_strings)              # (B,S,T,1)
            x = x + flag * self.event_embed[name].view(1, 1, 1, E)
        x = x + self.string_embed(torch.arange(S, device=x.device)).view(1, S, 1, E)

        if self.enc_cross is None:
            for within in self.enc_within:
                x = within(x.reshape(B * S, T, E)).reshape(B, S, T, E)
        else:
            for within, cross in zip(self.enc_within, self.enc_cross):
                x = within(x.reshape(B * S, T, E)).reshape(B, S, T, E)
                x = cross(x.permute(0, 2, 1, 3).reshape(B * T, S, E)).reshape(B, T, S, E).permute(0, 2, 1, 3)
        x = self.enc_norm(x)

        if cfg_drop_mask is not None:
            x = torch.where(cfg_drop_mask.view(-1, 1, 1, 1), self.null_embedding.view(1, 1, 1, E).expand_as(x), x)
        return x

    def forward(self, noisy_latent: Tensor, t: Tensor, controls: Dict[str, Tensor],
                cfg_drop_mask: Optional[Tensor] = None) -> Tensor:
        cond = self.encode_controls(controls, cfg_drop_mask)          # (B,S,T,E)
        B, S, T, E = cond.shape
        t_emb = self.t_embed(t)                                        # (B,E)

        if self.output_mode == "mix":
            # Collapse the string axis for the target, keep it for the conditioning.
            cond_mix = cond.mean(dim=1) + t_emb.unsqueeze(1)           # (B,T,E)
            x = self.input_proj(noisy_latent)                          # (B,T,d)
            for block in self.blocks:
                x = block(x, cond_mix)
            return self.final(x, cond_mix)                             # (B,T,latent_dim)

        cond = cond + t_emb.view(B, 1, 1, E)
        x = _to_strings(self.input_proj(noisy_latent), S)              # (B,S,T,d)
        d = x.shape[-1]
        cond_flat = cond.reshape(B * S, T, E)
        if self.dec_cross is None:
            for block in self.blocks:
                x = block(x.reshape(B * S, T, d), cond_flat).reshape(B, S, T, d)
        else:
            for block, cross in zip(self.blocks, self.dec_cross):
                x = block(x.reshape(B * S, T, d), cond_flat).reshape(B, S, T, d)
                x = cross(x.permute(0, 2, 1, 3).reshape(B * T, S, d)).reshape(B, T, S, d).permute(0, 2, 1, 3)
        out = self.final(x.reshape(B * S, T, d), cond_flat).reshape(B, S, T, -1)
        return _flatten_strings(out)                                   # (B, S*T, latent_dim)

    def cross_string_gates(self) -> Dict[str, List[float]]:
        """For inspecting whether the model actually opened the cross-string paths."""
        g = {"encoder": [] if self.enc_cross is None else [float(b.gate.item()) for b in self.enc_cross]}
        g["decoder"] = [] if self.dec_cross is None else [float(b.gate.item()) for b in self.dec_cross]
        return g
