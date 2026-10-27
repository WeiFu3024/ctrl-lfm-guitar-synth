# Small note-level regressor: score-only per-note features -> per-note loudness (the same
# quantity pseudo_velocity's supplied gain is, dataset/perf/perf_curve_dataset.py's
# per_note_gain_values). Tests whether the main pitch/envelope predictor's inference-time
# gain contract can be predicted instead of supplied from a real recording -- see
# dataset/perf/note_gain_dataset.py for the data this consumes and
# scripts/perf/train_note_gain_predictor.py for training.
#
# ONE MERGED, TIME-SORTED SEQUENCE ACROSS ALL 6 STRINGS. Replaces an earlier design that
# kept a per-string (S, N) note grid and reused EnvelopeRegressor's frame-synchronous
# _GatedCrossStringBlock's *assumption* only in spirit -- it never actually applied that
# block, because "index t is the same wall-clock instant on every string" (true for
# frames) is false for notes: note index k on string 1 and index k on string 3 are not
# simultaneous. That design fed cross-string context through a single hand-built
# `simultaneity` feature instead and stayed within-string-only in attention, while still
# paying the (S, N) grid's padding cost (per-string note counts on one recording vary
# widely, so most of a short string's row was padding). Measured result: variance_ratio
# 0.32, pearson 0.34 on the test split, and substituting its output for GT gain collapsed
# rendered envelope correlation 0.88 -> 0.62 (experiments/note_gain_predictor_v1/README.md)
# -- preserved as a negative result, not reproduced here.
#
# Here, every note from every string is merged into ONE sequence sorted by onset time --
# a note's sequence NEIGHBOURS are genuinely whoever is sounding closest in time, on
# whichever string -- and string identity is carried per-token by a learned embedding
# (`string_embed`, indexed by `string_id`) instead of a structural axis. A plain
# bidirectional transformer encoder (`_WithinStringBlock`, reused unchanged: it was always
# just "temporal self-attention + MLP", agnostic to what the sequence axis means) then
# mixes real cross-string context through actual attention, so the `simultaneity` feature
# is dropped entirely -- attention now does that job directly instead of through a
# hand-computed proxy. This is the simple, classic structure: attribute encoder -> add
# string embedding -> transformer encoder -> linear head. Deliberately NOT the archived
# experiments/perf_predictor_v1/note_ar_decoder.py's heavier alternative (autoregressive,
# causal, local-time-windowed cross-string attention) -- this predictor is offline and
# non-autoregressive, so no causal masking is needed, only ordinary key-padding masking.
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from model.common.axial import _WithinStringBlock
from model.common.dit import RotaryEmbedding


class NoteGainPredictor(nn.Module):
    def __init__(self, input_names: List[str], input_channels: Dict[str, int],
                 n_strings: int = 6, d_model: int = 64, depth: int = 2,
                 n_heads: int = 4, mlp_ratio: float = 4.0, rope_theta: float = 10000.0):
        super().__init__()
        self.input_names = list(input_names)
        self.n_strings = n_strings

        total_c = sum(input_channels[n] for n in self.input_names)
        self.in_proj = nn.Linear(total_c, d_model)
        # Per-token string identity -- looked up by `string_id`. There is no structural
        # axis to place it on any more (see module docstring), so it is added to every
        # token's projected features instead, the same role EnvelopeRegressor.string_embed
        # plays via broadcast over its S axis.
        self.string_embed = nn.Embedding(n_strings, d_model)
        rope = RotaryEmbedding(d_model // n_heads, theta=rope_theta)
        self.blocks = nn.ModuleList(
            [_WithinStringBlock(d_model, n_heads, mlp_ratio, rope) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, inputs: Dict[str, Tensor], string_id: Tensor, valid: Tensor) -> Tensor:
        """``inputs`` values are (B, M, C) note-indexed tensors, M = notes pooled across
        every string in one recording and sorted by onset time (see
        dataset/perf/note_gain_dataset.py). `string_id`: (B, M) int64 in [0, n_strings).
        `valid`: (B, M) bool, True for a real (non-padding) note. Returns (B, M, 1)
        predicted gain, one value per input row, in the same merged order.

        Padded notes are masked out as attention KEYS -- a zero-padded token reads as an
        "average note" in z-scored feature space, not a flagged sentinel, so real notes
        must never draw on it. Padded positions are still free to appear as QUERIES (never
        an all-masked row, so no NaN risk): their own output rows are simply discarded by
        the caller's loss/valid mask and never feed anything else.
        """
        stacked = torch.cat([inputs[n] for n in self.input_names], dim=-1)
        x = self.in_proj(stacked) + self.string_embed(string_id)
        attn_mask = valid[:, None, None, :]  # (B, 1, 1, M): broadcasts over heads and queries
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.head(self.norm(x))
