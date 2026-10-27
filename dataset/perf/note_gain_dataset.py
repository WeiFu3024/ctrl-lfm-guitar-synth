# Score-only note-level dataset for the note-gain predictor.
#
# One row per NOTE, POOLED ACROSS ALL 6 STRINGS AND SORTED BY ONSET TIME -- NOT one row
# per audio frame like ControlFlowDataset (dataset/perf/perf_curve_dataset.py), and NOT a
# padded per-string (S, N) grid either (an earlier design's choice, retired: see
# model/perf/note_gain_predictor.py's module docstring for why -- notes on different
# strings don't share a time axis the way frames do, so a shared per-string note INDEX
# never meant "simultaneous", and that grid paid a large, uneven padding cost for no
# attention benefit). Built to test whether a small model can predict pseudo_velocity's
# per-note gain from score-only information, so the main pitch/envelope predictor no
# longer needs a real recording's ground-truth loudness at inference. See
# model/perf/note_gain_predictor.py for the model this feeds, and
# scripts/perf/train_note_gain_predictor.py for training.
#
# The target is per_note_gain_values() (perf_curve_dataset.py) -- the SAME onset-to-
# next-onset statistic of the anchor channel that pseudo_velocity's supplied conditioning
# already uses -- so this dataset's ground truth and the deployment gain contract can never
# define "loudness" differently by accident. Duration/IOI features are still computed
# WITHIN a note's own string (its own previous/next onset, its own annotated duration):
# only the SEQUENCE ORDER presented to the model is merged across strings; each note's own
# content features describe its own string's local context, same as before.
#
# Every input feature here must be derivable from the score alone: no cached CREPE pitch, no
# measured envelope shape -- that is exactly what dataset/perf/detail_features.py's ORACLE
# features are, used elsewhere for ablations but never for anything meant to run without a
# real recording. String identity IS fed, but as a categorical `string_id` per token (looked
# up through the model's own embedding table), not a continuous feature channel and not a
# structural axis -- there is no axis left once strings are merged into one sequence.
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import torch
import zarr

from dataset.control.signal import notes_for_sample, onset_to_frame
from dataset.dataset_register import build_raw_dataset
from dataset.perf.perf_curve_dataset import (
    EXTRA_CHANNEL_SPECS, GAIN_MODES, MIDI_MAX, MIDI_MIN, N_STRINGS,
    broadcast_note_values_to_frames, per_note_gain_values,
)
from dataset.split import split_registry

# Sentinel IOI for a string's first/last note -- same clamp convention as the archived
# experiments/perf_predictor_v1/note_score_dataset.py's cross-string merge (its own
# MAX_GAP_S), reimplemented here rather than imported: live code must never import
# experiments/ (CLAUDE.md).
MAX_GAP_S = 5.0
_EPS = 1e-6

# python -m scripts.perf.calibrate_note_gain_stats --split train
# Re-run and paste again if note extraction or the IOI/duration convention changes. Both
# constants are computed from each note's own WITHIN-STRING duration/IOI, unaffected by the
# cross-string merge below -- still valid as-is.
LOG_DURATION_MEAN, LOG_DURATION_STD = -1.4278, 0.6258
LOG_IOI_MEAN, LOG_IOI_STD = -0.8131, 0.9533
# Median consecutive-note gap (not mean: the gap distribution is heavily right-skewed by
# rests) -- default distance-decay constant for the note-to-note delta loss's neighbour
# kernel A(i,j) = exp(-|onset_i-onset_j|/tau) (see _note_delta_neighbors below), used by
# the --delta-loss weighted note-gain training ablation.
IOI_DELTA_WEIGHT_TAU = 0.3812

# score-only, per note, within that note's own string. `simultaneity` (a hand-built
# chord-proxy feature the retired per-string design needed) is gone: cross-string context
# now reaches the model through real attention over the merged sequence instead.
INPUT_CHANNELS = ["midi_pitch", "log_duration", "log_ioi_prev", "log_ioi_next"]


def _log_z(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.log(x + _EPS) - mean) / std).astype(np.float32)


def _note_delta_neighbors(
    onset_times: np.ndarray, tau: float, window_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every note i (in `onset_times`' own ascending order), find every OTHER note j
    with |onset_i - onset_j| <= window_s and weight it A(i,j) = exp(-|onset_i-onset_j|/tau)
    -- the note-to-note delta loss's neighbourhood, generalising a single "next note on
    this string" pointer to every note "near" i in time, on ANY string (the whole point of
    the merged sequence is that a note's real neighbours can be on a different string --
    chord/strum context -- not just its own).

    `window_s` is precomputed ONCE from (tau, threshold) by the caller as the distance at
    which A(i,j) decays below `threshold` (A is monotonic in gap, so "A(i,j) > threshold"
    is exactly "gap < window_s") -- so both the RANGE (window_s) and each pair's WEIGHT are
    fixed, cacheable data, not something the loss function chooses per step.

    LINEAR in the number of notes: `onset_times` is sorted, so as i increases the window
    [onset_i-window_s, onset_i+window_s] only ever slides forward -- a two-pointer scan
    over the sorted array finds every note's window bounds in O(M) total pointer movement,
    never an O(M^2) all-pairs comparison. Materialising the actual (i, j, weight) triples
    is necessarily proportional to how many pairs qualify (the output size), which no
    algorithm can avoid; only the BOUND-FINDING is what stays linear.

    Returns `nbr_idx`/`nbr_weight`/`nbr_valid`, each `(M, K)` where K = the largest
    neighbour count any single note has (0 if no note has a neighbour, e.g. M <= 1).
    """
    M = onset_times.shape[0]
    neighbor_j: List[List[int]] = [[] for _ in range(M)]
    neighbor_w: List[List[float]] = [[] for _ in range(M)]
    lo = 0
    hi = 0
    for i in range(M):
        t = onset_times[i]
        while lo < M and onset_times[lo] < t - window_s:
            lo += 1
        hi = max(hi, lo)
        while hi < M and onset_times[hi] <= t + window_s:
            hi += 1
        for j in range(lo, hi):
            if j == i:
                continue
            neighbor_j[i].append(j)
            neighbor_w[i].append(math.exp(-abs(onset_times[j] - t) / tau))

    K = max((len(js) for js in neighbor_j), default=0)
    nbr_idx = np.zeros((M, K), dtype=np.int64)
    nbr_weight = np.zeros((M, K), dtype=np.float32)
    nbr_valid = np.zeros((M, K), dtype=bool)
    for i, (js, ws) in enumerate(zip(neighbor_j, neighbor_w)):
        n = len(js)
        if n:
            nbr_idx[i, :n] = js
            nbr_weight[i, :n] = ws
            nbr_valid[i, :n] = True
    return nbr_idx, nbr_weight, nbr_valid


def _same_string_next_neighbors(
    positions: Dict[int, List[int]], per_string_notes: Dict[int, List[tuple]],
    tau: float, M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K=1 neighbour table: every note's ONLY neighbour is the immediate NEXT note on ITS
    OWN STRING (forward-only, no cross-string pairing at all), weight = exp(-gap/tau) --
    the ORIGINAL note-gain predictor's delta-loss design (kept as an explicit ablation
    scope, `delta_scope="same_string_next"`, for controlled comparison against the default
    cross-string window `_note_delta_neighbors` above). Unlike that window, there is no
    threshold cutoff here: v2 always paired a note with its next same-string note
    regardless of gap size (the weight alone decays for a distant pair; the pair itself is
    never dropped), so this reproduces that unconditionally.
    """
    nbr_idx = np.zeros((M, 1), dtype=np.int64)
    nbr_weight = np.zeros((M, 1), dtype=np.float32)
    nbr_valid = np.zeros((M, 1), dtype=bool)
    for s_idx, notes in per_string_notes.items():
        n = len(notes)
        if n <= 1:
            continue
        pos = np.array(positions[s_idx], dtype=np.int64)
        onsets_s = np.array([o for o, _d, _p in notes], dtype=np.float64)
        gap = np.clip(onsets_s[1:] - onsets_s[:-1], 0.0, None)
        nbr_idx[pos[:-1], 0] = pos[1:]
        nbr_weight[pos[:-1], 0] = np.exp(-gap / tau).astype(np.float32)
        nbr_valid[pos[:-1], 0] = True
    return nbr_idx, nbr_weight, nbr_valid


class NoteGainDataset(torch.utils.data.Dataset):
    """One item per recording: a single (M, 1) flat, time-sorted sequence pooling every
    note from all 6 strings (M = total note count in the recording), plus a parallel
    (M,) `string_id`. Mirrors ControlFlowDataset's per-recording structure but indexes by
    NOTE, not frame, and -- unlike a per-string frame grid -- has no string axis at all."""

    def __init__(self, cfg, split: Optional[str] = None, deterministic: bool = False,
                 anchor_channel: str = "envelope", gain_mode: str = "max",
                 delta_tau: float = IOI_DELTA_WEIGHT_TAU, delta_weight_threshold: float = 0.05,
                 delta_scope: str = "cross_string"):
        assert gain_mode in [m for m in GAIN_MODES if m != "none"], \
            f"gain_mode must be 'mean' or 'max' (no-gain has no target to predict), got {gain_mode!r}"
        assert anchor_channel in EXTRA_CHANNEL_SPECS, \
            f"anchor_channel must be one of {list(EXTRA_CHANNEL_SPECS)}"
        assert 0.0 < delta_weight_threshold < 1.0, \
            f"delta_weight_threshold must be in (0, 1), got {delta_weight_threshold!r}"
        assert delta_scope in ("cross_string", "same_string_next"), \
            f"delta_scope must be 'cross_string' or 'same_string_next', got {delta_scope!r}"
        self.cfg = cfg
        self.deterministic = deterministic
        self.anchor_channel = anchor_channel
        self._spec = EXTRA_CHANNEL_SPECS[anchor_channel]
        self.gain_mode = gain_mode
        self.n_strings = N_STRINGS
        self.delta_tau = delta_tau
        self.delta_scope = delta_scope
        # A(i,j) = exp(-gap/tau) is monotonically decreasing in gap, so "A(i,j) >
        # threshold" is exactly "gap < this window" -- computed once, not searched.
        # Only used by delta_scope="cross_string"; same_string_next has no cutoff (see
        # _same_string_next_neighbors).
        self._delta_window_s = -delta_tau * math.log(delta_weight_threshold)

        types = cfg.data.type if isinstance(cfg.data.type, (list, tuple)) else [cfg.data.type]
        self.hex_dataset = types[0]

        raw = build_raw_dataset(cfg.data)
        self.sample_by_id = {s.sample_id: s for s in raw.list_samples()}
        self.control_store = zarr.open(str(Path(cfg.cache.root) / cfg.cache.control_store), mode="r")

        self._recording_strings = self._build_recording_index()
        self._index = sorted(self._recording_strings)
        if split is not None:
            self._index = self._filter_by_split(self._index, split)
        self._fr: Optional[float] = None

    def _build_recording_index(self) -> Dict[str, Dict[int, str]]:
        by_rec: Dict[str, Dict[int, str]] = {}
        for sample_id, sample in self.sample_by_id.items():
            if sample.meta.get("dataset") != self.hex_dataset or "string" not in sample.meta:
                continue
            if sample_id not in self.control_store:
                continue
            grp = self.control_store[sample_id]
            if self._spec["cache_key"] not in grp:
                continue
            by_rec.setdefault(sample_id.split("/")[1], {})[sample.meta["string"] - 1] = sample_id
        return by_rec

    def _filter_by_split(self, index: List[str], split: str) -> List[str]:
        entry = {e.dataset: e for e in self.cfg.dataset.split.entries}[self.hex_dataset]
        splitter = split_registry.get(entry.splitter)(entry.params)
        rep = {r: self.sample_by_id[next(iter(s.values()))] for r, s in self._recording_strings.items()}
        assignment = splitter.assign(list(rep.values()))
        keep = {r for r, s in rep.items() if assignment.get(s.sample_id) == split}
        return [r for r in index if r in keep]

    def _frame_rate(self) -> float:
        if self._fr is None:
            sid = next(iter(next(iter(self._recording_strings.values())).values()))
            self._fr = float(
                self.control_store[sid][self._spec["cache_key"]].attrs.get(
                    "frame_rate", 21.533203125
                )
            )
        return self._fr

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        recording_id = self._index[idx]
        strings = self._recording_strings[recording_id]
        fr = self._frame_rate()

        per_string_notes: Dict[int, List[tuple]] = {
            s_idx: sorted(notes_for_sample(self.sample_by_id[sid], {}), key=lambda n: n[0])
            for s_idx, sid in strings.items()
        }

        # Merge every string's notes into ONE onset-time-sorted order -- this recording's
        # actual multi-string sequence, not a per-string index. `positions[s_idx][k]` is
        # where that string's k-th (onset-order) note landed in the merge, used below to
        # place per-string features at the right merged row.
        merged = sorted(
            ((onset_s, s_idx, k) for s_idx, notes in per_string_notes.items()
             for k, (onset_s, _dur, _pitch) in enumerate(notes)),
            key=lambda row: row[0],
        )
        n_real = len(merged)
        M = max(n_real, 1)
        onset_times = np.array([row[0] for row in merged], dtype=np.float64)  # (n_real,)
        positions: Dict[int, List[int]] = {s_idx: [] for s_idx in per_string_notes}
        for pos, (_onset_s, s_idx, _k) in enumerate(merged):
            positions[s_idx].append(pos)

        feats = {name: np.zeros((M, 1), dtype=np.float32) for name in INPUT_CHANNELS}
        target = np.zeros((M, 1), dtype=np.float32)
        string_id = np.zeros(M, dtype=np.int64)
        note_valid = np.zeros(M, dtype=bool)

        for s_idx, notes in per_string_notes.items():
            if not notes:
                continue
            n = len(notes)
            pos = np.array(positions[s_idx], dtype=np.int64)
            onsets_s = np.array([o for o, _d, _p in notes], dtype=np.float64)
            durs = np.array([d for _o, d, _p in notes], dtype=np.float64)
            pitches = np.array([p for _o, _d, p in notes], dtype=np.float32)

            feats["midi_pitch"][pos, 0] = np.clip(
                (pitches - MIDI_MIN) / (MIDI_MAX - MIDI_MIN), 0.0, 1.0
            )
            feats["log_duration"][pos, 0] = _log_z(durs, LOG_DURATION_MEAN, LOG_DURATION_STD)

            ioi_prev = np.empty(n, dtype=np.float64)
            ioi_prev[0] = MAX_GAP_S
            ioi_prev[1:] = onsets_s[1:] - onsets_s[:-1]
            ioi_next = np.empty(n, dtype=np.float64)
            ioi_next[:-1] = onsets_s[1:] - onsets_s[:-1]
            ioi_next[-1] = MAX_GAP_S
            feats["log_ioi_prev"][pos, 0] = _log_z(
                np.clip(ioi_prev, 0.0, None), LOG_IOI_MEAN, LOG_IOI_STD
            )
            feats["log_ioi_next"][pos, 0] = _log_z(
                np.clip(ioi_next, 0.0, None), LOG_IOI_MEAN, LOG_IOI_STD
            )

            string_id[pos] = s_idx
            note_valid[pos] = True

            # target: same onset-to-next-onset statistic pseudo_velocity's supplied gain uses
            grp = self.control_store[strings[s_idx]]
            T = grp[self._spec["cache_key"]].shape[0]
            env = grp[self._spec["cache_key"]][:T].reshape(-1)[:T]
            onset_frames = [min(T - 1, max(0, onset_to_frame(o, fr))) for o in onsets_s]
            raw_gain = per_note_gain_values(env, onset_frames, T, self.gain_mode)
            target[pos, 0] = (raw_gain - self._spec["mean"]) / self._spec["std"]

        # note-to-note delta loss's neighbourhood (model/perf/perf_losses.py's
        # note_delta_loss). Two scopes:
        #   "cross_string" (default): every note within `_delta_window_s` of note i, ANY
        #     string, precomputed IOI-decay weight -- see _note_delta_neighbors for the
        #     linear-time construction. Built on the REAL onset count (n_real), not the
        #     padded M: the degenerate all-silent recording (n_real == 0, M clamped to 1)
        #     gets an empty (0, 0) table rather than a bogus self-referential row.
        #   "same_string_next": the ORIGINAL note-gain predictor design, kept as an
        #     explicit ablation -- see _same_string_next_neighbors. Already (M, 1)-shaped
        #     (writes only real rows, so no separate n_real->M padding step is needed).
        if self.delta_scope == "same_string_next":
            nbr_idx, nbr_weight, nbr_valid = _same_string_next_neighbors(
                positions, per_string_notes, self.delta_tau, M
            )
        else:
            nbr_idx, nbr_weight, nbr_valid = _note_delta_neighbors(
                onset_times, self.delta_tau, self._delta_window_s
            )
            if n_real < M:
                pad_row = np.zeros((M - n_real, nbr_idx.shape[1]), dtype=nbr_idx.dtype)
                nbr_idx = np.concatenate([nbr_idx, pad_row], axis=0)
                nbr_weight = np.concatenate(
                    [nbr_weight,
                     np.zeros((M - n_real, nbr_weight.shape[1]), dtype=nbr_weight.dtype)],
                    axis=0,
                )
                nbr_valid = np.concatenate(
                    [nbr_valid, np.zeros((M - n_real, nbr_valid.shape[1]), dtype=bool)], axis=0
                )

        return {
            "recording_id": recording_id,
            "n_strings": self.n_strings,
            "n_notes": M,
            "controls": {k: torch.from_numpy(v) for k, v in feats.items()},
            "target": torch.from_numpy(target),
            "note_valid": torch.from_numpy(note_valid),
            "string_id": torch.from_numpy(string_id),
            "nbr_idx": torch.from_numpy(nbr_idx),
            "nbr_weight": torch.from_numpy(nbr_weight),
            "nbr_valid": torch.from_numpy(nbr_valid),
        }

    def predict_gain_frames(self, model, recording_id: str, T: int, device) -> np.ndarray:
        """Run `model` on `recording_id`'s score-only note features and broadcast its
        prediction onto a (S, T) RAW-unit frame grid -- the eval-time substitute for
        pseudo_velocity's GT-derived conditioning (see eval/renderers/performance.py's
        --gain-source predicted). `T` is the CALLER's own ControlFlowDataset item's frame
        count, passed in explicitly rather than recomputed here: the two datasets' internal
        T can differ by a frame at a silence-padded tail, and the caller's grid is the one
        that actually has to line up with pseudo_velocity's other conditioning channels.

        Returns raw (un-z-scored) units, matching `per_note_gain_frames`'s own convention --
        the caller re-applies `(x - anchor_mean) / anchor_std` itself, the same z-scoring
        pseudo_velocity's GT path uses, so both paths are normalised identically.
        """
        idx = self._index.index(recording_id)
        item = self[idx]
        controls = {k: v.unsqueeze(0).to(device) for k, v in item["controls"].items()}
        string_id = item["string_id"].unsqueeze(0).to(device)
        valid = item["note_valid"].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred_z = model(controls, string_id, valid)[0, :, 0].cpu().numpy()  # (M,), z-scored

        strings = self._recording_strings[recording_id]
        fr = self._frame_rate()
        string_id_np = item["string_id"].numpy()
        note_valid_np = item["note_valid"].numpy()
        frames = np.full((self.n_strings, T), self._spec["mean"], dtype=np.float32)
        for s_idx, sid in strings.items():
            notes = sorted(notes_for_sample(self.sample_by_id[sid], {}), key=lambda n: n[0])
            if not notes:
                continue
            # boolean masking preserves ascending merged-index order, which -- because the
            # merge is globally onset-sorted -- is exactly this string's own onset order,
            # the same order `notes` is independently sorted in.
            mask = note_valid_np & (string_id_np == s_idx)
            onsets_s = [o for o, _d, _p in notes]
            onset_frames = [min(T - 1, max(0, onset_to_frame(o, fr))) for o in onsets_s]
            values_raw = pred_z[mask] * self._spec["std"] + self._spec["mean"]
            frames[s_idx] = broadcast_note_values_to_frames(
                values_raw, onset_frames, T, self._spec["mean"]
            )
        return frames


def collate_note_gain(batch: List[Dict]) -> Dict:
    """Pads every recording's flat, time-sorted, multi-string note sequence to the batch's
    max note count -- ordinary tail-padding, like any other variable-length sequence batch.
    (The retired per-string grid padded far more unevenly: a single recording's own
    per-string note counts vary widely, so a short string was mostly padding even before
    batching; here the padding fraction only reflects note-count variation ACROSS
    recordings, which is smaller.) The delta-loss neighbour tables (nbr_idx/nbr_weight/
    nbr_valid) are additionally padded along their own K axis (each recording finds a
    different max neighbour count) to the batch's max K."""
    max_M = max(item["n_notes"] for item in batch)
    max_K = max(item["nbr_valid"].shape[1] for item in batch)

    def pad(x: torch.Tensor, m: int) -> torch.Tensor:
        if m == max_M:
            return x
        pad_shape = (max_M - m,) + tuple(x.shape[1:])
        return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)

    def pad_nbr(x: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        out = torch.zeros((max_M, max_K), dtype=x.dtype)
        out[:m, :k] = x
        return out

    controls = {
        name: torch.stack([pad(item["controls"][name], item["n_notes"]) for item in batch])
        for name in batch[0]["controls"]
    }
    return {
        "recording_id": [item["recording_id"] for item in batch],
        "controls": controls,
        "target": torch.stack([pad(item["target"], item["n_notes"]) for item in batch]),
        "note_valid": torch.stack([pad(item["note_valid"], item["n_notes"]) for item in batch]),
        "string_id": torch.stack([pad(item["string_id"], item["n_notes"]) for item in batch]),
        "nbr_idx": torch.stack([pad_nbr(item["nbr_idx"]) for item in batch]),
        "nbr_weight": torch.stack([pad_nbr(item["nbr_weight"]) for item in batch]),
        "nbr_valid": torch.stack([pad_nbr(item["nbr_valid"]) for item in batch]),
        "n_strings": batch[0]["n_strings"],
        "n_notes": max_M,
    }
