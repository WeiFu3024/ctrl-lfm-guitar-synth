# Arm C: score -> frame-level control curves, with NO intermediate per-note representation.
#
# Targets the two control curves the synth actually consumes -- (pitch, envelope) -- on the
# 21.53 Hz grid, for all six strings jointly. Conditioning is score-derived only: quantised
# MIDI pitch, pseudo-velocity, voicing, and two SEGMENT-ALIGNED channels built from the
# realigned onsets (dataset/control/onset_align.py). No audio touches the conditioning.
#
# Layout is the same flattened (S*T, D) trick as dataset/synth/joint_synth_dataset.py, and for the
# same reason: model/common/flow_matching.py assumes rank-3 (B, L, D) throughout, so presenting
# (B, S*T, D) lets FlowMatching, the loss and CFG dropout run COMPLETELY UNCHANGED, with the
# flow timestep drawn once per recording rather than once per string. model/synth/joint_synth.py
# unflattens internally for its axial attention, so JointSynthDecoder is reused as-is with
# latent_dim=2 -- arm C needs no new model file.
#
# TWO REPRESENTATION CHOICES THAT ARE NOT COSMETIC:
#
# 1. Pitch is predicted as MIDI, held forward through unvoiced frames -- not raw Hz. Hz is
#    log-distributed and the cache's silence convention is 0 Hz, which has no log. Worse,
#    feeding a 0 Hz-between-notes curve to the synth measured catastrophic (pitch following
#    -0.02, and the untouched envelope channel's re-extraction correlation collapsing
#    0.996 -> 0.42); holding forward recovered it. So hold-forward is both the numerically
#    sane choice and the one matching what the synth wants to be fed.
#
# 2. Both channels are z-scored with constants calibrated on the train split
#    (scripts/calibrate_control_flow_stats.py). FlowMatching's loss is an unweighted mean
#    over the channel axis, and in raw units log-RMS has ~134x smaller variance than f0 --
#    exactly the imbalance that starved loudness's gradient in the note-rate AR work.
#    Normalising makes the two channels contribute equally by construction.
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from dataset.control.signal import (
    hz_to_midi, midi_to_hz, notes_for_sample, note_to_frame_span,
)
from dataset.perf.detail_features import (
    broadcast_detail_features, detail_feature_dim, extract_pitch_detail_features,
)
from dataset.synth.dataset import _pad_to
from dataset.dataset_register import build_raw_dataset
from dataset.split import split_registry
from model.perf.perf_losses import PITCH_MASK_LO, PITCH_MASK_HI

N_STRINGS = 6
# GuitarSet has 6 players (dataset/dataset_register.py's GuitarSetLoader parses the leading
# "00".."05" filename digit into sample.meta["player"]). Every player appears in train, val,
# AND test (dataset/splits/guitarset_ddsp_*_filenames.txt splits by recording/tune, never by
# player), so a learned per-player embedding is identity conditioning, not held-out-speaker
# generalisation -- the model can only ever see players it was also trained on.
N_PLAYERS = 6
# Target array is (S, T, 1 + len(extra_channels)): column 0 is always pitch_midi, columns
# 1.. are each entry of extra_channels in order. No fixed list any more -- see
# ControlFlowDataset.__init__'s extra_channels parameter.
COND_CHANNELS = ["midi_pitch", "pseudo_velocity", "voiced", "segment_phase", "segment_onset"]

# HOW THE PER-NOTE GAIN REACHES THE MODEL -- part of the system input contract.
#
# "max" (the DEFAULT since this codebase aligned with DDSP-Guitar) sets pseudo_velocity to
# the supplied per-note PEAK envelope level -- matching Jonason et al.'s convention: "Since
# velocity annotation is not included in GuitarSet, we use the peak unit scaled dB(A)
# loudness of every note as a proxy for MIDI velocity." GuitarSet derives the supplied value
# from the matching recording because its note_midi annotation has no separate velocity
# field. The predictor is intentionally asked to predict the SHAPE within each note given
# its height.
#
# "mean" is the ORIGINAL representation (every result reported before this default changed
# used it) -- kept for reproducing those runs, not for new work.
#
# "none" removes the channel entirely for the no-gain ablation: the model must produce
# absolute level without the supplied per-note control.
#
# All three are valid provided train, evaluation, and inference use the same supplied-gain
# contract.
GAIN_MODES = ("mean", "max", "none")

# --- calibrated on the train split ---------------------------------------------------------
# python -m scripts.perf.calibrate_perf_stats --split train
# (1208436 frames, 306 recordings; raw-unit variance ratio pitch/envelope 58.4x)
# Re-run and paste again if the control cache or the hold-forward convention changes.
PITCH_MIDI_MEAN, PITCH_MIDI_STD = 55.1702, 10.3574
ENVELOPE_MEAN, ENVELOPE_STD = -4.7507, 1.3552

# python -m scripts.perf.calibrate_perf_stats --split train --signal ddsp_loudness_fixed
# (1208436 frames, 306 recordings; raw-unit variance ratio pitch/ddsp_loudness_fixed 1380.1x)
# official-run/pitch-loudness/ -- the pitch+ddsp-loudness(fixed) arm, paralleling pitch+envelope.
DDSP_LOUDNESS_FIXED_MEAN, DDSP_LOUDNESS_FIXED_STD = 1.3958, 0.2788

MIDI_MIN, MIDI_MAX = 40.0, 88.0     # GuitarSet's playable range, for scaling the score channel
SEGMENT_ONSET_DECAY = 0.6           # per-frame decay of the onset impulse conditioning channel
# Deadband width as a multiple of the recording's OWN measured jitter, not a fixed constant --
# same convention as experiments/corruption_pilot's D3 condition.
DEADBAND_MULT = 2.0

# EnvelopeSignal computes log(sqrt(mean(x^2) + eps) + eps) with eps 1e-5, so a silent frame
# floors at log(2e-5). Used to pad a short string instead of zero, which in log-RMS is RMS 1.0.
ENVELOPE_SILENCE = float(np.log(2e-5))
# ddsp_loudness_fixed is _scale_db(db) = db/DB_RANGE + 1 with DB_RANGE=-80 (dataset/control/
# ddsp_guitar.py) -- INCREASING as db gets more negative (scale_db(0)=1.0, scale_db(-80)=2.0,
# scale_db(-100)=2.25), i.e. a LARGER scaled value means QUIETER, the reverse of what its name
# suggests (the same orientation this session's "why does loudness trend opposite to ours"
# question was about; scripts/data/plot_feature_sets.py inverts it back with (1-scaled)*80
# for display). So the SILENT end of this channel is scale_db(-80) = 2.0, not 0.0 -- 0.0 would
# be scale_db(+80), an unreachable, implausibly LOUD value. Measured real voiced audio (this
# calibration's own p1..p99) sits in [0.85, 1.81], i.e. strictly below 2.0, consistent with
# true silence sitting just past the quietest real frame.
DDSP_LOUDNESS_FIXED_SILENCE = 2.0

# python -m scripts.perf.calibrate_perf_stats --split train --signal ddsp_centroid_fixed
# (1208436 frames, 306 recordings; raw-unit variance ratio pitch/ddsp_centroid_fixed 5818.3x)
DDSP_CENTROID_FIXED_MEAN, DDSP_CENTROID_FIXED_STD = 0.7416, 0.1358
# centroid_scaled's digital-silence value is analytically 0.0, not measured: a fully-silent
# (NaN) frame is repaired to raw Hz 0 by _replace_nan_with_previous (dataset/control/
# ddsp_guitar.py), and _hz_to_unit clips any input at/below CENTROID_MIN_HZ to the unit
# range's own floor -- so silence maps to exactly 0.0 after scaling, the same kind of
# derived-not-measured floor as DDSP_LOUDNESS_FIXED_SILENCE above.
DDSP_CENTROID_FIXED_SILENCE = 0.0
# UNUSED by design, not calibrated: mask_lo/hi/invert exist on every EXTRA_CHANNEL_SPECS
# entry for schema uniformity, but "is centroid itself above threshold X" is not a coherent
# audibility gate the way "is envelope/loudness above X" is -- centroid measures spectral
# brightness, not loudness. scripts/perf/train_perf_predictor.py masks EVERY loss term
# (pitch and every extra channel, centroid included) using extra_channels[0]'s (the
# loudness-like anchor channel's) mask, never a non-anchor channel's own mask_lo/hi. See
# that script's run_split for the reasoning.
DDSP_CENTROID_FIXED_MASK_LO, DDSP_CENTROID_FIXED_MASK_HI = -3.0, 3.0
DDSP_CENTROID_FIXED_MASK_INVERT = False

# model/perf/perf_losses.py's loudness_weight() turns a z-scored second-channel value into a
# "how audible is this frame" mask, via a (lo, hi) pair in that channel's OWN z-scored units
# -- PITCH_MASK_LO/HI (-0.5, 1.0) is envelope's. ddsp_loudness_fixed's z-score runs the
# OPPOSITE direction (see DDSP_LOUDNESS_FIXED_SILENCE's comment above: higher raw/z =
# QUIETER), so its mask needs its OWN (lo, hi) AND loudness_weight's invert=True -- reusing
# envelope's (-0.5, 1.0) unmodified would score quiet frames as audible and loud frames as
# silent, inverting every pitch-masking decision downstream.
#
# These two numbers are NOT a proportional guess from the raw percentiles -- a first attempt
# at that (-0.8, 1.0) gave a mask with 33.6% FALSE POSITIVE rate against the real cached
# `voiced` flag (unvoiced/hold-forward frames wrongly scored "audible"), against envelope's
# own mask's 6.6% on the SAME 60-recording sample -- and it trained a stage-1 predictor to
# 113.9 pitch-cents, 3.3x worse than the envelope arm's 34.3, because over half its "audible"
# frames were actually hold-forward filler. Fixed by grid-searching (lo, hi) directly against
# the real `voiced` flag (recall - FPR, scanning lo over the channel's own z-range and hi over
# several widths) to match envelope's own operating point: envelope's mask gets recall 0.879 /
# FPR 0.066 on real voiced frames; this grid search's best pair matches it closely at recall
# 0.897 / FPR 0.084. Re-run the same search (or extend calibrate_perf_stats.py to do it) if
# the control cache or hold-forward convention changes.
DDSP_LOUDNESS_FIXED_MASK_LO, DDSP_LOUDNESS_FIXED_MASK_HI = -1.25, 0.25

# Everything an EXTRA (non-pitch) target/conditioning channel needs: which cache array to
# read, the RAW-unit (mean, std, silence-floor) triple to normalise it with, and the
# loudness_weight() (lo, hi, invert) triple to mask pitch supervision by it. Every consumer
# downstream is calibrated assuming this channel is z-scored to roughly N(0,1) -- so a new
# entry MUST get its own fresh calibration via calibrate_perf_stats.py, not skip
# normalisation just because its raw units already happen to be bounded (ddsp_controls'
# OWN 4-feature predictor, train_ddsp4_predictor.py, can skip it because it does not share
# loudness_weight's calibration; this dataset's masking does).
#
# ControlFlowDataset.extra_channels is a LIST of these keys (default ["envelope"], matching
# every arm before this dict supported more than one) -- target column 0 is always pitch,
# columns 1.. are each entry in list order. extra_channels[0] additionally anchors
# pseudo_velocity's per-note gain (see _get_full): a note's "loudness" for gain purposes is
# always the first/primary channel, never a channel like centroid that isn't loudness-like.
EXTRA_CHANNEL_SPECS = {
    "envelope": dict(cache_key="envelope", mean=ENVELOPE_MEAN, std=ENVELOPE_STD,
                     silence=ENVELOPE_SILENCE, mask_lo=PITCH_MASK_LO, mask_hi=PITCH_MASK_HI,
                     mask_invert=False),
    "ddsp_loudness_fixed": dict(cache_key="ddsp_loudness_fixed", mean=DDSP_LOUDNESS_FIXED_MEAN,
                                std=DDSP_LOUDNESS_FIXED_STD, silence=DDSP_LOUDNESS_FIXED_SILENCE,
                                mask_lo=DDSP_LOUDNESS_FIXED_MASK_LO,
                                mask_hi=DDSP_LOUDNESS_FIXED_MASK_HI, mask_invert=True),
    "ddsp_centroid_fixed": dict(cache_key="ddsp_centroid_fixed", mean=DDSP_CENTROID_FIXED_MEAN,
                                std=DDSP_CENTROID_FIXED_STD, silence=DDSP_CENTROID_FIXED_SILENCE,
                                mask_lo=DDSP_CENTROID_FIXED_MASK_LO,
                                mask_hi=DDSP_CENTROID_FIXED_MASK_HI,
                                mask_invert=DDSP_CENTROID_FIXED_MASK_INVERT),
}


def filter_pitch(pitch_hz: np.ndarray, voiced: np.ndarray, mode: str) -> np.ndarray:
    """Clean frame-level jitter out of the cached CREPE pitch before it becomes a target.

    Motivated by measurement, not taste: run through the frozen synth, a Hampel-filtered
    pitch curve scored BETTER than the raw noisy one (pitch_corr_voiced 0.903 vs 0.901,
    all-frame 0.920 vs 0.914), while plain moving-average smoothing scored worse (0.866) --
    so removing outliers helps and blurring real structure hurts. Asking a regressor to fit
    frame noise it cannot possibly predict just spends capacity on the unpredictable, which
    is the same conditional-mean trap that flattens vibrato.

    Filters live in dataset/control/corrupt.py and operate per voiced span, so none of them
    smear across a silence gap into the next note.
    """
    if mode in ("none", "", None):
        return pitch_hz
    from dataset.control.corrupt import (deadband_pitch_within_notes,
                                         hampel_pitch_within_notes,
                                         measure_pitch_jitter_std,
                                         smooth_pitch_within_notes)
    p2 = pitch_hz.reshape(-1, 1)
    v = voiced.reshape(-1, 1)
    if mode == "hampel":
        return hampel_pitch_within_notes(p2, v).reshape(-1)
    if mode == "deadband":
        band = DEADBAND_MULT * measure_pitch_jitter_std(p2, v)
        return deadband_pitch_within_notes(p2, v, band).reshape(-1)
    if mode == "mean":
        return smooth_pitch_within_notes(p2, v, window=3).reshape(-1)
    raise ValueError(f"unknown pitch filter {mode!r}")


def hold_forward_pitch(pitch_hz: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Raw cached pitch (0 Hz in silence) -> continuous MIDI, holding the last voiced value
    through unvoiced stretches and back-filling before the first note. See this module's
    header for why 0 Hz is not an option."""
    hz = np.asarray(pitch_hz, dtype=np.float64).reshape(-1)
    ok = (np.asarray(voiced).reshape(-1) > 0.5) & (hz > 1.0)
    if not ok.any():
        return np.full(len(hz), MIDI_MIN, dtype=np.float32)
    idx = np.where(ok, np.arange(len(hz)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = hz[idx]
    first = int(np.argmax(ok))
    filled[:first] = hz[first]                       # back-fill the lead-in
    return hz_to_midi(filled).astype(np.float32)


def per_note_gain_frames(env: np.ndarray, onsets: List[int], T: int, stat: str,
                          fill: float) -> np.ndarray:
    """Per-note statistic of `env` (a raw anchor-channel curve), broadcast onto a (T,)
    frame grid, held constant from each note's onset up to the NEXT note's onset -- not its
    own annotated duration end, since a ringing string is still "owned" by this note until
    something else starts. This is the single definition of "per-note loudness" shared by
    pseudo_velocity's supplied gain (_get_full below) and the note-gain predictor's
    ground-truth target (dataset/perf/note_gain_dataset.py) -- both call this, so the two
    can never define loudness differently by accident.

    `onsets` are frame indices, sorted, one per note in this string's sequence. `stat` is
    "max" or "mean" (this module's GAIN_MODES, minus "none"). `fill` covers any frames
    before the first onset.
    """
    vel = np.full(T, fill, dtype=np.float32)
    for k, a in enumerate(onsets):
        b = onsets[k + 1] if k + 1 < len(onsets) else T
        if b > a:
            seg = env[a:b]
            vel[a:b] = seg.max() if stat == "max" else seg.mean()
    return vel


def per_note_gain_values(env: np.ndarray, onsets: List[int], T: int, stat: str) -> np.ndarray:
    """Same onset-to-next-onset statistic as `per_note_gain_frames`, but returns one scalar
    per note (onset order) instead of a broadcast frame array -- what the note-gain
    predictor's dataset wants, without materialising a whole (T,) array per string."""
    values = np.empty(len(onsets), dtype=np.float32)
    for k, a in enumerate(onsets):
        b = onsets[k + 1] if k + 1 < len(onsets) else T
        seg = env[a:b] if b > a else env[a:a + 1]
        values[k] = seg.max() if stat == "max" else seg.mean()
    return values


def broadcast_note_values_to_frames(values: np.ndarray, onsets: List[int], T: int,
                                    fill: float) -> np.ndarray:
    """Broadcast one value per note (onset order) onto a (T,) frame grid, held constant
    onset-to-next-onset -- the inverse of `per_note_gain_values`: the values are already
    computed (e.g. a note-gain predictor's output), not derived from a raw curve here. Same
    span rule as `per_note_gain_frames`/`per_note_gain_values`, so a predicted-gain render
    and the GT-gain render it is compared against are never broadcast differently."""
    frames = np.full(T, fill, dtype=np.float32)
    for k, a in enumerate(onsets):
        b = onsets[k + 1] if k + 1 < len(onsets) else T
        if b > a:
            frames[a:b] = values[k]
    return frames


def denormalize(x: np.ndarray) -> Dict[str, np.ndarray]:
    """(T, 2) model output in normalised space -> {"pitch": Hz, "envelope": log-RMS}, the
    units config/ctrl-lfm-guitarset-joint-pe.yaml's synth expects to be conditioned on."""
    midi = x[..., 0] * PITCH_MIDI_STD + PITCH_MIDI_MEAN
    env = x[..., 1] * ENVELOPE_STD + ENVELOPE_MEAN
    return {"pitch": midi_to_hz(midi).astype(np.float32), "envelope": env.astype(np.float32)}


def collate_control_flow(batch: List[Dict]) -> Dict:
    """Pads to the batch's max FRAME count, then re-flattens -- same reason
    collate_joint_synth cannot use collate_variable_length: padding the flattened S*T axis
    would append frames to the last string instead of extending all six."""
    S = batch[0]["n_strings"]
    max_T = max(i["n_frames"] for i in batch)

    def pad_flat(x: torch.Tensor, T: int) -> torch.Tensor:
        v = x.reshape(S, T, -1)
        if T < max_T:
            v = torch.cat([v, torch.zeros((S, max_T - T, v.shape[-1]), dtype=v.dtype)], dim=1)
        return v.reshape(S * max_T, -1)

    return {
        "recording_id": [i["recording_id"] for i in batch],
        "target": torch.stack([pad_flat(i["target"], i["n_frames"]) for i in batch]),
        "controls": {n: torch.stack([pad_flat(i["controls"][n], i["n_frames"]) for i in batch])
                     for n in batch[0]["controls"]},
        "valid_mask": torch.stack([
            pad_flat(i["valid_mask"].unsqueeze(-1).float(), i["n_frames"]).squeeze(-1).bool()
            for i in batch]),
        "string_valid": torch.stack([i["string_valid"] for i in batch]),
        "player_id": torch.stack([i["player_id"] for i in batch]),
        "n_strings": S,
        "n_frames": max_T,
    }


class ControlFlowDataset(torch.utils.data.Dataset):
    """One item per recording: (S*T, 1+len(extra_channels)) normalised control-curve target
    + score-only conditioning, sharing ONE crop window across all six strings so they stay
    aligned. Target column 0 is always pitch; columns 1.. are extra_channels in order."""

    def __init__(self, cfg, split: Optional[str] = None, deterministic: bool = False,
                 pitch_filter: str = "none", gain_mode: str = "max",
                 full_length: bool = False, extra_channels: Optional[List[str]] = None,
                 detail_features: str = "none"):
        assert gain_mode in GAIN_MODES, f"gain_mode must be one of {GAIN_MODES}"
        self.extra_channels = list(extra_channels) if extra_channels else ["envelope"]
        assert self.extra_channels, "extra_channels must be non-empty"
        unknown = [c for c in self.extra_channels if c not in EXTRA_CHANNEL_SPECS]
        assert not unknown, \
            f"extra_channels must all be one of {list(EXTRA_CHANNEL_SPECS)}, got unknown {unknown}"
        self.cfg = cfg
        # extra_channels[0] is the loudness-like ANCHOR: it backs pseudo_velocity's per-note
        # gain and every loss term's audibility mask (see EXTRA_CHANNEL_SPECS's own comment
        # and scripts/perf/train_perf_predictor.py's run_split) -- a non-loudness channel
        # like centroid must never be placed first.
        self._sc_list = [EXTRA_CHANNEL_SPECS[c] for c in self.extra_channels]
        self.deterministic = deterministic
        # Evaluation must not crop to the TRAINING window. scripts/synth/infer.py's header makes
        # the same point: max_segment_length is a training-time augmentation bound, and
        # 256 frames is only 11.9 s at 21.53 Hz, so leaving it on silently truncates every
        # ~30 s GuitarSet recording to a third of its length. That is what limited an
        # earlier full-pipeline comparison to 238 s of audio in total, which was too little
        # to estimate a 128-dim FAD covariance from.
        self.full_length = full_length
        self.pitch_filter = pitch_filter
        self.gain_mode = gain_mode
        self.detail_features = detail_features
        self.detail_dim = detail_feature_dim(detail_features)
        # the channel is dropped from the conditioning entirely rather than zero-filled, so a
        # "no gain" run cannot quietly learn to exploit a constant input
        self.cond_channels = [c for c in COND_CHANNELS
                              if not (c == "pseudo_velocity" and gain_mode == "none")]
        self.n_strings = N_STRINGS
        self.min_segment_length = cfg.dataset.min_segment_length
        self.max_segment_length = cfg.dataset.max_segment_length
        self.random_crop = cfg.dataset.random_crop
        self.note_shape_padding_frames = int(
            getattr(cfg.dataset, "note_shape_padding_frames", 2)
        )

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
        self._cache: Dict[str, Dict] = {}

    def _build_recording_index(self) -> Dict[str, Dict[int, str]]:
        by_rec: Dict[str, Dict[int, str]] = {}
        for sample_id, sample in self.sample_by_id.items():
            if sample.meta.get("dataset") != self.hex_dataset or "string" not in sample.meta:
                continue
            if sample_id not in self.control_store:
                continue
            grp = self.control_store[sample_id]
            required = ("pitch", "voiced") + tuple(sc["cache_key"] for sc in self._sc_list)
            if not all(n in grp for n in required):
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
            self._fr = float(self.control_store[sid]["pitch"].attrs.get("frame_rate", 21.533203125))
        return self._fr

    def _get_full(self, recording_id: str) -> Dict:
        if recording_id in self._cache:
            return self._cache[recording_id]
        strings = self._recording_strings[recording_id]
        fr = self._frame_rate()
        S = self.n_strings
        T = max(self.control_store[sid]["pitch"].shape[0] for sid in strings.values())

        target = np.zeros((S, T, 1 + len(self.extra_channels)), dtype=np.float32)
        # RAW cached pitch in Hz, kept alongside the normalised target: this is exactly what
        # the frozen synth was conditioned on at training time, so anything feeding curves
        # back through it must use these values, not the hold-forward MIDI reconstruction.
        pitch_hz = np.zeros((S, T), dtype=np.float32)
        cond = {n: np.zeros((S, T, 1), dtype=np.float32) for n in self.cond_channels}
        score_pitch_reference = np.full((S, T, 1), MIDI_MIN, dtype=np.float32)
        string_valid = np.zeros(S, dtype=bool)
        frame_valid = np.zeros((S, T), dtype=bool)
        note_spans: List[tuple[int, int, int]] = []
        note_core_spans: List[tuple[int, int, int]] = []
        note_detail_features: List[np.ndarray] = []

        for s_idx, sid in strings.items():
            grp = self.control_store[sid]
            n = grp["pitch"].shape[0]
            # T is the max over strings, so a string whose cache is a frame shorter gets
            # padded. _pad_to fills with ZERO, which is right for pitch (Hz) and voiced but
            # catastrophic for envelope: envelope is log-RMS, so 0 means RMS 1.0 -- FULL
            # SCALE, not silence. That put a six-string full-scale impulse in the final frame
            # of every full-length item; the loss never saw it (frame_valid excludes it) but
            # the oracle render path feeds the target straight to the synth, so all 36 oracle
            # renders ended with a bang 9-12 dB louder than the whole rest of the recording,
            # and that peak then set the gain of every peak-normalised render.
            pitch = _pad_to(grp["pitch"][:T], T).reshape(-1)[:T]
            extra_raw = []
            for sc in self._sc_list:
                e = grp[sc["cache_key"]][:T].reshape(-1)[:T]
                if e.shape[0] < T:                      # pad with the silence floor, not zero
                    e = np.concatenate([e, np.full(T - e.shape[0], sc["silence"], dtype=e.dtype)])
                extra_raw.append(e)
            env = extra_raw[0]        # anchor channel's raw values -- pseudo_velocity below
            voiced = _pad_to(grp["voiced"][:T], T).reshape(-1)[:T]

            # filter BEFORE hold-forward: the filters work per voiced span, and hold-forward
            # would otherwise hand them long constant stretches that are not real signal
            pitch = filter_pitch(pitch, voiced, self.pitch_filter)
            pitch_hz[s_idx] = pitch
            midi = hold_forward_pitch(pitch, voiced)
            target[s_idx, :, 0] = (midi - PITCH_MIDI_MEAN) / PITCH_MIDI_STD
            for i, (sc, raw) in enumerate(zip(self._sc_list, extra_raw)):
                target[s_idx, :, 1 + i] = (raw - sc["mean"]) / sc["std"]
            string_valid[s_idx] = True
            frame_valid[s_idx, :n] = True

            # ---- score-only conditioning -----------------------------------------------
            notes = list(notes_for_sample(self.sample_by_id[sid], {}))
            score_midi = np.zeros(T, dtype=np.float32)
            active = np.zeros(T, dtype=bool)
            onsets: List[int] = []
            for onset_s, dur_s, p in notes:
                span = note_to_frame_span(onset_s, dur_s, fr, T)
                if span is not None:
                    a, b = span
                    score_midi[a:b] = p
                    active[a:b] = True
                    note_core_spans.append((s_idx, a, b))
                    if self.detail_dim:
                        note_detail_features.append(extract_pitch_detail_features(
                            pitch[a:b], p, fr, self.detail_features, voiced=voiced[a:b],
                        ))
                    onsets.append(a)
                    note_a, note_b = note_to_frame_span(
                        onset_s, dur_s, fr, T, self.note_shape_padding_frames
                    )
                    note_spans.append((s_idx, note_a, note_b))

            onsets = sorted(set(onsets))

            cond["midi_pitch"][s_idx, :, 0] = np.clip(
                (score_midi - MIDI_MIN) / (MIDI_MAX - MIDI_MIN), 0.0, 1.0)
            cond["voiced"][s_idx, :, 0] = active.astype(np.float32)
            if active.any():
                positions = np.arange(T)
                last = np.where(active, positions, 0)
                np.maximum.accumulate(last, out=last)
                held_score = score_midi[last]
                first = int(np.argmax(active))
                held_score[:first] = score_midi[first]
                score_pitch_reference[s_idx, :, 0] = held_score

            # per-note gain, when it is supplied at all -- see GAIN_MODES above. Always
            # keyed to the ANCHOR channel (extra_channels[0]), never a later one: gain means
            # "how loud", and only the anchor is guaranteed to be a loudness-like measure.
            if self.gain_mode != "none":
                anchor = self._sc_list[0]
                vel = per_note_gain_frames(env, onsets, T, self.gain_mode, anchor["mean"])
                cond["pseudo_velocity"][s_idx, :, 0] = (vel - anchor["mean"]) / anchor["std"]

            # segment-aligned channels: where in the current excitation segment are we, and
            # how recently did it start. These are what make arm C "segment-aligned" without
            # any intermediate per-note representation.
            phase = np.zeros(T, dtype=np.float32)
            imp = np.zeros(T, dtype=np.float32)
            for k, a in enumerate(onsets):
                b = onsets[k + 1] if k + 1 < len(onsets) else T
                if b > a:
                    phase[a:b] = np.linspace(0.0, 1.0, b - a, dtype=np.float32)
                    imp[a] = 1.0
            for i in range(1, T):  # exponential decay, so "just plucked" is a smooth cue
                imp[i] = max(imp[i], imp[i - 1] * SEGMENT_ONSET_DECAY)
            cond["segment_phase"][s_idx, :, 0] = phase
            cond["segment_onset"][s_idx, :, 0] = imp

        detail = (
            broadcast_detail_features(
                note_detail_features, note_core_spans, S, T, self.detail_features,
            )
            if self.detail_dim else np.zeros((S, T, 0), dtype=np.float32)
        )
        entry = {"target": target, "pitch_hz": pitch_hz, "controls": cond,
                 "score_pitch_midi": score_pitch_reference, "detail_features": detail,
                 "note_spans": note_spans, "note_core_spans": note_core_spans,
                 "string_valid": string_valid, "frame_valid": frame_valid, "T": T}
        self._cache[recording_id] = entry
        return entry

    def _pick_window(self, available: int):
        if self.full_length:
            return 0, available
        if self.deterministic:
            return 0, min(self.max_segment_length, available)
        length = min(random.randint(self.min_segment_length, self.max_segment_length), available)
        start = random.randint(0, available - length) if (self.random_crop and available > length) else 0
        return start, length
    def _note_masks_for_window(
        self, full: Dict, start: int, length: int, span_key: str = "note_spans",
    ) -> torch.Tensor:
        masks = []
        for string_index, note_start, note_end in full[span_key]:
            if note_start < start or note_end > start + length:
                continue
            row = np.zeros((self.n_strings, length), dtype=bool)
            row[string_index, note_start - start:note_end - start] = True
            masks.append(row.reshape(self.n_strings * length))
        if not masks:
            return torch.zeros((0, self.n_strings * length), dtype=torch.bool)
        return torch.from_numpy(np.asarray(masks, dtype=bool))

    def _player_id(self, recording_id: str) -> int:
        sid = next(iter(self._recording_strings[recording_id].values()))
        return int(self.sample_by_id[sid].meta["player"])

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict:
        rec = self._index[idx]
        full = self._get_full(rec)
        S = self.n_strings
        start, length = self._pick_window(full["T"])
        sl = slice(start, start + length)
        valid = full["frame_valid"][:, sl] & full["string_valid"][:, None]

        note_mask = self._note_masks_for_window(full, start, length)
        note_core_mask = self._note_masks_for_window(
            full, start, length, span_key="note_core_spans"
        )
        controls = {n: torch.from_numpy(
            np.ascontiguousarray(full["controls"][n][:, sl].reshape(S * length, -1)))
            for n in self.cond_channels}
        controls["score_pitch_midi"] = torch.from_numpy(
            np.ascontiguousarray(full["score_pitch_midi"][:, sl].reshape(S * length, -1))
        )
        return {
            "recording_id": rec,
            "n_strings": S,
            "n_frames": length,
            "target": torch.from_numpy(
                np.ascontiguousarray(full["target"][:, sl].reshape(S * length, -1))),
            "detail_features": torch.from_numpy(
                np.ascontiguousarray(full["detail_features"][:, sl].reshape(
                    S * length, self.detail_dim
                ))
            ),
            "controls": controls,
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid.reshape(S * length))),
            "note_mask": note_mask,
            "note_core_mask": note_core_mask,
            "string_valid": torch.from_numpy(full["string_valid"].copy()),
            "player_id": torch.tensor(self._player_id(rec), dtype=torch.long),
        }
