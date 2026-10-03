# Score -> audio, through the full three-model pipeline this repo trains and evaluates
# with (see eval/renderers/full_pipeline.py, the official evaluation render, and
# scripts/synth/technique_steering.py, whose build_score_cond/build_event_channels this
# reuses the design of):
#
#   1. NoteGainPredictor  (score-only note features -> per-note loudness, "velocity")
#   2. EnvelopeRegressor  (score + predicted loudness -> per-frame pitch/envelope curves)
#   3. JointSynthDecoder  (curves -> six-string audio latents, by rectified-flow sampling)
#
# All three are FROZEN checkpoints from this repo's own training runs; nothing here is
# trained. Unlike full_pipeline.py (which reads score conditioning out of a cached
# GuitarSet recording), every input here comes from a GuitarPro file's own notes -- there
# is no real recording backing this audio at all, by construction.
#
# WINDOWING. full_pipeline.py windows only the SYNTH (quadratic attention over S*T,
# already OOMs past ~5905 GuitarSet frames) and runs the perf predictor whole-recording in
# one shot, because no GuitarSet recording exceeds that. A full song (this module's
# target) can run past 4000 frames per string -- 4-5x anything either model has been run
# on before, in a codebase that has already been bitten once by out-of-distribution
# sequence length (eval/renderers/full_pipeline.py's own docstring: a 55-frame clip under
# min_segment_length produced 417 cents of pitch error). So this module windows both.
#
# The SYNTH keeps full_pipeline.py's sliding hop-overlap scheme (render_synth_windowed):
# validated there via measured tail-loudness/quiet-block behaviour, and correctness here
# was confirmed by a systematic 36-recording GuitarSet check (no envelope/pitch
# degradation past a window boundary -- see docs/reports/active/perf-note-gain-redesign.md's
# sibling investigation).
#
# STAGE 2 (the perf predictor, predict_curves_chunked) does NOT use that scheme. Measured
# directly on this module's own long song: a sliding hop-192/window-256 stitch (matching
# the synth's own parameters) gave RAW, UNBLENDED per-window pitch MAE of 8.7 cents for
# the window starting at frame 0, and 24-123 cents for every other window -- a real,
# repeatable 3-14x gap, not a blending artefact (confirmed by comparing the SAME real
# frames predicted via two different window calls: 8.8 cents when they fell late inside a
# window that started at frame 0, 89.0 cents when the identical frames fell early inside a
# window that started mid-song). The fix that closed nearly all of that gap: prepending
# 0.5s of SILENCE to every non-initial window's conditioning. GuitarSet recordings
# virtually always open with a brief silence before the first note, and this predictor's
# own validation loop (ControlFlowDataset._pick_window with deterministic=True) always
# evaluates crops starting at frame 0 of a recording -- so "silence immediately followed by
# real content" is the ONLY start-of-sequence shape this checkpoint's selected epoch was
# ever actually measured against. A window starting abruptly mid-phrase, with no such lead-
# in, sits in a regime nothing ever validated -- and the model's BIDIRECTIONAL attention
# (no causal mask) lets that unfamiliar opening drag down the whole window's prediction,
# not just its first few frames (measured: with the silence prefix, error was uniformly
# low across the entire window, 8-14 cents throughout, not just near the front).
#
# predict_curves_chunked implements that fix as a real windowing scheme instead of a
# one-off patch: cut only at NOTE BOUNDARIES (never mid-sustain, on any string) so a chunk
# never contains a truncated note; carry the last ~1s of already-finalised notes forward as
# CONTEXT into the next chunk (so the model isn't reasoning from zero musical memory); and
# prefix every non-initial chunk with 0.5s of silence so it presents the same "recording
# start" shape the checkpoint was actually validated on. Each chunk's context+pad prefix is
# re-predicted for the model's benefit but discarded from the stitched output -- only a
# chunk's own new content is kept, concatenated directly (no overlap-add needed, since
# chunks no longer overlap in what they contribute).
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from config.config import load_config
from dataset.control.signal import note_to_frame_span, onset_to_frame
from dataset.latent.sao3 import latent_registry
from dataset.perf.note_gain_dataset import (
    LOG_DURATION_MEAN, LOG_DURATION_STD, LOG_IOI_MEAN, LOG_IOI_STD, MAX_GAP_S,
)
from dataset.perf.perf_curve_dataset import (
    EXTRA_CHANNEL_SPECS, MIDI_MAX, MIDI_MIN, N_STRINGS, SEGMENT_ONSET_DECAY,
    broadcast_note_values_to_frames,
)
from model.common.flow_matching import FlowMatching
from model.perf.note_gain_predictor import NoteGainPredictor
from model.perf.perf_predictor import EnvelopeRegressor
from model.synth.joint_synth import JointSynthDecoder
from model.synth.mix_reverb import MultiChannelReverb
from scripts.perf.train_perf_predictor import denorm_env, denorm_pitch

from inference.gp_score import GuitarPart

PERF_CONFIG = "config/perf.yaml"
PERF_CKPT = "checkpoints/perf/pe_note_shape_fixed13.pt"
SYNTH_CONFIG = "config/synth.yaml"
SYNTH_CKPT = "checkpoints/synth/joint_synth_peov_embed.pt"
# The work report's user-selected reference checkpoint (docs/reports/active/
# perf-note-gain-redesign.md): same_string_next delta scope, fixed weight=1.0. Chosen
# there for simplicity + performance among the note-gain-predictor's own per-note metrics
# -- NOT because it closes the audio-quality gap that work ID measured and left open (see
# this module's own README for what that means for the audio this script produces).
GAIN_CKPT = "checkpoints/gain/note_gain_v5_samestring_fixed.pt"

NFE, CFG_SCALE = 50, 2.0
WINDOW_FRAMES, HOP_FRAMES = 256, 192   # synth's own sliding-window blend (unchanged)
# predict_curves_chunked's note-boundary scheme (see module docstring): every non-initial
# chunk gets this much silence prefixed, then this much already-finalised note context.
PAD_S, CONTEXT_S = 0.5, 1.0
GAIN_INPUT_NAMES = ["midi_pitch", "log_duration", "log_ioi_prev", "log_ioi_next"]


def _log_z(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.log(x + 1e-6) - mean) / std).astype(np.float32)


@dataclass
class Models:
    device: str
    gain_model: NoteGainPredictor
    perf_model: EnvelopeRegressor
    synth_controls: List[str]
    flow: FlowMatching
    encoder: object
    latent_dim: int
    reverb: Optional[MultiChannelReverb]


def load_models(device: str, perf_config: str = PERF_CONFIG, perf_ckpt: str = PERF_CKPT,
                 synth_config: str = SYNTH_CONFIG, synth_ckpt: str = SYNTH_CKPT,
                 gain_ckpt: str = GAIN_CKPT, reverb_ckpt: Optional[str] = None) -> Models:
    gck = torch.load(gain_ckpt, map_location=device, weights_only=False)
    assert gck["input_names"] == GAIN_INPUT_NAMES, (
        f"note-gain checkpoint expects inputs {gck['input_names']}, this module builds "
        f"{GAIN_INPUT_NAMES}"
    )
    gain_model = NoteGainPredictor(
        gck["input_names"], {n: 1 for n in gck["input_names"]},
        d_model=gck["d_model"], depth=gck["depth"], n_heads=gck["n_heads"],
    ).to(device)
    gain_model.load_state_dict(gck["model"])
    gain_model.eval()

    pcfg = load_config(perf_config)
    pck = torch.load(perf_ckpt, map_location=device, weights_only=False)
    assert pck["targets"] == ["pitch", "envelope"], \
        f"expected a pitch+envelope regression checkpoint, got targets={pck['targets']}"
    assert not pck.get("n_bins") and not pck.get("hybrid"), \
        "classification/hybrid perf-predictor heads are not supported here"
    assert pck["gain_mode"] == gck["gain_mode"], (
        f"perf checkpoint gain_mode={pck['gain_mode']!r} must match note-gain checkpoint "
        f"gain_mode={gck['gain_mode']!r}"
    )
    perf_model = EnvelopeRegressor(
        pck["input_names"], {n: 1 for n in pck["input_names"]}, out_dim=len(pck["targets"]),
        n_players=pck.get("n_players"), d_model=pck.get("d_model", 256), depth=pck.get("depth", 6),
    ).to(device)
    perf_model.load_state_dict(pck["model"])
    perf_model.eval()

    scfg = load_config(synth_config)
    synth_controls = list(scfg.dataset.control_signals)
    synth = JointSynthDecoder(scfg.model, synth_controls, {n: 1 for n in synth_controls}).to(device)
    synth.load_state_dict(torch.load(synth_ckpt, map_location=device, weights_only=False)["model"])
    synth.eval()
    flow = FlowMatching(synth, scfg.model.flow_matching)

    encoder = latent_registry.get(pcfg.data.latent)(pcfg.latent)
    encoder.load()

    reverb = None
    if reverb_ckpt:
        rck = torch.load(reverb_ckpt, map_location=device, weights_only=False)
        reverb = MultiChannelReverb(encoder.sample_rate, residual=bool(rck.get("residual", False))).to(device)
        reverb.load_state_dict(rck["model"])
        reverb.eval()

    return Models(
        device=device, gain_model=gain_model, perf_model=perf_model,
        synth_controls=synth_controls, flow=flow, encoder=encoder,
        latent_dim=scfg.model.latent_dim, reverb=reverb,
    )


# --------------------------------------------------------------------------------------
# Score -> frame grid (mirrors scripts/synth/technique_steering.py's build_score_cond /
# build_event_channels, generalised to a runtime-computed T and an externally supplied,
# PREDICTED pseudo_velocity instead of that script's constant).
# --------------------------------------------------------------------------------------

def frame_count_for_notes(notes_by_string: Dict[int, List[Tuple[float, float, float]]], fr: float) -> int:
    spans = [
        note_to_frame_span(onset_s, dur_s, fr, 1_000_000_000)
        for notes in notes_by_string.values() for onset_s, dur_s, _p in notes
    ]
    spans = [s for s in spans if s is not None]
    return max((b for _a, b in spans), default=1)


def _windows(T: int, window: int, hop: int) -> List[int]:
    starts = list(range(0, max(1, T - window + 1), hop))
    if starts[-1] + window < T:
        starts.append(max(0, T - window))
    return starts


def build_gain_inputs(notes_by_string: Dict[int, List[Tuple[float, float, float]]], n_strings: int = N_STRINGS):
    """Score-only per-note features for NoteGainPredictor, merged across every string of
    ONE guitar into a single onset-time-sorted sequence -- same construction as
    dataset/perf/note_gain_dataset.py's NoteGainDataset.__getitem__, built directly from
    plain note tuples instead of a cached GuitarSet recording."""
    per_string = {s: notes for s, notes in notes_by_string.items() if notes}
    merged = sorted(
        ((onset, s, k) for s, notes in per_string.items() for k, (onset, _d, _p) in enumerate(notes)),
        key=lambda row: row[0],
    )
    M = max(len(merged), 1)
    positions: Dict[int, List[int]] = {s: [] for s in per_string}
    for pos, (_onset, s, _k) in enumerate(merged):
        positions[s].append(pos)

    feats = {name: np.zeros((M, 1), dtype=np.float32) for name in GAIN_INPUT_NAMES}
    string_id = np.zeros(M, dtype=np.int64)
    note_valid = np.zeros(M, dtype=bool)

    for s, notes in per_string.items():
        n = len(notes)
        pos = np.array(positions[s], dtype=np.int64)
        onsets_s = np.array([o for o, _d, _p in notes], dtype=np.float64)
        durs = np.array([d for _o, d, _p in notes], dtype=np.float64)
        pitches = np.array([p for _o, _d, p in notes], dtype=np.float32)

        feats["midi_pitch"][pos, 0] = np.clip((pitches - MIDI_MIN) / (MIDI_MAX - MIDI_MIN), 0.0, 1.0)
        feats["log_duration"][pos, 0] = _log_z(durs, LOG_DURATION_MEAN, LOG_DURATION_STD)
        ioi_prev = np.empty(n, dtype=np.float64)
        ioi_prev[0] = MAX_GAP_S
        ioi_prev[1:] = onsets_s[1:] - onsets_s[:-1]
        ioi_next = np.empty(n, dtype=np.float64)
        ioi_next[:-1] = onsets_s[1:] - onsets_s[:-1]
        ioi_next[-1] = MAX_GAP_S
        feats["log_ioi_prev"][pos, 0] = _log_z(np.clip(ioi_prev, 0.0, None), LOG_IOI_MEAN, LOG_IOI_STD)
        feats["log_ioi_next"][pos, 0] = _log_z(np.clip(ioi_next, 0.0, None), LOG_IOI_MEAN, LOG_IOI_STD)
        string_id[pos] = s
        note_valid[pos] = True

    return feats, string_id, note_valid, per_string


def predict_pseudo_velocity(gain_model: NoteGainPredictor,
                             notes_by_string: Dict[int, List[Tuple[float, float, float]]],
                             T: int, fr: float, device: str, n_strings: int = N_STRINGS) -> np.ndarray:
    """Stage 1: predicted per-note loudness, broadcast to a z-scored (S, T) frame grid --
    the substitute for a real recording's pseudo_velocity (dataset/perf/note_gain_dataset.py's
    predict_gain_frames, reimplemented against plain note tuples instead of a cached
    GuitarSet recording)."""
    feats, string_id, note_valid, per_string = build_gain_inputs(notes_by_string, n_strings)
    controls = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in feats.items()}
    sid_t = torch.from_numpy(string_id).unsqueeze(0).to(device)
    valid_t = torch.from_numpy(note_valid).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_z = gain_model(controls, sid_t, valid_t)[0, :, 0].cpu().numpy()

    spec = EXTRA_CHANNEL_SPECS["envelope"]
    frames = np.full((n_strings, T), spec["mean"], dtype=np.float32)
    for s, notes in per_string.items():
        mask = note_valid & (string_id == s)
        onset_frames = [min(T - 1, max(0, onset_to_frame(o, fr))) for o, _d, _p in notes]
        values_raw = pred_z[mask] * spec["std"] + spec["mean"]
        frames[s] = broadcast_note_values_to_frames(values_raw, onset_frames, T, spec["mean"])
    return ((frames - spec["mean"]) / spec["std"]).astype(np.float32)


def build_score_cond(notes_by_string: Dict[int, List[Tuple[float, float, float]]], T: int, fr: float,
                      pseudo_velocity_z: np.ndarray, n_strings: int = N_STRINGS) -> Dict[str, np.ndarray]:
    """Stage 2's score-only conditioning (dataset/perf/perf_curve_dataset.py's COND_CHANNELS)."""
    cond = {n: np.zeros((n_strings, T, 1), dtype=np.float32) for n in ("midi_pitch", "voiced", "segment_phase", "segment_onset")}
    cond["pseudo_velocity"] = pseudo_velocity_z.reshape(n_strings, T, 1).astype(np.float32)
    for s_idx, notes in notes_by_string.items():
        if not notes:
            continue
        score_midi = np.zeros(T, dtype=np.float32)
        active = np.zeros(T, dtype=bool)
        onsets: List[int] = []
        for onset_s, dur_s, p in notes:
            span = note_to_frame_span(onset_s, dur_s, fr, T)
            if span is None:
                continue
            a, b = span
            score_midi[a:b] = p
            active[a:b] = True
            onsets.append(a)
        onsets = sorted(set(onsets))
        cond["midi_pitch"][s_idx, :, 0] = np.clip((score_midi - MIDI_MIN) / (MIDI_MAX - MIDI_MIN), 0.0, 1.0)
        cond["voiced"][s_idx, :, 0] = active.astype(np.float32)
        phase = np.zeros(T, dtype=np.float32)
        imp = np.zeros(T, dtype=np.float32)
        for k, a in enumerate(onsets):
            b = onsets[k + 1] if k + 1 < len(onsets) else T
            if b > a:
                phase[a:b] = np.linspace(0.0, 1.0, b - a, dtype=np.float32)
                imp[a] = 1.0
        for i in range(1, T):
            imp[i] = max(imp[i], imp[i - 1] * SEGMENT_ONSET_DECAY)
        cond["segment_phase"][s_idx, :, 0] = phase
        cond["segment_onset"][s_idx, :, 0] = imp
    return cond


def build_event_channels(notes_by_string: Dict[int, List[Tuple[float, float, float]]], T: int, fr: float,
                          n_strings: int = N_STRINGS) -> Tuple[np.ndarray, np.ndarray]:
    """Annotation-derived onset/voiced SYNTH controls -- score information the synth was
    trained on, a different construction from build_score_cond's predictor-facing 'voiced'."""
    onset = np.zeros((n_strings, T, 1), dtype=np.float32)
    voiced = np.zeros((n_strings, T, 1), dtype=np.float32)
    frame_dur = 1.0 / fr
    for s_idx, notes in notes_by_string.items():
        for onset_s, dur_s, _p in notes:
            of = int(onset_s / frame_dur)
            if 0 <= of < T:
                onset[s_idx, of, 0] = 1.0
            start_f = max(0, int(np.floor(onset_s / frame_dur)))
            end_f = min(T, int(np.ceil((onset_s + dur_s) / frame_dur)))
            voiced[s_idx, start_f:end_f, 0] = 1.0
    return onset, voiced


# --------------------------------------------------------------------------------------
# Stage 2 (note-boundary chunked, see module docstring) and stage 3 (windowed,
# latent-domain blend as in full_pipeline.py)
# --------------------------------------------------------------------------------------

def _note_boundaries(active: np.ndarray) -> np.ndarray:
    """Every frame position where it is SAFE to start or end a chunk: every note's own
    onset and end frame, on ANY string, plus 0 and T. `active`: (S, T) bool, exactly
    build_score_cond's 'voiced' channel (True over each note's own span). Cutting only at
    one of these positions guarantees a chunk boundary never falls inside a note's
    sustain, on any of the 6 strings -- the same guarantee whether the boundary is a
    chunk's start (context/new-content edge) or its end (new-content/next-chunk edge)."""
    S, T = active.shape
    edges = {0, T}
    for s in range(S):
        row = active[s]
        diff = np.diff(row.astype(np.int8))
        starts = np.nonzero(diff == 1)[0] + 1
        ends = np.nonzero(diff == -1)[0] + 1
        if row[0]:
            starts = np.concatenate([[0], starts])
        if row[-1]:
            ends = np.concatenate([ends, [T]])
        edges.update(int(x) for x in starts)
        edges.update(int(x) for x in ends)
    return np.array(sorted(edges), dtype=np.int64)


def predict_curves_chunked(perf_model: EnvelopeRegressor, cond: Dict[str, np.ndarray], S: int, T: int,
                            device: str, fr: float, max_frame: int = WINDOW_FRAMES,
                            pad_s: float = PAD_S, context_s: float = CONTEXT_S):
    """Note-boundary chunked inference (see module docstring for why -- a sliding hop
    window like the synth's gave a measured 3-14x raw quality gap between chunks that
    start at frame 0 and chunks that don't). Each non-initial chunk is built as
    [pad_frames of silence] + [up to context_s of already-finalised note context] +
    [new content, up to max_frame total] -- cut only at _note_boundaries positions, so no
    note is ever split mid-sustain on any string, on either edge. Only the new-content
    portion of each chunk's output is kept; chunks are concatenated directly (no
    overlap-add), since they no longer overlap in what they contribute."""
    active = cond["voiced"][:, :, 0] > 0.5
    boundaries = _note_boundaries(active)
    pad_frames = int(round(pad_s * fr))
    context_budget = int(round(context_s * fr))

    def largest_leq(x: int, lower_exclusive: int):
        idx = int(np.searchsorted(boundaries, x, side="right")) - 1
        if idx < 0:
            return None
        b = int(boundaries[idx])
        return b if b > lower_exclusive else None

    def smallest_geq(x: int, upper_inclusive: int):
        idx = int(np.searchsorted(boundaries, x, side="left"))
        if idx >= len(boundaries):
            return None
        b = int(boundaries[idx])
        return b if b <= upper_inclusive else None

    pitch_full = np.zeros((S, T), dtype=np.float32)
    env_full = np.zeros((S, T), dtype=np.float32)

    cursor = 0
    first = True
    while cursor < T:
        if first:
            pad, context_start = 0, cursor
        else:
            pad = pad_frames
            context_start = smallest_geq(max(0, cursor - context_budget), cursor)
            if context_start is None:
                context_start = cursor          # no safe boundary in the lookback -> no context
        context_len = cursor - context_start
        remaining = max_frame - pad - context_len
        if remaining <= 0:
            context_start, context_len = cursor, 0     # context alone blew the budget -> drop it
            remaining = max_frame - pad

        end = largest_leq(min(cursor + remaining, T), cursor)
        if end is None:
            # no safe boundary at all within budget (one note longer than the whole chunk) --
            # accept a rare hard cut rather than loop forever
            end = max(min(cursor + remaining, T), cursor + 1)

        w = pad + context_len + (end - cursor)
        sub = {}
        for k, v in cond.items():
            pad_block = np.zeros((S, pad), dtype=np.float32)
            ctx_block = v[:, context_start:cursor, 0]
            new_block = v[:, cursor:end, 0]
            sub[k] = torch.from_numpy(
                np.concatenate([pad_block, ctx_block, new_block], axis=1).reshape(1, S * w, 1)
            ).float().to(device)
        with torch.no_grad():
            pred = perf_model(sub, player_id=None)
        pitch_w = denorm_pitch(pred[..., 0:1]).reshape(S, w).detach().cpu().numpy()
        env_w = denorm_env(pred[..., 1:2]).reshape(S, w).detach().cpu().numpy()

        keep_from = pad + context_len
        pitch_full[:, cursor:end] = pitch_w[:, keep_from:]
        env_full[:, cursor:end] = env_w[:, keep_from:]

        first = False
        cursor = end
    return pitch_full, env_full


def render_synth_windowed(flow: FlowMatching, encoder, synth_controls: List[str], ctl: Dict[str, torch.Tensor],
                           S: int, T: int, D: int, device: str, seed: int, nfe: int = NFE, cfg_scale: float = CFG_SCALE,
                           window: int = WINDOW_FRAMES, hop: int = HOP_FRAMES) -> Tuple[List[np.ndarray], int]:
    torch.manual_seed(seed)
    noise = torch.randn((1, S, T, D), device=device)
    acc = torch.zeros((S, T, D), device=device)
    wsum = torch.zeros((S, T, 1), device=device)
    for a in _windows(T, window, hop):
        w = min(window, T - a)
        sub = {k: ctl[k][:, :, a:a + w].reshape(1, S * w, 1) for k in synth_controls}
        zw = flow.sample(sub, (1, S * w, D), n_steps=nfe, cfg_scale=cfg_scale, device=device,
                          x0=noise[:, :, a:a + w].reshape(1, S * w, D))
        ramp = torch.hann_window(w, periodic=False, device=device).clamp(min=1e-3).view(1, w, 1)
        acc[:, a:a + w] += zw[0].reshape(S, w, D) * ramp
        wsum[:, a:a + w] += ramp
    z = (acc / wsum.clamp(min=1e-6)).cpu().numpy()
    streams = [encoder.decode(z[s]) for s in range(S)]
    ln = min(len(a) for a in streams)
    return streams, ln


def render_guitar_part(part: GuitarPart, models: Models, out_dir: Path, seed: int = 0,
                        nfe: int = NFE, cfg_scale: float = CFG_SCALE) -> dict:
    device = models.device
    fr = models.encoder.frame_rate
    S = N_STRINGS
    T = frame_count_for_notes(part.notes_by_string, fr)

    pseudo_velocity_z = predict_pseudo_velocity(models.gain_model, part.notes_by_string, T, fr, device)
    cond = build_score_cond(part.notes_by_string, T, fr, pseudo_velocity_z)
    onset_ctl, voiced_ctl = build_event_channels(part.notes_by_string, T, fr)
    pitch_hz, env_lograms = predict_curves_chunked(models.perf_model, cond, S, T, device, fr)

    raw = {
        "pitch": pitch_hz.reshape(1, S, T, 1), "envelope": env_lograms.reshape(1, S, T, 1),
        "onset": onset_ctl.reshape(1, S, T, 1), "voiced": voiced_ctl.reshape(1, S, T, 1),
    }
    ctl = {k: torch.from_numpy(np.ascontiguousarray(raw[k])).float().to(device) for k in models.synth_controls}

    streams, ln = render_synth_windowed(
        models.flow, models.encoder, models.synth_controls, ctl, S, T, models.latent_dim, device, seed, nfe, cfg_scale,
    )
    dry = np.sum([a[:ln] for a in streams], axis=0)
    if models.reverb is not None:
        with torch.no_grad():
            stack = torch.from_numpy(np.stack([a[:ln] for a in streams])).float().unsqueeze(0).to(device)
            mix = models.reverb(stack)[0].cpu().numpy()
    else:
        mix = dry

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "mix.wav"), mix, models.encoder.sample_rate, subtype="FLOAT")
    if models.reverb is not None:
        sf.write(str(out_dir / "mix_dry.wav"), dry, models.encoder.sample_rate, subtype="FLOAT")
    for s, a in enumerate(streams):
        sf.write(str(out_dir / f"string_{s + 1}.wav"), a[:ln], models.encoder.sample_rate, subtype="FLOAT")
    np.savez(out_dir / "fed_curves.npz", **{k: raw[k][0, :, :, 0] for k in models.synth_controls})

    return {
        "name": part.name, "program": part.program, "n_notes": part.n_notes,
        "strings_used": part.strings_used, "n_frames": T, "frame_rate": fr,
        "sample_rate": models.encoder.sample_rate, "duration_s": ln / models.encoder.sample_rate,
        "seed": seed, "nfe": nfe, "cfg_scale": cfg_scale,
    }
