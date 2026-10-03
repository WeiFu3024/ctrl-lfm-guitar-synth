# this file is used for control extraction
# pitch: run crepe
# envelope: compute the audio envelope (loudness or log-RMS)
# onset/voiced: derived from ground-truth MIDI/annotation note events, not audio heuristics
# preserve the possibilities of adding more control signal
# IMPORTANT: the control signal resolution should match the latent resolution by default,
# unless specified in the config
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import json
import os

import numpy as np
import torch

from dataset.registry import Registry

if TYPE_CHECKING:
    from dataset.dataset_register import RawSample


@dataclass
class ControlResult:
    values: np.ndarray  # (T, C)
    frame_rate: float


class MissingAnnotationError(Exception):
    """Raised by a MIDI/annotation-derived signal when its required source file isn't
    available for a given sample (e.g. a SynthTab-hex song with no rendered MIDI)."""


class ControlSignal(ABC):
    name: str

    def __init__(self, sample_rate: int, hop_length: int, **params):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.params = params

    @abstractmethod
    def compute(self, audio: np.ndarray, sample: "RawSample") -> ControlResult: ...

    @property
    def is_gpu(self) -> bool:
        """True routes this signal into ControlCacheExtractor's dedicated sequential
        GPU pass instead of the parallel CPU worker pool -- see extractor.py."""
        return False


control_registry = Registry("control_signal")


@control_registry.register("pitch")
class PitchSignal(ControlSignal):
    """Frame-wise fundamental frequency via CREPE (torchcrepe)."""

    name = "pitch"

    def __init__(self, sample_rate: int, hop_length: int, **params):
        super().__init__(sample_rate, hop_length, **params)
        self.fmin = params.get("fmin", 50.0)
        self.fmax = params.get("fmax", 1100.0)
        self.model = params.get("model", "full")
        self.batch_size = params.get("batch_size", 2048)
        # CREPE on CPU is the extraction pipeline's real bottleneck -- it's a CNN run
        # per-frame and gets no benefit from the CPU worker pool's process-level
        # parallelism, just N processes each doing slow single-threaded-equivalent
        # convolution work. Default to GPU; ControlCacheExtractor routes is_gpu signals
        # into one dedicated sequential pass (see extractor.py) rather than the CPU
        # pool, so this doesn't multiply into N competing CUDA contexts. Pass
        # params={"device": "cpu"} to opt back into the old CPU-pool behavior.
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # viterbi (torchcrepe's default) moves probabilities back to CPU and calls
        # librosa.sequence.viterbi -- a pure-CPU dynamic-programming step, so it's
        # still CPU-bound regardless of device. Measured against the GPU-only
        # alternative (weighted_argmax, decodes each frame independently, no
        # cross-frame smoothing): ~15x faster per file, but 14.8% of frames disagreed
        # by more than an octave on real GuitarSet audio -- viterbi's smoothing is
        # worth more than the speedup here. Kept CPU-bound on purpose; it still only
        # runs in ControlCacheExtractor's single sequential is_gpu pass (not the
        # 8-worker pool), so it no longer multiplies across processes even though the
        # decode step itself is CPU. Pass params={"decoder": "weighted_argmax"} to
        # opt into the faster, less accurate path.
        self.decoder = params.get("decoder", "viterbi")

    @property
    def is_gpu(self) -> bool:
        return self.device != "cpu"

    def compute(self, audio: np.ndarray, sample: "RawSample") -> ControlResult:
        import torchcrepe

        decoder_fn = {
            "viterbi": torchcrepe.decode.viterbi,
            "weighted_argmax": torchcrepe.decode.weighted_argmax,
            "argmax": torchcrepe.decode.argmax,
        }[self.decoder]

        audio_t = torch.from_numpy(audio).float().unsqueeze(0)  # (1, N)
        with torch.no_grad():
            pitch = torchcrepe.predict(
                audio_t,
                self.sample_rate,
                self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                decoder=decoder_fn,
                model=self.model,
                batch_size=self.batch_size,
                device=self.device,
            )
        values = pitch.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1, 1)
        return ControlResult(values=values, frame_rate=self.sample_rate / self.hop_length)


@control_registry.register("envelope")
class EnvelopeSignal(ControlSignal):
    """Frame-wise loudness: log-RMS (or plain RMS) over non-overlapping hop_length frames."""

    name = "envelope"

    def __init__(self, sample_rate: int, hop_length: int, **params):
        super().__init__(sample_rate, hop_length, **params)
        self.log = params.get("log", True)
        self.eps = params.get("eps", 1e-5)

    def compute(self, audio: np.ndarray, sample: "RawSample") -> ControlResult:
        hop = self.hop_length
        n_frames = len(audio) // hop
        trimmed = audio[: n_frames * hop]
        frames = trimmed.reshape(n_frames, hop)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + self.eps)
        values = np.log(rms + self.eps) if self.log else rms
        values = values.astype(np.float32).reshape(-1, 1)
        return ControlResult(values=values, frame_rate=self.sample_rate / hop)


def _notes_from_guitarset_jams(sample: "RawSample", annotation_root: str) -> List[Tuple[float, float, float]]:
    """[(onset_s, duration_s, midi_pitch), ...] from GuitarSet's note_midi JAMS annotation
    for this sample's string. data_source (0-5) empirically confirmed to match
    meta["string"]-1 by cross-checking MIDI pitch against cached CREPE pitch."""
    recording_id = sample.sample_id.split("/")[1]
    jams_path = Path(annotation_root) / f"{recording_id}.jams"
    if not jams_path.exists():
        raise MissingAnnotationError(str(jams_path))

    with open(jams_path) as f:
        data = json.load(f)

    data_source = str(sample.meta["string"] - 1)
    for ann in data["annotations"]:
        if ann["namespace"] == "note_midi" and ann["annotation_metadata"].get("data_source") == data_source:
            return [(d["time"], d["duration"], d["value"]) for d in ann["data"]]

    raise MissingAnnotationError(f"no note_midi annotation for string {sample.meta['string']} in {jams_path}")


def _synthtab_track_for_channel(midi_path: str, channel: int) -> "int | None":
    """SynthTab MIDI files are type-1, one MIDI channel per track, string N on channel
    N-1 — verified empirically (a single-string song's lone track used exactly the
    channel matching its string number; a 4-string song had tracks/channels 0..3 in
    matching order). Returns the track index carrying `channel`, or None if absent
    (that string wasn't played in this tab)."""
    import mido

    mid = mido.MidiFile(midi_path)
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type in ("note_on", "note_off") and msg.channel == channel:
                return i
    return None


def _notes_from_synthtab_midi(sample: "RawSample", midi_root: str) -> List[Tuple[float, float, float]]:
    """[(onset_s, duration_s, midi_pitch), ...] from SynthTab-hex's rendered MIDI
    (data/SynthTab, one multi-channel .mid per song, six strings = six channels).
    Verified: MIDI onset times align with independently-detected audio onsets within a
    few ms. Filters out a spurious ~1ms pitch-24 "marker" note observed at t=0 in every
    file (not a real performed note — well below guitar range and far too short)."""
    import pretty_midi

    song = sample.meta["song"]
    midi_path = Path(midi_root) / song / f"{song}.mid"
    if not midi_path.exists():
        raise MissingAnnotationError(str(midi_path))

    channel = sample.meta["string"] - 1
    track_idx = _synthtab_track_for_channel(str(midi_path), channel)
    if track_idx is None:
        raise MissingAnnotationError(f"no notes on channel {channel} (string {sample.meta['string']}) in {midi_path}")

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = pm.instruments[track_idx].notes
    return [(n.start, n.end - n.start, float(n.pitch)) for n in notes if (n.end - n.start) >= 0.01]


def default_guitarset_annotation_root() -> str:
    """Where GuitarSet's JAMS live when a caller does not say.

    config's DataConfig.root already lets the AUDIO root move, but almost every caller of
    notes_for_sample passes {}, so the annotation root had no way to follow it and stayed
    pinned to the repo's data/ directory. That breaks on any machine whose GuitarSet is
    somewhere else -- on `fatcat`, data/ is a mount of a FULL disk and the real tree is
    nested differently, so nothing can even be symlinked into place.

    GUITARSET_ROOT (or GUITARSET_ANNOTATION_ROOT for the annotation dir alone) overrides it
    per machine, which is the right granularity: the layout difference is a property of the
    machine, not of the experiment, so it should not fork the configs.
    """
    explicit = os.environ.get("GUITARSET_ANNOTATION_ROOT")
    if explicit:
        return explicit
    return str(Path(os.environ.get("GUITARSET_ROOT", "data/GuitarSet")) / "annotation")


def notes_for_sample(sample: "RawSample", params: dict) -> List[Tuple[float, float, float]]:
    """[(onset_s, duration_s, midi_pitch), ...] for this sample's string, dispatched by
    dataset type. Public: also used by experiments/note_rate/note_window.py to build per-note windows,
    not just by the onset/voiced signals below."""
    dataset = sample.meta.get("dataset")
    if dataset == "GuitarSet":
        return _notes_from_guitarset_jams(sample, params.get("guitarset_annotation_root",
                                                             default_guitarset_annotation_root()))
    if dataset == "SynthTab-hex":
        return _notes_from_synthtab_midi(sample, params.get("synthtab_midi_root", "data/SynthTab"))
    raise MissingAnnotationError(f"no MIDI/annotation source configured for dataset '{dataset}'")


def onset_to_frame(onset_s: float, frame_rate: float) -> int:
    """Frame index a note onset belongs to: FLOOR, i.e. the frame whose span contains the
    onset instant -- never the nearest one.

    Frames are non-overlapping [k*hop, (k+1)*hop), so the floor frame is the one the attack
    physically happens inside. Rounding to nearest sends an onset in the second half of a
    frame UP to the next one, which leaves the frame carrying the attack assigned to the
    PREVIOUS note -- notes tile the timeline by onset interval here, so a boundary placed one
    frame late does not lose the attack, it files it under the wrong note. That is the exact
    failure experiments/oneoff_analyses/plot_onset_realign.py was built to look at.

    It matters more since realignment: flux-detected onsets sit a median 27 ms before the
    JAMS annotation (dataset/control/onset_align.py), which is 0.6 of a 46 ms control frame --
    right in the range where round() and floor() disagree.

    Matches OnsetSignal/VoicedSignal above, which already floor.
    """
    return int(np.floor(onset_s * frame_rate))


def note_to_frame_span(
    onset_s: float,
    duration_s: float,
    frame_rate: float,
    n_frames: int,
    padding_frames: int = 0,
) -> Optional[Tuple[int, int]]:
    """Convert one MIDI annotation to a clipped half-open frame span."""
    start = onset_to_frame(onset_s, frame_rate)
    end = min(n_frames, start + max(1, int(round(duration_s * frame_rate))))
    if start < 0 or start >= n_frames or end <= start:
        return None
    return max(0, start - padding_frames), min(n_frames, end + padding_frames)


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    """Continuous (non-quantized) MIDI note number from frequency in Hz."""
    hz = np.clip(hz, 1e-6, None)
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """Inverse of hz_to_midi."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


@control_registry.register("onset")
class OnsetSignal(ControlSignal):
    """Frame-wise onset impulse derived from ground-truth MIDI/annotation note events
    (GuitarSet JAMS note_midi, or SynthTab-hex rendered MIDI): 1.0 at the frame
    containing each note's onset time, else 0.0. Raises MissingAnnotationError (caught
    by the extractor) when no MIDI/annotation source exists for a sample."""

    name = "onset"

    def compute(self, audio: np.ndarray, sample: "RawSample") -> ControlResult:
        n_frames = len(audio) // self.hop_length
        frame_dur = self.hop_length / self.sample_rate
        values = np.zeros(n_frames, dtype=np.float32)
        for onset_s, _duration_s, _pitch in notes_for_sample(sample, self.params):
            frame = int(onset_s / frame_dur)
            if 0 <= frame < n_frames:
                values[frame] = 1.0
        return ControlResult(values=values.reshape(-1, 1), frame_rate=self.sample_rate / self.hop_length)


@control_registry.register("voiced")
class VoicedSignal(ControlSignal):
    """Frame-wise binary voicing flag derived from ground-truth MIDI/annotation note
    events: 1.0 for frames within [onset, onset+duration) of any note, else 0.0. Raises
    MissingAnnotationError (caught by the extractor) when no MIDI/annotation source
    exists for a sample."""

    name = "voiced"

    def compute(self, audio: np.ndarray, sample: "RawSample") -> ControlResult:
        n_frames = len(audio) // self.hop_length
        frame_dur = self.hop_length / self.sample_rate
        values = np.zeros(n_frames, dtype=np.float32)
        for onset_s, duration_s, _pitch in notes_for_sample(sample, self.params):
            start_frame = max(0, int(np.floor(onset_s / frame_dur)))
            end_frame = min(n_frames, int(np.ceil((onset_s + duration_s) / frame_dur)))
            values[start_frame:end_frame] = 1.0
        return ControlResult(values=values.reshape(-1, 1), frame_rate=self.sample_rate / self.hop_length)


# Moved here from experiments/note_rate/note_window.py when the note-rate stack was archived: the only
# remaining LIVE caller is dataset/control/corrupt.py's time-warp corruption, and live code
# must never import experiments/. Direction-agnostic by design -- it builds canonical
# (T_note -> K) targets and upsamples decoded curves back (K -> T_note) with the same code.
def phase_resample(x: np.ndarray, k: int) -> np.ndarray:
    """Resamples x -- (T,) or (T, C) -- onto k points evenly spaced over normalized phase
    [0, 1]. Direction-agnostic: used both to build canonical (T_note -> K) training
    targets here, and later, unchanged, to upsample a decoded canonical curve back to a
    note's real frame count (K -> T_note) for synthesis."""
    x = np.asarray(x)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    T, C = x.shape

    if T == 1:
        return np.repeat(x, k, axis=0).squeeze(-1) if squeeze else np.repeat(x, k, axis=0)

    src_phase = np.linspace(0.0, 1.0, T)
    tgt_phase = np.linspace(0.0, 1.0, k)
    # np.interp always returns float64 regardless of input dtype -- cast back so callers
    # get the same dtype they passed in (e.g. float32 targets feeding a torch backward).
    out = np.stack([np.interp(tgt_phase, src_phase, x[:, c]) for c in range(C)], axis=-1).astype(x.dtype)
    return out.squeeze(-1) if squeeze else out
