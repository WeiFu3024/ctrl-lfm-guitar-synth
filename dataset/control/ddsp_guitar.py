# DDSP-Guitar's four synthesis-conditioning features, ported EXACTLY from the vendored
# reference/ddsp-guitar/ implementation rather than re-derived: F0, loudness,
# periodicity, and spectral centroid, the set its synthesis model is conditioned on and
# its ctr-syn-rg / ctr-syn-cl control models predict (reference/ddsp-guitar/repro/
# configs.py's SYNTHESIS_CONFIG["acoustic_features"]).
#
# Every helper below names the reference file and symbol it mirrors, so drift from the
# reference stays traceable. The point of this module is fidelity, not elegance -- where
# their implementation does something surprising (midi-domain scaling, a 4x-hop STFT
# window, NaN repair on centroid), it is reproduced rather than "fixed".
#
# EXTRACTION RATE: features are computed directly on our latent grid (44100 Hz, hop 2048
# = 21.533203125 Hz), NOT at the reference's 48kHz/128Hz. An earlier version did extract at
# their rate and then resample the features down by 5.94x; that was replaced because the
# resampling was a measurable confound -- at that ratio linear interpolation degenerates
# into point-sampling (a lone transient kept only 48% of its energy) while mean pooling
# smeared it, and the two methods differed by 7-35% of each feature's std, more than the
# aliased energy itself. Extracting natively removes the step and the choice entirely, so
# frames land on the latent grid by construction.
#
# What that changes, precisely:
#   - f0 / periodicity: unaffected in character. torchcrepe's analysis window is a fixed
#     1024 samples @16kHz regardless of hop, so only frame spacing differs.
#   - loudness / centroid: the analysis window is set by WINDOW_HOP_RATIO, chosen by a
#     measured sweep rather than by carrying over either of DDSP's numbers -- see the
#     table at that constant.
#
# VERIFIED against the reference implementation itself (not just "it runs"): feeding the
# same audio to reference/ddsp-guitar's own compute_loudness / compute_spectral_centroid /
# scaling helpers reproduces loudness_scaled, centroid_scaled and periodicity BIT-EXACTLY
# (max|diff| <= 1.2e-7), and hz_to_unit matches to 0.0 given identical input.
#
# f0 is the one exception, and it is NOT a porting error: torchcrepe's decoder applies
# deliberate random DITHER when converting pitch bins to cents ("Trade quantization error
# for noise" -- torchcrepe.convert.bins_to_cents), so torchcrepe.predict returns slightly
# different Hz on every call for identical input. Measured on one GuitarSet string: median
# 7.9 cents, max 35 cents run-to-run, with the network probabilities and the periodicity
# output bit-identical -- i.e. the noise enters only at bin->cents. The reference disagrees
# with ITSELF by the same margin (4.1-4.9 Hz max, on CPU and GPU alike), and by slightly
# more than it disagrees with this port. DDSP-Guitar inherits this, as does our own
# PitchSignal. Harmless here because extraction happens once into the cache and training
# reads that fixed cache -- but it does mean re-extracting f0 will not reproduce bit-exactly.
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from dataset.control.signal import ControlResult, ControlSignal, control_registry

if TYPE_CHECKING:
    from dataset.dataset_register import RawSample

# reference/ddsp-guitar/util.py
EPS = 1e-7
DB_RANGE = -80.0
# reference/ddsp-guitar/data.py
GUITAR_F0_MIN_HZ = 35
GUITAR_F0_MAX_HZ = 1200
CENTROID_MIN_HZ = 20  # reference/ddsp-guitar/data.py __getitem__: hz_to_unit(centroid, 20, sr//2)
# reference/ddsp-guitar/repro/prepare_data.py
# The reference's own operating point, kept for documentation and for the fidelity test
# in the module docstring. Extraction no longer runs here -- see WINDOW_HOP_RATIO.
DDSP_SAMPLE_RATE = 48000
DDSP_FEATURE_FRAME_RATE = 128
DDSP_HOP = DDSP_SAMPLE_RATE // DDSP_FEATURE_FRAME_RATE          # 375
DDSP_WINDOW = DDSP_HOP * 4                                       # 1500, data.py's loudness_window_frames
# data.py uses loudness_window_frames = hop_frames * 4 (31ms at their 7.8ms hop). Neither
# their absolute window nor their 4x ratio transfers cleanly to our 46ms hop, so this was
# chosen by measurement -- window swept over {0.67, 1, 2, 4}x hop on three GuitarSet
# strings, holding everything else fixed:
#
#   win    ms   xhop   loudness_std   centroid_std
#   1378   31   0.67      0.0037         0.1171     <- shorter than hop: GAPS, audio
#   2048   46   1.00      0.0037         0.1137        between frames never analysed
#   4096   93   2.00      0.0040         0.1058
#   8192  186   4.00      0.0043         0.0952     <- DDSP's ratio rule at our hop
#
# 1x is chosen: every sample is counted exactly once (no gaps, no redundancy), centroid
# keeps all but 3% of its best variance, and the loudness spread across the whole sweep
# is within noise. It also matches EnvelopeSignal's existing framing in
# dataset/control/signal.py, which reshapes audio into contiguous hop-sized frames.
# 0.67x is rejected outright regardless of its numbers: leaving audio unanalysed can drop
# a note attack entirely.
WINDOW_HOP_RATIO = 1

FEATURE_NAMES = ["f0_scaled", "loudness_scaled", "periodicity", "centroid_scaled"]

def _hz_to_midi(f: torch.Tensor) -> torch.Tensor:
    """reference/ddsp-guitar/util.py's hz_to_midi -- note the clip to EPS, which maps
    silence (0 Hz) to a very negative midi value rather than -inf."""
    f = torch.clip(f, EPS, float("inf"))
    return 12 * torch.log2(f / 440) + 69


def _midi_to_unit(m: torch.Tensor, midi_min: float, midi_max: float, clip: bool = True) -> torch.Tensor:
    """reference/ddsp-guitar/util.py's midi_to_unit."""
    midi = torch.clip(m, midi_min, midi_max) if clip else m
    return (midi - midi_min) / (midi_max - midi_min)


def _hz_to_unit(f: torch.Tensor, hz_min: float, hz_max: float, clip: bool = True) -> torch.Tensor:
    """reference/ddsp-guitar/util.py's hz_to_unit. Scaling is MIDI-domain (log), not
    linear in Hz -- easy to get wrong when reimplementing from the paper alone."""
    midi_min = _hz_to_midi(torch.tensor(float(hz_min)))
    midi_max = _hz_to_midi(torch.tensor(float(hz_max)))
    return _midi_to_unit(_hz_to_midi(f), float(midi_min), float(midi_max), clip)


def _scale_db(db: torch.Tensor) -> torch.Tensor:
    """reference/ddsp-guitar/util.py's scale_db: scales [-DB_RANGE, 0] to [0, 1].
    DB_RANGE is negative (-80), so this is db/(-80) + 1."""
    return (db / DB_RANGE) + 1.0


def _replace_nan_with_previous(x: torch.Tensor) -> torch.Tensor:
    """reference/ddsp-guitar/data.py's replace_nan_with_previous, called by
    repair_spectral_centroid. torchaudio's spectral_centroid divides by the frame's total
    magnitude, so an all-zero (digitally silent) frame yields NaN -- common in hex-pickup
    audio where an unplayed string is genuinely silent."""
    x = x.clone()
    if torch.isnan(x[0]):
        x[0] = 0
    for i in range(1, x.shape[0]):
        if torch.isnan(x[i]):
            x[i] = x[i - 1]
    return x


def _compute_loudness(signal: torch.Tensor, sample_rate: int, hop_length: int, n_fft: int,
                      corrected: bool = False) -> torch.Tensor:
    """reference/ddsp-guitar/feature.py's compute_loudness -- A-weighted mean of the
    log-magnitude STFT. signal: (channels, samples) -> (channels, frames).

    corrected=True switches to magenta's original definition instead, because theirs (copied
    from acids-ircam/ddsp_pytorch) departs from it in two ways that both matter:

      * A-weighting is added in the LOG domain, so mean(ln|S| + A) = mean(ln|S|) + mean(A):
        it collapses to a constant offset and applies NO frequency weighting at all, despite
        being the whole point of the function. magenta multiplies in the linear power domain
        (power * 10**(A/10)) BEFORE the mean, where the weighting actually reweights.
      * np.log instead of 20*log10, which compresses the dynamic range 8.686x. Measured on
        GuitarSet string 4: their output spans -9.79..-6.67 (range 3.12) where a 20*log10
        version of the same audio spans -45.1..-27.7 (range 17.4 dB). That is why scale_db
        lands at ~1.08..1.14 rather than [0, 1].

    The corrected form exists to test whether the 4-feature synth arm's plateau and overfit
    (val_loss 0.7632 @ep217 against pitch+envelope's 0.7508 @ep483) is caused by a degenerate
    loudness channel rather than by the representation being genuinely worse. Keep
    corrected=False for anything compared against reference/ddsp-guitar's own checkpoints,
    which were trained on the uncorrected feature."""
    import librosa

    S = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
        return_complex=True,
        pad_mode="constant",
        window=torch.hann_window(n_fft),
    )
    if corrected:
        # magenta: weight in linear power, average, THEN convert to dB
        power = torch.abs(S) ** 2
        aw = torch.from_numpy(librosa.A_weighting(
            librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)).reshape(1, -1, 1))
        weighted = power * (10.0 ** (aw.to(power.dtype) / 10.0))
        return 10.0 * torch.log10(weighted.mean(dim=1) + 1e-10)
    S = torch.log(torch.abs(S) + 1e-7)
    f = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weight = librosa.A_weighting(f)
    S = S + torch.from_numpy(a_weight.reshape(1, -1, 1)).to(S.dtype)
    return torch.mean(S, 1)


def _compute_spectral_centroid(signal: torch.Tensor, win_length: int, hop_length: int, sample_rate: int) -> torch.Tensor:
    """reference/ddsp-guitar/feature.py's compute_spectral_centroid."""
    import torchaudio

    return torchaudio.functional.spectral_centroid(
        signal,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
        window=torch.hann_window(win_length),
        pad=0,
    )


def _compute_pitch(signal: torch.Tensor, sample_rate: int, fmin: float, fmax: float, hop_frames: int,
                   device: str, batch_size: int):
    """reference/ddsp-guitar/feature.py's compute_pitch, crepe branch (the module-level
    PITCH_EXTRACTION_ALGORITHM is set to "crepe" there; the penn branch is dead and needs
    a package that isn't installed). Returns (pitch_hz, periodicity), each (channels, frames).
    Their call leaves torchcrepe's decoder at its default (viterbi) and passes pad=True."""
    import torchcrepe

    pitch, periodicity = [], []
    for i in range(signal.shape[0]):
        p, p2 = torchcrepe.predict(
            signal[i][None, ...],
            sample_rate,
            hop_frames,
            fmin,
            fmax,
            "full",
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
            pad=True,
        )
        pitch.append(p.cpu())
        periodicity.append(p2.cpu())
    return torch.cat(pitch, dim=0), torch.cat(periodicity, dim=0)


def compute_ddsp_features(audio: np.ndarray, sample_rate: int, hop: int, device: str = "cpu",
                          corrected_loudness: bool = False,
                          batch_size: int = 2048) -> Dict[str, torch.Tensor]:
    """Mono `audio` at `sample_rate` -> the four features on the `hop`-spaced grid, each
    (frames,). Extraction happens at the CALLER's rate and hop -- no audio resampling and
    no feature resampling -- so frames land directly on the latent grid. See the module
    docstring for why that replaced the earlier extract-at-48kHz-then-downsample path."""
    y = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)  # extractors want (channels, samples)
    window = hop * WINDOW_HOP_RATIO

    pitch, periodicity = _compute_pitch(
        y, sample_rate=sample_rate, fmin=GUITAR_F0_MIN_HZ, fmax=GUITAR_F0_MAX_HZ,
        hop_frames=hop, device=device, batch_size=batch_size,
    )
    loudness = _compute_loudness(y, sample_rate=sample_rate, hop_length=hop, n_fft=window,
                                 corrected=corrected_loudness)
    centroid = _compute_spectral_centroid(y, win_length=window, hop_length=hop, sample_rate=sample_rate)

    # reference/ddsp-guitar/data.py's repair_spectral_centroid, applied before scaling.
    centroid = _replace_nan_with_previous(centroid.reshape(-1))

    # reference/ddsp-guitar/preprocessing.py's preprocess_model_inputs is where the raw
    # features become the *_scaled ones the synthesis model actually consumes.
    return {
        "f0_scaled": _hz_to_unit(pitch.reshape(-1), GUITAR_F0_MIN_HZ, GUITAR_F0_MAX_HZ),
        "loudness_scaled": _scale_db(loudness.reshape(-1)),
        "periodicity": periodicity.reshape(-1),  # used raw -- preprocessing.py applies no scaling
        "centroid_scaled": _hz_to_unit(centroid, CENTROID_MIN_HZ, sample_rate // 2),
    }


@control_registry.register("ddsp_controls")
class DDSPGuitarControlSignal(ControlSignal):
    """DDSP-Guitar's 4-feature conditioning set as a single control signal.

    Emitted as ONE multi-channel signal rather than four registered signals because F0
    and periodicity come from a single torchcrepe call -- splitting them would run CREPE
    twice, and CREPE is this pipeline's extraction bottleneck (see PitchSignal's own
    comments in dataset/control/signal.py). Pass params={"features": [...]} to emit a
    subset for ablations without paying for a second pass.
    """

    name = "ddsp_controls"
    corrected_loudness = False

    def __init__(self, sample_rate: int, hop_length: int, **params):
        super().__init__(sample_rate, hop_length, **params)
        # subclass default, overridable per-config for one-off ablations
        self.corrected_loudness = bool(params.get("corrected_loudness",
                                                  type(self).corrected_loudness))
        self.features: List[str] = list(params.get("features", FEATURE_NAMES))
        unknown = [f for f in self.features if f not in FEATURE_NAMES]
        if unknown:
            raise ValueError(f"unknown ddsp_controls feature(s) {unknown}; valid: {FEATURE_NAMES}")
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = params.get("batch_size", 2048)

    @property
    def is_gpu(self) -> bool:
        # Same reasoning as PitchSignal: route into ControlCacheExtractor's single
        # sequential GPU pass instead of the CPU worker pool, so CREPE doesn't multiply
        # into N competing CUDA contexts.
        return self.device != "cpu"

    def compute(self, audio: np.ndarray, sample: "RawSample") -> ControlResult:
        feats = compute_ddsp_features(audio, self.sample_rate, self.hop_length,
                                      device=self.device, batch_size=self.batch_size,
                                      corrected_loudness=self.corrected_loudness)
        # Extraction is already on our grid, so this only trims/pads the off-by-one that
        # torchcrepe's own framing can introduce -- there is no rate conversion left.
        n_frames = len(audio) // self.hop_length
        cols = []
        for name in self.features:
            v = feats[name].reshape(-1)
            if v.shape[0] < n_frames:
                v = F.pad(v, (0, n_frames - v.shape[0]), mode="replicate" if v.shape[0] > 1 else "constant")
            cols.append(v[:n_frames])
        values = torch.stack(cols, dim=-1).numpy().astype(np.float32)
        return ControlResult(values=values, frame_rate=self.sample_rate / self.hop_length)


@control_registry.register("ddsp_controls_fixed")
class DDSPGuitarControlSignalFixed(DDSPGuitarControlSignal):
    """The same four features, but with loudness computed magenta's way (see
    _compute_loudness's `corrected` branch).

    Registered as a SEPARATE signal rather than a flag on the original so both live in the
    cache side by side: anything compared against reference/ddsp-guitar's own checkpoints
    must keep using `ddsp_controls`, because those checkpoints were trained on the
    uncorrected feature. This one exists to answer a narrower question -- whether the
    4-feature synth arm's plateau and overfit is caused by a degenerate loudness channel
    (constant A-weighting, 8.7x compressed range) rather than by the 4-feature
    representation being genuinely worse than pitch+envelope. That distinction matters
    before claiming 2 features beat 4.
    """

    name = "ddsp_controls_fixed"
    corrected_loudness = True


@control_registry.register("ddsp_loudness_fixed")
class DDSPLoudnessFixedSignal(DDSPGuitarControlSignalFixed):
    """Just the corrected loudness_scaled channel, alone -- for a pitch+loudness
    conditioning pair that parallels pitch+envelope, rather than the full 4-feature set.

    NOT built via DDSPGuitarControlSignal's own params={"features": [...]} subsetting
    mechanism, even though that exists for exactly this kind of ablation: that mechanism
    reuses the SAME registry name ("ddsp_controls_fixed"), and cache already holds a
    4-channel array under that name -- ControlCacheExtractor skips a signal that already
    exists, so a differently-shaped request under the same name would silently return the
    stale 4-channel array instead of the requested 1-channel one (the exact trap
    CLAUDE.md's Environments section warns about). A separate registry name gets its own
    cache key, so no collision is possible.
    """

    name = "ddsp_loudness_fixed"

    def __init__(self, sample_rate: int, hop_length: int, **params):
        params = {**params, "features": ["loudness_scaled"]}
        super().__init__(sample_rate, hop_length, **params)


@control_registry.register("ddsp_centroid_fixed")
class DDSPCentroidFixedSignal(DDSPGuitarControlSignalFixed):
    """Just the centroid_scaled channel, alone -- for a pitch+envelope+centroid
    conditioning triple, the same reason `DDSPLoudnessFixedSignal` exists rather than
    reusing `ddsp_controls_fixed`'s params={"features": [...]} subsetting (see that
    class's docstring): a separate registry name gets its own cache key, so a
    differently-shaped request under the shared `ddsp_controls_fixed` name can't
    silently return the stale 4-channel array instead."""

    name = "ddsp_centroid_fixed"

    def __init__(self, sample_rate: int, hop_length: int, **params):
        params = {**params, "features": ["centroid_scaled"]}
        super().__init__(sample_rate, hop_length, **params)
