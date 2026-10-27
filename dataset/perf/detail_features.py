"""Oracle per-note pitch-detail features shared by performance datasets and studies."""
from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks

from dataset.control.signal import hz_to_midi

DetailFeatureSet = Literal["none", "energy", "peaks", "phase"]
DETAIL_FEATURE_DIMS: Mapping[str, int] = {
    "none": 0,
    "energy": 4,
    "peaks": 13,
    "phase": 19,
}

N_FFT = 256
MIN_NOTE_FRAMES = 8
N_PEAKS = 3
BANDS_HZ = ((0.5, 3.0), (3.0, 8.0), (8.0, None))
PEAK_MIN_SNR = 3.0
PEAK_MIN_RELATIVE_PROMINENCE = 0.1
_EPS = 1e-8


def detail_feature_dim(feature_set: str) -> int:
    try:
        return DETAIL_FEATURE_DIMS[feature_set]
    except KeyError as exc:
        raise ValueError(
            f"unknown detail feature set {feature_set!r}; expected one of {tuple(DETAIL_FEATURE_DIMS)}"
        ) from exc


def _interpolate_measured_pitch(pitch_midi: np.ndarray, present: np.ndarray) -> np.ndarray:
    positions = np.arange(len(pitch_midi))
    return np.interp(positions, positions[present], pitch_midi[present])


def _magnitude_spectrum(residual: np.ndarray, frame_rate: float, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float64) - float(np.mean(residual))
    window = np.hanning(len(residual))
    fft_size = max(n_fft, len(residual))
    magnitude = np.abs(np.fft.rfft(residual * window, n=fft_size))
    magnitude /= max(float(window.sum()), 1e-12)
    magnitude[1:] *= 2.0
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / frame_rate)
    return frequencies, magnitude


def _band_log_energies(frequencies: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    values = []
    nyquist = float(frequencies[-1])
    for low, high in BANDS_HZ:
        upper = nyquist if high is None else min(float(high), nyquist)
        # Adjacent bands share only their named boundary. Assign it to the upper band so no
        # FFT bin contributes twice, while the high band still includes Nyquist.
        band = (frequencies >= low) & (frequencies < upper)
        if high is None:
            band = (frequencies >= low) & (frequencies <= upper)
        energy = float(np.mean(np.square(magnitude[band]))) if np.any(band) else 0.0
        values.append(np.log(energy + _EPS))
    return np.asarray(values, dtype=np.float32)


def _select_spectral_peaks(frequencies: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    eligible = (frequencies >= BANDS_HZ[0][0]) & (frequencies <= frequencies[-1])
    noise_floor = float(np.median(magnitude[eligible])) if np.any(eligible) else 0.0
    noise_floor = max(noise_floor, _EPS)
    candidates, properties = find_peaks(
        magnitude,
        height=PEAK_MIN_SNR * noise_floor,
        prominence=noise_floor,
    )
    keep = eligible[candidates]
    candidates = candidates[keep]
    prominences = properties["prominences"][keep]
    if candidates.size:
        relative = prominences >= PEAK_MIN_RELATIVE_PROMINENCE * magnitude[candidates]
        candidates = candidates[relative]
    # Magnitude descending and frequency ascending makes ties deterministic.
    order = np.lexsort((frequencies[candidates], -magnitude[candidates]))
    return candidates[order[:N_PEAKS]]
def _refine_peak_frequency(
    frequencies: np.ndarray, magnitude: np.ndarray, peak_index: int,
) -> float:
    """Quadratic interpolation of log magnitude around one FFT-bin peak."""
    if peak_index <= 0 or peak_index >= len(magnitude) - 1:
        return float(frequencies[peak_index])
    y0, y1, y2 = np.log(magnitude[peak_index - 1:peak_index + 2] + _EPS)
    denominator = y0 - 2.0 * y1 + y2
    delta = 0.0 if abs(denominator) < _EPS else 0.5 * (y0 - y2) / denominator
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(frequencies[peak_index] + delta * (frequencies[1] - frequencies[0]))




def _fit_sinusoid(residual: np.ndarray, frame_rate: float, frequency: float) -> tuple[float, float]:
    t = np.arange(len(residual), dtype=np.float64) / frame_rate
    angle = 2.0 * np.pi * frequency * t
    design = np.column_stack((np.cos(angle), np.sin(angle)))
    cosine, sine = np.linalg.lstsq(design, residual, rcond=None)[0]
    magnitude = float(np.hypot(cosine, sine))
    # a*cos(wt)+b*sin(wt) == A*cos(wt+phase), where b == -A*sin(phase).
    phase = float(np.arctan2(-sine, cosine))
    return magnitude, phase


def extract_pitch_detail_features(
    pitch_hz: np.ndarray,
    midi_center: float,
    frame_rate: float,
    feature_set: DetailFeatureSet,
    *,
    voiced: np.ndarray | None = None,
    min_note_frames: int = MIN_NOTE_FRAMES,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Extract one fixed-width feature vector from an exact annotated note span.

    Missing pitch frames are linearly interpolated only within the note. Notes outside the
    supported FFT length, shorter than ``min_note_frames``, or with fewer than two measured
    frames return all-zero continuous values and an explicit zero ``spectral_valid`` bit.
    """
    dim = detail_feature_dim(feature_set)
    out = np.zeros(dim, dtype=np.float32)
    if dim == 0:
        return out

    pitch_hz = np.asarray(pitch_hz, dtype=np.float64).reshape(-1)
    present = np.isfinite(pitch_hz) & (pitch_hz > 0.0)
    if voiced is not None:
        present &= np.asarray(voiced, dtype=bool).reshape(-1)
    if not min_note_frames <= len(pitch_hz) <= n_fft or present.sum() < 2:
        return out

    pitch_midi = hz_to_midi(np.maximum(pitch_hz, 1e-6))
    measured = _interpolate_measured_pitch(pitch_midi, present)
    residual = measured - float(midi_center)
    residual = residual - float(np.mean(residual))
    frequencies, magnitude = _magnitude_spectrum(residual, frame_rate, n_fft)

    out[:3] = _band_log_energies(frequencies, magnitude)
    out[3] = 1.0
    if dim == DETAIL_FEATURE_DIMS["energy"]:
        return out

    selected = _select_spectral_peaks(frequencies, magnitude)
    phases = np.zeros((N_PEAKS, 2), dtype=np.float32)
    for slot, peak_index in enumerate(selected):
        frequency = _refine_peak_frequency(frequencies, magnitude, int(peak_index))
        fitted_magnitude, phase = _fit_sinusoid(residual, frame_rate, frequency)
        base = 4 + 3 * slot
        out[base:base + 3] = (frequency, np.log(fitted_magnitude + _EPS), 1.0)
        phases[slot] = (np.sin(phase), np.cos(phase))
    if dim == DETAIL_FEATURE_DIMS["phase"]:
        out[DETAIL_FEATURE_DIMS["peaks"]:] = phases.reshape(-1)
    return out


def broadcast_detail_features(
    note_features: Sequence[np.ndarray],
    note_spans: Sequence[tuple[int, int, int]],
    n_strings: int,
    n_frames: int,
    feature_set: DetailFeatureSet,
) -> np.ndarray:
    """Broadcast vectors in annotation order; later notes own rounded overlap frames."""
    dim = detail_feature_dim(feature_set)
    if len(note_features) != len(note_spans):
        raise ValueError("note_features and note_spans must have equal length")
    dense = np.zeros((n_strings, n_frames, dim), dtype=np.float32)
    for vector, (string_index, start, end) in zip(note_features, note_spans):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (dim,):
            raise ValueError(f"detail vector has shape {vector.shape}, expected {(dim,)}")
        if not (0 <= string_index < n_strings and 0 <= start < end <= n_frames):
            raise ValueError(f"invalid note span {(string_index, start, end)}")
        dense[string_index, start:end] = vector
    return dense
