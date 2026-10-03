import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless, no display on a training server
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


def _plot_note_roll(
    ax,
    pitch_hz: np.ndarray,
    voiced: np.ndarray,
    onset: np.ndarray,
    time: np.ndarray,
    pred_pitch_hz: Optional[np.ndarray] = None,
) -> None:
    """Piano-roll-style view of pitch+voiced+onset: a continuous pitch line (in MIDI
    note units) drawn only where voiced, with tick markers at onset frames — the
    natural single-voice analog of a MIDI roll, reusing the already-cached ground-truth
    pitch/voiced/onset signals directly rather than re-deriving note boxes. If
    pred_pitch_hz is given (pitch re-extracted from the generated audio — there's no
    analogous "predicted voiced", since voiced comes from ground-truth MIDI, not audio),
    it's overlaid using the SAME target voiced mask, so both lines are directly
    comparable rather than each using its own noisy silence detection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69 + 12 * np.log2(np.maximum(pitch_hz, 1e-6) / 440.0)
    midi = np.where(voiced > 0.5, midi, np.nan)

    has_legend = False
    ax.plot(time, midi, linewidth=1.5, color="tab:blue", label="target", zorder=2)

    if pred_pitch_hz is not None:
        n = min(len(time), len(pred_pitch_hz))
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_midi = 69 + 12 * np.log2(np.maximum(pred_pitch_hz[:n], 1e-6) / 440.0)
        pred_midi_full = np.full_like(midi, np.nan)
        pred_midi_full[:n] = np.where(voiced[:n] > 0.5, pred_midi, np.nan)
        ax.plot(time, pred_midi_full, linewidth=1.2, color="tab:orange", linestyle="--", alpha=0.85, label="predicted", zorder=2)
        has_legend = True

    onset_mask = onset > 0.5
    if onset_mask.any():
        ax.scatter(time[onset_mask], midi[onset_mask], marker="|", s=180, color="tab:red", zorder=3, label="onset")
        has_legend = True
    if has_legend:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("MIDI note")
    ax.set_title("note roll (pitch target vs predicted, target voicing + onset)")


def _spectrogram_db(audio: np.ndarray, sample_rate: int, n_fft: int, max_freq_hz: Optional[float]) -> Tuple[np.ndarray, float]:
    noverlap = n_fft * 3 // 4
    freqs, _, Zxx = stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=noverlap)
    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-8)
    if max_freq_hz is not None:
        keep = freqs <= max_freq_hz
        mag_db = mag_db[keep]
        freqs = freqs[keep]
    return mag_db, freqs[-1]


def plot_aligned_comparison(
    target_latent: np.ndarray,
    pred_latent: np.ndarray,
    controls: Dict[str, np.ndarray],
    frame_rate: float,
    target_audio: np.ndarray,
    pred_audio: np.ndarray,
    audio_sample_rate: int,
    out_path: str,
    n_fft: int = 1024,
    max_freq_hz: Optional[float] = 8000.0,
    pred_controls: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """One figure, sharex=True: target/predicted latent heatmaps, target/predicted decoded-
    audio spectrograms, and one line panel per control signal, all on a shared seconds-based
    time axis.

    Colorbar alignment fix: a plain fig.colorbar(im, ax=ax) shrinks *that* axes'
    physical width to make room for the colorbar, while sibling sharex panels without a
    colorbar stay full width — even make_axes_locatable, applied post-hoc, leaves a
    residual mismatch once tight_layout() redistributes space (measured ~2% off).  The
    only fix that guarantees exact pixel alignment is giving every main-content axes an
    identical GridSpec column from construction, with colorbars in a separate dedicated
    column that main axes never share width with.

    Spectrogram time alignment: computed via our own STFT (not e.g. ax.specgram's
    internal windowing) and plotted with an explicit extent=[time[0], time[-1], ...] using
    the SAME time array as the latent panels — target_audio/pred_audio are decoded from
    exactly the same [start, start+length) crop as target_latent/pred_latent, so t=0 of
    the decoded audio already corresponds to t=0 of the latent/control panels; the extent
    just needs to assert that explicitly rather than trust scipy's own internal time bins,
    which don't necessarily reach exactly frame_rate-derived boundaries.

    pred_controls (optional): control signals re-extracted from pred_audio (e.g. via
    eval.metrics.control.compute_control_following already computes these to get its
    MAE/corr metrics — pass its second return value here rather than re-extracting.
    Overlaid on the target curve in each matching line panel, and on the note-roll's
    pitch line, for a direct predicted-vs-target read instead of eyeballing two plots.
    """
    T = target_latent.shape[0]
    time = np.arange(T) / frame_rate

    target_spec, spec_fmax_t = _spectrogram_db(target_audio, audio_sample_rate, n_fft, max_freq_hz)
    pred_spec, spec_fmax_p = _spectrogram_db(pred_audio, audio_sample_rate, n_fft, max_freq_hz)
    spec_fmax = max(spec_fmax_t, spec_fmax_p)

    roll_keys = {"pitch", "voiced", "onset"}
    has_roll = roll_keys.issubset(controls.keys())
    line_signals = {k: v for k, v in controls.items() if not (has_roll and k in roll_keys)}

    n_panels = 4 + (1 if has_roll else 0) + len(line_signals)
    fig = plt.figure(figsize=(10, 2.5 * n_panels))
    gs = fig.add_gridspec(n_panels, 2, width_ratios=[40, 1], wspace=0.05, hspace=0.4)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_panels)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    vmin = min(target_latent.min(), pred_latent.min())
    vmax = max(target_latent.max(), pred_latent.max())

    for i, (data, title) in enumerate([(target_latent, "target latent"), (pred_latent, "predicted latent")]):
        ax = axes[i]
        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            extent=[time[0], time[-1], 0, data.shape[1]],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_ylabel("channel")
        cax = fig.add_subplot(gs[i, 1])
        fig.colorbar(im, cax=cax)

    spec_vmin = min(target_spec.min(), pred_spec.min())
    spec_vmax = max(target_spec.max(), pred_spec.max())
    for i, (spec, title) in enumerate([(target_spec, "target spectrogram"), (pred_spec, "predicted spectrogram")], start=2):
        ax = axes[i]
        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[time[0], time[-1], 0, spec_fmax],
            vmin=spec_vmin,
            vmax=spec_vmax,
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_ylabel("Hz")
        cax = fig.add_subplot(gs[i, 1])
        fig.colorbar(im, cax=cax, label="dB")

    pred_controls = pred_controls or {}

    remaining_axes = axes[4:]
    if has_roll:
        roll_ax, remaining_axes = remaining_axes[0], remaining_axes[1:]
        pred_pitch = pred_controls["pitch"][:, 0] if "pitch" in pred_controls else None
        _plot_note_roll(roll_ax, controls["pitch"][:, 0], controls["voiced"][:, 0], controls["onset"][:, 0], time, pred_pitch_hz=pred_pitch)
        roll_ax.set_xlim(time[0], time[-1])

    for ax, (name, values) in zip(remaining_axes, line_signals.items()):
        ax.plot(time, values[:, 0], label="target")
        if name in pred_controls:
            n = min(len(time), pred_controls[name].shape[0])
            ax.plot(time[:n], pred_controls[name][:n, 0], linestyle="--", alpha=0.85, label="predicted")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name)
        ax.set_xlim(time[0], time[-1])

    axes[-1].set_xlabel("time (s)")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_curve(log_path: str, out_path: str) -> None:
    if not Path(log_path).exists():
        return

    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            epochs.append(entry["epoch"])
            train_losses.append(entry.get("train_loss"))
            val_losses.append(entry.get("val_loss"))

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in train_losses):
        ax.plot(epochs, train_losses, label="train")
    if any(v is not None for v in val_losses):
        ax.plot(epochs, val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training curve")
    ax.legend()
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
