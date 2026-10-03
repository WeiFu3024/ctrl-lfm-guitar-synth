# The four loss terms this experiment compares, plus the frozen-synth FM velocity critic.
#
# All differences are taken ALONG TIME WITHIN A STRING. The batch tensors are flattened to
# (B, S*T, 1), so a naive torch.diff would also difference across the boundary between one
# string's last frame and the next string's first -- a large fake discontinuity six times per
# item. Everything here unflattens to (B,S,T) first.
#
# THE DELTA AND JITTER TERMS DELIBERATELY ACT ON DIFFERENT ORDERS:
#   L_delta   first differences, MATCHED to the target -- controls slope statistics (decay
#             rate, attack steepness). Matching, not minimising: the real curve has its own
#             roughness and the goal is to reproduce it, not to flatten it.
#   L_jitter  SECOND differences, penalised -- controls zigzag only. A linear ramp has
#             second difference zero, so this cannot fight the decay slope that L_delta is
#             trying to match. Put both on first differences and they pull against each other.
# By default L_jitter is HINGED: it only charges for being ROUGHER than the target, never for
# being smoother, so it cannot reintroduce the over-smoothing that motivated L_delta.
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _unflatten(x: Tensor, n_strings: int) -> Tensor:
    """(B, S*T, 1) -> (B, S, T)."""
    B, L, _ = x.shape
    return x.reshape(B, n_strings, L // n_strings)


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


# --- loudness weighting for the PITCH channel ---------------------------------------------
# Pitch is meaningless where the string is not sounding: CREPE still emits a value, but it is
# tracking noise, and the cached curve is hold-forward there anyway (see
# dataset/control_flow_dataset.py). Charging full price for those frames would spend most of
# the pitch budget on frames that carry no pitch.
#
# A soft ramp, not a hard threshold, so the gradient does not cliff at the boundary. Bounds
# are in NORMALISED envelope units (mean -4.75, std 1.355 log-RMS): the train split's p1 sits
# near -0.74 and its p99 near +2.39, so 0 weight below -0.5 and full weight above +1.0 spans
# roughly "silence" to "clearly sounding".
PITCH_MASK_LO, PITCH_MASK_HI = -0.5, 1.0


def loudness_weight(env_target: Tensor, lo: float = PITCH_MASK_LO,
                    hi: float = PITCH_MASK_HI, invert: bool = False) -> Tensor:
    """(B, L, 1) normalised real second-channel target -> (B, L) per-frame weight in [0, 1].

    `invert` is for a channel whose RAW value DECREASES with loudness (e.g.
    ddsp_loudness_fixed -- see dataset/perf/perf_curve_dataset.py's EXTRA_CHANNEL_SPECS
    docstring for why): lo/hi still bound the same "quiet -> loud" span in the channel's OWN
    z-scored units, but the weight ramps the opposite direction across it."""
    w = ((env_target.squeeze(-1) - lo) / (hi - lo)).clamp(0.0, 1.0)
    return (1.0 - w) if invert else w


def l1_loss(pred: Tensor, target: Tensor, mask: Tensor,
            weight: Optional[Tensor] = None) -> Tensor:
    err = (pred - target).abs().squeeze(-1)
    if weight is None:
        return _masked_mean(err, mask)
    # weight folded into the mask so the denominator counts effective frames, not raw ones --
    # otherwise a mostly-silent item looks like it has near-zero pitch error rather than
    # near-zero pitch evidence
    w = weight * mask.to(weight.dtype)
    return (err * w).sum() / w.sum().clamp(min=1e-6)


def per_note_l1_loss(
    pred: Tensor,
    target: Tensor,
    note_mask: Tensor,
    valid_mask: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Mean note-wise MAE.

    Each MIDI note contributes one scalar regardless of duration. ``note_mask`` has shape
    (B, N, L), where each row selects one note's annotation span (including any configured
    boundary padding). This differs from ``l1_loss`` only in aggregation: frame L1 weights
    long notes more heavily, while this loss weights every note equally.
    """
    err = (pred - target).abs().squeeze(-1).unsqueeze(1)
    effective = note_mask.to(err.dtype) * valid_mask.unsqueeze(1).to(err.dtype)
    if weight is not None:
        effective = effective * weight.unsqueeze(1)
    denom = effective.sum(dim=-1)
    per_note = (err * effective).sum(dim=-1) / denom.clamp(min=1e-6)
    present = denom > 0
    if not present.any():
        return pred.sum() * 0.0
    return per_note[present].mean()


def _note_window_geometry(
    note_mask: Tensor, valid_mask: Tensor, min_frames: int, n_fft: int,
):
    """Shared setup for the per-note spectral losses below: which notes qualify, where each
    note's exact annotation span sits in the flattened (B, S*T) sequence, and the Hann window
    to apply once it is gathered into a fixed n_fft buffer. A note's time SHIFT within that
    buffer only changes FFT phase, never magnitude, so no alignment is needed beyond gathering
    each note to buffer position 0.
    """
    if min_frames < 3:
        raise ValueError("min_frames must be at least 3")
    if n_fft < min_frames:
        raise ValueError("n_fft must be at least min_frames")
    effective = note_mask & valid_mask.unsqueeze(1)
    lengths = effective.sum(dim=-1)
    present = (lengths >= min_frames) & (lengths <= n_fft)

    batch_size, n_notes, flat_length = effective.shape
    starts = effective.to(torch.int64).argmax(dim=-1)
    offsets = torch.arange(n_fft, device=note_mask.device)
    raw_indices = starts.unsqueeze(-1) + offsets
    indices = raw_indices.clamp(max=flat_length - 1)
    sequence_mask = torch.gather(effective, 2, indices) & (raw_indices < flat_length)
    mask = sequence_mask.to(torch.float32)
    denominator = lengths.clamp(min=1).to(torch.float32)
    phase = offsets.view(1, 1, -1) / (denominator.unsqueeze(-1) - 1.0).clamp(min=1.0)
    window = (0.5 - 0.5 * torch.cos(2.0 * torch.pi * phase)) * mask
    return present, batch_size, n_notes, flat_length, indices, mask, denominator, window


def _gather_note_sequence(
    curve: Tensor, indices: Tensor, batch_size: int, n_notes: int, flat_length: int,
) -> Tensor:
    expanded = curve.squeeze(-1).unsqueeze(1).expand(batch_size, n_notes, flat_length)
    return torch.gather(expanded, 2, indices)


def _spectral_l1(pred_magnitude: Tensor, target_magnitude: Tensor, present: Tensor) -> Tensor:
    # Bin 0 (DC) is excluded: it carries the note's overall level/offset, which the
    # pointwise L1 term already supervises directly and far more efficiently.
    per_note = (pred_magnitude[..., 1:] - target_magnitude[..., 1:]).abs().mean(dim=-1)
    return per_note[present].mean()


def per_note_residual_spectrum_loss(
    pred: Tensor,
    target: Tensor,
    score_pitch: Tensor,
    note_mask: Tensor,
    valid_mask: Tensor,
    min_frames: int = 8,
    n_fft: int = 256,
) -> Tensor:
    """Phase-insensitive pitch-modulation loss, averaged equally across MIDI notes.

    Inputs use the same pitch units. Each exact annotation span is gathered into a fixed-size
    buffer, mean-centered, Hann-windowed, and transformed with a magnitude FFT. Pointwise L1
    remains responsible for note-center detuning; this term supervises non-DC modulation
    energy without requiring its phase to align. Fixed FFT size keeps the loss scale invariant
    to the random crop length and makes training match the spectrum study.
    """
    present, batch_size, n_notes, flat_length, indices, mask, denominator, window = (
        _note_window_geometry(note_mask, valid_mask, min_frames, n_fft)
    )
    if not present.any():
        return pred.sum() * 0.0

    def magnitude(curve: Tensor) -> Tensor:
        residual = curve - score_pitch
        sequence = _gather_note_sequence(residual, indices, batch_size, n_notes, flat_length)
        mean = (sequence * mask).sum(dim=-1) / denominator
        windowed = (sequence - mean.unsqueeze(-1)) * window
        spectrum = torch.fft.rfft(windowed, n=n_fft, dim=-1).abs()
        spectrum = spectrum / window.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cat([spectrum[..., :1], 2.0 * spectrum[..., 1:]], dim=-1)

    return _spectral_l1(magnitude(pred), magnitude(target), present)


def _masked_lpf_trend(sequence: Tensor, mask: Tensor, kernel_frames: int) -> Tensor:
    """Box-filtered low-pass trend, normalised by the count of VALID samples in each window so
    a note's boundary does not pull the trend toward zero. `kernel_frames` must be odd so
    symmetric padding preserves the buffer length exactly."""
    if kernel_frames % 2 == 0:
        raise ValueError("kernel_frames must be odd")
    batch_size, n_notes, length = sequence.shape
    flat_seq = (sequence * mask).reshape(batch_size * n_notes, 1, length)
    flat_mask = mask.reshape(batch_size * n_notes, 1, length)
    kernel = torch.ones(1, 1, kernel_frames, device=sequence.device, dtype=sequence.dtype)
    pad = kernel_frames // 2
    summed = F.conv1d(flat_seq, kernel, padding=pad)
    counted = F.conv1d(flat_mask, kernel, padding=pad)
    return (summed / counted.clamp(min=1e-6)).reshape(batch_size, n_notes, length)


def per_note_envelope_spectrum_loss(
    pred: Tensor,
    target: Tensor,
    note_mask: Tensor,
    valid_mask: Tensor,
    min_frames: int = 8,
    n_fft: int = 256,
    lpf_frames: int = 9,
) -> Tensor:
    """Phase-insensitive envelope-modulation loss, averaged equally across MIDI notes.

    Unlike pitch, envelope has no external per-frame reference to subtract (no "score
    envelope"): its dominant low-frequency content is the note's own attack/decay shape, which
    would otherwise dominate the magnitude spectrum and swamp any faster texture. Each gathered
    note is detrended against a masked box-filtered low-pass of ITSELF (`lpf_frames` wide,
    ~`lpf_frames / frame_rate` seconds) before the same Hann-windowed magnitude FFT comparison
    `per_note_residual_spectrum_loss` uses.
    """
    present, batch_size, n_notes, flat_length, indices, mask, denominator, window = (
        _note_window_geometry(note_mask, valid_mask, min_frames, n_fft)
    )
    if not present.any():
        return pred.sum() * 0.0

    def magnitude(curve: Tensor) -> Tensor:
        sequence = _gather_note_sequence(curve, indices, batch_size, n_notes, flat_length)
        trend = _masked_lpf_trend(sequence, mask, lpf_frames)
        detrended = (sequence - trend) * mask
        # The box-filtered trend already removes most of the level; re-center defensively
        # against the small residual DC an asymmetric note boundary can leave behind.
        mean = (detrended * mask).sum(dim=-1) / denominator
        windowed = (detrended - mean.unsqueeze(-1)) * window
        spectrum = torch.fft.rfft(windowed, n=n_fft, dim=-1).abs()
        spectrum = spectrum / window.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.cat([spectrum[..., :1], 2.0 * spectrum[..., 1:]], dim=-1)

    return _spectral_l1(magnitude(pred), magnitude(target), present)

def build_fir_lowpass_kernel(
    cutoff_hz: float, frame_rate: float, num_taps: int,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Zero-phase windowed-sinc low-pass, built once and reused every step.

    A moving-average (box) filter is NOT used here: its frequency response is a sinc with
    large sidelobes, so it leaks meaningful energy from the 3-8 Hz vibrato band into a
    "trend" term meant to isolate content below it. A Hann-windowed FIR gives an actual
    passband/stopband instead. `num_taps` must be odd so symmetric `pad=num_taps//2` padding
    preserves sequence length exactly, matching `_masked_lpf_trend`'s convention. At this
    project's frame rate (~21.53 Hz), a clean ~2.5 Hz cutoff needs on the order of 40+ taps
    (~2 s) -- longer than the median annotated note (~4.75 frames / ~0.22 s, see
    config/schema.py's NoteVAEModelConfig.canonical_k comment), so a GLOBAL trend loss built
    from this kernel necessarily draws most of its support from neighbouring notes. That is
    intentional for long-term drift/intonation; a per-note-scoped trend term needs its own,
    narrower kernel instead of this one.
    """
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd")
    if not 0 < cutoff_hz < frame_rate / 2:
        raise ValueError("cutoff_hz must be in (0, nyquist)")
    from scipy.signal import firwin
    taps = firwin(num_taps, cutoff_hz, fs=frame_rate, window="hann")
    return torch.as_tensor(taps, device=device, dtype=dtype).view(1, 1, -1)


def _masked_convolve(sequence: Tensor, mask: Tensor, kernel: Tensor) -> Tensor:
    """Apply a fixed 1D `kernel` (1,1,K) along the last axis of `sequence`/`mask` (…, length),
    normalising by the LOCAL SIGNED sum of `mask` under `kernel` -- the direct generalisation
    of `_masked_lpf_trend`'s box-filter convention (there `kernel` is uniform, so this reduces
    to "divide by the count of valid samples"). A box filter's local sum can only shrink
    toward zero at a boundary; an oscillating-sign kernel's local sum can shrink OR change
    sign when very little of the window is valid, which would blow up unstably through the
    `clamp`. Callers MUST additionally gate on `_convolution_coverage` and exclude
    low-coverage positions from the loss -- this function alone does not guard that case.
    `kernel` must have odd length."""
    if kernel.shape[-1] % 2 == 0:
        raise ValueError("kernel must have odd length")
    *lead, length = sequence.shape
    flat_seq = (sequence * mask).reshape(-1, 1, length)
    flat_mask = mask.reshape(-1, 1, length)
    pad = kernel.shape[-1] // 2
    summed = F.conv1d(flat_seq, kernel.to(sequence.dtype), padding=pad)
    counted = F.conv1d(flat_mask, kernel.to(sequence.dtype), padding=pad)
    return (summed / counted.clamp(min=1e-6)).reshape(*lead, length)


def _convolution_coverage(mask: Tensor, kernel: Tensor) -> Tensor:
    """Fraction (0..1) of `kernel`'s total ABSOLUTE weight landing on valid samples at each
    position. Near 1: the window is fully real data, `_masked_convolve`'s signed
    renormalisation is well-conditioned. Near 0: the filtered value would be extrapolated
    from little (or sign-oscillating) support -- callers should exclude those positions
    rather than trust the clamped output."""
    *lead, length = mask.shape
    flat_mask = mask.reshape(-1, 1, length)
    pad = kernel.shape[-1] // 2
    abs_kernel = kernel.abs().to(mask.dtype)
    coverage = F.conv1d(flat_mask, abs_kernel, padding=pad) / abs_kernel.sum().clamp(min=1e-6)
    return coverage.reshape(*lead, length)


def masked_pitch_trend_loss(
    pred: Tensor, target: Tensor, valid_mask: Tensor, n_strings: int, kernel: Tensor,
    weight: Optional[Tensor] = None, min_coverage: float = 0.7,
) -> Tensor:
    """L1 between low-pass-filtered pred and target, over the FULL per-string sequence (not
    note-scoped) -- the phase-sensitive macro-trajectory term: note-center/intonation, slow
    bend, long-term drift. Reuses the same `valid_mask`/`weight` convention as `l1_loss`, so
    silence is excluded the same way and audibility weighting composes identically. Positions
    whose filter window is under `min_coverage` real data (recording edges, short strings)
    are excluded rather than compared on an unstable extrapolated trend value."""
    p, t = _unflatten(pred, n_strings), _unflatten(target, n_strings)
    m = _unflatten(valid_mask.unsqueeze(-1).float(), n_strings)
    p_trend = _masked_convolve(p, m, kernel)
    t_trend = _masked_convolve(t, m, kernel)
    covered = (_convolution_coverage(m, kernel) >= min_coverage) & (m > 0.5)
    err = (p_trend - t_trend).abs()
    if weight is None:
        return _masked_mean(err, covered)
    w = _unflatten(weight.unsqueeze(-1), n_strings) * covered.to(weight.dtype)
    return (err * w).sum() / w.sum().clamp(min=1e-6)


def per_note_pitch_trend_loss(
    pred: Tensor, target: Tensor, note_mask: Tensor, valid_mask: Tensor, kernel: Tensor,
    min_frames: int = 8, n_fft: int = 256, min_coverage: float = 0.7,
) -> Tensor:
    """Note-scoped counterpart to `masked_pitch_trend_loss`: each note is gathered into its
    own fixed buffer (same geometry as the spectral losses) before filtering, so the trend
    cannot draw support from a NEIGHBOURING note -- a cleaner per-note intonation estimate at
    the cost of losing genuinely cross-note drift. Equal-note-weighted like the spectral
    losses, not frame-weighted. A short note can leave every position under-covered by the
    kernel; such notes contribute zero-weight (excluded, not extrapolated) rather than
    silently forcing `min_frames` up for every other loss sharing that argument."""
    present, batch_size, n_notes, flat_length, indices, mask, denominator, _ = (
        _note_window_geometry(note_mask, valid_mask, min_frames, n_fft)
    )
    if not present.any():
        return pred.sum() * 0.0

    def trend(curve: Tensor) -> Tensor:
        sequence = _gather_note_sequence(curve, indices, batch_size, n_notes, flat_length)
        return _masked_convolve(sequence, mask, kernel)

    covered = (_convolution_coverage(mask, kernel) >= min_coverage) & (mask > 0.5)
    covered = covered.to(mask.dtype)
    err = (trend(pred) - trend(target)).abs()
    note_covered = covered.sum(dim=-1)
    per_note = (err * covered).sum(dim=-1) / note_covered.clamp(min=1e-6)
    present = present & (note_covered > 0)
    if not present.any():
        return pred.sum() * 0.0
    return per_note[present].mean()




class EMAMagnitudeBalancer:
    """Scale an auxiliary loss to a requested EMA magnitude ratio against a primary loss."""

    def __init__(self, target_ratio: float, decay: float, eps: float = 1e-8):
        if target_ratio < 0:
            raise ValueError("target_ratio must be non-negative")
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        self.target_ratio = target_ratio
        self.decay = decay
        self.eps = eps
        self.primary_ema = None
        self.auxiliary_ema = None

    @torch.no_grad()
    def weight(
        self, primary_loss: Tensor, auxiliary_loss: Tensor, update: bool,
    ) -> Tensor:
        primary = primary_loss.detach()
        auxiliary = auxiliary_loss.detach()
        if self.primary_ema is None:
            self.primary_ema = primary.clone()
            self.auxiliary_ema = auxiliary.clone()
        elif update:
            self.primary_ema.mul_(self.decay).add_(primary, alpha=1.0 - self.decay)
            self.auxiliary_ema.mul_(self.decay).add_(auxiliary, alpha=1.0 - self.decay)
        ratio = (
            self.target_ratio * self.primary_ema
            / self.auxiliary_ema.clamp(min=self.eps)
        )
        return torch.where(
            self.auxiliary_ema > self.eps, ratio, torch.zeros_like(ratio)
        )


class FixedWeightBalancer:
    """Return one fixed multiplier for an auxiliary loss."""

    def __init__(self, weight: float):
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.weight_value = float(weight)

    def weight(
        self, primary_loss: Tensor, auxiliary_loss: Tensor, update: bool,
    ) -> Tensor:
        del auxiliary_loss, update
        return primary_loss.new_tensor(self.weight_value)


def delta_loss(pred: Tensor, target: Tensor, mask: Tensor, n_strings: int,
               weight: Optional[Tensor] = None) -> Tensor:
    """‖Δpred − Δtarget‖₁ along time. Point-wise loss is nearly blind to jaggedness (a small
    high-frequency wiggle contributes almost nothing to L1), and this also supervises the
    decay slope directly."""
    p, t = _unflatten(pred, n_strings), _unflatten(target, n_strings)
    m = _unflatten(mask.unsqueeze(-1).float(), n_strings)
    dm = m[:, :, 1:] * m[:, :, :-1]                       # both endpoints must be valid
    err = (p.diff(dim=-1) - t.diff(dim=-1)).abs()
    if weight is None:
        return _masked_mean(err, dm > 0.5)
    w = _unflatten(weight.unsqueeze(-1), n_strings)
    w = (w[:, :, 1:] * w[:, :, :-1]) * dm                 # both endpoints must be audible too
    return (err * w).sum() / w.sum().clamp(min=1e-6)


def note_delta_loss(pred: Tensor, target: Tensor, nbr_idx: Tensor, nbr_weight: Tensor,
                    nbr_valid: Tensor) -> Tensor:
    """sum_i sum_{j in neighbours(i)} A(i,j) * |Δpred(i,j) - Δtarget(i,j)| -- NOT
    delta_loss's per-FRAME version above, and not a single "next note" pointer either:
    every note pair "near" each other in time (dataset/perf/note_gain_dataset.py's
    linear-time onset-window precompute, `_note_delta_neighbors`) contributes, weighted by
    A(i,j) = exp(-|onset_i-onset_j|/tau). Neighbours are found over the WHOLE merged
    sequence, any string -- a note's real temporal neighbours can be on a different string
    (chord/strum context), which is the entire point of the merged representation. Matches
    the target's own note-to-note change rather than penalising roughness -- delta_loss's
    family, not jitter_loss's.

    pred/target: (B, M, 1). `nbr_idx`: (B, M, K) int64, merged-sequence index of each of
    note i's up to K neighbours (arbitrary in-range placeholder where `nbr_valid` is
    False -- never gathered incorrectly, always masked out below). `nbr_weight`: (B, M, K)
    float32, `A(i,j)`, already zero where invalid. `nbr_valid`: (B, M, K) bool. All three
    are precomputed, fixed data (dataset-built), not chosen by this function.
    """
    M = pred.shape[1]
    pred_flat, target_flat = pred[..., 0], target[..., 0]                       # (B, M)
    # gather(dim=2) against a zero-copy (B, M, M) broadcast view of the (B, M) source picks
    # out pred_flat[b, nbr_idx[b, i, k]] for every (i, k) without materialising an M x M
    # tensor.
    pred_nbr = torch.gather(pred_flat.unsqueeze(1).expand(-1, M, M), 2, nbr_idx)
    target_nbr = torch.gather(target_flat.unsqueeze(1).expand(-1, M, M), 2, nbr_idx)
    dpred = pred_flat.unsqueeze(-1) - pred_nbr                                  # (B, M, K)
    dtarget = target_flat.unsqueeze(-1) - target_nbr
    w = nbr_valid.to(dpred.dtype) * nbr_weight
    err = (dpred - dtarget).abs()
    return (err * w).sum() / w.sum().clamp(min=1e-6)



def jitter_loss(pred: Tensor, target: Tensor, mask: Tensor, n_strings: int,
                mode: str = "hinge") -> Tensor:
    """Second-difference roughness penalty. mode="hinge" charges only the EXCESS roughness
    over the target's own; mode="plain" penalises all of it (kept so the difference is
    measurable rather than asserted -- plain is expected to over-smooth)."""
    p, t = _unflatten(pred, n_strings), _unflatten(target, n_strings)
    m = _unflatten(mask.unsqueeze(-1).float(), n_strings)
    dm = m[:, :, 2:] * m[:, :, 1:-1] * m[:, :, :-2]
    d2p = p.diff(dim=-1).diff(dim=-1).abs()
    if mode == "plain":
        return _masked_mean(d2p, dm > 0.5)
    d2t = t.diff(dim=-1).diff(dim=-1).abs()
    return _masked_mean(F.relu(d2p - d2t), dm > 0.5)


def roughness_ratio(pred: Tensor, target: Tensor, mask: Tensor, n_strings: int) -> float:
    """mean|Δ²pred| / mean|Δ²target|. The jitter METRIC, reported for every variant whether
    or not jitter is being penalised -- 1.0 means the output is exactly as rough as the real
    envelope, >1 jittery, <1 over-smoothed."""
    p, t = _unflatten(pred, n_strings), _unflatten(target, n_strings)
    m = _unflatten(mask.unsqueeze(-1).float(), n_strings)
    dm = (m[:, :, 2:] * m[:, :, 1:-1] * m[:, :, :-2]) > 0.5
    a = _masked_mean(p.diff(dim=-1).diff(dim=-1).abs(), dm)
    b = _masked_mean(t.diff(dim=-1).diff(dim=-1).abs(), dm)
    return float(a / b.clamp(min=1e-8))


def variance_ratio(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    """var(pred)/var(target) over valid frames -- the mode-collapse check. A deterministic
    regressor trained on a multimodal target collapses toward the conditional mean, which
    shows up here as a ratio well under 1 (the note-rate AR work measured 0.20-0.68)."""
    m = mask.reshape(-1)
    p = pred.reshape(-1)[m]
    t = target.reshape(-1)[m]
    if t.numel() < 2:
        return float("nan")
    return float(p.var(unbiased=False) / t.var(unbiased=False).clamp(min=1e-8))


class FMVelocityCritic:
    """Audio-informed gradient from the FROZEN synth, with no ODE sampling at all.

    One forward pass through the synth on a noised REAL audio latent:
        x0 ~ N(0,I),  x_t = (1-t)x0 + t*x1,  u = x1 - x0
        L_A = ‖v_θ(x_t, t, ĉ) − u‖²                     (GT velocity)
        L_B = ‖v_θ(x_t, t, ĉ) − sg[v_θ(x_t, t, c_gt)]‖²  (teacher velocity)
    v_θ is frozen; gradient reaches only the predicted envelope inside ĉ. x1 is always the
    REAL audio latent, never a generated one, so the synth's own sampling artifacts cannot
    enter the loss -- what can enter is its modeling error, as gradient bias.

    A and B differ in variance and in optimum. B is A's Rao-Blackwell form: it drops the
    zero-mean (v* − u) term, so it has far lower variance, but its optimum is ĉ = c_gt by
    construction -- it will faithfully copy any error in c_gt. A can in principle correct
    such errors (GuitarSet's per-string curves carry hexaphonic bleed) at the cost of a much
    noisier gradient and an optimum displaced by the synth's modeling error.

    B's variance advantage REQUIRES both branches to share one (t, x0, x_t) draw -- common
    random numbers. Re-drawing per branch destroys the cancellation and costs an extra
    forward for nothing. That sharing is what `_shared` below exists to guarantee.
    """

    def __init__(self, synth, control_names, device: str, target: str = "A",
                 t_dist: str = "uniform", beta_a: float = 2.0, beta_b: float = 5.0):
        assert target in ("A", "B"), target
        assert t_dist in ("uniform", "low", "high", "logit_normal"), t_dist
        self.synth = synth
        self.control_names = control_names
        self.device = device
        self.target = target
        self.t_dist = t_dist
        self.beta_a, self.beta_b = beta_a, beta_b
        for p in self.synth.parameters():
            p.requires_grad_(False)
        self.synth.eval()

    def sample_t(self, B: int) -> Tensor:
        if self.t_dist == "uniform":
            return torch.rand(B, device=self.device)
        if self.t_dist == "low":     # mass near 0: coarse structure, strongest gradient
            return torch.distributions.Beta(self.beta_a, self.beta_b).sample((B,)).to(self.device)
        if self.t_dist == "high":    # mass near 1: fine detail, weakest conditioning signal
            return torch.distributions.Beta(self.beta_b, self.beta_a).sample((B,)).to(self.device)
        z = torch.randn(B, device=self.device)
        return torch.sigmoid(z)

    def _shared(self, x1: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = self.sample_t(B)
        x_t = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        return x0, t, x_t, x1 - x0

    def __call__(self, pred_env_hz_units: Tensor, gt_pitch_hz: Tensor, x1: Tensor,
                 mask: Tensor, gt_env: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """pred_env_hz_units and gt_env are in the synth's own units (log-RMS), gt_pitch_hz in
        Hz -- NOT the normalised space the regressor works in. Callers denormalise first."""
        x0, t, x_t, u = self._shared(x1)
        c_hat = {"pitch": gt_pitch_hz, "envelope": pred_env_hz_units}
        v_hat = self.synth(x_t, t, c_hat, cfg_drop_mask=None)

        m = mask.unsqueeze(-1).to(v_hat.dtype)
        denom = (m.sum() * v_hat.shape[-1]).clamp(min=1)
        out = {"L_A": (((v_hat - u) ** 2) * m).sum() / denom}

        if self.target == "B" or gt_env is not None:
            with torch.no_grad():   # same x_t and t -- common random numbers
                v_star = self.synth(x_t, t, {"pitch": gt_pitch_hz, "envelope": gt_env},
                                    cfg_drop_mask=None)
            out["L_B"] = (((v_hat - v_star) ** 2) * m).sum() / denom
        out["loss"] = out["L_B"] if self.target == "B" else out["L_A"]
        return out
