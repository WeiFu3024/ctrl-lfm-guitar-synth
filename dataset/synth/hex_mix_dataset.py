# Paired (six per-string streams, mono mic recording) audio for the 6->1 mixing stage.
#
# Three input variants, because what the reverb is trained on decides what it learns:
#
#   raw       the real hex-pickup audio          -> the IR is pure pickup->mic acoustics,
#                                                   but this is NOT the domain it meets at
#                                                   inference (arm A's audio is decoded)
#   decoded   SAO round-trip of the real hex     -> acoustics, in the codec domain the
#                                                   reverb actually sees. The honest default.
#   predicted arm A's generated hex              -> acoustics PLUS compensation for that
#                                                   synth's systematic error. This is what
#                                                   DDSP-Guitar effectively does, since its
#                                                   reverb is trained end-to-end with its
#                                                   synth, so it is the matched setting for
#                                                   a headline comparison -- and the gap
#                                                   against `decoded` measures how much the
#                                                   mixer is papering over the synth.
#
# Decoding is the expensive part, so it happens once per split and is cached to .npz;
# 306 recordings x 6 strings x 8 s is ~2.6 GB in float32, trivial against this machine's
# RAM but slow to recompute (one SAO decode per string).
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from dataset.audio_utils import load_audio
from dataset.dataset_register import dataset_registry
from dataset.split import split_registry

# +/-30 ms, set by measurement rather than by an acoustics argument. An earlier version
# used +/-10 ms reasoning that a mic under 2 m away implies under ~6 ms of delay. That
# bounds only the ACOUSTIC term: hex pickup and room mic went through separate converter
# chains, and device latency has no such bound. Measured over 324 recordings the offset
# spans [-16.2, +18.4] ms (median +3.71, IQR 4.31), and +/-10 ms pinned 18 of them at the
# cap. Widening to +/-30 ms leaves nothing at the cap, and +/-80 ms returns identical
# statistics -- so the distribution is fully contained and this window is not truncating.
#
# The window must still be finite: at very wide settings waveform cross-correlation locks
# onto the wrong pitch period on quasi-periodic guitar audio, which is why the coarse
# stage runs on envelopes and only the +/-2 ms refinement sees the waveform.
MAX_LAG_S = 0.030

# MEASURED VERDICT: align="none" is the default because per-recording alignment makes the
# fit WORSE. Fitting the IR under each scheme (306 train / 18 val, decoded hex, 150 epochs,
# lr 2e-4 with grad clipping). An earlier version of this table was produced at lr 1e-3,
# where every fit oscillated rather than converged, so those numbers meant nothing:
#
#   scheme     lag IQR   val spectral   val SI-SDR
#   none             0         6.8790        -1.41   <- best on both
#   twostage       216         6.9597       -12.47
#   envelope       190         6.9667       -17.09
#   waveform       315         6.9878       -17.00
#
# The offsets are real ([-16.2, +18.4] ms across 324 recordings) but the estimates are not
# reliable enough to act on: hex-to-mic waveform correlation is only 0.36, so the
# cross-correlation peak is weak and each per-recording shift injects more variance than
# the true spread contains. The SI-SDR column shows the cost most starkly -- shifting by a
# noisy estimate destroys phase coherence the filter would otherwise exploit (-1.41 dB
# unaligned against -12 to -17 dB for every alignment scheme).
#
# The schemes stay behind `align=` because a phase-sensitive objective (--time-weight)
# would change the trade-off; re-measure before enabling one.

MIX_ID_FMT = "guitarset-mic/{recording_id}"
N_STRINGS = 6


class HexMixDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split: str, variant: str = "decoded", seconds: float = 8.0,
                 cache_dir: str = "cache/mix_reverb", encoder=None,
                 pred_dir: Optional[str] = None, align: str = "none"):
        assert variant in ("raw", "decoded", "predicted"), variant
        assert align in ("none", "waveform", "envelope", "twostage"), align
        self.align = align
        self.cfg, self.split, self.variant = cfg, split, variant
        self.sr = cfg.data.sample_rate
        self.n = int(seconds * self.sr)
        self.pred_dir = Path(pred_dir) if pred_dir else None

        hex_samples = dataset_registry.get("GuitarSet")(cfg.data).list_samples()
        entry = next(e for e in cfg.dataset.split.entries if e.dataset == "GuitarSet")
        assign = split_registry.get(entry.splitter)(entry.params).assign(hex_samples)
        keep = {s.sample_id.split("/")[1] for s in hex_samples
                if assign.get(s.sample_id) == split}

        self.hex_by_rec: Dict[str, List] = {}
        for s in hex_samples:
            rec = s.sample_id.split("/")[1]
            if rec in keep:
                self.hex_by_rec.setdefault(rec, []).append(s)
        for rec in self.hex_by_rec:
            self.hex_by_rec[rec].sort(key=lambda s: s.sample_id)
        self.mic_by_rec = {s.sample_id.split("/")[-1]: s for s in
                           dataset_registry.get("GuitarSet-mic")(cfg.data).list_samples()}

        self.index = sorted(r for r in self.hex_by_rec
                            if r in self.mic_by_rec and len(self.hex_by_rec[r]) == N_STRINGS)
        self.cache_path = Path(cache_dir) / f"{split}_{variant}_{align}_{int(seconds)}s.npz"
        # shared across variants of the same split -- the lag belongs to the recording
        self.lag_path = Path(cache_dir) / f"{split}_lags_{align}_{int(seconds)}s.json"
        self._lag_cache: Dict[str, int] = (
            {k: int(v) for k, v in json.loads(self.lag_path.read_text()).items()}
            if self.lag_path.exists() else {})
        self._hex, self._mic = self._build(encoder)

    # a single fixed mid-file window per recording: the head of a GuitarSet take is often
    # silence, and a linear 66k-parameter filter does not need crop augmentation to fit
    def _window(self, total: int) -> slice:
        start = max(0, (total - self.n) // 2)
        return slice(start, start + self.n)

    def _recording_lag(self, rec: str, sl: slice) -> int:
        """Lag for this recording, measured once from real hex and memoised to JSON."""
        if rec in self._lag_cache:
            return self._lag_cache[rec]
        mic = load_audio(self.mic_by_rec[rec].audio_path, None, self.sr)
        m = np.pad(mic[sl], (0, max(0, self.n - len(mic[sl]))))[: self.n]
        acc = np.zeros(self.n, dtype=np.float32)
        for smp in self.hex_by_rec[rec]:
            x = load_audio(smp.audio_path, smp.channel, self.sr)[sl]
            acc[: len(x)] += x
        lag = self._best_lag(m, acc, int(MAX_LAG_S * self.sr), self.sr, self.align)
        self._lag_cache[rec] = lag
        return lag

    @staticmethod
    def _envelope(x: np.ndarray, sr: int, cutoff_hz: float = 50.0) -> np.ndarray:
        """Rectified, low-passed amplitude envelope.

        Alignment runs on this rather than on the waveform: guitar audio is quasi-periodic,
        so waveform cross-correlation has near-equal peaks one pitch period apart and can
        pick the wrong one. Envelopes are dominated by onsets, which are unambiguous.
        """
        n = len(x)
        k = max(1, int(sr / cutoff_hz))
        kern = np.hanning(k); kern /= kern.sum()
        return np.convolve(np.abs(x), kern, mode="same")[:n]

    @staticmethod
    def _xcorr_peak(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
        a = a - a.mean(); b = b - b.mean()
        n = len(a)
        fa, fb = np.fft.rfft(a, 2 * n), np.fft.rfft(b, 2 * n)
        cc = np.fft.irfft(fa * np.conj(fb), 2 * n)
        cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
        return int(np.argmax(cc) - max_lag)

    @staticmethod
    def _best_lag(mic: np.ndarray, hexsum: np.ndarray, max_lag: int, sr: int = 44100,
                  align: str = "twostage") -> int:
        """Samples the mic trails the pickup sum by, via cross-correlation.

        NOT a refinement -- the fit needs it. The mic sits a variable distance from the
        strings across GuitarSet sessions: measured over the 36 test recordings the lag is
        median +252 samples (5.73 ms) with an IQR of [+84, +395], i.e. 1.9-9.0 ms. A single
        global FIR has ONE impulse response, so an unaligned fit would have to place energy
        at every session's delay at once and would smear every attack. Aligning first lets
        the IR model the transfer instead of the geometry.
        """
        # Stage 1 (coarse, unambiguous): envelopes are onset-dominated, so they cannot
        # lock onto the wrong pitch period the way the waveform can.
        if align == "none":
            return 0
        if align == "waveform":
            return HexMixDataset._xcorr_peak(mic, hexsum, max_lag)
        coarse = HexMixDataset._xcorr_peak(HexMixDataset._envelope(mic, sr),
                                           HexMixDataset._envelope(hexsum, sr), max_lag)
        if align == "envelope":
            return coarse
        # Stage 2 (fine, phase-accurate): the envelope kernel is ~20 ms wide, far too
        # coarse for phase -- at 1 kHz a 1 ms error is a whole period, and the fitted IR
        # applies ONE phase correction to every recording. Refine on the waveform inside
        # +/-2 ms of the coarse estimate, a window narrower than a guitar pitch period at
        # the low E (12 ms), so the period ambiguity cannot reappear.
        fine_win = int(0.002 * sr)
        shifted = np.roll(hexsum, coarse)
        return coarse + HexMixDataset._xcorr_peak(mic, shifted, fine_win)

    def _build(self, encoder):
        if self.cache_path.exists():
            z = np.load(self.cache_path)
            print(f"[hex_mix] loaded {self.cache_path} ({len(z['hex'])} recordings)", flush=True)
            return z["hex"], z["mic"]

        hexes, mics, lags = [], [], []
        for k, rec in enumerate(self.index):
            mic = load_audio(self.mic_by_rec[rec].audio_path, None, self.sr)
            sl = self._window(len(mic))
            mics.append(np.pad(mic[sl], (0, max(0, self.n - len(mic[sl]))))[: self.n])

            if self.variant == "predicted":
                a = np.load(self.pred_dir / f"{rec}.npy")            # (S, N) generated hex
                cur = np.stack([np.pad(c[sl], (0, max(0, self.n - len(c[sl]))))[: self.n] for c in a])
            else:
                streams = []
                for s in self.hex_by_rec[rec]:
                    x = load_audio(s.audio_path, s.channel, self.sr)
                    if self.variant == "decoded":
                        # round-trip through SAO so the reverb is fit in the domain it will
                        # actually receive; encode/decode is why this gets cached
                        x = encoder.decode(encoder.encode(x, self.sr))
                    streams.append(np.pad(x[sl], (0, max(0, self.n - len(x[sl]))))[: self.n])
                cur = np.stack(streams)
            # Align this recording before it enters the fit (see _best_lag). The lag is
            # always measured from the REAL hex, never from `cur`: for variant="predicted"
            # the generated audio is perceptually right but phase-unrelated to the
            # recording, so cross-correlating it against the mic would return noise. The
            # lag is a property of the session's mic placement, not of the signal, so the
            # real-hex estimate is the correct one to apply to every variant.
            lag = self._recording_lag(rec, sl)
            cur = np.roll(cur, lag, axis=-1)
            if lag > 0:
                cur[:, :lag] = 0.0
            elif lag < 0:
                cur[:, lag:] = 0.0
            lags.append(lag)
            hexes.append(cur.astype(np.float32))
            if (k + 1) % 25 == 0:
                print(f"[hex_mix] {k+1}/{len(self.index)} recordings", flush=True)

        lg = np.array(lags)
        print(f"[hex_mix] alignment lag: median {np.median(lg):+.0f} samples "
              f"({np.median(lg)/self.sr*1000:+.2f} ms), IQR [{np.percentile(lg,25):+.0f}, "
              f"{np.percentile(lg,75):+.0f}]", flush=True)
        hex_a, mic_a = np.stack(hexes), np.stack(mics).astype(np.float32)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.lag_path.write_text(json.dumps(self._lag_cache))
        np.savez(self.cache_path, hex=hex_a, mic=mic_a)
        print(f"[hex_mix] wrote {self.cache_path}", flush=True)
        return hex_a, mic_a

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        return {"hex": torch.from_numpy(self._hex[i]), "mic": torch.from_numpy(self._mic[i]),
                "recording_id": self.index[i]}
