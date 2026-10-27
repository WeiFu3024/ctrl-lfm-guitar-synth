# Joint 6-string analogue of experiments/velocity_synth/velocity_synth_dataset.py's VelocitySynthDataset: the
# "no control-curve modeling at all" baseline, extended to the joint (cross-string) synth.
# midi_pitch and pseudo_velocity (mean definition) are derived on the fly from real
# notes/envelope, same as VelocitySynthDataset -- never cached, so JointSynthDataset's
# generic control_store[sid][name] read (which assumes every control name is precomputed)
# can't be reused as-is. Subclasses JointSynthDataset and overrides only the two methods
# that touch the control source: _build_recording_index (checks pitch/envelope presence,
# the SOURCE data, not the derived names) and _get_full (derives instead of reading).
# __getitem__, split filtering, and collate_joint_synth are all reused unchanged.
from typing import Dict

import numpy as np

from dataset.control.signal import midi_to_hz, notes_for_sample
from dataset.synth.joint_synth_dataset import JointSynthDataset

# Calibrated quantisation floor from experiments/velocity_synth/quantize.py. Keep the
# derived velocity arm's silence convention stable without making live dataset code
# import the archived experiments package.
ENVELOPE_MIN = -5.753


DERIVED_CONTROL_NAMES = frozenset(("midi_pitch", "pseudo_velocity"))


def _midi_pitch_curve(sample, frame_rate: float, t_total: int) -> np.ndarray:
    """Flat per-note nominal MIDI pitch, in Hz (matching control.zarr's "pitch" units) --
    no pitch_dev/vibrato/bend at all, the simplest possible pitch signal. 0 Hz
    (silence, control.zarr's own convention) between notes."""
    midi = np.zeros(t_total, dtype=np.float32)
    for onset_s, dur_s, pitch in notes_for_sample(sample, {}):
        a = int(round(onset_s * frame_rate))
        b = min(t_total, a + max(1, int(round(dur_s * frame_rate))))
        if b > a:
            midi[a:b] = pitch
    return midi_to_hz(midi)


def _pseudo_velocity_curve_mean(sample, envelope: np.ndarray, frame_rate: float, t_total: int) -> np.ndarray:
    """Per-note MEAN real envelope, broadcast across the note -- NOT the DDSP-Guitar-
    matching MAX convention _pseudo_velocity_curve above uses. This is the "pe
    definition" of velocity: the same masked-mean "level" every
    experiments/vqvae_ctrl/ module computes (e.g. model.py's
    SingleChannelCurveVAE.forward), applied per note instead of per arbitrary
    curve segment. Kept as a separate function rather than a flag on the existing
    one, so the already-reported MAX-based velocity baseline
    (config/archive/ctrl-lfm-guitarset-velocity.yaml) stays exactly reproducible."""
    curve = np.full(t_total, ENVELOPE_MIN, dtype=np.float32)
    for onset_s, dur_s, _pitch in notes_for_sample(sample, {}):
        a = int(round(onset_s * frame_rate))
        b = min(t_total, a + max(1, int(round(dur_s * frame_rate))))
        if b > a:
            curve[a:b] = envelope[a:b].mean()
    return curve

def _pseudo_velocity_curve_max(
    sample, envelope: np.ndarray, frame_rate: float, t_total: int,
) -> np.ndarray:
    """Per-note MAX real envelope, broadcast across the note."""
    curve = np.full(t_total, ENVELOPE_MIN, dtype=np.float32)
    for onset_s, dur_s, _pitch in notes_for_sample(sample, {}):
        a = int(round(onset_s * frame_rate))
        b = min(t_total, a + max(1, int(round(dur_s * frame_rate))))
        if b > a:
            curve[a:b] = envelope[a:b].max()
    return curve


def _pseudo_velocity_curve(
    sample, envelope: np.ndarray, frame_rate: float, t_total: int, mode: str,
) -> np.ndarray:
    if mode == "mean":
        return _pseudo_velocity_curve_mean(sample, envelope, frame_rate, t_total)
    if mode == "max":
        return _pseudo_velocity_curve_max(sample, envelope, frame_rate, t_total)
    raise ValueError(f"velocity_gain_mode must be 'mean' or 'max', got {mode!r}")



class JointVelocitySynthDataset(JointSynthDataset):
    def __init__(self, cfg, split=None, deterministic=False):
        self.velocity_gain_mode = getattr(cfg.dataset, "velocity_gain_mode", "mean")
        if self.velocity_gain_mode not in {"mean", "max"}:
            raise ValueError(
                "dataset.velocity_gain_mode must be 'mean' or 'max', "
                f"got {self.velocity_gain_mode!r}"
            )
        super().__init__(cfg, split=split, deterministic=deterministic)

    def _build_recording_index(self) -> Dict[str, Dict[int, str]]:
        by_rec: Dict[str, Dict[int, str]] = {}
        required_cached = [
            name for name in self.control_names
            if name not in DERIVED_CONTROL_NAMES
        ]
        for sample_id, sample in self.sample_by_id.items():
            if sample.meta.get("dataset") != self.hex_dataset or "string" not in sample.meta:
                continue
            if sample_id not in self.latent_store or self.latent_key not in self.latent_store[sample_id]:
                continue
            if sample_id not in self.control_store:
                continue
            grp = self.control_store[sample_id]
            # midi_pitch/pseudo_velocity are derived below; event channels such as
            # onset/voiced must already be present in the control cache.
            if "pitch" not in grp or "envelope" not in grp:
                continue
            if not all(name in grp for name in required_cached):
                continue
            by_rec.setdefault(sample_id.split("/")[1], {})[sample.meta["string"] - 1] = sample_id
        return by_rec

    def _get_full(self, recording_id: str) -> Dict:
        if recording_id in self._cache:
            return self._cache[recording_id]
        strings = self._recording_strings[recording_id]
        T = max(self.latent_store[sid][self.latent_key].shape[0] for sid in strings.values())
        D = self.latent_store[next(iter(strings.values()))][self.latent_key].shape[1]

        latent = np.zeros((self.n_strings, T, D), dtype=np.float32)
        # Derived controls are always scalar. Cached controls are initialized from
        # their first sample so this remains compatible with future one-channel
        # annotation-derived inputs without duplicating the dataset class.
        controls = {
            name: np.zeros((self.n_strings, T, 1), dtype=np.float32)
            if name in DERIVED_CONTROL_NAMES else None
            for name in self.control_names
        }
        string_valid = np.zeros(self.n_strings, dtype=bool)
        frame_valid = np.zeros((self.n_strings, T), dtype=bool)

        for s_idx, sid in strings.items():
            arr = self.latent_store[sid][self.latent_key]
            n = arr.shape[0]
            latent[s_idx, :n] = arr[:]
            string_valid[s_idx] = True
            frame_valid[s_idx, :n] = True

            sample = self.sample_by_id[sid]
            grp = self.control_store[sid]
            frame_rate = grp["pitch"].attrs["frame_rate"]
            real_envelope = grp["envelope"][:n, 0]
            if "midi_pitch" in controls:
                controls["midi_pitch"][s_idx, :n, 0] = _midi_pitch_curve(sample, frame_rate, n)
            if "pseudo_velocity" in controls:
                controls["pseudo_velocity"][s_idx, :n, 0] = _pseudo_velocity_curve(
                    sample, real_envelope, frame_rate, n, self.velocity_gain_mode
                )

            for name in self.control_names:
                if name in DERIVED_CONTROL_NAMES:
                    continue
                value = np.asarray(grp[name][:n], dtype=np.float32)
                if value.ndim == 1:
                    value = value[:, None]
                if controls[name] is None:
                    controls[name] = np.zeros(
                        (self.n_strings, T, value.shape[-1]), dtype=np.float32
                    )
                controls[name][s_idx, :n] = value

        for name in self.control_names:
            if controls[name] is None:
                controls[name] = np.zeros((self.n_strings, T, 1), dtype=np.float32)

        entry = {"latent": latent, "controls": controls, "string_valid": string_valid,
                 "frame_valid": frame_valid, "T": T}
        mix_id = f"{self.mix_prefix}/{recording_id}"
        if mix_id in self.latent_store and self.latent_key in self.latent_store[mix_id]:
            entry["mix_latent"] = self.latent_store[mix_id][self.latent_key][:].astype(np.float32)
        self._cache[recording_id] = entry
        return entry
