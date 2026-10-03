# Envelope-regression dataset: score -> log-RMS envelope, with pitch held at GROUND TRUTH
# throughout. Pitch is never predicted here -- it is fed to the model as input and to the
# frozen synth as conditioning -- so every number this experiment produces is about envelope
# and nothing else.
#
# Extends dataset/control_flow_dataset.py rather than re-deriving the score conditioning:
# same excitation-segment channels off the realigned onsets, same z-scoring constants. Adds
# the cached SAO audio latents, which the FM velocity critic (losses.py) needs as its x1.
#
# Latent and control arrays are cached at the same 21.53 Hz grid but can differ by a frame at
# the tail, so both are truncated to their common length before the crop is taken -- an
# off-by-one here would silently misalign the critic's audio target from the curve it is
# scoring.
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from dataset.perf.perf_curve_dataset import COND_CHANNELS, ControlFlowDataset

# The model sees GT pitch as an extra conditioning channel on top of the score-only ones --
# but ONLY when it is not being asked to predict pitch. Feeding the answer as an input in that
# case would make the pitch head trivial and the numbers meaningless.
INPUT_CHANNELS = COND_CHANNELS + ["gt_pitch"]
INPUT_CHANNELS_PE = list(COND_CHANNELS)

TARGETS_ENV = ["envelope"]
TARGETS_PE = ["pitch", "envelope"]


def input_channels_for(targets, gain_mode: str = "max"):
    """Model input names. gain_mode="none" drops the supplied pseudo_velocity channel for
    the explicit no-gain ablation."""
    names = INPUT_CHANNELS_PE if "pitch" in targets else INPUT_CHANNELS
    if gain_mode == "none":
        names = [n for n in names if n != "pseudo_velocity"]
    return list(names)


def collate_env_reg(batch: List[Dict]) -> Dict:
    """Pads to the batch's max FRAME count, then re-flattens. Padding the flattened S*T axis
    directly would append frames to the last string instead of extending all six."""
    S = batch[0]["n_strings"]
    max_T = max(i["n_frames"] for i in batch)

    def pad_flat(x: torch.Tensor, T: int) -> torch.Tensor:
        v = x.reshape(S, T, -1)
        if T < max_T:
            v = torch.cat([v, torch.zeros((S, max_T - T, v.shape[-1]), dtype=v.dtype)], dim=1)
        return v.reshape(S * max_T, -1)

    def pad_note_mask(key: str) -> torch.Tensor:
        max_notes = max(item[key].shape[0] for item in batch)
        padded = torch.zeros((len(batch), max_notes, S * max_T), dtype=torch.bool)
        for i, item in enumerate(batch):
            n_notes = item[key].shape[0]
            T = item["n_frames"]
            if n_notes:
                value = item[key].reshape(n_notes, S, T)
                if T < max_T:
                    value = torch.cat(
                        [value, torch.zeros(
                            (n_notes, S, max_T - T), dtype=torch.bool
                        )],
                        dim=2,
                    )
                padded[i, :n_notes] = value.reshape(n_notes, S * max_T)
        return padded

    note_mask = pad_note_mask("note_mask")
    note_core_mask = pad_note_mask("note_core_mask")
    detail_dim = batch[0]["detail_features"].shape[-1]
    detail_features = (
        torch.stack([pad_flat(i["detail_features"], i["n_frames"]) for i in batch])
        if detail_dim else torch.zeros((len(batch), S * max_T, 0), dtype=torch.float32)
    )

    return {
        "recording_id": [i["recording_id"] for i in batch],
        "target": torch.stack([pad_flat(i["target"], i["n_frames"]) for i in batch]),
        "detail_features": detail_features,
        "latent": torch.stack([pad_flat(i["latent"], i["n_frames"]) for i in batch]),
        "controls": {n: torch.stack([pad_flat(i["controls"][n], i["n_frames"]) for i in batch])
                     for n in batch[0]["controls"]},
        "gt_pitch_hz": torch.stack([pad_flat(i["gt_pitch_hz"], i["n_frames"]) for i in batch]),
        "valid_mask": torch.stack([
            pad_flat(i["valid_mask"].unsqueeze(-1).float(), i["n_frames"]).squeeze(-1).bool()
            for i in batch]),
        "note_mask": note_mask,
        "note_core_mask": note_core_mask,
        "player_id": torch.stack([i["player_id"] for i in batch]),
        "n_strings": S,
        "n_frames": max_T,
    }


class EnvelopeRegDataset(ControlFlowDataset):
    def __init__(self, cfg, split: Optional[str] = None, deterministic: bool = False,
                 need_latent: bool = True, targets=None, pitch_filter: str = "none",
                 gain_mode: str = "max", full_length: bool = False,
                 extra_channels: Optional[List[str]] = None,
                 detail_features: str = "none"):
        super().__init__(cfg, split=split, deterministic=deterministic,
                         pitch_filter=pitch_filter, gain_mode=gain_mode,
                         full_length=full_length, extra_channels=extra_channels,
                         detail_features=detail_features)
        self.need_latent = need_latent
        self.targets = list(targets) if targets else list(TARGETS_ENV)
        self.predict_pitch = "pitch" in self.targets
        # target array column i+1 is self.extra_channels[i] (ControlFlowDataset's layout);
        # self.targets must therefore list exactly ["pitch"]+extra_channels or extra_channels,
        # in that order, so _cols below indexes the right columns.
        expected = (["pitch"] + self.extra_channels) if self.predict_pitch else self.extra_channels
        assert self.targets == expected, \
            f"targets {self.targets} must equal {expected} (pitch, then extra_channels in order)"
        self.latent_key = cfg.data.latent
        self.latent_store = zarr.open(
            str(Path(cfg.cache.root) / cfg.cache.latent_store), mode="r") if need_latent else None
        if need_latent:
            # drop recordings missing any string's latent, rather than training the critic
            # against a zero target it would happily learn to match
            self._index = [r for r in self._index if self._has_latents(r)]
        self._lat_cache: Dict[str, Dict] = {}

    def _has_latents(self, rec: str) -> bool:
        return all(sid in self.latent_store and self.latent_key in self.latent_store[sid]
                   for sid in self._recording_strings[rec].values())

    def _get_latents(self, rec: str) -> np.ndarray:
        if rec in self._lat_cache:
            return self._lat_cache[rec]["latent"]
        strings = self._recording_strings[rec]
        arrs = {s: self.latent_store[sid][self.latent_key][:] for s, sid in strings.items()}
        T = max(a.shape[0] for a in arrs.values())
        D = next(iter(arrs.values())).shape[1]
        out = np.zeros((self.n_strings, T, D), dtype=np.float32)
        for s, a in arrs.items():
            out[s, : a.shape[0]] = a
        self._lat_cache[rec] = {"latent": out}
        return out

    def __getitem__(self, idx: int) -> Dict:
        rec = self._index[idx]
        full = self._get_full(rec)
        S = self.n_strings
        T = full["T"]

        latent = self._get_latents(rec) if self.need_latent else None
        if latent is not None:
            T = min(T, latent.shape[1])          # common length; see module header

        start, length = self._pick_window(T)
        sl = slice(start, start + length)
        valid = full["frame_valid"][:, sl] & full["string_valid"][:, None]

        def flat(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a.reshape(S * length, -1)))

        controls = {n: flat(full["controls"][n][:, sl]) for n in self.cond_channels}
        controls["score_pitch_midi"] = flat(full["score_pitch_midi"][:, sl])
        if not self.predict_pitch:
            # GT pitch as model input, in the same normalised MIDI units as target ch 0
            controls["gt_pitch"] = flat(full["target"][:, sl, 0:1])

        # channel order matches self.targets == ["pitch"]+extra_channels or extra_channels
        extra_cols = list(range(1, 1 + len(self.extra_channels)))
        cols = [0] + extra_cols if self.predict_pitch else extra_cols
        item = {
            "recording_id": rec,
            "n_strings": S,
            "n_frames": length,
            "target": flat(full["target"][:, sl][:, :, cols]),
            "gt_pitch_hz": flat(full["pitch_hz"][:, sl, None]),
            "detail_features": torch.from_numpy(np.ascontiguousarray(
                full["detail_features"][:, sl].reshape(S * length, self.detail_dim)
            )),
            "controls": controls,
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid.reshape(S * length))),
            "note_mask": self._note_masks_for_window(full, start, length),
            "note_core_mask": self._note_masks_for_window(
                full, start, length, span_key="note_core_spans"
            ),
            "player_id": torch.tensor(self._player_id(rec), dtype=torch.long),
        }
        item["latent"] = flat(latent[:, sl]) if latent is not None else torch.zeros(S * length, 1)
        return item
