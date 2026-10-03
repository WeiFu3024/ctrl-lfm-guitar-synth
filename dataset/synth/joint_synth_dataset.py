# Per-RECORDING dataset for the joint 6-string synth: all six strings' control signals
# and latents for one performance, time-aligned, in a single item. The per-string
# GuitarSynthDataset cannot express this -- it emits six independent samples that never
# meet, so no model built on it can know a strum hits six strings at once.
#
# Layout: strings are FLATTENED into the sequence axis as (S*T, ...), not kept as a
# separate tensor axis. That is deliberate and load-bearing: model/common/flow_matching.py
# assumes rank-3 (B, L, D) throughout (t.view(B,1,1), mask (B,L,1), B,_,_ = shape), so
# presenting (B, S*T, D) lets FlowMatching, the loss, and CFG dropout work COMPLETELY
# UNCHANGED, with t correctly sampled once per recording rather than once per string.
# model/synth/joint_synth.py unflattens to (B,S,T,D) internally for its axial attention.
#
# Structure follows experiments/frame_rate_perf_decoder/joint_perf_dataset.py, which
# solved the same per-recording/absent-string problem for control curves.
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from dataset.synth.dataset import _pad_to
from dataset.dataset_register import build_raw_dataset
from dataset.split import split_registry

N_STRINGS = 6
# Which loader supplies the mix target for each per-string dataset. GuitarSet's is a real
# room mic; SynthTab's is a rendered sum with no room, which is why SynthTab is only useful
# for PRETRAINING the mix arm before fine-tuning on GuitarSet.
MIX_PREFIX_BY_HEX = {
    "GuitarSet": "guitarset-mic",       # dataset_register.py's GuitarSetMicLoader
    "SynthTab-hex": "synthtab-mix",     # dataset_register.py's SynthTabMixLoader
}


def collate_joint_synth(batch: List[Dict]) -> Dict:
    """Pads to the batch's max FRAME count, then re-flattens. Cannot reuse
    dataset/synth/dataset.py's collate_variable_length: that pads along the flattened S*T axis,
    which would append frames to the last string instead of extending all six."""
    S = batch[0]["n_strings"]
    max_T = max(item["n_frames"] for item in batch)

    def pad_flat(x: torch.Tensor, T: int, fill: float = 0.0) -> torch.Tensor:
        v = x.reshape(S, T, -1)
        if T < max_T:
            pad = torch.full((S, max_T - T, v.shape[-1]), fill, dtype=v.dtype)
            v = torch.cat([v, pad], dim=1)
        return v.reshape(S * max_T, -1)

    out = {
        "recording_id": [i["recording_id"] for i in batch],
        "latent": torch.stack([pad_flat(i["latent"], i["n_frames"]) for i in batch]),
        "controls": {
            name: torch.stack([pad_flat(i["controls"][name], i["n_frames"]) for i in batch])
            for name in batch[0]["controls"]
        },
        "valid_mask": torch.stack([
            pad_flat(i["valid_mask"].unsqueeze(-1).float(), i["n_frames"]).squeeze(-1).bool()
            for i in batch
        ]),
        "string_valid": torch.stack([i["string_valid"] for i in batch]),
        "n_strings": S,
    }
    if "mix_latent" in batch[0]:
        out["mix_latent"] = torch.stack([
            torch.from_numpy(_pad_to(i["mix_latent"].numpy(), max_T)) for i in batch
        ])
        out["mix_valid_mask"] = torch.stack([
            torch.cat([torch.ones(i["n_frames"], dtype=torch.bool),
                       torch.zeros(max_T - i["n_frames"], dtype=torch.bool)])
            for i in batch
        ])
    return out


class JointSynthDataset(torch.utils.data.Dataset):
    """One item per recording. output_mode="per_string" targets the six hex latents;
    "mix" targets the mono mic latent (dataset_register.py's GuitarSet-mic) while keeping
    all six strings as conditioning."""

    def __init__(self, cfg, split: Optional[str] = None, deterministic: bool = False):
        self.cfg = cfg
        self.deterministic = deterministic
        self.output_mode = getattr(cfg.model, "joint", None) and cfg.model.joint.output_mode or "per_string"
        self.n_strings = N_STRINGS
        self.control_names = list(cfg.dataset.control_signals)
        self.latent_key = cfg.data.latent
        self.min_segment_length = cfg.dataset.min_segment_length
        self.max_segment_length = cfg.dataset.max_segment_length
        self.random_crop = cfg.dataset.random_crop

        # cfg.data.type may be a single key or a list; the per-string dataset is the one
        # this class indexes by recording, and it decides the mix target's key prefix
        types = cfg.data.type if isinstance(cfg.data.type, (list, tuple)) else [cfg.data.type]
        self.hex_dataset = next(t for t in types if t in MIX_PREFIX_BY_HEX)
        self.mix_prefix = MIX_PREFIX_BY_HEX[self.hex_dataset]

        raw = build_raw_dataset(cfg.data)
        self.sample_by_id = {s.sample_id: s for s in raw.list_samples()}
        self.latent_store = zarr.open(str(Path(cfg.cache.root) / cfg.cache.latent_store), mode="r")
        self.control_store = zarr.open(str(Path(cfg.cache.root) / cfg.cache.control_store), mode="r")

        self._recording_strings = self._build_recording_index()
        self._index = sorted(self._recording_strings)
        if split is not None:
            self._index = self._filter_by_split(self._index, split)
        if self.output_mode == "mix":
            # Drop recordings whose mic latent was never cached, rather than silently
            # training on a zero target.
            self._index = [r for r in self._index if f"{self.mix_prefix}/{r}" in self.latent_store]
        self._cache: Dict[str, Dict] = {}

    def _build_recording_index(self) -> Dict[str, Dict[int, str]]:
        """recording_id -> {string_idx: sample_id}, a string included only if BOTH its
        latent and every requested control signal are cached.

        On GuitarSet every recording yields all six (the hex pickup is a 6-channel file, so
        an unplayed string is silence, not a missing stream) and the absent-string path
        below never fires. SynthTab is the opposite: only ~56% of songs render all six, and
        some render one, so that path carries real weight there.
        """
        by_rec: Dict[str, Dict[int, str]] = {}
        for sample_id, sample in self.sample_by_id.items():
            if sample.meta.get("dataset") != self.hex_dataset or "string" not in sample.meta:
                continue
            if sample_id not in self.latent_store or self.latent_key not in self.latent_store[sample_id]:
                continue
            if sample_id not in self.control_store:
                continue
            grp = self.control_store[sample_id]
            if not all(n in grp for n in self.control_names):
                continue
            by_rec.setdefault(sample_id.split("/")[1], {})[sample.meta["string"] - 1] = sample_id
        return by_rec

    def _filter_by_split(self, index: List[str], split: str) -> List[str]:
        # One representative RawSample per recording: the guitarset_ddsp splitter assigns
        # by recording, so all six strings land in the same split by construction.
        entry = {e.dataset: e for e in self.cfg.dataset.split.entries}[self.hex_dataset]
        splitter = split_registry.get(entry.splitter)(entry.params)
        rep = {r: self.sample_by_id[next(iter(s.values()))] for r, s in self._recording_strings.items()}
        assignment = splitter.assign(list(rep.values()))
        keep = {r for r, s in rep.items() if assignment.get(s.sample_id) == split}
        return [r for r in index if r in keep]

    def __len__(self) -> int:
        return len(self._index)

    def _get_full(self, recording_id: str) -> Dict:
        if recording_id in self._cache:
            return self._cache[recording_id]
        strings = self._recording_strings[recording_id]
        T = max(self.latent_store[sid][self.latent_key].shape[0] for sid in strings.values())
        D = self.latent_store[next(iter(strings.values()))][self.latent_key].shape[1]

        latent = np.zeros((self.n_strings, T, D), dtype=np.float32)
        controls = {n: None for n in self.control_names}
        string_valid = np.zeros(self.n_strings, dtype=bool)
        frame_valid = np.zeros((self.n_strings, T), dtype=bool)

        for s_idx, sid in strings.items():
            arr = self.latent_store[sid][self.latent_key]
            n = arr.shape[0]
            latent[s_idx, :n] = arr[:]
            string_valid[s_idx] = True
            frame_valid[s_idx, :n] = True
            grp = self.control_store[sid]
            for name in self.control_names:
                c = _pad_to(grp[name][:T], T)
                if controls[name] is None:
                    controls[name] = np.zeros((self.n_strings, T, c.shape[-1]), dtype=np.float32)
                controls[name][s_idx] = c[:T]
        for name in self.control_names:  # a control with no present string at all
            if controls[name] is None:
                controls[name] = np.zeros((self.n_strings, T, 1), dtype=np.float32)

        entry = {"latent": latent, "controls": controls, "string_valid": string_valid,
                 "frame_valid": frame_valid, "T": T}
        mix_id = f"{self.mix_prefix}/{recording_id}"
        if mix_id in self.latent_store and self.latent_key in self.latent_store[mix_id]:
            entry["mix_latent"] = self.latent_store[mix_id][self.latent_key][:].astype(np.float32)
        self._cache[recording_id] = entry
        return entry

    def _pick_window(self, available: int):
        if self.deterministic:
            length = min(self.max_segment_length, available)
            return 0, length
        length = random.randint(self.min_segment_length, self.max_segment_length)
        length = min(length, available)
        start = random.randint(0, available - length) if (self.random_crop and available > length) else 0
        return start, length

    def __getitem__(self, idx: int) -> Dict:
        recording_id = self._index[idx]
        full = self._get_full(recording_id)
        S, T = self.n_strings, full["T"]
        # ONE window shared by all six strings -- that shared crop is what keeps them
        # time-aligned, and is the whole point of the joint framing.
        start, length = self._pick_window(T)
        sl = slice(start, start + length)

        latent = full["latent"][:, sl]                              # (S, L, D)
        fvalid = full["frame_valid"][:, sl]                          # (S, L)
        # absent strings are masked out entirely, so they contribute nothing to the loss
        valid = fvalid & full["string_valid"][:, None]

        item = {
            "recording_id": recording_id,
            "n_strings": S,
            "n_frames": length,
            "latent": torch.from_numpy(np.ascontiguousarray(latent.reshape(S * length, -1))),
            "controls": {
                n: torch.from_numpy(np.ascontiguousarray(full["controls"][n][:, sl].reshape(S * length, -1)))
                for n in self.control_names
            },
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid.reshape(S * length))),
            "string_valid": torch.from_numpy(full["string_valid"].copy()),
        }
        if self.output_mode == "mix":
            mix = full["mix_latent"]
            m = _pad_to(mix[sl], length) if mix.shape[0] >= start else np.zeros((length, mix.shape[1]), np.float32)
            item["mix_latent"] = torch.from_numpy(np.ascontiguousarray(m[:length]))
        return item
