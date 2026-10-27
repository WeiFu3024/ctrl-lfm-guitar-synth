# This is the dataset file, which will be called by the training loop directly,
# it should be able to adapt several config options
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

from dataset.dataset_register import RawSample, build_raw_dataset
from dataset.split import split_registry


def _pad_to(array: np.ndarray, length: int) -> np.ndarray:
    if array.shape[0] >= length:
        return array
    pad_width = [(0, length - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad_width, mode="constant")


def _pad_tensor(t: torch.Tensor, length: int) -> torch.Tensor:
    if t.shape[0] >= length:
        return t
    pad = torch.zeros((length - t.shape[0],) + tuple(t.shape[1:]), dtype=t.dtype)
    return torch.cat([t, pad], dim=0)


def collate_variable_length(batch: List[Dict]) -> Dict:
    """Pads every item in a batch up to that batch's own max length (items can have
    different lengths since GuitarSynthDataset draws a random length per __getitem__)."""
    max_len = max(item["latent"].shape[0] for item in batch)
    control_names = batch[0]["controls"].keys()

    return {
        "sample_id": [item["sample_id"] for item in batch],
        "latent": torch.stack([_pad_tensor(item["latent"], max_len) for item in batch]),
        "controls": {
            name: torch.stack([_pad_tensor(item["controls"][name], max_len) for item in batch])
            for name in control_names
        },
        "valid_mask": torch.stack([_pad_tensor(item["valid_mask"], max_len) for item in batch]),
        "meta": [item["meta"] for item in batch],
    }


class GuitarSynthDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split: Optional[str] = None, deterministic: bool = False):
        self.cfg = cfg
        self.deterministic = deterministic
        raw_dataset = build_raw_dataset(cfg.data)
        self.sample_by_id = {s.sample_id: s for s in raw_dataset.list_samples()}

        latent_store_path = str(Path(cfg.cache.root) / cfg.cache.latent_store)
        control_store_path = str(Path(cfg.cache.root) / cfg.cache.control_store)
        self.latent_store = zarr.open(latent_store_path, mode="r")
        self.control_store = zarr.open(control_store_path, mode="r")

        self.latent_key = cfg.data.latent
        self.control_names = list(cfg.dataset.control_signals)
        self.min_segment_length = cfg.dataset.min_segment_length
        self.max_segment_length = cfg.dataset.max_segment_length
        self.random_crop = cfg.dataset.random_crop
        self.pad_short = cfg.dataset.pad_short

        self._index = self._build_index()
        if split is not None:
            self._index = self._filter_by_split(self._index, split)

    def _is_fully_cached(self, sample_id: str) -> bool:
        if sample_id not in self.latent_store or self.latent_key not in self.latent_store[sample_id]:
            return False
        if sample_id not in self.control_store:
            return False
        ctrl_grp = self.control_store[sample_id]
        return all(name in ctrl_grp for name in self.control_names)

    def _build_index(self) -> List[str]:
        index = []
        for sample_id in self.sample_by_id:
            if not self._is_fully_cached(sample_id):
                continue
            latent_len = self.latent_store[sample_id][self.latent_key].shape[0]
            if latent_len < self.min_segment_length and not self.pad_short:
                continue
            index.append(sample_id)
        return index

    def _filter_by_split(self, index: List[str], split: str) -> List[str]:
        by_dataset_type: Dict[str, List[RawSample]] = {}
        for sample_id in index:
            dtype = self.sample_by_id[sample_id].meta["dataset"]
            by_dataset_type.setdefault(dtype, []).append(self.sample_by_id[sample_id])

        entry_by_type = {e.dataset: e for e in self.cfg.dataset.split.entries}
        keep = set()
        for dtype, samples in by_dataset_type.items():
            if dtype not in entry_by_type:
                raise ValueError(f"No dataset.split.entries entry configured for dataset type '{dtype}'")
            entry = entry_by_type[dtype]
            splitter = split_registry.get(entry.splitter)(entry.params)
            assignment = splitter.assign(samples)
            keep.update(sid for sid, assigned in assignment.items() if assigned == split)
        return [sid for sid in index if sid in keep]

    def __len__(self) -> int:
        return len(self._index)

    def _pick_length(self, available_length: int) -> int:
        if self.deterministic:
            return self.max_segment_length
        if self.pad_short:
            return random.randint(self.min_segment_length, self.max_segment_length)
        hi = max(self.min_segment_length, min(self.max_segment_length, available_length))
        return random.randint(self.min_segment_length, hi)

    def _pick_start(self, available_length: int, length: int) -> int:
        if self.deterministic or not self.random_crop or available_length <= length:
            return 0
        return random.randint(0, available_length - length)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self._index[idx]
        sample: RawSample = self.sample_by_id[sample_id]

        latent_arr = self.latent_store[sample_id][self.latent_key]
        available = latent_arr.shape[0]
        length = self._pick_length(available)
        start = self._pick_start(available, length)

        latent = _pad_to(latent_arr[start:start + length], length)

        valid_len = min(available - start, length)
        valid_mask = torch.zeros(length, dtype=torch.bool)
        valid_mask[:valid_len] = True

        ctrl_grp = self.control_store[sample_id]
        controls = {}
        for name in self.control_names:
            ctrl = ctrl_grp[name][start:start + length]
            controls[name] = torch.from_numpy(_pad_to(ctrl, length))

        return {
            "sample_id": sample_id,
            "latent": torch.from_numpy(latent),
            "controls": controls,
            "valid_mask": valid_mask,
            "meta": sample.meta,
            # Which crop this item took. Subclasses that substitute an alternative control
            # source (experiments/recon_synth/recon_mix_dataset.py) must slice it with the SAME window, or
            # the substituted curves silently misalign against the latent.
            # collate_variable_length ignores keys it does not name, so this is inert.
            "crop_start": start,
        }
