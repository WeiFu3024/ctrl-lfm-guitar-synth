from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from tqdm import tqdm

from dataset.audio_utils import load_audio
from dataset.dataset_register import RawDatasetLoader, RawSample


class CacheExtractor(ABC):
    """Shared orchestration for pre-caching a signal (control or latent) into a Zarr store.

    Subclasses implement compute_sample(); everything else (iterating the raw
    dataset, skip-if-cached, writing per-sample zarr groups) is identical
    between the control and latent extractors.
    """

    def __init__(self, cfg, raw_dataset: RawDatasetLoader, store_path: str):
        self.cfg = cfg
        self.raw_dataset = raw_dataset
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        self.store = zarr.open(store_path, mode="a")

    @abstractmethod
    def compute_sample(self, sample: RawSample, audio: np.ndarray) -> Dict[str, Tuple[np.ndarray, dict]]:
        """Returns {array_name: (data, attrs)} to be written under the sample's group."""
        ...

    @abstractmethod
    def expected_keys(self) -> List[str]:
        """Array names this extractor writes per sample, used to detect a complete cache entry."""
        ...

    def is_cached(self, sample: RawSample) -> bool:
        if sample.sample_id not in self.store:
            return False
        grp = self.store[sample.sample_id]
        return all(key in grp for key in self.expected_keys())

    def _pending_samples(self, limit: Optional[int] = None) -> List[RawSample]:
        samples = self.raw_dataset.list_samples()
        if limit is not None:
            samples = samples[:limit]
        return [s for s in samples if self.cfg.cache.overwrite or not self.is_cached(s)]

    def _write_outputs(self, sample_id: str, outputs: Dict[str, Tuple[np.ndarray, dict]]) -> None:
        grp = self.store.require_group(sample_id)
        for name, (data, attrs) in outputs.items():
            arr = grp.array(name, data=data, chunks=data.shape, overwrite=True)
            arr.attrs.update(attrs)

    def run(self, limit: Optional[int] = None) -> None:
        for sample in tqdm(self._pending_samples(limit)):
            audio = load_audio(sample.audio_path, sample.channel, self.cfg.data.sample_rate)
            outputs = self.compute_sample(sample, audio)
            self._write_outputs(sample.sample_id, outputs)
