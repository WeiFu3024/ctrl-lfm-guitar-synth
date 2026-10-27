# Build the latent pre-caching pipeline here
# The pre-caching pipeline should load the defined dataset, and the latent extraction method
# pre-extract the latent, then store the cache into the defined cache path
# please preserve the possiblities of adding more latent type
from typing import Dict, List, Tuple

import numpy as np

from dataset.cache_extractor import CacheExtractor
from dataset.dataset_register import RawSample
from dataset.latent.sao3 import BaseLatentEncoder


class LatentCacheExtractor(CacheExtractor):
    def __init__(self, cfg, raw_dataset, encoder: BaseLatentEncoder, store_path: str):
        super().__init__(cfg, raw_dataset, store_path)
        self.encoder = encoder

    def compute_sample(self, sample: RawSample, audio: np.ndarray) -> Dict[str, Tuple[np.ndarray, dict]]:
        latent = self.encoder.encode(audio, self.cfg.data.sample_rate)
        attrs = {
            "frame_rate": self.encoder.frame_rate,
            "hop_length": self.encoder.hop_length,
            "sample_rate": self.encoder.sample_rate,
            "model_id": self.cfg.latent.model_id,
        }
        return {self.cfg.data.latent: (latent, attrs)}

    def expected_keys(self) -> List[str]:
        return [self.cfg.data.latent]
