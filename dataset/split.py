import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set

from dataset.dataset_register import RawSample
from dataset.registry import Registry

SPLITS_DIR = Path(__file__).parent / "splits"


class Splitter(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def assign(self, samples: List[RawSample]) -> Dict[str, str]:
        """Returns {sample_id: one of 'train'/'val'/'test'}."""
        ...


split_registry = Registry("splitter")


def _read_names(filename: str) -> Set[str]:
    return set((SPLITS_DIR / filename).read_text().split())


@split_registry.register("guitarset_ddsp")
class GuitarSetDDSPSplitter(Splitter):
    """DDSP-Guitar's (erl-j/ddsp-guitar) exact GuitarSet split, vendored in dataset/splits/.
    Split is per recording (all strings of a recording share one assignment), since a
    recording_id like '00_SS2-88-F_comp' is shared by string_1..string_6's sample_ids."""

    def assign(self, samples: List[RawSample]) -> Dict[str, str]:
        test_names = _read_names("guitarset_ddsp_tst_filenames.txt")
        val_names = _read_names("guitarset_ddsp_val_filenames.txt")

        result = {}
        for sample in samples:
            recording_id = sample.sample_id.split("/")[1]
            if recording_id in test_names:
                result[sample.sample_id] = "test"
            elif recording_id in val_names:
                result[sample.sample_id] = "val"
            else:
                result[sample.sample_id] = "train"
        return result


@split_registry.register("synthtab_hex_random")
class SynthTabHexRandomSplitter(Splitter):
    """Seeded random split by song (no external reference split exists for SynthTab-hex).
    Grouped by song, not by individual string, so correlated string_1..string_6 content
    from one song doesn't leak across train/val."""

    def assign(self, samples: List[RawSample]) -> Dict[str, str]:
        val_fraction = self.params.get("val_fraction", 0.02)
        seed = self.params.get("seed", 0)

        songs = sorted({s.meta["song"] for s in samples})
        rng = random.Random(seed)
        rng.shuffle(songs)
        val_songs = set(songs[: round(val_fraction * len(songs))])

        return {s.sample_id: ("val" if s.meta["song"] in val_songs else "train") for s in samples}
