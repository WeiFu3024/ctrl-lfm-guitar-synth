# Manage the multiple dataset here, and write the function to load the different dataset
# into unified format, this file is important for scalability of multiple dataset usage
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dataset.registry import Registry

def default_guitarset_root() -> str:
    """GuitarSet's audio root when a config does not set data.root.

    Per-MACHINE, not per-experiment: `fatcat` mounts data/ from a full disk and nests
    GuitarSet two levels deeper than terra, so the repo-relative default finds zero files
    there and every job silently renders nothing. GUITARSET_ROOT points at a flat tree
    instead. dataset/control/signal.py reads the same variable for the annotation dir.
    """
    return os.environ.get("GUITARSET_ROOT", "data/GuitarSet")


_GUITARSET_HEX_RE = re.compile(r"^(?P<player>\d+)_(?P<tune>.+)_(?P<style>comp|solo)_hex_cln\.wav$")
_GUITARSET_MIC_RE = re.compile(r"^(?P<player>\d+)_(?P<tune>.+)_(?P<style>comp|solo)_mic\.wav$")
_STRING_FILE_RE = re.compile(r"^string_(?P<index>\d+)\.wav$")


@dataclass
class RawSample:
    sample_id: str
    audio_path: str
    channel: Optional[int]  # select this channel from a multi-channel file; None = file is already mono
    sample_rate: int
    meta: Dict[str, Any] = field(default_factory=dict)


class RawDatasetLoader(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def list_samples(self) -> List[RawSample]: ...

    def __len__(self) -> int:
        return len(self.list_samples())


dataset_registry = Registry("dataset")


@dataset_registry.register("GuitarSet")
class GuitarSetLoader(RawDatasetLoader):
    """audio_hex-pickup_debleeded/*.wav: one 6-channel file per recording, channel index = string."""

    NUM_STRINGS = 6

    def _root(self) -> Path:
        root = Path(self.cfg.root) if self.cfg.root else Path(default_guitarset_root())
        return root / "audio_hex-pickup_debleeded"

    def list_samples(self) -> List[RawSample]:
        samples = []
        for path in sorted(self._root().glob("*.wav")):
            m = _GUITARSET_HEX_RE.match(path.name)
            if not m:
                continue
            recording_id = path.stem.removesuffix("_hex_cln")
            for string_idx in range(self.NUM_STRINGS):
                samples.append(
                    RawSample(
                        sample_id=f"guitarset/{recording_id}/string_{string_idx + 1}",
                        audio_path=str(path),
                        channel=string_idx,
                        sample_rate=self.cfg.sample_rate,
                        meta={
                            "dataset": "GuitarSet",
                            "player": m.group("player"),
                            "tune": m.group("tune"),
                            "style": m.group("style"),
                            "string": string_idx + 1,
                        },
                    )
                )
        return samples


@dataset_registry.register("GuitarSet-mic")
class GuitarSetMicLoader(RawDatasetLoader):
    """audio_mono-mic/*_mic.wav: the room-mic recording of a whole performance -- one mono
    file per recording, i.e. all six strings already mixed, plus mic/room colouration.

    Exists so the joint synth can be trained to predict the real recording directly
    (dataset/synth/joint_synth_dataset.py's "mix" mode), which is the comparison point against
    DDSP-Guitar -- it renders per-string and sums, whereas this targets what a listener
    actually hears. One sample per recording, not six: recording_id here is the filename
    minus "_mic", which matches GuitarSetLoader's "_hex_cln"-stripped recording_id 1:1, so
    the two trees line up by key.

    Deliberately NOT part of the "GuitarSet" entry: mixing one-per-recording samples into a
    six-per-recording loader would break every existing per-string sample_id and split.
    """

    def _root(self) -> Path:
        root = Path(self.cfg.root) if self.cfg.root else Path(default_guitarset_root())
        return root / "audio_mono-mic"

    def list_samples(self) -> List[RawSample]:
        samples = []
        for path in sorted(self._root().glob("*.wav")):
            m = _GUITARSET_MIC_RE.match(path.name)
            if not m:
                continue
            recording_id = path.stem.removesuffix("_mic")
            samples.append(
                RawSample(
                    sample_id=f"guitarset-mic/{recording_id}",
                    audio_path=str(path),
                    channel=None,  # already mono
                    sample_rate=self.cfg.sample_rate,
                    meta={
                        "dataset": "GuitarSet-mic",
                        "player": m.group("player"),
                        "tune": m.group("tune"),
                        "style": m.group("style"),
                    },
                )
            )
        return samples


@dataset_registry.register("SynthTab-mix")
class SynthTabMixLoader(RawDatasetLoader):
    """mix.wav from each SynthTab song directory -- the rendered sum of the six strings.

    Not a room mic: no body radiation, no room. It exists so the mix-prediction arm can be
    PRETRAINED on 4550 songs before fine-tuning on GuitarSet's 306, which is the arm that
    overfits hardest (one target stream per recording instead of six, train/val gap -0.225
    by epoch 240; see docs/reports/archive/2026/project-history-through-2026-09-18.md). The domain gap to GuitarSet's mic is exactly what
    the fine-tune stage is for.
    """

    def _root(self) -> Path:
        return Path(self.cfg.root) if self.cfg.root else Path("data/SynthTab-hex")

    def list_samples(self) -> List[RawSample]:
        samples = []
        for song_dir in sorted(p for p in self._root().iterdir() if p.is_dir()):
            path = song_dir / "mix.wav"
            if not path.exists():
                continue
            samples.append(
                RawSample(
                    sample_id=f"synthtab-mix/{song_dir.name}",
                    audio_path=str(path),
                    channel=None,
                    sample_rate=self.cfg.sample_rate,
                    meta={"dataset": "SynthTab-mix", "song": song_dir.name},
                )
            )
        return samples


@dataset_registry.register("SynthTab-hex")
class SynthTabHexLoader(RawDatasetLoader):
    """One directory per song, containing mix.wav + string_N.wav (N not always 1..6)."""

    def _root(self) -> Path:
        return Path(self.cfg.root) if self.cfg.root else Path("data/SynthTab-hex")

    def list_samples(self) -> List[RawSample]:
        samples = []
        for song_dir in sorted(p for p in self._root().iterdir() if p.is_dir()):
            for path in sorted(song_dir.glob("string_*.wav")):
                m = _STRING_FILE_RE.match(path.name)
                if not m:
                    continue
                string_idx = int(m.group("index"))
                samples.append(
                    RawSample(
                        sample_id=f"synthtab-hex/{song_dir.name}/string_{string_idx}",
                        audio_path=str(path),
                        channel=None,
                        sample_rate=self.cfg.sample_rate,
                        meta={
                            "dataset": "SynthTab-hex",
                            "song": song_dir.name,
                            "string": string_idx,
                        },
                    )
                )
        return samples


class CompositeDatasetLoader(RawDatasetLoader):
    def __init__(self, loaders: List[RawDatasetLoader]):
        self.loaders = loaders

    def list_samples(self) -> List[RawSample]:
        samples: List[RawSample] = []
        for loader in self.loaders:
            samples.extend(loader.list_samples())
        return samples


def build_raw_dataset(cfg) -> RawDatasetLoader:
    types: Union[str, List[str]] = cfg.type
    if isinstance(types, str):
        return dataset_registry.get(types)(cfg)
    return CompositeDatasetLoader([dataset_registry.get(t)(cfg) for t in types])
