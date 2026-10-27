# define the extractor here, wire the control and build dataset control cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset.audio_utils import load_audio
from dataset.cache_extractor import CacheExtractor
from dataset.control.signal import MissingAnnotationError, control_registry
from dataset.dataset_register import RawSample
from dataset.latent.sao3 import BaseLatentEncoder


def _compute_control_signals(args):
    """Module-level (picklable) worker: load audio + compute all configured control
    signals for one sample. Runs in a separate process, so it does no zarr I/O itself
    — the caller writes results back on the main process."""
    sample, sample_rate, hop_length, signal_specs = args
    audio = load_audio(sample.audio_path, sample.channel, sample_rate)

    outputs = {}
    for name, params in signal_specs:
        signal = control_registry.get(name)(sample_rate=sample_rate, hop_length=hop_length, **params)
        try:
            result = signal.compute(audio, sample)
        except MissingAnnotationError:
            continue  # e.g. no MIDI available yet for this SynthTab-hex song
        outputs[name] = (
            result.values,
            {"frame_rate": result.frame_rate, "hop_length": signal.hop_length, "sample_rate": signal.sample_rate},
        )
    return sample.sample_id, outputs


class ControlCacheExtractor(CacheExtractor):
    def __init__(self, cfg, raw_dataset, store_path: str, encoder: Optional[BaseLatentEncoder] = None):
        super().__init__(cfg, raw_dataset, store_path)

        if self.cfg.control.match_latent_resolution:
            if encoder is None:
                raise ValueError("control.match_latent_resolution=True requires a latent encoder")
            hop_length = encoder.hop_length
        else:
            if self.cfg.control.hop_length is None:
                raise ValueError("control.hop_length must be set when match_latent_resolution=False")
            hop_length = self.cfg.control.hop_length

        self.hop_length = hop_length
        self.signals = [
            control_registry.get(entry.name)(
                sample_rate=self.cfg.data.sample_rate,
                hop_length=hop_length,
                **entry.params,
            )
            for entry in self.cfg.control.signals
        ]

    def _existing_signal_names(self, sample_id: str) -> set:
        if sample_id not in self.store:
            return set()
        return set(self.store[sample_id].array_keys())

    def _compute_signals(
        self, sample: RawSample, audio: np.ndarray, signals: List
    ) -> Dict[str, Tuple[np.ndarray, dict]]:
        existing = self._existing_signal_names(sample.sample_id)
        outputs = {}
        for signal in signals:
            # skip signals already cached for this sample (e.g. a signal added to config
            # after this sample was cached under an older config) so patching in a new
            # signal never recomputes ones that already exist
            if signal.name in existing and not self.cfg.cache.overwrite:
                continue
            try:
                result = signal.compute(audio, sample)
            except MissingAnnotationError:
                continue  # e.g. no MIDI available yet for this SynthTab-hex song
            outputs[signal.name] = (
                result.values,
                {
                    "frame_rate": result.frame_rate,
                    "hop_length": signal.hop_length,
                    "sample_rate": signal.sample_rate,
                },
            )
        return outputs

    def compute_sample(self, sample: RawSample, audio: np.ndarray) -> Dict[str, Tuple[np.ndarray, dict]]:
        return self._compute_signals(sample, audio, self.signals)

    def expected_keys(self) -> List[str]:
        return [signal.name for signal in self.signals]

    def run(self, limit: Optional[int] = None) -> None:
        cpu_signals = [s for s in self.signals if not s.is_gpu]

        if self.cfg.control.num_workers <= 1 or not cpu_signals:
            # Nothing to parallelize (or every configured signal is GPU-bound, which
            # always runs as a single sequential pass regardless -- see below). The
            # ordinary sequential path already can't hit any multi-process GPU
            # contention, since it's one process computing one signal at a time.
            super().run(limit)
            return

        samples = self._pending_samples(limit)
        if not samples:
            return

        gpu_signals = [s for s in self.signals if s.is_gpu]
        if gpu_signals:
            self._run_gpu_pass(samples, gpu_signals)

        all_specs = [
            (entry.name, OmegaConf.to_container(entry.params, resolve=True))
            for entry in self.cfg.control.signals
            if entry.name in {s.name for s in cpu_signals}
        ]
        tasks = []
        for s in samples:
            existing = self._existing_signal_names(s.sample_id)
            specs = [(name, params) for name, params in all_specs if name not in existing or self.cfg.cache.overwrite]
            if specs:
                tasks.append((s, self.cfg.data.sample_rate, self.hop_length, specs))
        if not tasks:
            return

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.cfg.control.num_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_compute_control_signals, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="control signals (cpu)"):
                sample_id, outputs = future.result()
                if outputs:
                    self._write_outputs(sample_id, outputs)

    def _run_gpu_pass(self, samples: List[RawSample], gpu_signals: List) -> None:
        """GPU-bound signals (currently just pitch/CREPE) run as one dedicated
        sequential pass in this process, never inside the CPU worker pool below.
        CREPE's own internal batching (batch_size frames per call) already saturates
        the GPU on its own; spinning it up inside N spawned CPU worker processes
        instead would mean N separate CUDA contexts fighting over the same VRAM --
        including against the concurrently-running latent extractor process -- for
        no speed benefit, which is exactly what made this CPU-bound in the first
        place: pitch previously defaulted to CPU specifically to avoid that."""
        for sample in tqdm(samples, desc="control signals (gpu)"):
            existing = self._existing_signal_names(sample.sample_id)
            pending = [s for s in gpu_signals if s.name not in existing or self.cfg.cache.overwrite]
            if not pending:
                continue
            audio = load_audio(sample.audio_path, sample.channel, self.cfg.data.sample_rate)
            outputs = self._compute_signals(sample, audio, pending)
            if outputs:
                self._write_outputs(sample.sample_id, outputs)
