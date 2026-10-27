from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(path: str, channel: Optional[int], target_sr: int) -> np.ndarray:
    """Load one channel (or the mono mix) of an audio file, resampled to target_sr."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)  # (N, C)
    audio = audio[:, channel] if channel is not None else audio.mean(axis=1)

    if sr != target_sr:
        audio_t = torchaudio.functional.resample(torch.from_numpy(audio), sr, target_sr)
        audio = audio_t.numpy()

    return audio.astype(np.float32)
