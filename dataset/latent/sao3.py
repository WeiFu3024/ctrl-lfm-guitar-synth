from abc import ABC, abstractmethod

import numpy as np
import torch
import torchaudio
from stable_audio_tools import get_pretrained_model

from access_token import get_hf_token
from dataset.registry import Registry


class BaseLatentEncoder(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self._loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

    @abstractmethod
    def decode(self, latent: np.ndarray) -> np.ndarray: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def hop_length(self) -> int: ...

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


latent_registry = Registry("latent_encoder")


@latent_registry.register("SAO3")
class SAO3LatentEncoder(BaseLatentEncoder):
    """Wraps a stable-audio-tools autoencoder (model.pretransform) for latent extraction.
    Generic over cfg.model_id — not specific to stable-audio-3 despite the class name."""
    def load(self) -> None:
        if self._loaded:
            return
        # `get_pretrained_model` reads HF_TOKEN through huggingface_hub itself. Calling
        # login() here performs a /whoami request for every renderer process and eventually
        # hits Hugging Face's strict rate limit; no persisted login is needed for downloads
        # when the repository's required HF_TOKEN environment variable is present.
        self.model, self.model_config = get_pretrained_model(self.cfg.model_id)
        self.device = self.cfg.device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        if self.cfg.half and self.device == "cuda":
            self.model = self.model.to(torch.float16)
        self.model.eval()
        self._loaded = True

    @property
    def sample_rate(self) -> int:
        self.load()
        return self.model_config["sample_rate"]

    @property
    def hop_length(self) -> int:
        self.load()
        return self.model.pretransform.downsampling_ratio

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self.load()
        audio_t = torch.from_numpy(audio).float()
        if sample_rate != self.sample_rate:
            audio_t = torchaudio.functional.resample(audio_t, sample_rate, self.sample_rate)

        io_channels = self.model.pretransform.io_channels
        audio_t = audio_t.unsqueeze(0).repeat(io_channels, 1).unsqueeze(0)  # (1, io_channels, N)
        audio_t = audio_t.to(self.device)
        if self.cfg.half and self.device == "cuda":
            audio_t = audio_t.half()

        with torch.no_grad():
            latent = self._encode_chunked(audio_t)  # (1, D, T_lat)

        return latent.squeeze(0).transpose(0, 1).float().cpu().numpy()  # (T_lat, D)

    # SynthTab songs are 133 s median and
    # up to 254 s, which OOMs a 24 GB card (a single 133 s encode asked for 3.93 GiB on top
    # of 20.5 GiB already resident). Chunking keeps peak memory bounded by CHUNK_S instead
    # of by song length.
    CHUNK_S = 20.0   # ~430 latent frames per chunk at hop 2048
    OVERLAP_S = 2.0  # discarded from each interior edge, so the convolutional encoder has
                     # real context either side of every frame it contributes

    def _encode_chunked(self, audio_t: torch.Tensor) -> torch.Tensor:
        n = audio_t.shape[-1]
        hop = self.hop_length
        # MUST be a whole number of latent frames. The encoder floors a partial trailing
        # frame away, so a chunk of int(20.0 * 44100) = 882000 samples emits 430 frames and
        # SILENTLY DISCARDS the 1360-sample remainder -- 30.8 ms of audio -- at every
        # interior boundary. Measured on 00_SS2-88-F_comp/string_3: encode->decode tracked
        # the input at lag 0 for the first 16 s, then -1360 samples, then -2720. Everything
        # after a boundary was shifted early against the control-signal timeline by that
        # amount, growing one boundary at a time.
        chunk = (int(self.CHUNK_S * self.sample_rate) // hop) * hop
        if n <= chunk:
            return self.model.pretransform.encode(audio_t)

        ov = (int(self.OVERLAP_S * self.sample_rate) // hop) * hop  # whole latent frames
        step = chunk - 2 * ov
        pieces, pos = [], 0
        while pos < n:
            end = min(pos + chunk, n)
            seg = audio_t[..., pos:end]
            if seg.shape[-1] < hop:
                break
            z = self.model.pretransform.encode(seg)
            # trim the context margins, keeping the interior that had full context
            lo = 0 if pos == 0 else ov // hop
            hi = z.shape[-1] if end == n else z.shape[-1] - ov // hop
            pieces.append(z[..., lo:hi])
            if end == n:
                break
            pos += step
        return torch.cat(pieces, dim=-1)

    # A single unwindowed decode() call OOMs a 24 GB card well before any length this
    # codebase's own GuitarSet-trained inference paths previously needed: measured
    # directly (inference), a 213 s/4595-frame song's single-shot
    # decode() succeeded, but a 301 s/6494-frame song's single-shot decode() asked for
    # 3.17 GiB on top of an already-resident 20.55 GiB and OOM'd. DECODE_CHUNK_FRAMES is
    # deliberately NOT _encode_chunked's own CHUNK_S (20s ~= 430 frames): that would
    # route every already-validated GuitarSet-scale decode (<=~1000 frames) and the
    # already-measured 4595-frame render through a new, previously-untested code path for
    # no benefit. 5000 frames sits strictly above every length this repo has already
    # exercised a single-shot decode() at and strictly below the measured OOM point, so
    # chunking only ever engages for genuinely-new, longer-than-ever-tested input.
    DECODE_CHUNK_FRAMES = 5000

    def _decode_chunked(self, z: torch.Tensor) -> torch.Tensor:
        t_lat = z.shape[-1]
        if t_lat <= self.DECODE_CHUNK_FRAMES:
            return self.model.pretransform.decode(z)

        ov = int(self.OVERLAP_S * self.frame_rate)  # latent frames of context trimmed per interior edge
        hop = self.hop_length
        step = self.DECODE_CHUNK_FRAMES - 2 * ov
        pieces, pos = [], 0
        while pos < t_lat:
            end = min(pos + self.DECODE_CHUNK_FRAMES, t_lat)
            audio = self.model.pretransform.decode(z[..., pos:end])
            # trim the context margins in the AUDIO domain (ov latent frames -> ov*hop
            # samples), keeping the interior that had full decoder context either side --
            # this decodes one fixed latent tensor's overlapping pieces, not independent
            # sampled realisations, so (unlike render_synth_windowed's own latent-domain
            # blend) there is no phase-coherence risk in stitching after decode.
            lo = 0 if pos == 0 else ov * hop
            hi = audio.shape[-1] if end == t_lat else audio.shape[-1] - ov * hop
            pieces.append(audio[..., lo:hi])
            if end == t_lat:
                break
            pos += step
        return torch.cat(pieces, dim=-1)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self.load()
        z = torch.from_numpy(latent).float().transpose(0, 1).unsqueeze(0)  # (1, D, T_lat)
        z = z.to(self.device)
        if self.cfg.half and self.device == "cuda":
            z = z.half()

        with torch.no_grad():
            audio = self._decode_chunked(z)  # (1, io_channels, N)

        return audio.squeeze(0).float().cpu().numpy().mean(axis=0)  # (N,) mono


# Same wrapper class, registered under a second key so a separate cache/config track can
# use stable-audio-open-1.0 (hop=1024, ~43Hz, latent_dim=64 -- 4x finer time resolution
# than stable-audio-3-medium's ~10.77Hz, at nearly identical total latent bandwidth)
# without disturbing the existing SAO3-based cache/experiments at all.
latent_registry.register("stable-audio-open-1.0")(SAO3LatentEncoder)
