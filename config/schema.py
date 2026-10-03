import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


@dataclass
class DataConfig:
    # registry key (or list of keys) into dataset_registry, e.g. "SynthTab-hex"
    # or ["GuitarSet", "SynthTab-hex"] to train on multiple sources combined
    type: Any = MISSING
    # registry key into latent_registry, e.g. "SAO3"
    latent: str = MISSING
    # common sample rate audio is loaded/resampled to before extraction
    sample_rate: int = 44100
    # override the default data/ root; None uses the repo's data/ directory
    root: Optional[str] = None


@dataclass
class CacheConfig:
    root: str = "cache"
    latent_store: str = "latent.zarr"
    control_store: str = "control.zarr"
    overwrite: bool = False


@dataclass
class ControlSignalEntry:
    name: str = MISSING
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlConfig:
    signals: List[ControlSignalEntry] = field(default_factory=list)
    # if True, hop_length is derived from the configured latent encoder so
    # control frames line up with latent frames; if False, hop_length below is used
    match_latent_resolution: bool = True
    hop_length: Optional[int] = None
    # >1 parallelizes control extraction across a process pool (control signals are
    # CPU-bound and independent per sample, unlike latent extraction which needs the GPU)
    num_workers: int = field(default_factory=lambda: min(8, os.cpu_count() or 4))


@dataclass
class LatentConfig:
    model_id: str = "stabilityai/stable-audio-3-medium"
    device: str = "cuda"
    half: bool = True


@dataclass
class SplitEntry:
    dataset: str = MISSING  # matches RawSample.meta["dataset"], e.g. "GuitarSet"
    splitter: str = MISSING  # split_registry key, e.g. "guitarset_ddsp"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    entries: List[SplitEntry] = field(default_factory=list)


@dataclass
class DatasetRuntimeConfig:
    # per __getitem__ call, a random window length in [min,max] latent frames is drawn
    # (natural length augmentation, exercises the model's RoPE length flexibility);
    # control arrays share this index space
    min_segment_length: int = 128
    max_segment_length: int = 256
    control_signals: List[str] = field(default_factory=lambda: ["pitch", "envelope"])
    # Probability that experiments/recon_synth/recon_mix_dataset.py serves an item with the note-rate
    # representation's RECONSTRUCTED pitch/envelope instead of the real cached ones, so
    # the synth learns to accept both. 0.0 (default) is plain GuitarSynthDataset
    # behaviour; only ReconMixDataset reads this.
    recon_mix_prob: float = 0.0
    random_crop: bool = True
    # Number of cached control frames added on both sides of each MIDI note for the
    # performance predictor's per-note shape loss. Zero preserves note-only spans.
    note_shape_padding_frames: int = 2
    pad_short: bool = True
    # Which dataset class scripts/synth/train_joint_synth.py instantiates -- "JointSynthDataset"
    # (default, cached control signals e.g. ddsp_controls/pitch+envelope) or
    # "JointVelocitySynthDataset" (midi_pitch + pseudo_velocity, derived on the fly).
    joint_dataset_cls: str = "JointSynthDataset"
    # Pseudo-velocity statistic for JointVelocitySynthDataset. Keep mean as the default
    # because the archived joint-velocity checkpoint used it; the peak experiment opts in
    # explicitly to match DDSP-Guitar's MIDI-velocity convention.
    velocity_gain_mode: str = "mean"
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class DiTConfig:
    d_model: int = 384
    depth: int = 8
    n_heads: int = 6
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0


@dataclass
class ControlEncoderConfig:
    embed_dim: int = 256
    depth: int = 3
    n_heads: int = 4
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0


@dataclass
class FlowMatchingConfig:
    t_distribution: str = "logit_normal"  # or "uniform"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    cfg_dropout_prob: float = 0.1
    sampling_steps: int = 32
    default_cfg_scale: float = 1.0


@dataclass
class JointConfig:
    # Only model/synth/joint_synth.py reads this; every other model ignores it, so the defaults
    # leave existing configs untouched.
    # "per_string": predict all 6 hex latents at once. "mix": predict the mono mic latent
    # (dataset_register.py's GuitarSet-mic) while still conditioning on all 6 strings.
    output_mode: str = "per_string"
    n_strings: int = 6
    # Names (a subset of dataset.control_signals) fed to JointSynthDecoder as an additive
    # learned embedding instead of concatenation -- see model/synth/joint_synth.py's
    # JointSynthConfig.event_channels docstring. Empty by default (plain concat), matching
    # every existing config's behavior exactly.
    event_channels: List[str] = field(default_factory=list)
    # Keep the conditioning encoder's cross-string path independent from this decoder/DiT
    # ablation. False omits decoder cross-string blocks and their parameters entirely.
    decoder_cross_string_attention: bool = True
    # Same ablation, applied to the conditioning encoder's cross-string blocks instead.
    # False omits enc_cross entirely -- combine with decoder_cross_string_attention=False
    # for a fully per-string model with no cross-string communication anywhere.
    encoder_cross_string_attention: bool = True


@dataclass
class ModelConfig:
    latent_dim: int = 256  # must match the cache's latent D (SAO3 -> 256)
    dit: DiTConfig = field(default_factory=DiTConfig)
    control_encoder: ControlEncoderConfig = field(default_factory=ControlEncoderConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    joint: JointConfig = field(default_factory=JointConfig)


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # optimizer_registry key
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class LRSchedulerConfig:
    name: str = "warmup_cosine"  # lr_scheduler_registry key
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1


@dataclass
class TrainerConfig:
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 100
    grad_clip_norm: float = 1.0
    num_dataloader_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class SaverConfig:
    output_dir: str = "runs/default"
    num_monitor_samples: int = 4
    sample_n_steps: int = 32
    sample_cfg_scale: float = 1.0
    checkpoint_keep_last_n: int = 3
    # extra cached control signals to fetch purely for the comparison plot, independent
    # of dataset.control_signals (model conditioning) — e.g. onset/voiced render as a
    # note roll alongside pitch even when not used to condition the model
    plot_extra_signals: List[str] = field(default_factory=lambda: ["onset", "voiced"])


@dataclass
class ExperimentConfig:
    name: str = MISSING
    data: DataConfig = field(default_factory=DataConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    dataset: DatasetRuntimeConfig = field(default_factory=DatasetRuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    saver: SaverConfig = field(default_factory=SaverConfig)
