# Populates the latent and control caches for a config's dataset, incrementally: anything
# already cached is skipped, so this is safe to re-run after adding a signal or a dataset.
#
# Run as: python -m scripts.data.build_cache --config config/perf.yaml [--only control]
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

from config.config import load_config
from dataset.control.extractor import ControlCacheExtractor
from dataset.dataset_register import build_raw_dataset
from dataset.latent.extractor import LatentCacheExtractor
from dataset.latent.sao3 import latent_registry


def _run_latent(config_path: str, overrides: List[str], limit: Optional[int]) -> None:
    cfg = load_config(config_path, overrides)
    raw_dataset = build_raw_dataset(cfg.data)
    encoder = latent_registry.get(cfg.data.latent)(cfg.latent)
    store_path = str(Path(cfg.cache.root) / cfg.cache.latent_store)
    print(f"[build_cache] extracting latents -> {store_path}")
    LatentCacheExtractor(cfg, raw_dataset, encoder, store_path).run(limit=limit)


def _run_control(config_path: str, overrides: List[str], limit: Optional[int]) -> None:
    # Force the control-side encoder onto CPU: it's only used to read hop_length (a
    # fixed property of the model architecture, independent of device placement), and
    # this avoids contending for GPU memory with the concurrently-running latent process.
    cfg = load_config(config_path, overrides + ["latent.device=cpu"])
    raw_dataset = build_raw_dataset(cfg.data)
    encoder = latent_registry.get(cfg.data.latent)(cfg.latent) if cfg.control.match_latent_resolution else None
    store_path = str(Path(cfg.cache.root) / cfg.cache.control_store)
    print(f"[build_cache] extracting control signals -> {store_path} (num_workers={cfg.control.num_workers})")
    ControlCacheExtractor(cfg, raw_dataset, store_path, encoder=encoder).run(limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract latents and control signals into Zarr caches.")
    parser.add_argument("--config", required=True, help="Path to an experiment config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N raw samples")
    parser.add_argument(
        "--only",
        choices=["latent", "control", "both"],
        default="both",
        help="Extract only one cache instead of both. 'control' is the useful one when "
             "adding a newly-registered signal to an existing cache: ControlCacheExtractor "
             "skips per-sample signals that are already present, so it patches rather than "
             "recomputes, and it avoids constructing the latent encoder (which needs HF_TOKEN) "
             "just to re-skip already-cached latents.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run latent extraction then control extraction one after another instead of concurrently",
    )
    parser.add_argument("overrides", nargs="*", help="Dot-list config overrides, e.g. data.type=GuitarSet")
    args = parser.parse_args()

    if args.only == "latent":
        _run_latent(args.config, args.overrides, args.limit)
        return
    if args.only == "control":
        _run_control(args.config, args.overrides, args.limit)
        return

    if args.sequential:
        _run_latent(args.config, args.overrides, args.limit)
        _run_control(args.config, args.overrides, args.limit)
        return

    ctx = mp.get_context("spawn")
    p_latent = ctx.Process(target=_run_latent, args=(args.config, args.overrides, args.limit))
    p_control = ctx.Process(target=_run_control, args=(args.config, args.overrides, args.limit))
    p_latent.start()
    p_control.start()
    p_latent.join()
    p_control.join()

    if p_latent.exitcode != 0 or p_control.exitcode != 0:
        raise RuntimeError(f"build_cache failed: latent exitcode={p_latent.exitcode}, control exitcode={p_control.exitcode}")


if __name__ == "__main__":
    main()
