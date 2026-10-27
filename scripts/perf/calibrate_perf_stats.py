# Prints the per-channel normalisation constants dataset/control_flow_dataset.py needs, over
# a whole split's real curves. Hand-copy the printed values into that module's constants, per
# this repo's convention for calibrated numbers.
#
# Why this exists at all: FlowMatching's loss is an UNWEIGHTED mean over the channel axis. In
# raw units log-RMS has ~134x smaller variance than f0, which is exactly the imbalance that
# left loudness with essentially no gradient in the note-rate AR work. Z-scoring both
# channels on the same split the model trains on makes them contribute equally by
# construction rather than by luck.
#
# Run as: python -m scripts.perf.calibrate_perf_stats --split train
import argparse

import numpy as np

from config.config import load_config
from dataset.perf.perf_curve_dataset import ControlFlowDataset, hold_forward_pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/perf.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--signal", default="envelope",
                    help="which cache array to calibrate -- 'envelope' (default) or e.g. "
                         "'ddsp_loudness_fixed'/'ddsp_centroid_fixed' for a new "
                         "EXTRA_CHANNEL_SPECS entry in dataset/perf/perf_curve_dataset.py. "
                         "Only ANCHOR-eligible (loudness-like) channels need a (mask_lo, "
                         "mask_hi) grid search too -- that is NOT done by this script; see "
                         "DDSP_LOUDNESS_FIXED_MASK_LO/HI's own derivation for the method if a "
                         "future channel needs one.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # extra_channels only gates _build_recording_index's membership check (does this signal
    # exist in the cache) and _get_full's silence-padding value -- neither is exercised by
    # this script, which reads the raw arrays directly, but it must still be a KNOWN name
    # (EXTRA_CHANNEL_SPECS) if --signal matches one, and ControlFlowDataset defaults to
    # "envelope" for anything else so an ad-hoc --signal (e.g. a one-off ablation) doesn't
    # crash on construction -- it is used here for READING, not for the dataset's own logic.
    from dataset.perf.perf_curve_dataset import EXTRA_CHANNEL_SPECS
    ds_extra_channel = args.signal if args.signal in EXTRA_CHANNEL_SPECS else "envelope"
    ds = ControlFlowDataset(cfg, split=args.split, deterministic=True,
                            extra_channels=[ds_extra_channel])
    print(f"{len(ds._index)} recordings in {args.split}", flush=True)

    midi_all, env_all = [], []
    for i, rec in enumerate(ds._index):
        for s_idx, sid in sorted(ds._recording_strings[rec].items()):
            grp = ds.control_store[sid]
            if args.signal not in grp:
                continue
            pitch = grp["pitch"][:].reshape(-1)
            env = grp[args.signal][:].reshape(-1)
            voiced = grp["voiced"][:].reshape(-1)
            n = min(len(pitch), len(env), len(voiced))
            midi_all.append(hold_forward_pitch(pitch[:n], voiced[:n]))
            env_all.append(env[:n].astype(np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds._index)}", flush=True)

    midi = np.concatenate(midi_all)
    env = np.concatenate(env_all)
    print(f"\nframes: {len(midi)}")
    print(f"pitch (MIDI, hold-forward): mean {midi.mean():.4f}  std {midi.std():.4f}  "
          f"p1 {np.percentile(midi,1):.2f}  p99 {np.percentile(midi,99):.2f}")
    print(f"{args.signal:<24}: mean {env.mean():.4f}  std {env.std():.4f}  "
          f"p1 {np.percentile(env,1):.2f}  p99 {np.percentile(env,99):.2f}")
    print(f"\nraw-unit variance ratio pitch/{args.signal}: {midi.var()/env.var():.1f}x  "
          f"(the imbalance this normalisation removes)")
    print("\n--- paste into dataset/perf/perf_curve_dataset.py ---")
    print(f"PITCH_MIDI_MEAN, PITCH_MIDI_STD = {midi.mean():.4f}, {midi.std():.4f}")
    if args.signal == "envelope":
        print(f"ENVELOPE_MEAN, ENVELOPE_STD = {env.mean():.4f}, {env.std():.4f}")
    else:
        const_name = args.signal.upper()
        print(f"{const_name}_MEAN, {const_name}_STD = {env.mean():.4f}, {env.std():.4f}")


if __name__ == "__main__":
    main()
