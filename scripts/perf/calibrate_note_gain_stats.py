# Prints the log-duration / log-IOI normalisation constants dataset/perf/note_gain_dataset.py
# needs, over a whole split's real notes. Hand-copy the printed values into that module's
# constants, per this repo's convention for calibrated numbers (CLAUDE.md).
#
# The per-note gain TARGET itself needs no separate calibration here: it reuses whichever
# anchor channel's (mean, std) are already calibrated in dataset/perf/perf_curve_dataset.py's
# EXTRA_CHANNEL_SPECS, so pseudo_velocity's supplied gain and this predictor's target always
# sit in the same normalised units.
#
# Run as: python -m scripts.perf.calibrate_note_gain_stats --split train
import argparse

import numpy as np

from config.config import load_config
from dataset.control.signal import notes_for_sample
from dataset.perf.note_gain_dataset import MAX_GAP_S, NoteGainDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/perf.yaml")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = NoteGainDataset(cfg, split=args.split, deterministic=True)
    print(f"{len(ds._index)} recordings in {args.split}", flush=True)

    durations, ioi_prev, ioi_next, gaps = [], [], [], []
    n_notes = 0
    for i, rec in enumerate(ds._index):
        strings = ds._recording_strings[rec]
        for s_idx, sid in strings.items():
            notes = sorted(notes_for_sample(ds.sample_by_id[sid], {}), key=lambda n: n[0])
            if not notes:
                continue
            onsets_s = np.array([o for o, _d, _p in notes], dtype=np.float64)
            durs = np.array([d for _o, d, _p in notes], dtype=np.float64)
            n = len(notes)
            n_notes += n

            durations.append(durs)

            prev = np.empty(n, dtype=np.float64)
            prev[0] = MAX_GAP_S
            prev[1:] = onsets_s[1:] - onsets_s[:-1]
            ioi_prev.append(np.clip(prev, 0.0, None))

            nxt = np.empty(n, dtype=np.float64)
            nxt[:-1] = onsets_s[1:] - onsets_s[:-1]
            nxt[-1] = MAX_GAP_S
            ioi_next.append(np.clip(nxt, 0.0, None))
            if n >= 2:
                gaps.append(onsets_s[1:] - onsets_s[:-1])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds._index)}", flush=True)

    durations = np.concatenate(durations)
    ioi = np.concatenate(ioi_prev + ioi_next)  # prev and next share one clamp/scale
    gaps = np.concatenate(gaps)  # consecutive REAL note pairs only, no boundary sentinel
    log_dur = np.log(durations + 1e-6)
    log_ioi = np.log(ioi + 1e-6)

    print(f"\nnotes: {n_notes} ({len(ds._index)} recordings)")
    print(f"duration (s):   mean {durations.mean():.4f}  std {durations.std():.4f}  "
          f"p1 {np.percentile(durations,1):.3f}  p99 {np.percentile(durations,99):.3f}")
    print(f"log(duration):  mean {log_dur.mean():.4f}  std {log_dur.std():.4f}")
    print(f"IOI prev+next (s): mean {ioi.mean():.4f}  std {ioi.std():.4f}  "
          f"p1 {np.percentile(ioi,1):.3f}  p99 {np.percentile(ioi,99):.3f}")
    print(f"log(IOI):       mean {log_ioi.mean():.4f}  std {log_ioi.std():.4f}")
    print(f"consecutive-note gap (s), no boundary sentinel: mean {gaps.mean():.4f}  "
          f"median {np.median(gaps):.4f}  std {gaps.std():.4f}  "
          f"p25 {np.percentile(gaps,25):.4f}  p75 {np.percentile(gaps,75):.4f}")
    print("\n--- paste into dataset/perf/note_gain_dataset.py ---")
    print(f"LOG_DURATION_MEAN, LOG_DURATION_STD = {log_dur.mean():.4f}, {log_dur.std():.4f}")
    print(f"LOG_IOI_MEAN, LOG_IOI_STD = {log_ioi.mean():.4f}, {log_ioi.std():.4f}")
    print(f"IOI_DELTA_WEIGHT_TAU = {np.median(gaps):.4f}  "
          f"# median consecutive-note gap -- weight decays to 1/e at this spacing")


if __name__ == "__main__":
    main()
