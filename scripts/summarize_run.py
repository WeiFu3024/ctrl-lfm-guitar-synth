# Prints the best-epoch validation metrics from a training run's log.jsonl -- the same
# per-channel interpretable numbers (pitch_mae_cents, mae_db, variance_ratio, ...) each
# training script already computes in its own validation loop and writes to
# <out_dir>/log.jsonl. Does not render audio or compute anything new; this is a thin
# reader over logs that already exist once training finishes.
#
# Run as: python -m scripts.summarize_run runs/perf/pe_note_shape_fixed13 --loss-key loss --split val
import argparse
import json
from pathlib import Path
from typing import List, Optional


def load_rows(log_path: Path, split: Optional[str]) -> List[dict]:
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if split is not None:
        rows = [r for r in rows if r.get("split") == split]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="training run directory, e.g. runs/perf/pe_note_shape_fixed13")
    ap.add_argument("--loss-key", required=True,
                     help="log.jsonl field to minimize over, e.g. 'loss' (perf predictor), "
                          "'loss_l1' (note-gain predictor), or 'val_loss' (joint synth / mix reverb)")
    ap.add_argument("--split", default=None,
                     help="filter to this 'split' field before taking the minimum "
                          "(perf predictor / note-gain predictor log both 'train' and 'val' rows "
                          "under one log.jsonl; pass --split val). Joint synth / mix reverb write "
                          "one row per epoch with no 'split' field -- omit this flag for those.")
    ap.add_argument("--fields", default=None,
                     help="comma-separated subset of fields to print (default: everything "
                          "in the best row)")
    args = ap.parse_args()

    log_path = Path(args.run_dir) / "log.jsonl"
    if not log_path.exists():
        raise SystemExit(f"no log.jsonl under {args.run_dir} -- has training started?")

    rows = load_rows(log_path, args.split)
    if not rows:
        raise SystemExit(f"no rows in {log_path}" + (f" with split={args.split!r}" if args.split else ""))

    best = min(rows, key=lambda r: r[args.loss_key])
    fields = args.fields.split(",") if args.fields else sorted(best.keys())
    print(f"{args.run_dir}  (best epoch {best.get('epoch')}, {len(rows)} logged rows)")
    for k in fields:
        if k in best:
            v = best[k]
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
