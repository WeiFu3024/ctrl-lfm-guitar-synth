#!/usr/bin/env bash
# Trains every model this repo ships (three pipeline stages + the optional mix-reverb
# post-process + the optional velocity-baseline ablation) with the exact hyperparameters
# that produced the released checkpoints, then prints each run's best-epoch validation
# metrics next to the released checkpoints' own numbers for you to compare.
#
# Assumes: GuitarSet is reachable (GUITARSET_ROOT or data/GuitarSet/), HF_TOKEN is set and
# has accepted stabilityai/stable-audio-open-1.0's license, and the cache has been built
# (this script builds it if missing -- see --skip-cache to skip).
#
# Run as: CUDA_VISIBLE_DEVICES=0 ./scripts/run_all.sh [--skip-cache] [--skip-baseline] [--skip-reverb]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SKIP_CACHE=0
SKIP_BASELINE=0
SKIP_REVERB=0
for arg in "$@"; do
  case "$arg" in
    --skip-cache) SKIP_CACHE=1 ;;
    --skip-baseline) SKIP_BASELINE=1 ;;
    --skip-reverb) SKIP_REVERB=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

: "${HF_TOKEN:?HF_TOKEN must be set -- export HF_TOKEN=\"\$(cat ~/.cache/huggingface/token)\"}"

if [ "$SKIP_CACHE" -eq 0 ]; then
  echo "== building cache (config/synth.yaml) =="
  python -m scripts.data.build_cache --config config/synth.yaml
fi

echo "== 1/4: joint six-string synth =="
python -m scripts.synth.train_joint_synth --config config/synth.yaml

if [ "$SKIP_REVERB" -eq 0 ]; then
  echo "== optional: learned 6->1 mix reverb =="
  python -m scripts.synth.train_mix_reverb --variant decoded
fi

echo "== 2/4: envelope regressor (perf predictor) =="
python -m scripts.perf.train_perf_predictor --name pe_note_shape_fixed13 \
  --config config/perf.yaml --out-root runs/perf \
  --targets pitch,envelope --note-shape-ratio 1 --note-shape-fixed-weight 1.3 \
  --epochs 150 --batch-size 4 --d-model 256 --depth 6 --lr 3e-4

echo "== 3/4: note-gain predictor =="
python -m scripts.perf.train_note_gain_predictor --name note_gain \
  --out-root runs/note-gain --epochs 150 --batch-size 8 \
  --delta-loss weighted --delta-scope same_string_next --delta-ratio 1.0

if [ "$SKIP_BASELINE" -eq 0 ]; then
  echo "== optional baseline: single-stage synth, flat MIDI pitch + peak velocity only =="
  python -m scripts.synth.train_joint_synth --config config/synth-velocity-baseline.yaml
fi

echo
echo "================= best-epoch validation metrics ================="
echo "-- reference numbers are from the released checkpoints' own training logs --"
echo

echo "[1] joint synth (config/synth.yaml)"
python -m scripts.summarize_run runs/synth --loss-key val_loss --fields epoch,train_loss,val_loss
echo "    released checkpoint:  epoch 486  train_loss 0.7610  val_loss 0.6951"
echo

if [ "$SKIP_REVERB" -eq 0 ]; then
  echo "[*] mix reverb (optional)"
  python -m scripts.summarize_run runs/mix-reverb-decoded --loss-key val_loss --fields epoch,train_loss,val_loss
  echo "    released checkpoint:  epoch 382  train_loss 6.7314  val_loss 7.0745"
  echo
fi

echo "[2] envelope regressor / perf predictor (pe_note_shape_fixed13)"
python -m scripts.summarize_run runs/perf/pe_note_shape_fixed13 --loss-key loss --split val \
  --fields epoch,loss,loss_l1,pitch_mae_cents,roughness_ratio,variance_ratio
echo "    released checkpoint:  epoch 145  loss 0.5828  loss_l1 0.1576  pitch_mae_cents 34.9407  roughness_ratio 0.7069  variance_ratio 1.0430"
echo

echo "[3] note-gain predictor (note_gain)"
python -m scripts.summarize_run runs/note-gain/note_gain --loss-key loss_l1 --split val \
  --fields epoch,loss_l1,mae_db,variance_ratio,delta_mae_db
echo "    released checkpoint:  epoch 90  loss_l1 0.4019  mae_db 4.7311  variance_ratio 0.3800  delta_mae_db 3.9763"
echo

if [ "$SKIP_BASELINE" -eq 0 ]; then
  echo "[*] velocity-baseline synth (optional)"
  python -m scripts.summarize_run runs/synth-velocity-baseline --loss-key val_loss --fields epoch,train_loss,val_loss
  echo "    released checkpoint:  epoch 404  train_loss 0.8045  val_loss 0.7028"
  echo
fi

echo "These are training-time validation losses only (GuitarSet val split, teacher-forced /"
echo "oracle-conditioned where applicable) -- not the paper's audio-level or full-pipeline"
echo "numbers, which need real rendering + the paper's own evaluation harness (not included"
echo "in this release). Close-but-not-identical numbers are expected: GPU nondeterminism,"
echo "torch/driver version drift, and dataloader worker count all perturb results by a few"
echo "percent even on literally the same hardware re-run."
