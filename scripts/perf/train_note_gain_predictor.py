# Trains the small note-gain predictor (model/perf/note_gain_predictor.py): score-only
# per-note features -> per-note loudness, the same quantity pseudo_velocity's supplied gain
# is (dataset/perf/perf_curve_dataset.py's per_note_gain_values). Standalone: does not touch
# the frozen pitch/envelope predictor (e.g. pe_pitch_best) or its checkpoint. See
# dataset/perf/note_gain_dataset.py's module docstring for the full plan this is part of.
#
# Reports the loss in three units every eval: z-scored (the training objective's own scale,
# comparable across anchor channels), raw anchor-channel units (log-RMS for the default
# "envelope" anchor), and dB (raw * 20/ln(10)) -- the interpretable unit for judging whether
# a given MAE is audible, matching the codebase's existing pitch_mae_cents convention
# (scripts/perf/train_perf_predictor.py).
#
# variance_ratio is logged every eval as the mode-collapse tripwire: the archived
# experiments/perf_predictor_v1/README.md found plain per-note regressors on this exact
# corpus recover only 20-68% of real variance (a flat "always predict the mean" collapse).
# A low variance_ratio here means the MAE number alone is not trustworthy evidence of a good
# predictor.
#
# --delta-loss weighted adds a NOTE-TO-NOTE CHANGE term: for every note i, sum over every
# note j "near" it in time (any string -- the merged sequence's whole point is that a
# note's real neighbours can be on a different string) with weight A(i,j) =
# exp(-|onset_i-onset_j|/tau), L1 between (pred_i-pred_j) and (target_i-target_j) --
# MATCHING the target's own note-to-note change, not penalising roughness
# (model/perf/perf_losses.py's note_delta_loss, delta_loss's family, not jitter_loss's) --
# on top of the base per-note L1. Only pairs with A(i,j) above --delta-weight-threshold
# count at all (dataset/perf/note_gain_dataset.py's `_note_delta_neighbors`, a linear-time
# onset-window precompute -- both the neighbour SET and its weights are dataset-built,
# fixed data, not chosen in this training loop).
#
# Weighting, decided before training: the delta term's magnitude is, BY DEFAULT, EMA-
# balanced to a requested ratio against the base L1 (--delta-ratio, default 1.0, decay
# 0.99) -- the same convention this codebase already uses for a new per-note auxiliary term
# (scripts/perf/train_perf_predictor.py's --note-shape-ratio, whose ratio=1 was
# pe_pitch_best's own winning choice), rather than a hand-picked fixed coefficient.
# --delta-weight-mode fixed switches to a constant --delta-fixed-weight multiplier instead
# (same fixed/ema toggle as train_perf_predictor.py's --note-shape-weight-mode) -- useful
# to isolate whether the EMA schedule itself (large early, shrinking as L1 falls) matters,
# versus just adding the delta term at all. The kernel's timescale (--delta-tau) defaults
# to IOI_DELTA_WEIGHT_TAU, the calibrated MEDIAN consecutive-note gap on the train split
# (scripts/perf/calibrate_note_gain_stats.py).
#
# Run as: python -m scripts.perf.train_note_gain_predictor --name note_gain_v1 \
#   --out-root runs/note-gain
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config import load_config
from dataset.perf.note_gain_dataset import (
    INPUT_CHANNELS, IOI_DELTA_WEIGHT_TAU, NoteGainDataset, collate_note_gain,
)
from dataset.perf.perf_curve_dataset import EXTRA_CHANNEL_SPECS
from model.perf.note_gain_predictor import NoteGainPredictor
from model.perf.perf_losses import (
    EMAMagnitudeBalancer, FixedWeightBalancer, l1_loss, note_delta_loss, variance_ratio,
)
from saver.perf_saver import Saver
from trainer.optim import lr_scheduler_registry, optimizer_registry

CONFIG = "config/perf.yaml"
_DB_PER_NEPER = 20.0 / np.log(10.0)  # raw log-RMS units -> dB, same conversion as elsewhere


def run_split(model, loader, device, anchor_std: float, delta_mode: str = "none",
             delta_balancer: "EMAMagnitudeBalancer" = None, opt=None, grad_clip: float = 1.0):
    train = opt is not None
    model.train(train)
    acc, n = {}, 0
    for batch in loader:
        controls = {k: v.to(device) for k, v in batch["controls"].items()}
        target = batch["target"].to(device)
        note_valid = batch["note_valid"].to(device)
        string_id = batch["string_id"].to(device)
        nbr_idx = batch["nbr_idx"].to(device)
        nbr_weight = batch["nbr_weight"].to(device)
        nbr_valid = batch["nbr_valid"].to(device)

        with torch.set_grad_enabled(train):
            pred = model(controls, string_id, note_valid)
            l1 = l1_loss(pred, target, note_valid)

            loss = l1
            delta = torch.zeros((), device=device)
            lam = torch.zeros((), device=device)
            if delta_mode != "none":
                delta = note_delta_loss(pred, target, nbr_idx, nbr_weight, nbr_valid)
                lam = delta_balancer.weight(l1, delta, update=train)
                loss = l1 + lam * delta

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        with torch.no_grad():
            terms = {
                "loss": loss,
                "loss_l1": l1,
                "loss_delta": delta,
                "lambda_delta": lam,
                "variance_ratio": torch.tensor(variance_ratio(pred, target, note_valid)),
            }
            terms["mae_raw"] = terms["loss_l1"] * anchor_std
            terms["mae_db"] = terms["mae_raw"] * _DB_PER_NEPER
            terms["delta_mae_db"] = terms["loss_delta"] * anchor_std * _DB_PER_NEPER
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        n += 1
    return {k: v / max(1, n) for k, v in acc.items()}


def plot_note_predictions(saver, epoch: int, recording_id: str, pred_raw: np.ndarray,
                          target_raw: np.ndarray, note_valid: np.ndarray,
                          string_id: np.ndarray, n_strings: int):
    """One PNG per eval step: predicted vs target per-note gain (RAW anchor units) for one
    FIXED validation recording (the caller always passes the same one), one subplot per
    string. `pred_raw`/`target_raw`/`note_valid`/`string_id`: (M,) arrays -- one flat,
    time-sorted, multi-string sequence (dataset/perf/note_gain_dataset.py), already
    denormalised where relevant. Each subplot selects its string's own tokens out of the
    merged sequence; boolean selection preserves onset order within that string."""
    fig, axes = plt.subplots(n_strings, 1, figsize=(9, 1.6 * n_strings), sharex=False)
    for s in range(n_strings):
        ax = axes[s]
        sel = note_valid & (string_id == s)
        if sel.any():
            idx = np.arange(int(sel.sum()))
            ax.plot(idx, target_raw[sel], "o-", ms=3, lw=1, color="black", label="target")
            ax.plot(idx, pred_raw[sel], "o-", ms=3, lw=1, color="tab:orange", alpha=0.85,
                    label="pred")
        ax.set_ylabel(f"str {s + 1}", fontsize=8)
        ax.tick_params(labelsize=7)
        if s == 0:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("note index (onset order)")
    fig.suptitle(f"{recording_id} — epoch {epoch}", fontsize=10)
    fig.tight_layout()
    path = saver.out / "plots" / f"notes_ep{epoch:04d}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out-root", default="runs/note-gain")
    ap.add_argument("--anchor-channel", default="envelope", choices=tuple(EXTRA_CHANNEL_SPECS))
    ap.add_argument("--gain-mode", default="max", choices=["mean", "max"],
                    help="which per-note statistic to predict -- must match the deployment "
                         "contract's --gain-mode (scripts/perf/train_perf_predictor.py)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--delta-loss", default="none", choices=["none", "weighted"],
                    help="add a note-to-note gain-CHANGE L1 term, summed over every "
                         "nearby-in-time note pair above --delta-weight-threshold, on top "
                         "of the base per-note L1 -- see module docstring")
    ap.add_argument("--delta-ratio", type=float, default=1.0,
                    help="--delta-weight-mode ema target magnitude ratio of the delta term "
                         "against the base per-note L1 -- same convention as "
                         "train_perf_predictor.py's --note-shape-ratio")
    ap.add_argument("--delta-weight-mode", choices=["ema", "fixed"], default="ema",
                    help="auxiliary delta-term multiplier policy -- same convention as "
                         "train_perf_predictor.py's --note-shape-weight-mode")
    ap.add_argument("--delta-fixed-weight", type=float, default=1.0,
                    help="fixed multiplier used when --delta-weight-mode fixed")
    ap.add_argument("--delta-ema-decay", type=float, default=0.99)
    ap.add_argument("--delta-tau", type=float, default=IOI_DELTA_WEIGHT_TAU,
                    help="neighbour kernel timescale (s); default = calibrated MEDIAN "
                         "consecutive-note gap on the train split")
    ap.add_argument("--delta-weight-threshold", type=float, default=0.05,
                    help="A(i,j)=exp(-gap/tau) cutoff below which a note pair is not a "
                         "neighbour at all; sets the precomputed window's radius "
                         "(dataset/perf/note_gain_dataset.py's _note_delta_neighbors) -- "
                         "only used by --delta-scope cross_string")
    ap.add_argument("--delta-scope", choices=["cross_string", "same_string_next"],
                    default="cross_string",
                    help="'cross_string' (default): every nearby-in-time note pair, any "
                         "string. 'same_string_next': the ORIGINAL note-gain predictor "
                         "design -- each note paired ONLY with the next note on its own "
                         "string, no --delta-weight-threshold cutoff -- kept as an explicit "
                         "ablation scope (dataset/perf/note_gain_dataset.py's "
                         "_same_string_next_neighbors)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)
    out = Path(args.out_root) / args.name
    saver = Saver(out)
    saver.save_config(vars(args))

    dataset_args = {
        "anchor_channel": args.anchor_channel, "gain_mode": args.gain_mode,
        "delta_tau": args.delta_tau, "delta_weight_threshold": args.delta_weight_threshold,
        "delta_scope": args.delta_scope,
    }
    delta_balancer = None
    if args.delta_loss != "none":
        delta_balancer = (
            EMAMagnitudeBalancer(args.delta_ratio, args.delta_ema_decay)
            if args.delta_weight_mode == "ema"
            else FixedWeightBalancer(args.delta_fixed_weight)
        )
    tr = NoteGainDataset(cfg, split="train", **dataset_args)
    va = NoteGainDataset(cfg, split="val", deterministic=True, **dataset_args)
    print(f"train {len(tr)} / val {len(va)} recordings", flush=True)

    worker_args = {"num_workers": args.num_workers, "persistent_workers": args.num_workers > 0}
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_note_gain,
                    drop_last=True, generator=train_generator, **worker_args)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_note_gain,
                    **worker_args)

    model = NoteGainPredictor(
        INPUT_CHANNELS, {n: 1 for n in INPUT_CHANNELS},
        d_model=args.d_model, depth=args.depth, n_heads=args.n_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}", flush=True)

    opt = optimizer_registry.get(cfg.trainer.optimizer.name)(model.parameters(), cfg.trainer.optimizer)
    for g in opt.param_groups:
        g["lr"] = args.lr
    sched = lr_scheduler_registry.get(cfg.trainer.lr_scheduler.name)(
        opt, cfg.trainer.lr_scheduler, total_steps=args.epochs)

    anchor_std = EXTRA_CHANNEL_SPECS[args.anchor_channel]["std"]

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_terms = run_split(model, tl, device, anchor_std, delta_mode=args.delta_loss,
                             delta_balancer=delta_balancer, opt=opt, grad_clip=args.grad_clip)
        sched.step()
        secs = round(time.time() - t0, 1)
        saver.log({"split": "train", "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                   "secs": secs, **tr_terms})

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            va_terms = run_split(model, vl, device, anchor_std, delta_mode=args.delta_loss,
                                 delta_balancer=delta_balancer)
            saver.log({"split": "val", "epoch": epoch, **va_terms})
            saver.save_checkpoint(
                model, opt, epoch, va_terms["loss_l1"],
                tag=f"ep{epoch:04d}" if epoch % 25 == 0 else None,
                metadata={
                    "input_names": INPUT_CHANNELS,
                    "anchor_channel": args.anchor_channel,
                    "gain_mode": args.gain_mode,
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "n_heads": args.n_heads,
                    "delta_loss": args.delta_loss,
                    "delta_ratio": args.delta_ratio,
                    "delta_weight_mode": args.delta_weight_mode,
                    "delta_fixed_weight": args.delta_fixed_weight,
                    "delta_tau": args.delta_tau,
                    "delta_weight_threshold": args.delta_weight_threshold,
                    "delta_scope": args.delta_scope,
                },
            )
            saver.plot_loss_terms()
            model.eval()
            fixed_item = va[0]
            fixed_batch = collate_note_gain([fixed_item])
            with torch.no_grad():
                fixed_pred = model(
                    {k: v.to(device) for k, v in fixed_batch["controls"].items()},
                    fixed_batch["string_id"].to(device),
                    fixed_batch["note_valid"].to(device),
                )
            anchor = EXTRA_CHANNEL_SPECS[args.anchor_channel]
            plot_note_predictions(
                saver, epoch, fixed_item["recording_id"],
                (fixed_pred[0, ..., 0] * anchor["std"] + anchor["mean"]).cpu().numpy(),
                (fixed_batch["target"][0, ..., 0] * anchor["std"] + anchor["mean"]).numpy(),
                fixed_batch["note_valid"][0].numpy(), fixed_batch["string_id"][0].numpy(),
                fixed_batch["n_strings"],
            )
            print(f"ep{epoch} train {tr_terms['loss_l1']:.4f} val {va_terms['loss_l1']:.4f} "
                  f"mae_db {va_terms['mae_db']:.3f} var {va_terms['variance_ratio']:.3f} "
                  f"delta_db {va_terms['delta_mae_db']:.3f} ({secs}s/ep)", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
