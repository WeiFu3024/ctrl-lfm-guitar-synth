# Trains model/synth/mix_reverb.py's MultiChannelReverb: six per-string streams -> the mono mic
# recording. This is the mixing stage for the per-string route -- synthesise each string,
# then learn the pickup->mic transfer and the summation in one linear operator, rather than
# predicting the mix latent directly (arm B) and inheriting the mic domain's worse codec
# floor (0.4875 vs 0.3376 FAD at matched clips; see docs/reports/archive/2026/project-history-through-2026-09-18.md).
#
# The measurement worth taking before anything else is --variant decoded on the TEST split:
# applying the fitted IR to oracle round-tripped hex bounds what any per-string-then-mix
# system can achieve, the way the codec floor bounds everything else.
#
# Run as: python -m scripts.synth.train_mix_reverb --variant decoded [--epochs 200]
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from config.config import load_config
from dataset.synth.hex_mix_dataset import HexMixDataset
from model.synth.mix_reverb import MultiChannelReverb, multiscale_spectral_loss

SCALES = [64, 128, 256, 512, 1024, 2048]


def _si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    r, e = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
    noise = e - a * r
    return float(10 * np.log10((np.sum((a * r) ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/synth.yaml")
    ap.add_argument("--variant", default="decoded", choices=["raw", "decoded", "predicted"])
    ap.add_argument("--pred-dir", default=None, help="required for --variant predicted")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    # 1e-3 oscillated instead of converging on decoded inputs (val 8.7 -> 15.4 -> 9.7 ->
    # 12.4 across the first 30 epochs): the log term in the multiscale spectral loss spikes
    # when a band is near-silent, and a 66k linear model has no normalisation to absorb it.
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # On raw/decoded inputs the hex and mic recordings are simultaneous, so the fit is
    # phase-aligned and time-domain MSE is a convex objective on exactly this linear model.
    # It is off by default because it does NOT transfer to --variant predicted, where the
    # generated audio is perceptually right but phase-unrelated to the recording.
    ap.add_argument("--time-weight", type=float, default=0.0)
    ap.add_argument("--residual", action="store_true",
                    help="output = dry_sum + reverb(dry), DDSP-Guitar's actual topology "
                         "(synthesis_model.py:143). Anchors the overall gain to the dry path "
                         "so the learned IR only has to model reflections.")
    ap.add_argument("--init-from", default=None, help="checkpoint to warm-start (e.g. fine-tuning decoded -> predicted)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out or f"runs/mix-reverb-{args.variant}"
                   + ("-residual" if args.residual else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = None
    if args.variant == "decoded":
        from dataset.latent.sao3 import latent_registry
        encoder = latent_registry.get(cfg.data.latent)(cfg.latent)

    kw = dict(variant=args.variant, seconds=args.seconds, encoder=encoder, pred_dir=args.pred_dir)
    train_ds = HexMixDataset(cfg, "train", **kw)
    val_ds = HexMixDataset(cfg, "val", **kw)
    print(f"variant={args.variant}  train {len(train_ds)}  val {len(val_ds)}  "
          f"{args.seconds:.0f}s @ {cfg.data.sample_rate} Hz", flush=True)

    model = MultiChannelReverb(cfg.data.sample_rate, residual=args.residual).to(args.device)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location=args.device)["model"])
        print(f"warm-started from {args.init_from}", flush=True)
    print(f"{sum(p.numel() for p in model.parameters())/1e3:.1f}k params "
          f"({model.n_channels} x {model.ir_samples} taps)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    def run(loader, train: bool):
        model.train(train)
        tot, n = 0.0, 0
        for b in loader:
            hx, mic = b["hex"].to(args.device), b["mic"].to(args.device)
            pred = model(hx)
            loss = multiscale_spectral_loss(pred, mic, SCALES)
            if args.time_weight:
                loss = loss + args.time_weight * torch.nn.functional.mse_loss(pred, mic)
            if train:
                opt.zero_grad(); loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            tot += float(loss) * hx.shape[0]; n += hx.shape[0]
        return tot / max(n, 1)

    # truncate rather than append: the log is opened in "a" mode each epoch, so a rerun
    # into the same output dir would interleave two runs' curves in one file
    best, log_path = float("inf"), out_dir / "log.jsonl"
    log_path.unlink(missing_ok=True)
    for ep in range(args.epochs):
        tr = run(tl, True)
        with torch.no_grad():
            va = run(vl, False)
        sched.step()
        rec = {"epoch": ep, "train_loss": tr, "val_loss": va, "lr": sched.get_last_lr()[0]}
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": va,
                        "residual": args.residual},
                       out_dir / "best.pt")

    # what the fitted mixer actually achieves on held-out audio, and what it sounds like
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=args.device)["model"])
    model.eval()
    test_ds = HexMixDataset(cfg, "test", **kw)
    sisdr, spec, spec_sum = [], [], []
    with torch.no_grad():
        for i in range(len(test_ds)):
            it = test_ds[i]
            hx = it["hex"][None].to(args.device)
            mic = it["mic"][None].to(args.device)
            out = model(hx)
            p = out[0].cpu().numpy()
            spec.append(float(multiscale_spectral_loss(out, mic, SCALES)))
            # the baseline the IR must beat: just add the six strings up
            spec_sum.append(float(multiscale_spectral_loss(hx.sum(1), mic, SCALES)))
            sisdr.append(_si_sdr(it["mic"].numpy(), p))
            if i < 3:
                sf.write(str(out_dir / f"{it['recording_id']}_pred.wav"), p, cfg.data.sample_rate)
                sf.write(str(out_dir / f"{it['recording_id']}_real.wav"), it["mic"].numpy(),
                         cfg.data.sample_rate)
    # spectral loss is the headline: it is what the model optimises. SI-SDR is reported for
    # completeness but a phase-invariant objective has no incentive to preserve phase, so a
    # good mixer can score badly on it (measured: fitted IR -20.57 dB vs raw sum -20.54).
    print(f"\nbest val {best:.4f}")
    print(f"test spectral  IR {np.mean(spec):.4f}  vs plain-sum baseline {np.mean(spec_sum):.4f}  "
          f"({100*(1-np.mean(spec)/np.mean(spec_sum)):+.1f}% )")
    print(f"test SI-SDR    {np.mean(sisdr):+.2f} dB over {len(sisdr)} recordings (see note above)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
