# Trains model/synth/joint_synth.py's JointSynthDecoder: all six strings jointly, either
# predicting the six hex latents (output_mode=per_string) or the mono mic mix
# (output_mode=mix).
#
# Self-contained rather than reusing trainer/trainer.py, because Trainer/Saver cannot
# represent this data: Trainer._sample_monitor_batch does `_, T, D = latent.shape` (which
# would read T as S*T), indexes latent_store by a per-string sample_id (joint items are
# keyed by recording), and Saver.on_epoch_end decodes each item as ONE audio stream --
# six concatenated strings. Complicating the class every other run depends on, to serve
# one experiment, is the worse trade; scripts/train_perf_decoder_joint.py made the same
# call for the same reason.
#
# Run as: python -m scripts.synth.train_joint_synth --config config/synth.yaml
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import load_config
from dataset.synth.joint_synth_dataset import JointSynthDataset, collate_joint_synth
from dataset.synth.joint_velocity_synth_dataset import JointVelocitySynthDataset

JOINT_DATASET_CLASSES = {"JointSynthDataset": JointSynthDataset,
                         "JointVelocitySynthDataset": JointVelocitySynthDataset}
from dataset.latent.sao3 import latent_registry
from model.common.flow_matching import FlowMatching
from model.synth.joint_synth import JointSynthDecoder
from saver.plotting import plot_training_curve
from trainer.optim import lr_scheduler_registry, optimizer_registry


def _target_and_mask(batch, output_mode: str, device):
    if output_mode == "mix":
        return batch["mix_latent"].to(device), batch["mix_valid_mask"].to(device)
    return batch["latent"].to(device), batch["valid_mask"].to(device)


@torch.no_grad()
def save_snapshot(model, flow, ds, recording_id, cfg, encoder, device, out_dir: Path, epoch: int):
    """Decodes each string separately (or the single mix stream) -- the reason this script
    does not reuse Saver, which would render six concatenated strings as one clip."""
    model.eval()
    idx = ds._index.index(recording_id)
    item = ds[idx]
    batch = collate_joint_synth([item])
    controls = {k: v.to(device) for k, v in batch["controls"].items()}
    target, _ = _target_and_mask(batch, ds.output_mode, device)
    pred = flow.sample(controls, tuple(target.shape), n_steps=cfg.saver.sample_n_steps,
                       cfg_scale=cfg.saver.sample_cfg_scale, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    S, T = item["n_strings"], item["n_frames"]
    if ds.output_mode == "mix":
        streams = {"mix_pred": pred[0].cpu().numpy(), "mix_target": target[0].cpu().numpy()}
    else:
        p = pred[0].reshape(S, T, -1).cpu().numpy()
        t = target[0].reshape(S, T, -1).cpu().numpy()
        streams = {}
        for s in range(S):
            if not bool(item["string_valid"][s]):
                continue  # unplayed string: nothing to listen to
            streams[f"string{s+1}_pred"] = p[s]
            streams[f"string{s+1}_target"] = t[s]
    for name, lat in streams.items():
        audio = encoder.decode(lat)
        sf.write(str(out_dir / f"{recording_id}_{name}.wav"), audio, encoder.sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/synth.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--snapshot-every", type=int, default=25, help="epochs between audio snapshots (decoding is slow)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    epochs = args.epochs or cfg.trainer.epochs
    device = args.device or (cfg.trainer.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.trainer.seed)

    DatasetCls = JOINT_DATASET_CLASSES[cfg.dataset.joint_dataset_cls]
    train_ds = DatasetCls(cfg, split="train")
    val_ds = DatasetCls(cfg, split="val", deterministic=True)
    print(f"mode={train_ds.output_mode}  train {len(train_ds)} recordings, val {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("empty split -- is ddsp_controls cached, and (for mix) the guitarset-mic latents?")

    control_names = list(cfg.dataset.control_signals)
    control_channels = {n: train_ds[0]["controls"][n].shape[-1] for n in control_names}
    # joint_cfg left None so JointSynthDecoder reads cfg.model.joint directly (the loaded
    # OmegaConf config, every field including new ones) -- constructing a JointSynthConfig
    # by hand here once already silently dropped event_channels for a whole training run,
    # because this call site was not updated when that field was added to the dataclass.
    model = JointSynthDecoder(cfg.model, control_names, control_channels).to(device)
    flow = FlowMatching(model, cfg.model.flow_matching)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  controls {control_channels}")

    train_loader = DataLoader(train_ds, batch_size=cfg.trainer.batch_size, shuffle=True,
                              collate_fn=collate_joint_synth, num_workers=cfg.trainer.num_dataloader_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.trainer.val_batch_size, shuffle=False,
                            collate_fn=collate_joint_synth, num_workers=cfg.trainer.num_dataloader_workers)

    total_steps = epochs * max(1, len(train_loader))
    opt = optimizer_registry.get(cfg.trainer.optimizer.name)(model.parameters(), cfg.trainer.optimizer)
    sched = lr_scheduler_registry.get(cfg.trainer.lr_scheduler.name)(opt, cfg.trainer.lr_scheduler, total_steps)

    out_dir = Path(cfg.saver.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    encoder = latent_registry.get(cfg.data.latent)(cfg.latent)
    snapshot_rec = sorted(val_ds._index)[0]

    best = float("inf")
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch} [train]"):
            controls = {k: v.to(device) for k, v in batch["controls"].items()}
            target, mask = _target_and_mask(batch, train_ds.output_mode, device)
            loss = flow.compute_loss(target, controls, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip_norm)
            opt.step()
            sched.step()
            losses.append(loss.item())

        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch in val_loader:
                controls = {k: v.to(device) for k, v in batch["controls"].items()}
                target, mask = _target_and_mask(batch, val_ds.output_mode, device)
                vlosses.append(flow.compute_loss(target, controls, mask, cfg_dropout_prob=0.0).item())

        train_loss, val_loss = float(np.mean(losses)), float(np.mean(vlosses))
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": sched.get_last_lr()[0], **{f"gate_{k}": v for k, v in model.cross_string_gates().items()}}
        print(json.dumps(rec))
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        plot_training_curve(str(log_path), str(out_dir / "plots" / "training_curve.png"))

        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")
        if args.snapshot_every and epoch % args.snapshot_every == 0:
            save_snapshot(model, flow, val_ds, snapshot_rec, cfg, encoder, device,
                          out_dir / "audio" / f"epoch_{epoch:04d}", epoch)

    print(f"done. best val_loss={best:.6f}, checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
