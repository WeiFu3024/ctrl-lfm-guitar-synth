# Performance predictor loss experiments.
#
#   L_frame = current frame-wise L1 + delta/jitter terms
#   L_total = L_frame + λ_note(t) * L_note + λ_fm * L_FM
#   λ_note(t) = requested_ratio * EMA(L_frame) / EMA(L_note)
#
# L_note computes MAE inside each MIDI note's annotated span plus the configured boundary
# padding, averages frames within each note, then averages notes. The architecture and all
# frame-wise losses stay unchanged, so a PE run with --note-shape-ratio 1 isolates whether
# equal note weighting improves within-note curve shape.
#
# Run as: python -m scripts.perf.train_perf_predictor --name pe_note_shape \
#   --out-root runs/perf --targets pitch,envelope --note-shape-ratio 1
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.config import load_config
from dataset.perf.perf_curve_dataset import (ENVELOPE_MEAN, ENVELOPE_STD, N_PLAYERS,
                                          PITCH_MIDI_MEAN, PITCH_MIDI_STD, EXTRA_CHANNEL_SPECS)
from dataset.perf.detail_features import DETAIL_FEATURE_DIMS, detail_feature_dim
from model.common.flow_matching import FlowMatching  # noqa: F401  (kept: documents the shared contract)
from model.synth.joint_synth import JointSynthDecoder
from dataset.control.signal import midi_to_hz
from dataset.perf.perf_dataset import (EnvelopeRegDataset, collate_env_reg,
                                        input_channels_for)
from model.perf.perf_losses import (
    EMAMagnitudeBalancer, FixedWeightBalancer, FMVelocityCritic,
    build_fir_lowpass_kernel, delta_loss,
    jitter_loss, l1_loss, loudness_weight, masked_pitch_trend_loss,
    per_note_envelope_spectrum_loss, per_note_l1_loss, per_note_pitch_trend_loss,
    per_note_residual_spectrum_loss, roughness_ratio, variance_ratio,
)
from model.perf.perf_predictor import EnvelopeRegressor
from saver.perf_saver import Saver
from trainer.optim import lr_scheduler_registry, optimizer_registry

CONFIG = "config/perf.yaml"
SYNTH_CONFIG = "config/synth.yaml"
SYNTH_CKPT = "checkpoints/synth/joint_synth_peov_embed.pt"


def denorm_env(x: torch.Tensor) -> torch.Tensor:
    """normalised -> log-RMS, the units the frozen synth is conditioned on."""
    return x * ENVELOPE_STD + ENVELOPE_MEAN


def denorm_pitch(x: torch.Tensor) -> torch.Tensor:
    """normalised MIDI -> Hz, the units the frozen synth is conditioned on."""
    return midi_to_hz(x * PITCH_MIDI_STD + PITCH_MIDI_MEAN)




def build_critic(args, device):
    if args.lambda_fm <= 0:
        return None
    if args.extra_channels != ["envelope"]:
        raise NotImplementedError(
            f"--lambda-fm > 0 with --extra-channels {args.extra_channels!r}: the FM critic "
            "is wired to the pitch+envelope synth (SYNTH_CONFIG/SYNTH_CKPT above) and has no "
            "equivalent checkpoint for this channel set yet. pe_noalign (this script's own "
            "precedent) trains with lambda_fm=0, so this path is untested for anything but "
            "plain envelope -- add a matching synth checkpoint before enabling it.")
    scfg = load_config(SYNTH_CONFIG)
    synth = JointSynthDecoder(scfg.model, ["pitch", "envelope"],
                              {"pitch": 1, "envelope": 1}).to(device)
    state = torch.load(SYNTH_CKPT, map_location=device, weights_only=False)
    synth.load_state_dict(state["model"])
    return FMVelocityCritic(synth, ["pitch", "envelope"], device,
                            target=args.fm_target, t_dist=args.t_dist)

def _ablate_detail(
    detail: torch.Tensor,
    note_mask: torch.Tensor,
    feature_set: str,
    mode: str,
) -> torch.Tensor:
    if mode == "oracle" or detail.shape[-1] == 0:
        return detail
    if mode == "zero":
        return torch.zeros_like(detail)
    if mode not in {"shuffle", "phase_shuffle"}:
        raise ValueError(f"unknown detail mode {mode!r}")

    out = detail.clone() if mode == "phase_shuffle" else torch.zeros_like(detail)
    for batch_index in range(detail.shape[0]):
        masks = note_mask[batch_index]
        present = masks.any(dim=-1)
        masks = masks[present]
        if len(masks) < 2:
            continue
        starts = masks.to(torch.int64).argmax(dim=-1)
        vectors = detail[batch_index, starts]
        shifted = torch.roll(vectors, shifts=1, dims=0)
        for note_index, frame_mask in enumerate(masks):
            if mode == "shuffle":
                out[batch_index, frame_mask] = shifted[note_index]
            elif feature_set == "phase":
                out[batch_index, frame_mask, 13:19] = shifted[note_index, 13:19]
    return out



def run_split(
    model, loader, args, critic, device, opt=None, scaler=None, shape_balancer=None,
    detail_mode: str = "oracle", pitch_trend_kernel=None,
):
    """One pass. Returns every term separately; ``detail_mode`` supports utilization controls."""
    train = opt is not None
    model.train(train)
    # (lo, hi, invert) for loudness_weight() -- envelope and ddsp_loudness_fixed run in
    # OPPOSITE directions (see EXTRA_CHANNEL_SPECS's own docstring), so this must be looked
    # up per anchor channel rather than relying on loudness_weight's envelope-shaped defaults.
    # extra_channels[0] (the anchor) is ALWAYS the one whose mask backs every masked term --
    # pitch's, and any further extra channel's -- see EXTRA_CHANNEL_SPECS's own comment on why
    # a non-loudness channel like centroid cannot supply a coherent audibility gate itself.
    mspec = EXTRA_CHANNEL_SPECS[args.extra_channels[0]]
    acc, n = {}, 0
    if train:
        opt.zero_grad(set_to_none=True)
    n_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch["controls"].items()}
        target = batch["target"].to(device)
        mask = batch["valid_mask"].to(device)
        note_mask = batch["note_mask"].to(device)
        note_core_mask = batch["note_core_mask"].to(device)
        S = batch["n_strings"]

        with torch.set_grad_enabled(train):
            player_id = batch["player_id"].to(device)
            detail = _ablate_detail(
                batch["detail_features"].to(device), note_core_mask,
                args.detail_features, detail_mode,
            )
            pred = model(inputs, player_id=player_id, detail_features=detail)
            pp = args.targets.index("pitch") if "pitch" in args.targets else None
            ep = args.targets.index(args.extra_channels[0])
            pe, te = pred[..., ep:ep + 1], target[..., ep:ep + 1]

            terms = {
                "loss_l1": l1_loss(pe, te, mask),
                "loss_delta": delta_loss(pe, te, mask, S),
                "loss_jitter": jitter_loss(pe, te, mask, S, mode=args.jitter_mode),
            }
            total = terms["loss_l1"] + args.lambda_delta * terms["loss_delta"] \
                + args.lambda_jitter * terms["loss_jitter"]

            if args.lambda_envelope_spectrum > 0:
                terms["loss_envelope_spectrum"] = per_note_envelope_spectrum_loss(
                    pe,
                    te,
                    note_core_mask,
                    mask,
                    min_frames=args.envelope_spectrum_min_frames,
                    n_fft=args.envelope_spectrum_n_fft,
                    lpf_frames=args.envelope_spectrum_lpf_frames,
                )
                terms["loss_envelope_spectrum_weighted"] = (
                    args.lambda_envelope_spectrum * terms["loss_envelope_spectrum"]
                )
                terms["envelope_spectrum_effective_ratio"] = (
                    terms["loss_envelope_spectrum_weighted"].detach()
                    / total.detach().clamp(min=1e-8)
                )
                total = total + terms["loss_envelope_spectrum_weighted"]

            # Anchor's own audibility mask -- used below for pitch AND for any further extra
            # channel (e.g. centroid), never for the anchor's own loss above: the anchor
            # (envelope/loudness) must learn to predict true silence too, the same reason
            # this term has never been masked.
            w = loudness_weight(te, lo=mspec["mask_lo"], hi=mspec["mask_hi"],
                                invert=mspec["mask_invert"])
            note_terms = {
                args.extra_channels[0]: per_note_l1_loss(pe, te, note_mask, mask)
            }

            if pp is not None:
                # Pitch is weighted by the REAL loudness: where the string is not sounding the
                # cached pitch is hold-forward filler, not evidence, and charging full price
                # there would spend the pitch budget on frames with no pitch in them.
                pcurve, tcurve = pred[..., pp:pp + 1], target[..., pp:pp + 1]
                terms["loss_pitch_l1"] = l1_loss(pcurve, tcurve, mask, weight=w)
                terms["loss_pitch_delta"] = delta_loss(pcurve, tcurve, mask, S, weight=w)
                # Independent from envelope's --lambda-delta so the trend/spectrum ablation can
                # zero pitch's raw-derivative supervision without touching envelope (which this
                # study leaves untouched). None (default) falls back to the shared value, so
                # every pre-existing invocation is byte-identical.
                pitch_delta_weight = (
                    args.lambda_delta if args.lambda_pitch_delta is None
                    else args.lambda_pitch_delta
                )
                total = total + args.lambda_pitch * (
                    terms["loss_pitch_l1"] + pitch_delta_weight * terms["loss_pitch_delta"])
                # Always computed for cross-arm comparability in the log, even when excluded
                # from the objective below (--exclude-pitch-from-note-shape) -- cheap, and
                # detached before accumulation like every other logged term.
                note_terms["pitch"] = per_note_l1_loss(
                    pcurve, tcurve, note_mask, mask, weight=w
                )
                if args.lambda_pitch_spectrum > 0:
                    score_pitch = (
                        inputs["score_pitch_midi"] - PITCH_MIDI_MEAN
                    ) / PITCH_MIDI_STD
                    terms["loss_pitch_spectrum"] = per_note_residual_spectrum_loss(
                        pcurve,
                        tcurve,
                        score_pitch,
                        note_core_mask,
                        mask,
                        min_frames=args.pitch_spectrum_min_frames,
                        n_fft=args.pitch_spectrum_n_fft,
                    )
                    terms["loss_pitch_spectrum_weighted"] = (
                        args.lambda_pitch_spectrum * terms["loss_pitch_spectrum"]
                    )
                    terms["pitch_spectrum_effective_ratio"] = (
                        terms["loss_pitch_spectrum_weighted"].detach()
                        / total.detach().clamp(min=1e-8)
                    )
                    total = total + terms["loss_pitch_spectrum_weighted"]

                if args.lambda_pitch_trend > 0:
                    if args.pitch_trend_note_scoped:
                        terms["loss_pitch_trend"] = per_note_pitch_trend_loss(
                            pcurve, tcurve, note_core_mask, mask, pitch_trend_kernel,
                            min_frames=args.pitch_trend_min_frames,
                            n_fft=args.pitch_trend_n_fft,
                        )
                    else:
                        terms["loss_pitch_trend"] = masked_pitch_trend_loss(
                            pcurve, tcurve, mask, S, pitch_trend_kernel, weight=w,
                        )
                    terms["loss_pitch_trend_weighted"] = (
                        args.lambda_pitch_trend * terms["loss_pitch_trend"]
                    )
                    terms["pitch_trend_effective_ratio"] = (
                        terms["loss_pitch_trend_weighted"].detach()
                        / total.detach().clamp(min=1e-8)
                    )
                    total = total + terms["loss_pitch_trend_weighted"]

            # Any channel beyond the anchor (e.g. centroid for the "pec" arm): masked like
            # pitch, not left unmasked like the anchor -- it is only meaningful when the
            # string is actually sounding, same reasoning as pitch's silence being filler.
            for ch in args.extra_channels[1:]:
                ci = args.targets.index(ch)
                ccurve, tc2 = pred[..., ci:ci + 1], target[..., ci:ci + 1]
                terms[f"loss_{ch}_l1"] = l1_loss(ccurve, tc2, mask, weight=w)
                terms[f"loss_{ch}_delta"] = delta_loss(ccurve, tc2, mask, S, weight=w)
                total = total + args.lambda_extra * (
                    terms[f"loss_{ch}_l1"] + args.lambda_delta * terms[f"loss_{ch}_delta"])
                note_terms[ch] = per_note_l1_loss(
                    ccurve, tc2, note_mask, mask, weight=w
                )
            terms.update({f"loss_note_{name}": value for name, value in note_terms.items()})
            note_shape = note_terms[args.extra_channels[0]]
            if pp is not None and not args.exclude_pitch_from_note_shape:
                note_shape = note_shape + args.lambda_pitch * note_terms["pitch"]
            for ch in args.extra_channels[1:]:
                note_shape = note_shape + args.lambda_extra * note_terms[ch]
            terms["loss_frame"] = total
            terms["loss_note_shape"] = note_shape
            if shape_balancer is not None:
                shape_weight = shape_balancer.weight(total, note_shape, update=train)
                terms["lambda_note_shape"] = shape_weight
                terms["loss_note_shape_weighted"] = shape_weight * note_shape
                terms["note_shape_effective_ratio"] = (
                    terms["loss_note_shape_weighted"].detach()
                    / terms["loss_frame"].detach().clamp(min=1e-8)
                )
                total = total + terms["loss_note_shape_weighted"]

            if critic is not None:
                # when pitch is predicted the critic scores the PREDICTED pitch too, otherwise
                # it would be judging an envelope against audio conditioned on a pitch the
                # model never has to get right
                fed_pitch = (denorm_pitch(pred[..., pp:pp + 1]) if pp is not None
                             else batch["gt_pitch_hz"].to(device))
                out = critic(denorm_env(pe), fed_pitch,
                             batch["latent"].to(device), mask,
                             gt_env=denorm_env(te))
                terms["loss_fm"] = out["loss"]
                terms["loss_fm_A"] = out["L_A"]
                if "L_B" in out:
                    terms["loss_fm_B"] = out["L_B"]
                total = total + args.lambda_fm * terms["loss_fm"]

            terms["loss"] = total

        if train:
            group_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, n_batches - group_start)
            (total / group_size).backward()
            if (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            terms["roughness_ratio"] = torch.tensor(roughness_ratio(pe, te, mask, S))
            terms["variance_ratio"] = torch.tensor(variance_ratio(pe, te, mask))
            if pp is not None:
                pc, tc = pred[..., pp:pp + 1], target[..., pp:pp + 1]
                # pitch error in CENTS over audible frames -- the interpretable unit, and the
                # one that caught a real bug earlier in this project where correlation on a
                # near-constant pitch looked fine while absolute tracking was far off
                wm = (loudness_weight(te, lo=mspec["mask_lo"], hi=mspec["mask_hi"],
                                      invert=mspec["mask_invert"])
                     * mask.to(te.dtype)) > 0.5
                cents = (pc - tc).squeeze(-1).abs() * PITCH_MIDI_STD * 100.0
                terms["pitch_mae_cents"] = (cents * wm).sum() / wm.sum().clamp(min=1)
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        n += 1
    return {k: v / max(1, n) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out-root", default="runs/perf")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--detail-features", default="none", choices=tuple(DETAIL_FEATURE_DIMS),
                    help="oracle per-note pitch-detail conditioning")
    ap.add_argument("--full-length", action="store_true",
                    help="train and validate on complete recordings")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="microbatches per optimizer update")
    ap.add_argument("--init-checkpoint",
                    help="warm-start model weights; optimizer/schedule restart intentionally")
    ap.add_argument("--eval-detail-controls", action="store_true",
                    help="also evaluate zero/shuffled detail at every validation point")
    # --- the knobs under study ---
    ap.add_argument("--lambda-delta", type=float, default=1.0)
    ap.add_argument("--lambda-jitter", type=float, default=0.0)
    ap.add_argument("--jitter-mode", default="hinge", choices=["hinge", "plain"])
    ap.add_argument("--lambda-fm", type=float, default=0.0)
    ap.add_argument(
        "--note-shape-ratio", type=float, default=0.0,
        help="EMA mode target weighted per-note/frame loss magnitude ratio; 0 disables the term",
    )
    ap.add_argument(
        "--note-shape-ema-decay", type=float, default=0.99,
        help="EMA decay used by --note-shape-weight-mode ema",
    )
    ap.add_argument(
        "--note-shape-weight-mode", choices=["ema", "fixed"], default="ema",
        help="auxiliary per-note multiplier policy",
    )
    ap.add_argument(
        "--note-shape-fixed-weight", type=float, default=1.0,
        help="fixed multiplier used when --note-shape-weight-mode fixed",
    )
    ap.add_argument(
        "--lambda-pitch-spectrum", type=float, default=0.0,
        help="weight of the per-note non-DC pitch-residual magnitude FFT loss",
    )
    ap.add_argument(
        "--pitch-spectrum-min-frames", type=int, default=8,
        help="ignore shorter notes whose FFT cannot resolve modulation",
    )
    ap.add_argument(
        "--pitch-spectrum-n-fft", type=int, default=256,
        help="fixed FFT size; notes longer than this are excluded",
    )
    ap.add_argument(
        "--lambda-pitch-trend", type=float, default=0.0,
        help="weight of the frequency-role-decomposition pitch trend loss: L1 between a "
             "shared FIR-lowpassed prediction and target -- phase-sensitive supervision for "
             "note-center/intonation, slow bend, and long-term drift, meant to run alongside "
             "--lambda-pitch-spectrum's phase-invariant modulation-energy term rather than "
             "raw full-band pitch L1. 0 disables it.",
    )
    ap.add_argument(
        "--pitch-trend-cutoff-hz", type=float, default=2.5,
        help="FIR low-pass cutoff (Hz); content above this is left to --lambda-pitch-spectrum",
    )
    ap.add_argument(
        "--pitch-trend-taps", type=int, default=43,
        help="odd FIR filter length; a clean ~2.5 Hz cutoff at this project's ~21.53 Hz frame "
             "rate needs on the order of 40+ taps (~2 s) -- LONGER than the median annotated "
             "note (~4.75 frames), so the (default) global trend draws support from "
             "neighbouring notes by design. Use --pitch-trend-note-scoped for a note-local "
             "estimate instead, which can use a shorter kernel.",
    )
    ap.add_argument(
        "--pitch-trend-note-scoped", action="store_true",
        help="compute the trend loss per annotated note (own fixed buffer, no cross-note "
             "support) instead of over the full per-string sequence",
    )
    ap.add_argument(
        "--pitch-trend-min-frames", type=int, default=8,
        help="--pitch-trend-note-scoped only: ignore shorter notes",
    )
    ap.add_argument(
        "--pitch-trend-n-fft", type=int, default=256,
        help="--pitch-trend-note-scoped only: fixed per-note buffer size",
    )
    ap.add_argument(
        "--lambda-pitch-delta", type=float, default=None,
        help="pitch's own first-difference weight, independent of envelope's --lambda-delta. "
             "None (default) falls back to --lambda-delta, matching every existing run. Set "
             "to 0 to isolate the trend/spectrum decomposition from raw-derivative "
             "supervision without touching envelope's delta term.",
    )
    ap.add_argument(
        "--exclude-pitch-from-note-shape", action="store_true",
        help="drop pitch's per-note L1 from the EMA-balanced note-shape term (envelope's own "
             "note-shape contribution is unaffected). Still logged as loss_note_pitch for "
             "cross-arm comparison; just excluded from the optimized total.",
    )
    ap.add_argument(
        "--lambda-envelope-spectrum", type=float, default=0.0,
        help="weight of the per-note non-DC LPF-detrended envelope magnitude FFT loss",
    )
    ap.add_argument(
        "--envelope-spectrum-min-frames", type=int, default=8,
        help="ignore shorter notes whose FFT cannot resolve modulation",
    )
    ap.add_argument(
        "--envelope-spectrum-n-fft", type=int, default=256,
        help="fixed FFT size; notes longer than this are excluded",
    )
    ap.add_argument(
        "--envelope-spectrum-lpf-frames", type=int, default=9,
        help="odd box-filter width (frames) removing the note's own attack/decay trend "
             "before the spectral comparison",
    )
    ap.add_argument(
        "--player-embed", action="store_true",
        help="condition on a learned per-GuitarSet-player identity embedding "
             "(dataset/perf/perf_curve_dataset.py's N_PLAYERS)",
    )
    ap.add_argument(
        "--no-cross-string-attention", action="store_true",
        help="ablation: omit model/common/axial.py's _GatedCrossStringBlock entirely "
             "(architecture and params, not just zero-gating) -- mirrors "
             "config/schema.py's decoder_cross_string_attention for the synth",
    )
    ap.add_argument("--targets", default="envelope",
                    help="'envelope' (default), 'pitch,envelope', or 'pitch,envelope,"
                         "ddsp_centroid_fixed' -- must equal ['pitch']+extra_channels (if "
                         "pitch is included) or extra_channels, in that order.")
    ap.add_argument("--extra-channels", default="envelope",
                    help="comma list of non-pitch target columns, in the order they appear "
                         "in --targets -- the first entry is the ANCHOR (must be a "
                         "loudness-like channel: it backs pseudo_velocity's gain and every "
                         "masked loss term's audibility gate, unmasked itself). Each name "
                         "must be a key in dataset/perf/perf_curve_dataset.py's "
                         "EXTRA_CHANNEL_SPECS, e.g. 'envelope' or 'envelope,ddsp_centroid_fixed'.")
    ap.add_argument("--lambda-pitch", type=float, default=1.0)
    ap.add_argument("--lambda-extra", type=float, default=1.0,
                    help="loss weight for every extra_channels entry beyond the anchor "
                         "(e.g. centroid for the pec arm) -- same role as --lambda-pitch.")
    ap.add_argument("--gain-mode", default="max", choices=["mean", "max", "none"],
                    help="how the supplied per-note gain reaches the model. 'mean'/'max' "
                         "select the statistic defined by the inference contract; 'none' "
                         "removes the channel for a no-gain ablation. Train, evaluation, and "
                         "inference must use the same mode.")
    ap.add_argument("--pitch-filter", default="none",
                    choices=["none", "hampel", "deadband", "mean"],
                    help="clean frame jitter out of the pitch TARGET before training on it")
    ap.add_argument("--fm-target", default="A", choices=["A", "B"])
    ap.add_argument("--t-dist", default="uniform",
                    choices=["uniform", "low", "high", "logit_normal"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.extra_channels = [c.strip() for c in args.extra_channels.split(",") if c.strip()]
    expected = (["pitch"] + args.extra_channels) if "pitch" in args.targets else args.extra_channels
    assert args.targets == expected, \
        f"--targets {args.targets} must equal {expected} (pitch, then --extra-channels in order)"
    if args.lambda_pitch_spectrum > 0 and "pitch" not in args.targets:
        raise ValueError("--lambda-pitch-spectrum requires pitch in --targets")
    if args.lambda_pitch_trend > 0 and "pitch" not in args.targets:
        raise ValueError("--lambda-pitch-trend requires pitch in --targets")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be positive")
    if args.full_length and args.batch_size != 1:
        raise ValueError("--full-length currently requires --batch-size 1: axial attention "
                         "does not mask cross-recording padding")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config)
    out = Path(args.out_root) / args.name
    saver = Saver(out)
    saver.save_config(vars(args))

    need_latent = args.lambda_fm > 0
    dataset_args = {
        "need_latent": need_latent,
        "targets": args.targets,
        "pitch_filter": args.pitch_filter,
        "gain_mode": args.gain_mode,
        "full_length": args.full_length,
        "extra_channels": args.extra_channels,
        "detail_features": args.detail_features,
    }
    tr = EnvelopeRegDataset(cfg, split="train", **dataset_args)
    va = EnvelopeRegDataset(cfg, split="val", deterministic=True, **dataset_args)
    print(f"train {len(tr)} / val {len(va)} recordings", flush=True)
    pitch_trend_kernel = None
    if args.lambda_pitch_trend > 0:
        pitch_trend_kernel = build_fir_lowpass_kernel(
            args.pitch_trend_cutoff_hz, tr._frame_rate(), args.pitch_trend_taps,
            device=device,
        )
    print(
        "pitch trend loss: "
        + ("off" if args.lambda_pitch_trend <= 0 else
           f"lambda={args.lambda_pitch_trend:g}, cutoff={args.pitch_trend_cutoff_hz:g}Hz, "
           f"taps={args.pitch_trend_taps}, "
           f"scope={'note' if args.pitch_trend_note_scoped else 'global'}"),
        flush=True,
    )
    print(
        "pitch decoupling: "
        f"lambda_pitch_delta="
        + ("(shared with --lambda-delta)" if args.lambda_pitch_delta is None
           else f"{args.lambda_pitch_delta:g}")
        + f", pitch_in_note_shape={not args.exclude_pitch_from_note_shape}",
        flush=True,
    )

    # Parsing GuitarSet JAMS dominates the first epoch. Keep workers alive so their dataset
    # caches survive subsequent epochs; recreating them repeated that work every epoch.
    worker_args = {
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    tl = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_env_reg,
        drop_last=True, generator=train_generator, **worker_args,
    )
    vl = DataLoader(
        va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_env_reg,
        **worker_args,
    )

    in_names = input_channels_for(args.targets, args.gain_mode)
    n_players = N_PLAYERS if args.player_embed else None
    detail_dim = detail_feature_dim(args.detail_features)
    model = EnvelopeRegressor(
        in_names, {n: 1 for n in in_names},
        d_model=args.d_model, depth=args.depth, out_dim=len(args.targets),
        n_players=n_players, detail_dim=detail_dim,
        cross_string_attention=not args.no_cross_string_attention,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = (
            {"detail_proj.weight", "detail_proj.bias"} if detail_dim else set()
        )
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible init checkpoint: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"warm-started {args.init_checkpoint}; new={sorted(allowed_missing)}",
            flush=True,
        )
    n_params = sum(p.numel() for p in model.parameters())

    opt = optimizer_registry.get(cfg.trainer.optimizer.name)(model.parameters(),
                                                             cfg.trainer.optimizer)
    for g in opt.param_groups:
        g["lr"] = args.lr
    sched = lr_scheduler_registry.get(cfg.trainer.lr_scheduler.name)(
        opt, cfg.trainer.lr_scheduler, total_steps=args.epochs)
    critic = build_critic(args, device)
    shape_balancer = None
    if args.note_shape_ratio > 0:
        shape_balancer = (
            EMAMagnitudeBalancer(args.note_shape_ratio, args.note_shape_ema_decay)
            if args.note_shape_weight_mode == "ema"
            else FixedWeightBalancer(args.note_shape_fixed_weight)
        )
    print(f"FM critic: {'off' if critic is None else f'{args.fm_target}, t={args.t_dist}'}",
          flush=True)
    print(
        "per-note shape loss: "
        + ("off" if shape_balancer is None else
           (f"ratio={args.note_shape_ratio:g}, EMA={args.note_shape_ema_decay:g}, "
            if args.note_shape_weight_mode == "ema"
            else f"fixed_weight={args.note_shape_fixed_weight:g}, ")
           + f"mode={args.note_shape_weight_mode}, "
           f"padding={cfg.dataset.note_shape_padding_frames} frames"),
        flush=True,
    )
    print(
        "pitch residual spectrum loss: "
        + ("off" if args.lambda_pitch_spectrum <= 0 else
           f"lambda={args.lambda_pitch_spectrum:g}, min_frames="
           f"{args.pitch_spectrum_min_frames}, n_fft={args.pitch_spectrum_n_fft}"),
        flush=True,
    )
    print(
        "envelope spectrum loss: "
        + ("off" if args.lambda_envelope_spectrum <= 0 else
           f"lambda={args.lambda_envelope_spectrum:g}, min_frames="
           f"{args.envelope_spectrum_min_frames}, n_fft={args.envelope_spectrum_n_fft}, "
           f"lpf_frames={args.envelope_spectrum_lpf_frames}"),
        flush=True,
    )
    print(
        "player embedding: "
        + ("off" if n_players is None else f"on, n_players={n_players}"),
        flush=True,
    )
    print(
        "cross-string attention: "
        + ("on" if not args.no_cross_string_attention else "OFF (ablated: blocks omitted)"),
        flush=True,
    )
    print(
        f"oracle detail: {args.detail_features}, dim={detail_dim}, "
        f"full_length={args.full_length}, grad_accum={args.grad_accum_steps}",
        flush=True,
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_terms = run_split(
            model, tl, args, critic, device, opt=opt, shape_balancer=shape_balancer,
            pitch_trend_kernel=pitch_trend_kernel,
        )
        sched.step()
        secs = round(time.time() - t0, 1)
        saver.log({"split": "train", "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                   "secs": secs, **tr_terms})

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            va_terms = run_split(
                model, vl, args, critic, device, shape_balancer=shape_balancer,
                pitch_trend_kernel=pitch_trend_kernel,
            )
            saver.log({"split": "val", "epoch": epoch, **va_terms})
            if args.eval_detail_controls and detail_dim:
                control_modes = ["zero", "shuffle"]
                if args.detail_features == "phase":
                    control_modes.append("phase_shuffle")
                for mode in control_modes:
                    controlled = run_split(
                        model, vl, args, critic, device, shape_balancer=shape_balancer,
                        detail_mode=mode, pitch_trend_kernel=pitch_trend_kernel,
                    )
                    saver.log({"split": f"val_{mode}", "epoch": epoch, **controlled})
            saver.save_checkpoint(
                model, opt, epoch, va_terms["loss"],
                tag=f"ep{epoch:04d}" if epoch % 25 == 0 else None,
                metadata={
                    "n_players": n_players,
                    "detail_features": args.detail_features,
                    "detail_dim": detail_dim,
                    "input_names": in_names,
                    "targets": args.targets,
                    "extra_channels": args.extra_channels,
                    "gain_mode": args.gain_mode,
                    "note_shape_weight_mode": args.note_shape_weight_mode,
                    "note_shape_ratio": args.note_shape_ratio,
                    "note_shape_fixed_weight": args.note_shape_fixed_weight,
                    "d_model": args.d_model,
                    "depth": args.depth,
                    "cross_string_attention": not args.no_cross_string_attention,
                },
            )

            # per-eval-step curve plot on a FIXED val recording
            model.eval()
            item = va[0]
            batch = collate_env_reg([item])
            with torch.no_grad():
                pred = model(
                    {k: v.to(device) for k, v in batch["controls"].items()},
                    player_id=batch["player_id"].to(device),
                    detail_features=batch["detail_features"].to(device),
                )
            ei = args.targets.index(args.extra_channels[0])  # plot the anchor channel
            saver.plot_curves(epoch, item["recording_id"],
                              pred[0, :, ei].cpu().numpy(),
                              batch["target"][0, :, ei].numpy(),
                              batch["valid_mask"][0].numpy(),
                              item["n_strings"],
                              extra={"rough": va_terms["roughness_ratio"],
                                     "var": va_terms["variance_ratio"]},
                              active=batch["controls"]["voiced"][0, :, 0].numpy(),
                              onset=batch["controls"]["segment_onset"][0, :, 0].numpy(),
                              pitch=((pred[0, :, args.targets.index("pitch")].cpu().numpy(),
                                      batch["target"][0, :, args.targets.index("pitch")].numpy())
                                     if "pitch" in args.targets else None))
            saver.plot_loss_terms()
            extra_p = (f" pcents {va_terms['pitch_mae_cents']:.1f}"
                       if "pitch_mae_cents" in va_terms else "")
            print(f"ep{epoch} train {tr_terms['loss']:.4f} val {va_terms['loss']:.4f} "
                  f"l1 {va_terms['loss_l1']:.4f} rough {va_terms['roughness_ratio']:.3f} "
                  f"var {va_terms['variance_ratio']:.3f}{extra_p} ({secs}s/ep)",
                  flush=True)

    print(f"done. gates={['%.3f' % g for g in model.cross_string_gates()]}")


if __name__ == "__main__":
    main()
