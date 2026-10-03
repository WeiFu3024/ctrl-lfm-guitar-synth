# Demo: real guitar score (GuitarPro or plain MIDI) -> velocity predictor -> control
# predictor -> frozen synth -> per-guitar audio, for a file this repo has never seen (no
# cached recording, no annotations -- see README.md in this directory for the pipeline
# this drives).
#
# Run as:
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
#     python -m inference.infer --gp examples/008_PSswc.xml \
#       --out runs/score-inference-demo
import argparse
import json
import re
from pathlib import Path

from inference import gp_score, midi_score, musicxml_score
from inference.pipeline import (
    CFG_SCALE, GAIN_CKPT, NFE, PERF_CKPT, PERF_CONFIG, SYNTH_CKPT, SYNTH_CONFIG,
    load_models, render_guitar_part,
)

# .gp* carries an exact string/fret field (gp_score.py); plain MIDI has none, so
# midi_score.py infers it (see that module's docstring); a MusicXML with a tab staff
# (musicxml_score.py) also carries an exact field, read directly like .gp*. Dispatched
# by extension so the same --gp flag and the rest of this script are format-agnostic.
LOADERS = {
    ".gp3": gp_score.load_guitar_parts, ".gp4": gp_score.load_guitar_parts,
    ".gp5": gp_score.load_guitar_parts,
    ".mid": midi_score.load_guitar_parts, ".midi": midi_score.load_guitar_parts,
    ".xml": musicxml_score.load_guitar_parts, ".musicxml": musicxml_score.load_guitar_parts,
}


def load_score(path: str):
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise SystemExit(f"unrecognised score extension {ext!r} for {path}; supported: {sorted(LOADERS)}")
    return loader(path)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "guitar"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gp", default="examples/008_PSswc.xml",
                    help="score file: GuitarPro (.gp3/.gp4/.gp5, exact string/fret), "
                         "MusicXML with a tab staff (.xml/.musicxml, exact string/fret), "
                         "or plain MIDI (.mid/.midi, string assigned by midi_score.py)")
    ap.add_argument("--out", default="runs/score-inference-demo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gain-ckpt", default=GAIN_CKPT)
    ap.add_argument("--perf-config", default=PERF_CONFIG)
    ap.add_argument("--perf-ckpt", default=PERF_CKPT)
    ap.add_argument("--synth-config", default=SYNTH_CONFIG)
    ap.add_argument("--synth-ckpt", default=SYNTH_CKPT)
    ap.add_argument("--reverb-ckpt", default=None,
                    help="optional learned six-string mix reverb; omitted by default "
                         "(dry string sum) since it is not one of the three predicted stages")
    ap.add_argument("--guitars", default=None,
                    help="comma-separated guitar part names to render (default: every "
                         "detected guitar part)")
    ap.add_argument("--nfe", type=int, default=NFE)
    ap.add_argument("--cfg-scale", type=float, default=CFG_SCALE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    parts = load_score(args.gp)
    if not parts:
        raise SystemExit(f"no acoustic/electric guitar track found in {args.gp}")
    print(f"detected {len(parts)} guitar part(s): "
          + ", ".join(f"{p.name!r} ({p.n_notes} notes, strings {p.strings_used})" for p in parts))

    wanted = None
    if args.guitars:
        wanted = {g.strip() for g in args.guitars.split(",") if g.strip()}
        missing = wanted - {p.name for p in parts}
        if missing:
            raise SystemExit(f"--guitars named parts not found: {sorted(missing)}; "
                              f"available: {[p.name for p in parts]}")

    print("loading checkpoints (note-gain predictor -> perf predictor -> synth)...", flush=True)
    models = load_models(
        args.device, perf_config=args.perf_config, perf_ckpt=args.perf_ckpt,
        synth_config=args.synth_config, synth_ckpt=args.synth_ckpt,
        gain_ckpt=args.gain_ckpt, reverb_ckpt=args.reverb_ckpt,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for part in parts:
        if wanted is not None and part.name not in wanted:
            continue
        d = out_root / _slug(part.name)
        print(f"rendering {part.name!r} -> {d} ({part.n_notes} notes)...", flush=True)
        meta = render_guitar_part(part, models, d, seed=args.seed, nfe=args.nfe, cfg_scale=args.cfg_scale)
        meta["source_gp"] = args.gp
        meta["output_dir"] = str(d)
        with open(d / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)
        print(f"  wrote {meta['duration_s']:.1f}s -> {d / 'mix.wav'}")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done. {len(manifest)} guitar(s) rendered under {out_root}")


if __name__ == "__main__":
    main()
