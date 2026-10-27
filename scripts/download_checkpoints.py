# Downloads the five pretrained checkpoints (see CHECKPOINTS.md) from a Hugging Face
# model repo into checkpoints/, matching the layout every training script and
# inference/pipeline.py already expects.
#
# Run as: python -m scripts.download_checkpoints --repo-id Will3024/ctrl-lfm-guitar-synth
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = [
    "checkpoints/gain/note_gain_v5_samestring_fixed.pt",
    "checkpoints/perf/pe_note_shape_fixed13.pt",
    "checkpoints/synth/joint_synth_peov_embed.pt",
    "checkpoints/reverb/mix_reverb.pt",
    "checkpoints/synth/joint_synth_velocity_peak.pt",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, help="Hugging Face model repo id, e.g. Will3024/ctrl-lfm-guitar-synth")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--skip-baseline", action="store_true",
                     help="skip joint_synth_velocity_peak.pt (the single-stage baseline, "
                          "not needed to run the three-stage score-to-audio pipeline)")
    args = ap.parse_args()

    for rel_path in FILES:
        if args.skip_baseline and rel_path.endswith("joint_synth_velocity_peak.pt"):
            continue
        dest = Path(rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download_checkpoints] {rel_path} <- {args.repo_id}")
        local = hf_hub_download(repo_id=args.repo_id, filename=rel_path, revision=args.revision)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(Path(local).resolve())

    print("done.")


if __name__ == "__main__":
    main()
