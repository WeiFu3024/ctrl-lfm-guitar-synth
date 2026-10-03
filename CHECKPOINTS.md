# Pretrained checkpoints

Four checkpoints are hosted on Hugging Face (not committed to this git
repository -- see `scripts/download_checkpoints.py`). They implement the
three-stage score-to-audio pipeline plus one baseline/ablation:

| checkpoint | stage | trained by | config |
|---|---|---|---|
| `checkpoints/gain/note_gain_v5_samestring_fixed.pt` | 1. note-gain predictor (score -> per-note loudness) | `scripts/perf/train_note_gain_predictor.py` | `config/perf.yaml` |
| `checkpoints/perf/pe_note_shape_fixed13.pt` | 2. envelope regressor (score + loudness -> pitch/envelope curves) | `scripts/perf/train_perf_predictor.py` | `config/perf.yaml` |
| `checkpoints/synth/joint_synth_peov_embed.pt` | 3. joint six-string synth (curves -> audio latents, rectified flow) | `scripts/synth/train_joint_synth.py` | `config/synth.yaml` |
| `checkpoints/reverb/mix_reverb.pt` | optional post-process (six dry strings -> one mic-domain mix) | `scripts/synth/train_mix_reverb.py` | -- |
| `checkpoints/synth/joint_synth_velocity_peak.pt` | baseline/ablation: single-stage synth conditioned on flat MIDI pitch + peak pseudo-velocity only, no predicted expressive curves | `scripts/synth/train_joint_synth.py` | `config/synth-velocity-baseline.yaml` |

All five are the exact weights this repository's own training commands
reproduce (see each script's `# Run as:` header and each config's own header
comment for the full training invocation and hyperparameters).

## License

These checkpoints are released under **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/),
matching the license of GuitarSet, the dataset they were trained on. You may
use, modify, and redistribute them, including commercially, provided you give
appropriate attribution -- cite this repository and GuitarSet:

> Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "GuitarSet: A Dataset
> for Guitar Transcription", ISMIR 2018.

**Upstream dependency you must separately accept.** Running inference with
any of these checkpoints requires downloading `stabilityai/stable-audio-open-1.0`
from Hugging Face at runtime (the frozen latent encoder/decoder, SAO3). That
model is governed by the **Stability AI Community License**, not by this
repository's license: free for organizations/individuals with under $1M USD
annual revenue, subject to Stability's Acceptable Use Policy; above that
threshold requires a separate Stability AI Enterprise License. You must
accept Stability's license yourself on Hugging Face (`huggingface-cli login`
+ accepting the gated model) before `HF_TOKEN` will be able to download it.
Generated audio's downstream use is bound by Stability's Acceptable Use
Policy in addition to this repository's own terms.
