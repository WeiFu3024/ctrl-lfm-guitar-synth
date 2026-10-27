---
license: cc-by-4.0
base_model: stabilityai/stable-audio-open-1.0
tags:
- audio
- music-generation
- guitar
- flow-matching
- pytorch
pipeline_tag: audio-to-audio
---

# guitar-synth checkpoints

Pretrained weights for [ctrl-lfm-guitar-synth](https://github.com/WeiFu3024/ctrl-lfm-guitar-synth)
(code + [listening demo](https://weifu3024.github.io/ctrl-lfm-guitar-synth/)):
score (GuitarPro / MusicXML / MIDI) -> expressive performance curves -> audio,
for solo acoustic/electric guitar. Three frozen models chained together, plus
one optional post-process and one baseline/ablation.

## Files

| file | stage | trained by | config |
|---|---|---|---|
| `checkpoints/gain/note_gain_v5_samestring_fixed.pt` | 1. note-gain predictor (score -> per-note loudness) | `scripts/perf/train_note_gain_predictor.py` | `config/perf.yaml` |
| `checkpoints/perf/pe_note_shape_fixed13.pt` | 2. envelope regressor (score + loudness -> pitch/envelope curves) | `scripts/perf/train_perf_predictor.py` | `config/perf.yaml` |
| `checkpoints/synth/joint_synth_peov_embed.pt` | 3. joint six-string synth (curves -> audio latents, rectified flow) | `scripts/synth/train_joint_synth.py` | `config/synth.yaml` |
| `checkpoints/reverb/mix_reverb.pt` | optional post-process (six dry strings -> one mic-domain mix) | `scripts/synth/train_mix_reverb.py` | -- |
| `checkpoints/synth/joint_synth_velocity_peak.pt` | baseline/ablation: single-stage synth conditioned on flat MIDI pitch + peak pseudo-velocity only | `scripts/synth/train_joint_synth.py` | `config/synth-velocity-baseline.yaml` |

These are the exact weights the repository's own training commands reproduce
(see each script's `# Run as:` header and each config's header comment).

## Usage

```bash
pip install -r requirements.txt   # from the code repo
python -m scripts.download_checkpoints --repo-id Will3024/ctrl-lfm-guitar-synth
CUDA_VISIBLE_DEVICES=0 python -m inference.infer --gp examples/008_PSswc.xml --out runs/demo
```

Or fetch a single file directly:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Will3024/ctrl-lfm-guitar-synth", filename="checkpoints/synth/joint_synth_peov_embed.pt")
```

## License

CC BY 4.0, matching [GuitarSet](https://zenodo.org/records/3371780), the dataset
these were trained on. Use, modify, and redistribute, including commercially,
provided you give attribution -- cite this repository and GuitarSet:

> Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "GuitarSet: A Dataset
> for Guitar Transcription", ISMIR 2018.

**Upstream dependency you must separately accept.** Running inference with any
of these checkpoints requires downloading
[`stabilityai/stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0)
at runtime (the frozen latent encoder/decoder, SAO3). That model is governed
by the **Stability AI Community License**, not this repository's license: free
for organizations/individuals under $1M USD annual revenue, subject to
Stability's Acceptable Use Policy; above that threshold requires a separate
Stability AI Enterprise License. You must accept Stability's license yourself
on Hugging Face (`huggingface-cli login` + accepting the gated model) before
`HF_TOKEN` can download it. Generated audio's downstream use is bound by
Stability's Acceptable Use Policy in addition to this repository's own terms.

## Citation

```bibtex
@software{fu_ctrl_lfm_guitar_synth,
  title        = {ctrl-lfm-guitar-synth: Expressive Guitar Synthesis via Controllable Latent Flow Matching},
  author       = {Fu, Hsueh-Wei},
  affiliation  = {National Taiwan University},
  url          = {https://github.com/WeiFu3024/ctrl-lfm-guitar-synth}
}
```
