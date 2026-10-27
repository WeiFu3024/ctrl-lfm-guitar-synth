# guitar-synth

Score -> expressive performance curves -> audio, for solo acoustic/electric guitar.

[Demo](https://weifu3024.github.io/ctrl-lfm-guitar-synth/) &middot; [Checkpoints (Hugging Face)](https://huggingface.co/Will3024/ctrl-lfm-guitar-synth)

![Two-stage architecture: a performance-control predictor (left) turns per-string score curves into predicted pitch/envelope curves via an axial transformer encoder, and a latent flow-matching decoder (right) renders those curves into six-string audio through an axial DiT, an ODE solver, and a neural audio codec decoder.](docs/architecture.png)

Three frozen models chained together (the diagram's "performance-control predictor"
box is the first two stages below; "flow-matching decoder" is the third):

1. **note-gain predictor** -- score-only note features -> per-note loudness ("velocity")
2. **envelope regressor** -- score + predicted loudness -> per-frame pitch/envelope curves
3. **joint six-string synth** -- curves -> six-string audio latents (rectified-flow sampling)

...then a frozen SAO3 decode turns the latents into per-string + mixed audio, with an
optional learned six-string mix reverb folding that into a mic-domain mix.

Trained on [GuitarSet](https://zenodo.org/records/3371780) (CC BY 4.0), conditioned
through [`stabilityai/stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0)'s
frozen latent space.

## Install

Python 3.10 (pinned by `stable-audio-tools`' `zarr<3` constraint).

```bash
pip install -r requirements.txt
huggingface-cli login   # accept stabilityai/stable-audio-open-1.0's gated license once
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"   # needed by any command that builds/uses SAO3
```

## Quickstart: inference only (no training)

```bash
python -m scripts.download_checkpoints --repo-id Will3024/ctrl-lfm-guitar-synth
CUDA_VISIBLE_DEVICES=0 python -m inference.infer --gp examples/008_PSswc.xml --out runs/demo
```

Renders every detected guitar part in the score to `runs/demo/<guitar>/mix.wav`
(plus per-string audio, the exact curves fed to the synth, and a manifest).
Accepts GuitarPro (`.gp3/.gp4/.gp5`), MusicXML with a guitar-tab staff
(`.xml/.musicxml`), or plain MIDI (`.mid/.midi`) -- see `inference/README.md`
for format-specific detail and the windowing scheme used for full-length songs.
`--reverb-ckpt checkpoints/reverb/mix_reverb.pt` reproduces the mic-domain mix
exactly; omitted by default (dry six-string sum) since it is a post-process on
top of the three predicted stages, not one of them.

## Training from scratch

### 1. Get GuitarSet

Download from [Zenodo](https://zenodo.org/records/3371780) and either place it at
`data/GuitarSet/` or point `GUITARSET_ROOT` at it:

```bash
export GUITARSET_ROOT=/path/to/GuitarSet
```

### 2. Build the cache

Extracts per-string audio latents (through SAO3) and control signals (pitch,
envelope, onset, voiced, DDSP-style controls) into local Zarr stores.
Incremental -- safe to re-run after editing `config/*.yaml`'s `control.signals`.

```bash
python -m scripts.data.build_cache --config config/synth.yaml
```

### 3. Train the three (plus one optional) models

Exact hyperparameters below reproduce the released checkpoints. The three
stages train independently from the cache -- none needs another's checkpoint
first (the synth only becomes a training-time dependency of stage 2 if you
pass `--lambda-fm > 0` to enable its optional flow-matching critic term,
which the released `pe_note_shape_fixed13` checkpoint does not use).

```bash
# joint six-string synth (curves -> audio latents)
python -m scripts.synth.train_joint_synth --config config/synth.yaml

# optional: learned 6 -> 1 mix reverb (post-process, not required for the 3-stage pipeline)
python -m scripts.synth.train_mix_reverb --variant decoded

# Stage 2: envelope regressor (score + loudness -> pitch/envelope curves)
python -m scripts.perf.train_perf_predictor --name pe_note_shape_fixed13 \
  --config config/perf.yaml --out-root runs/perf \
  --targets pitch,envelope --note-shape-ratio 1 --note-shape-fixed-weight 1.3 \
  --epochs 150 --batch-size 4 --d-model 256 --depth 6 --lr 3e-4

# Stage 1: note-gain predictor (score -> per-note loudness)
python -m scripts.perf.train_note_gain_predictor --name note_gain \
  --out-root runs/note-gain --epochs 150 --batch-size 8 \
  --delta-loss weighted --delta-scope same_string_next --delta-ratio 1.0

# optional baseline/ablation: single-stage synth on flat MIDI pitch + peak velocity only
python -m scripts.synth.train_joint_synth --config config/synth-velocity-baseline.yaml
```

Or run everything (cache + all four training commands) plus a metrics summary you can
compare against the released checkpoints' own numbers in one shot:

```bash
./scripts/run_all.sh
```

Point `inference/pipeline.py`'s `PERF_CKPT` / `SYNTH_CKPT` / `GAIN_CKPT`
constants (or `inference.infer`'s matching `--perf-ckpt` / `--synth-ckpt` /
`--gain-ckpt` flags) at your own checkpoints to run inference against them
instead of the released ones.

`scripts/perf/calibrate_perf_stats.py` / `calibrate_note_gain_stats.py` print
the per-channel normalization constants hand-pasted into
`dataset/perf/perf_curve_dataset.py` / `dataset/perf/note_gain_dataset.py` --
only needed again if you change the training corpus.

## Repository layout

```
config/       typed OmegaConf schema (config/schema.py) + the two live YAML configs
model/        network definitions: common/ (DiT, axial cross-string attention, flow
              matching), synth/ (joint six-string decoder, mix reverb), perf/ (envelope
              regressor, note-gain predictor)
dataset/      cache building (control/, latent/) and training datasets (synth/, perf/)
trainer/      optimizer/scheduler registries
saver/        checkpointing, logging, training-curve plots
scripts/      training and cache-building CLI entry points
inference/    the full score -> audio CLI (GuitarPro / MusicXML / MIDI parsers + pipeline)
examples/     four MIT-licensed GAPS demo pieces (MusicXML) for the quickstart
checkpoints/  pretrained weights (gitignored -- see CHECKPOINTS.md / download_checkpoints.py)
docs/         GitHub Pages listening demo (static HTML/CSS/JS, see docs/README.md) --
              https://weifu3024.github.io/ctrl-lfm-guitar-synth/
```

## License

Code: MIT (`LICENSE`). Pretrained checkpoints: CC BY 4.0, with a required
upstream Stability AI Community License acceptance at inference time -- see
`CHECKPOINTS.md`. Third-party runtime dependencies: `THIRD_PARTY_LICENSES.md`.

If you use this code or these checkpoints, please cite GuitarSet:

> Q. Xi, R. Bittner, J. Pauwels, X. Ye, and J. P. Bello, "GuitarSet: A Dataset
> for Guitar Transcription", ISMIR 2018.

`examples/` ships four pieces from [GAPS](https://zenodo.org/records/13962272) (MIT
licensed):

> X. Riley, Z. Guo, D. Edwards, and S. Dixon, "GAPS: A Large and Diverse Classical
> Guitar Dataset and Benchmark Transcription Model", ISMIR 2024.
