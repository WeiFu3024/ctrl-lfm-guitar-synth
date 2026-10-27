# guitar-synth listening demo

Static GitHub Pages site (no build step — plain HTML/CSS/JS, like this project's other
browser gadgets `tools/audio_control_viewer` and `tools/score_inference_demo` in the
`guitar-synth` dev repo). Lives at `docs/` in this repo's root so GitHub Pages can serve
it via "Deploy from a branch: master, /docs" without a build step; the rest of the repo
(`config/`, `model/`, `dataset/`, `scripts/`, `inference/`, ...) is the training +
inference code the top-level `README.md` documents.

## What it shows

5 recordings from `runs/official-run/listening-test/` (one — `03_Rock1-130-A_comp` — is
the dedicated practice recording, unscored; the other four are real scored listening-test
items): `03_Rock1-130-A_comp`, `04_Jazz3-150-C_comp`, `00_SS2-88-F_solo`,
`01_BN1-147-Gb_comp`, `05_Funk2-108-Eb_comp`.

Per sample: a FluidSynth "query" hint rendered from the score MIDI, a color-coded
six-string MIDI piano roll, and four selectable systems (`main_pipeline` — the paper's
model, default; `velocity_joint_peak`; `ddsp_guitar`; `ground_truth`). Each system shows
its own audio plus two curve groups for a selectable string: the 5-channel score-derived
input the performance predictor conditions on, and the 2-channel CREPE-pitch/log-RMS
envelope ground truth it was trained toward. **Those curve groups are dataset/score-level,
not model output** — they are identical across all four systems by construction, since
that is exactly what `scripts/pipeline/plot_ctrl_preview.py` computes (score-conditioned
input + real recorded target, not a per-system prediction). Only the audio and the shared
vertical playhead change with the system tab.

Playback is exclusive (starting any clip pauses whatever else is playing) and the
playhead is one shared clock driving the MIDI roll and both curve groups at once,
regardless of which of the five audio elements (query + 4 systems) is the one actually
sounding.

## Regenerating `data/`

`data/` (audio mp3s, per-string MIDI-note JSON, per-string control-curve JSON, and
`manifest.json`) is generated entirely by the `guitar-synth` dev repo's
`scripts/pipeline/export_demo_assets.py` — nothing in this repo computes it. From
`guitar-synth/`:

```bash
conda activate guitar-synth   # needs mido + that repo's dataset/control cache access
python -m scripts.pipeline.export_demo_assets --out demo_site/docs/data
```

Needs `ffmpeg` (wav -> 128kbps mp3) and reads from
`runs/official-run/listening-test/` (four-arm audio + per-string MIDI) and
`subjective-eval/guitar_mos/static/audio/<recording>/query.mp3` (query hint) in the main
repo. No GPU/HF_TOKEN needed — it only reads already-cached/rendered artifacts.

To add or change the five recordings, edit `RECORDINGS` at the top of that script and
re-run; the recording ID must have an entry in
`runs/official-run/listening-test/manifest.json` with all four arms available.

## Running locally

```bash
cd docs && python3 -m http.server 8910
```

then open `http://localhost:8910/`. `fetch()` of `data/*.json` needs an actual HTTP
server — opening `index.html` directly via `file://` will fail on CORS in most browsers.

## Deploying

GitHub Pages is configured to deploy from `master`, path `/docs` (Settings → Pages).
All asset references in this directory are relative, so it keeps working regardless of
repo name.
