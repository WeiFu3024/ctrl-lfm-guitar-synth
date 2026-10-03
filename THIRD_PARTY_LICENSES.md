# Third-party licenses

This repository does not bundle any third-party source code or weights.
Everything below is installed separately (via `pip install -r requirements.txt`
or downloaded from Hugging Face at runtime) and retains its own license.

| dependency | role | license |
|---|---|---|
| `stabilityai/stable-audio-open-1.0` (Hugging Face, downloaded at runtime) | frozen latent encoder/decoder (SAO3) | Stability AI Community License -- see CHECKPOINTS.md |
| `stable-audio-tools` (PyPI) | wraps the above model | Apache-2.0 |
| GuitarSet (dataset, downloaded separately) | training data | CC BY 4.0 -- cite Xi et al., ISMIR 2018 |
| GAPS (4 MusicXML pieces bundled in `examples/`) | demo inference inputs | MIT -- cite Riley et al., ISMIR 2024 |
| `torch`, `torchaudio` | tensor/audio ops | BSD-3-Clause |
| `torchcrepe` | pitch extraction | MIT |
| `music21` | MusicXML parsing | BSD-3-Clause |
| `PyGuitarPro` | GuitarPro file parsing | MIT |
| `pretty_midi`, `mido` | MIDI parsing | MIT |
| `omegaconf` | typed config loading | BSD-3-Clause |
| `zarr`, `numpy`, `scipy`, `soundfile`, `matplotlib`, `tqdm`, `einops`, `huggingface_hub` | standard scientific/ML tooling | BSD/MIT/Apache-2.0 (per-package) |

Verify current terms directly from each project before redistributing a
derivative work; the table above reflects the licenses in effect as of this
repository's release and is not a substitute for reading them yourself.
