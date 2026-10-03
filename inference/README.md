# Score-to-audio inference

Renders real, unseen guitar audio from a **GuitarPro score, a tab-bearing MusicXML file,
or a plain MIDI file** (no cached recording, no annotations, no ground-truth curves)
through this repo's full three-model pipeline:

```
score (per string, per guitar)
   |
   v
1. NoteGainPredictor   score-only note features -> per-note loudness ("velocity")
   |
   v
2. EnvelopeRegressor   score + predicted loudness -> per-frame pitch/envelope curves
   |
   v
3. JointSynthDecoder   curves -> six-string audio latents (rectified-flow sampling)
   |
   v
SAO decode -> one audio file per guitar
```

All three checkpoints are frozen, pretrained artifacts (see `pipeline.py`'s
`PERF_CKPT`/`SYNTH_CKPT`/`GAIN_CKPT`, or `scripts/download_checkpoints.py` to
fetch the released ones); nothing is trained here.

## Three input formats

`--gp` accepts any of the three, dispatched by extension (`infer.py`'s `LOADERS`).

**GuitarPro** (`.gp3`/`.gp4`/`.gp5`) carries string/fret assignment and note ties as
**exact fields** — `gp_score.py` reads them directly, no heuristic, no guessing which of
several overlapping pitch bands a note belongs to. (Guitar Pro 7/8's zipped-XML
container, `.gp`, is a different format that PyGuitarPro's `guitarpro` package does not
parse — `.gp5`, GuitarPro's older binary format, is what `gp_score.py` targets.)

**MusicXML with a tab staff** (`.xml`/`.musicxml`) is also exact, when the file has one.
A file exported from tab software often carries TWO `<part>` elements for one guitar: a
plain notation staff (no string/fret) and a tab staff (every note tagged).
`musicxml_score.py`'s `_tab_coverage` identifies which part(s) are real tab staves (>=95%
of notes carrying a `<technical><string>` mark) rather than assuming a fixed part index,
and reads pitch from the file's own `<pitch>` element rather than deriving it from fret +
an assumed standard tuning (some real pieces are not in standard tuning, which deriving
pitch from fret+tuning would get wrong). Uses `music21` for voice/tie/chord resolution
and handles genuine mid-piece tempo changes via a breakpoint-integrated tempo map, not a
single constant multiplier. If a MusicXML file has no tab staff, `load_guitar_parts`
returns no parts for it (surfaced by `infer.py` as "no acoustic/electric guitar track
found") rather than silently misreading the notation staff.

**Plain MIDI** (`.mid`/`.midi`) has no string/fret field at all — often not even a
channel convention, just one track's worth of (possibly polyphonic, possibly
chord-laden) notes. `midi_score.py` reconstructs string assignment with a fingering
heuristic: standard-tuning candidate (string, fret) pairs per note, notes grouped into
chords by onset proximity and solved jointly (a small combinatorial search) so two
simultaneous notes never compete for the same string, minimising distance from a running
hand-position estimate plus a chord-span penalty. See `midi_score.py`'s module docstring
for the full algorithm and its fallback behaviour. Leading silence is trimmed per part so
the model's frame budget isn't spent decoding dead air. Prefer the MusicXML or GuitarPro
path when available — it is exact; plain MIDI is a fallback for when no exact source
exists.

## Guitar detection and grouping

`gp_score.load_guitar_parts` keeps every GuitarPro track whose MIDI program is in the
GM guitar family (0-indexed programs 24-31: acoustic nylon/steel, electric jazz/clean/
muted/overdriven/distortion, harmonics — "either acoustic or electric guitar"), and drops
the percussion channel plus every non-guitar instrument (bass, drums, vocals, ...). Each
GuitarPro `Track` already contains all six of one guitar's strings, so "group the same
guitar" reduces to keeping one `GuitarPart` per track; a defensive merge unions any two
across multiple tracks by name instead of keeping it in one `Track`. A GuitarPro file
with two differently-named guitar tracks (e.g. "Lead Guitar" and "Rhythm Guitar") renders
each **one at a time**, via `infer.py`'s per-part loop, each to its own output directory.

Tied GuitarPro notes are merged into the sustaining note's own duration rather than
counted as a new pluck (see `gp_score.py`'s module docstring); muted ("dead") notes are
dropped, since they carry no definite pitch for the pitch/envelope curves this pipeline
predicts.

## Windowing

A full song can run past 4000 frames per string — far longer than the training
segment-length range either the perf predictor or the synth is validated against. Three
separate mechanisms handle this at three different stages, each fixed after a real,
measured failure at longer-than-ever-tested lengths (not pre-emptively):

1. **Perf predictor (stage 2).** `pipeline.py`'s `predict_curves_chunked` cuts only at
   note boundaries (never mid-sustain, on any string), prefixes every non-initial chunk
   with 0.5s of silence plus ~1s of already-finalised note context, and concatenates each
   chunk's own new content directly. A naive sliding hop-192/window-256 blend measured a
   3-14x raw quality gap between the chunk starting at frame 0 and every other chunk —
   see `predict_curves_chunked`'s own module docstring for the full measurement.
2. **Synth sampling.** `render_synth_windowed` uses a 256-frame/192-hop raised-cosine
   latent-domain blend, verified not to degrade across a window boundary.
3. **Synth decode.** `SAO3LatentEncoder.decode` (`dataset/latent/sao3.py`) chunks the
   latent-to-audio decode itself once a song exceeds `DECODE_CHUNK_FRAMES` (5000) — a
   single unwindowed `decode()` call OOMs a 24 GB card well before that, independent of
   and after the sampling loop above.

## Usage

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  python -m inference.infer --gp examples/008_PSswc.xml --out runs/demo
```

`examples/` ships four real MusicXML pieces from the [GAPS dataset](https://zenodo.org/records/13962272)
(004_JSswc, 008_PSswc, 009_xSswc, 010_QSswc, 63-185s each, MIT licensed) — unseen by any
of the three checkpoints during training, with no recording, no annotations, and no
ground-truth curves attached.

GuitarPro and plain MIDI work the same way, dispatched by extension:

```bash
python -m inference.infer --gp path/to/piece.gp5 --out runs/demo-gp
python -m inference.infer --gp path/to/piece.mid --out runs/demo-midi
```

Needs the Python environment that has `HF_TOKEN` available, since it constructs the SAO
decoder. Renders every detected guitar part by default; `--guitars "<name>"` restricts to
a subset. `--reverb-ckpt` is optional — omitted by default, since the learned six-string
mix reverb is a post-process on top of the three predicted stages, not one of them;
passing `checkpoints/reverb/mix_reverb.pt` reproduces the released pipeline's mixing
exactly.

Each guitar's output directory holds `mix.wav` (the six-string sum), `string_{1..6}.wav`
(each string's own decoded audio — mostly near-silent residual on strings the score never
uses), `fed_curves.npz` (the exact pitch/envelope/onset/voiced curves fed to the synth),
and `metadata.json`. `manifest.json` at the output root indexes every rendered guitar.

## What this does and does not demonstrate

Substituting a predicted loudness (stage 1's output) for a real recording's ground-truth
gain does not fully close the audio-quality gap against oracle-gain rendering; this
pipeline demonstrates score-to-audio **inference** on data the model has never seen, not
a claim that predicted-gain rendering matches oracle-gain quality.
