# GuitarPro (.gp3/.gp4/.gp5) -> per-guitar, per-string note lists.
#
# PyGuitarPro (`guitarpro.parse`) only reads gp3/gp4/gp5's binary format, not GP6+'s
# zipped-XML ".gp"/".gpx" container -- data/examples/test.gp is the latter (its
# `Content/score.gpif` member confirms GP8), so this module targets test.gp5, the gp5
# re-export the user supplied instead. A GuitarPro file gives EXACT string/fret ground
# truth per note, so this module never has to infer string identity. A plain MIDI file
# (no exact tab field) goes through midi_score.py's fingering-assignment heuristic
# instead -- see that module's docstring for why a channel-lookup heuristic isn't enough
# for real (non-hex-exported) performance MIDI.
#
# GUITAR DETECTION. GM patch numbers 24-31 (0-indexed) are the guitar family (24/25
# acoustic nylon/steel, 26-31 electric incl. distortion/harmonics) -- covers "either
# acoustic or electric guitar" per the brief. Channel 9 (GM's fixed percussion channel)
# is excluded regardless of program, since a drum kit's Track can carry program 0.
#
# GROUPING. Each PyGuitarPro Track already IS one guitar (all of its strings live in one
# Track object, unlike a MIDI export that can split one guitar's strings across several
# MIDI channels) -- so "detect multiple guitar tracks and group the same guitar" reduces
# to: keep every Track whose program is in the guitar family, and defensively merge any
# two guitar Tracks that share a Name (in case an export ever splits one guitar's strings
# across tracks by name convention rather than this format's own per-Track strings list).
#
# TIES. A GuitarPro "tied" note is not a new pluck: it is the previous note's sustain
# continuing to sound. Counting it as a fresh onset would inject spurious onset/envelope
# impulses into the score-only conditioning the stage-2 predictor reads. Tied beats
# instead extend the still-open note's own duration through the tied beat's end.
#
# TEMPO. This file (like the accompanying test.mid) carries exactly one tempo marking, at
# the first bar -- verified against test.gp's GP8 XML export of the same song (single
# <Automation><Type>Tempo</Type>) and against PyGuitarPro's own MeasureHeader objects
# (which expose no per-measure tempo field for this file, i.e. no tempo automation
# PyGuitarPro's gp5 reader surfaces). A constant-tempo tick->second conversion is exact
# for this file; a genuinely tempo-automated gp5 would need a tempo map, not implemented
# here.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import guitarpro
from guitarpro.models import Duration, NoteType

# 0-indexed GM patch numbers: Acoustic Guitar (nylon/steel), Electric Guitar
# (jazz/clean/muted/overdriven/distortion), Guitar harmonics.
GUITAR_PROGRAMS = frozenset(range(24, 32))
PERCUSSION_CHANNEL = 9
MAX_STRINGS = 6
MIN_NOTE_S = 1e-3  # floor so a zero-length grace/rounding artefact never collapses a span

# PyGuitarPro's fixed internal tick base (quarter note = 960 ticks), read off the library
# itself rather than hardcoded, so this stays correct if that constant ever changes.
QUARTER_NOTE_TICKS = Duration(value=4).time

Note = Tuple[float, float, float]  # (onset_s, duration_s, midi_pitch)


@dataclass
class GuitarPart:
    name: str
    program: int
    instrument_name: str
    # 0-indexed string, low pitch -> high pitch (string 0 = lowest open string), matching
    # this repo's GuitarSet/tab convention (dataset/perf/note_gain_dataset.py, technique_
    # steering.py's NOTE_STRING comment).
    notes_by_string: Dict[int, List[Note]] = field(default_factory=dict)

    @property
    def n_notes(self) -> int:
        return sum(len(v) for v in self.notes_by_string.values())

    @property
    def strings_used(self) -> List[int]:
        return sorted(s for s, notes in self.notes_by_string.items() if notes)


def _is_guitar_track(track) -> bool:
    if track.channel is None:
        return False
    if track.channel.channel == PERCUSSION_CHANNEL:
        return False
    return track.channel.instrument in GUITAR_PROGRAMS


def _extract_track(track) -> GuitarPart:
    # Ascending open-pitch order -> string 0 is the lowest string, matching this repo's
    # convention regardless of the file's own 1-indexed high-to-low string numbering.
    strings_sorted = sorted(track.strings, key=lambda s: s.value)
    if len(strings_sorted) > MAX_STRINGS:
        # Drop the lowest extra string(s) (e.g. a 7-string's low B) -- the synth this
        # feeds is hardcoded to 6 strings throughout the codebase.
        strings_sorted = strings_sorted[-MAX_STRINGS:]
    string_number_to_idx = {s.number: i for i, s in enumerate(strings_sorted)}
    open_pitch = {i: float(s.value) for i, s in enumerate(strings_sorted)}
    sec_per_tick = 60.0 / (float(track.song.tempo) * QUARTER_NOTE_TICKS)

    open_notes: Dict[int, dict] = {}  # string_idx -> still-extendable note dict
    raw: Dict[int, List[dict]] = {}

    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if not beat.notes:
                    continue
                onset_s = beat.start * sec_per_tick
                dur_s = max(beat.duration.time * sec_per_tick, MIN_NOTE_S)
                for note in beat.notes:
                    if note.type == NoteType.dead:
                        continue  # muted/percussive hit: no definite pitch
                    s_idx = string_number_to_idx.get(note.string)
                    if s_idx is None:
                        continue  # note on a dropped extra string
                    if note.type == NoteType.tie:
                        prev = open_notes.get(s_idx)
                        if prev is not None:
                            prev["dur"] = max((onset_s + dur_s) - prev["onset"], MIN_NOTE_S)
                            continue
                        # malformed tie with no open predecessor: fall through, treat as new
                    pitch = open_pitch[s_idx] + float(note.value)
                    entry = {"onset": onset_s, "dur": dur_s, "pitch": pitch}
                    raw.setdefault(s_idx, []).append(entry)
                    open_notes[s_idx] = entry

    notes_by_string = {
        s_idx: sorted(
            ((e["onset"], e["dur"], e["pitch"]) for e in entries), key=lambda n: n[0]
        )
        for s_idx, entries in raw.items()
    }
    name = (track.name or "").strip() or f"track_{track.number}"
    return GuitarPart(
        name=name, program=track.channel.instrument, instrument_name=name,
        notes_by_string=notes_by_string,
    )


def load_guitar_parts(path: str) -> List[GuitarPart]:
    """Every guitar (acoustic or electric) track in a GuitarPro file, grouped by name,
    ignoring bass/drums/vocals/other instruments."""
    song = guitarpro.parse(str(Path(path)))
    parts = [_extract_track(t) for t in song.tracks if _is_guitar_track(t)]

    grouped: Dict[str, GuitarPart] = {}
    for part in parts:
        if part.name not in grouped:
            grouped[part.name] = part
            continue
        # Same-named guitar split across >1 Track (defensive; not exercised by test.gp5,
        # whose two guitars already carry distinct names) -- union their per-string notes.
        target = grouped[part.name]
        for s_idx, notes in part.notes_by_string.items():
            merged = sorted(target.notes_by_string.get(s_idx, []) + notes, key=lambda n: n[0])
            target.notes_by_string[s_idx] = merged
    return list(grouped.values())
