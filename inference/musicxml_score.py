# MusicXML with an explicit guitar-tab staff -> per-guitar, per-string note lists,
# matching gp_score.py's GuitarPart output exactly.
#
# WHY THIS EXISTS. data/GAPS ships two representations of the same performance: MIDI
# (midi_score.py's target, no string information at all) and MusicXML. The MusicXML is
# NOT a single staff -- every GAPS file examined (003_HSswc, 001_mvswc, 399_g41wc) has
# exactly two <part> elements sharing one instrument name (e.g. "Steel Guitar" x2): one
# plain notation staff with zero <technical><fret>/<string> tags, and one tab staff where
# EVERY note (100% coverage, not the ~50% a naive whole-file count suggests before
# noticing the duplicate part) carries the real fret and string a guitarist actually
# played. Reading the tab part directly gives GuitarPro-level exact ground truth --
# strictly better than midi_score.py's fingering heuristic for this data, and simpler:
# no combinatorial search, because the string is a fact, not something to infer.
#
# WHY music21, NOT ELEMENTTREE. A tab part's own note sequence still needs the same
# things PyGuitarPro already resolves for gp_score.py: multi-voice interleaving (backup/
# forward elements), tie-into-duration merging, and chord grouping. Hand-rolling that
# against raw MusicXML would re-implement music21's own note-flattening for no benefit;
# music21.converter.parse + Part.flatten() already resolves voices/backup/forward into one
# offset-correct timeline.
#
# READING PITCH FROM <pitch>, NOT FROM fret+open-string. Cross-checked <pitch>-derived
# MIDI numbers against standard-tuning-implied pitch (open_string[string] + fret) across
# 2505 notes/chord-tones in 3 files: 2494 matched exactly, but 11 (all in one file,
# 399_g41wc.xml) were consistently 2 semitones flat -- that file uses a non-standard
# (likely drop-tuned) instrument. <pitch> is always the ground truth; deriving pitch from
# fret+string would silently assume standard tuning and get exactly those 11 notes wrong.
# <string> is still needed (and used) to pick which of this repo's 6 per-guitar string
# slots a note belongs to.
#
# CHORD ARTICULATION ORDER. music21 attaches a Chord's per-note <technical> marks to the
# CHORD object's own .articulations as a flat [Fret, String, Fret, String, ...] list, not
# to the individual constituent Note objects (verified empirically: chord-tone
# Note.articulations is always empty). Cross-checked that this list's order matches
# Chord.pitches's order via the same standard-tuning cross-check above (a length or order
# mismatch would have shown up as scattered, not one-file-consistent, disagreements) --
# _element_voices asserts the lengths match as a live guard against a file where it
# doesn't.
#
# ACCURACY AGAINST THE REAL PERFORMANCE (data/GAPS/midi/, the same pieces' audio-aligned
# transcription). Systematically cross-checked note span/count/pitch-content against the
# matching MIDI file, 79-120 files at a time (see _build_tempo_map and _expand_repeats'
# own docstrings for the two real bugs this found and fixed: MetronomeMark referent
# scaling, and unexpanded repeat barlines). After both fixes, an 80-file random sample
# categorised as: 20 (25%) matching MIDI span/count/pitch within 10-15%; 10 (13%) where
# music21's expandRepeats() itself fails on a real repeat structure (all had <ending>
# volta markings, e.g. a combined "2, 3" ending spanning two passes -- now a printed
# warning, not a silent truncation); the remainder mostly NOT a parsing defect --
# verified on 112_mf1wc.xml (810 tab notes, exactly matching the raw XML's own note
# count) that its MIDI performance's pitch sequence is the XML's note sequence repeated
# twice back-to-back with no encoded repeat marking of any kind (no barline, no <ending>,
# no D.C./Fine text direction) -- the performer played the whole piece twice for reasons
# the score does not encode. This module reads the score as notated; it cannot recover a
# structural choice the notated score never represented in the first place.
from collections import defaultdict
from typing import Dict, List, Tuple

import music21

from inference.gp_score import MAX_STRINGS, MIN_NOTE_S, GuitarPart, Note

TAB_COVERAGE_THRESHOLD = 0.95  # fraction of pitches needing a <string> mark to count as a tab part


def _string_indications(element) -> List[int]:
    return [a.number for a in element.articulations if type(a).__name__ == "StringIndication"]


def _tab_coverage(part) -> float:
    total, with_string = 0, 0
    for el in part.flatten().getElementsByClass(("Note", "Chord")):
        n_pitches = len(el.pitches) if isinstance(el, music21.chord.Chord) else 1
        total += n_pitches
        with_string += min(n_pitches, len(_string_indications(el)))
    return with_string / total if total else 0.0


def _element_voices(element):
    """One (pitch, 1-indexed string number, tie-or-None) tuple per note this element
    sounds -- one for a plain Note, one per constituent pitch for a Chord."""
    strings = _string_indications(element)
    if isinstance(element, music21.chord.Chord):
        pitches, ties = list(element.pitches), [n.tie for n in element.notes]
        assert len(strings) == len(pitches), (
            f"chord at offset {element.offset}: {len(pitches)} pitches but {len(strings)} "
            f"<string> marks -- _element_voices's pairing assumption doesn't hold for this file"
        )
        return list(zip(pitches, strings, ties))
    assert len(strings) == 1, f"note at offset {element.offset}: expected exactly 1 <string> mark, got {len(strings)}"
    return [(element.pitch, strings[0], element.tie)]


def _build_tempo_map(score):
    """Sorted (offset_in_quarters, cumulative_seconds_at_that_offset, bpm_from_there)
    breakpoints, deduplicated (both parts of a GAPS file carry their own copy of every
    tempo direction at the same offset). Real classical-guitar scores DO change tempo
    mid-piece (e.g. 001_mvswc.xml: 190bpm -> 300bpm at offset 328.5) -- a single constant-
    tempo assumption (gp_score.py's, valid there because that file has exactly one
    marking) would silently mis-time every note after such a change here.

    BUG FOUND BY CROSS-CHECKING AGAINST data/GAPS/midi/ (the same pieces' real performed
    timing): MetronomeMark.number is not always quarter-notes-per-minute -- it is
    relative to `.referent`, the note value the marking is actually written against
    (e.g. "dotted quarter = 190" in a 6/8 piece, referent.quarterLength=1.5). Reading
    .number alone without scaling by .referent.quarterLength silently assumed every mark
    means quarter-notes-per-minute; for 001_mvswc.xml (3/8, marked eighth=190,
    referent.quarterLength=0.5) that produced a 108.3s render for a piece whose own MIDI
    performance runs 227.8s (short by very close to the 2x this bug implies). A random
    25-file sample found 2 (8%) with a non-1.0 referent -- not a one-file edge case."""
    marks = sorted({
        (float(m.offset), float(m.number) * float(m.referent.quarterLength))
        for m in score.flatten().getElementsByClass("MetronomeMark")
    })
    if not marks:
        raise ValueError("no tempo marking found in this MusicXML file")
    offsets = [o for o, _ in marks]
    bpms = [b for _, b in marks]
    cum_seconds = [0.0]
    for i in range(1, len(offsets)):
        cum_seconds.append(cum_seconds[-1] + (offsets[i] - offsets[i - 1]) * 60.0 / bpms[i - 1])
    return offsets, bpms, cum_seconds


def _seconds_at(offset_ql: float, offsets: List[float], bpms: List[float], cum_seconds: List[float]) -> float:
    import bisect
    i = max(bisect.bisect_right(offsets, offset_ql) - 1, 0)
    return cum_seconds[i] + (offset_ql - offsets[i]) * 60.0 / bpms[i]


def _guess_program(part_name: str) -> int:
    # midiProgram is never encoded in the GAPS files examined; these names are always
    # some flavour of acoustic guitar, so pick GM's nylon/steel acoustic split by name --
    # cosmetic metadata only (render_guitar_part's synth input never reads .program).
    return 25 if "steel" in part_name.lower() else 24


def _extract_tab_part(part, tempo_map) -> Dict[int, List[Note]]:
    offsets, bpms, cum_seconds = tempo_map
    open_notes: Dict[int, dict] = {}  # string_idx -> still-extendable note dict (ties)
    raw: Dict[int, List[dict]] = defaultdict(list)

    elements = sorted(part.flatten().getElementsByClass(("Note", "Chord")), key=lambda e: e.offset)
    for el in elements:
        onset_s = _seconds_at(el.offset, offsets, bpms, cum_seconds)
        end_s = _seconds_at(el.offset + el.duration.quarterLength, offsets, bpms, cum_seconds)
        for pitch, string_number, tie in _element_voices(el):
            # GAPS is 1-indexed high-to-low (string 1 = high e); this repo's convention is
            # 0-indexed low-to-high (string 0 = low E) -- gp_score.py's own convention note.
            string_idx = MAX_STRINGS - string_number
            if tie is not None and tie.type in ("stop", "continue") and string_idx in open_notes:
                open_notes[string_idx]["dur"] = max(end_s - open_notes[string_idx]["onset"], MIN_NOTE_S)
                if tie.type == "stop":
                    open_notes.pop(string_idx, None)
                continue
            entry = {"onset": onset_s, "dur": max(end_s - onset_s, MIN_NOTE_S), "pitch": float(pitch.midi)}
            raw[string_idx].append(entry)
            if tie is not None and tie.type == "start":
                open_notes[string_idx] = entry
            else:
                open_notes.pop(string_idx, None)

    return {s: sorted(((e["onset"], e["dur"], e["pitch"]) for e in entries), key=lambda n: n[0])
            for s, entries in raw.items()}


def _expand_repeats(score, path: str):
    """Real performances play through repeat barlines; a bare Part.flatten() reads the
    score in notated order and does not. BUG FOUND BY CROSS-CHECKING AGAINST data/GAPS/
    midi/ (this module's own module docstring explains why that check exists): a random
    43-file sample found 34 (79%) with a note-span or note-count ratio against their own
    MIDI performance outside 0.9-1.1 -- e.g. 029_tpswc.xml at 0.56x span / 0.50x notes,
    044_Zc1wc.xml at 0.53x/0.51x. All had simple start/end repeat barlines (verified:
    029_tpswc.xml has 2 repeated sections; music21's own expandRepeats() roughly doubles
    its note count, from 456 to 912 elements). Confirmed harmless on a file with ZERO
    repeat marks (001_mvswc.xml): identical pitch count (1868) before and after.

    NOT every repeat structure is machine-expandable: music21 raises ExpanderException
    ("badly formed repeats or repeat expressions") for some files with real repeat marks
    (observed on 167_KM1wc.xml, 191_bk1wc.xml -- both still show <=0.5x span/note-count
    against their own MIDI after this fix, i.e. genuinely still repeat-truncated, not
    rubato). Falling back silently would produce a too-short render with no signal
    anything was wrong; this prints a warning instead so it is at least visible per file
    rather than a hidden accuracy gap."""
    try:
        return score.expandRepeats()
    except Exception as e:
        print(f"musicxml_score: {path!r} repeat expansion failed ({type(e).__name__}: {e}) -- "
              f"falling back to the unexpanded (possibly repeat-truncated) reading", flush=True)
        return score


def load_guitar_parts(path: str) -> List[GuitarPart]:
    """Every guitar tab staff in a MusicXML file (see module docstring for how a tab
    staff is identified) -> a GuitarPart with exact, not inferred, per-string notes."""
    score = _expand_repeats(music21.converter.parse(path), path)
    tempo_map = _build_tempo_map(score)

    parts = []
    for part in score.parts:
        if _tab_coverage(part) < TAB_COVERAGE_THRESHOLD:
            continue  # a plain notation staff (or anything else without real tab data)
        name = (part.partName or "").strip() or f"part_{len(parts)}"
        notes_by_string = _extract_tab_part(part, tempo_map)
        parts.append(GuitarPart(
            name=name, program=_guess_program(name), instrument_name=name,
            notes_by_string=notes_by_string,
        ))
    return parts
