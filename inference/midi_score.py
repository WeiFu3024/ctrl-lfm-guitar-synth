# Plain MIDI (single track, no per-string channel separation) -> per-guitar, per-string
# note lists, matching gp_score.py's GuitarPart output exactly.
#
# WHY THIS EXISTS. gp_score.py's own header used to say a MIDI fallback had "nothing to
# add" for this tool, because the only MIDI this repo had tested against
# (data/examples/test.mid) turned out to be a "hex" export -- one MIDI channel per
# string, verified in docs/reports/active/oneoff-score-inference-demo.md. data/GAPS's
# performance MIDI (data/GAPS/midi/*.mid) is a different, more common shape: one track,
# one channel, real polyphony (chords/overlapping sustain), no string information at all
# (verified: 003_HSswc.mid has notes 59 and 62 sounding simultaneously for 7s). A
# channel-lookup heuristic has nothing to look up here -- this module instead solves the
# actual "which string plays this note" problem: guitar tablature/fingering assignment.
#
# ALGORITHM. Standard 6-string tuning gives each note 1-3 candidate (string, fret) pairs
# (most pitches are reachable on more than one string). Notes are grouped into "chords" by
# onset proximity (CHORD_EPS_S) and solved JOINTLY per group -- greedy note-by-note
# assignment would let two simultaneous notes both grab the same string. Within a group, a
# small combinatorial search (at most 6 notes x 6 strings, trivially fast) picks the
# assignment that minimises total distance from a running "hand position" (an EMA over
# already-assigned fret positions, so the estimate follows the melody smoothly instead of
# snapping to each note) plus a small penalty on the chord's fret span (discourages
# fingerings wider than a hand can stretch). A string already sounding a note through the
# new note's onset is excluded as a candidate -- one string plays one note at a time.
#
# FALLBACKS (all counted and reported, never silent): a pitch reachable but with every
# candidate string currently busy relaxes the busy constraint (a real player would mute
# early); a pitch unreachable on every string (outside MIDI_MIN..MIDI_MAX's fret-0..24
# window) clips to the nearest boundary string/fret, matching build_score_cond's own
# clipping convention; a chord bigger than 6 notes (more simultaneous pitches than
# strings -- not physically playable, and not expected from real guitar transcription)
# drops whichever notes the search cannot place.
#
# TRIMMING. Real performance MIDI carries the recording's own lead-in silence before the
# first note (data/GAPS ranges from 0.07s to 19.7s across a random sample) -- left in,
# this wastes the model's frame budget rendering silence. Every note's onset is shifted so
# the earliest note in a part starts at t=0. No symmetric trailing trim is needed: this
# module (like gp_score.py) builds notes_by_string purely from note events, and
# frame_count_for_notes (pipeline.py) already ends exactly at the last note's own end --
# there is no "tail" past the last note for a note-event representation to carry in the
# first place, regardless of what silence follows the last note in the source file.
from collections import defaultdict
from typing import Dict, List, Tuple

from inference.gp_score import GUITAR_PROGRAMS, MAX_STRINGS, MIN_NOTE_S, GuitarPart, Note

# Standard 6-string tuning open-string MIDI pitches, low->high (string 0 = lowest, matching
# gp_score.py's "string 0 = lowest open string" convention): E2 A2 D3 G3 B3 E4. Cross-
# checked against technique_steering.py's NOTE_STRING comment and against
# docs/reports/active/oneoff-score-inference-demo.md's independent `mido` read of
# test.mid's 6 hex channels (disjoint pitch bands 40-43,45-50,50-55,55-60,59-62,64-67 --
# exactly these 6 open pitches in the same low-to-high order).
STANDARD_TUNING_MIDI = [40, 45, 50, 55, 59, 64]
MAX_FRET = 24  # matches dataset/perf/perf_curve_dataset.py's MIDI_MAX=88 (=64+24)

CHORD_EPS_S = 0.02       # onsets within this fall into one jointly-solved chord group
HAND_EMA_ALPHA = 0.3     # how fast the tracked hand position follows newly played frets
SPREAD_WEIGHT = 0.3      # per-fret penalty on a chord's (max_fret - min_fret) span
TIE_EPS_S = 1e-6         # float tolerance for "this string is free exactly at this onset"


def _group_chords(notes: List[Note], eps_s: float) -> List[List[Note]]:
    """Onset-sorted notes -> groups of near-simultaneous notes, each solved jointly so two
    notes in one chord never compete greedily for the same string."""
    ordered = sorted(notes, key=lambda n: n[0])
    groups, i = [], 0
    while i < len(ordered):
        t0 = ordered[i][0]
        j = i + 1
        while j < len(ordered) and ordered[j][0] <= t0 + eps_s:
            j += 1
        groups.append(ordered[i:j])
        i = j
    return groups


def _solve_chord_assignment(candidates: List[List[Tuple[int, int]]], hand_position: float,
                             spread_weight: float):
    """Best (string, fret) per note in one chord group, each candidates[i] a list of
    (string, fret) options for note i. Maximises how many notes get a string (a chord
    bigger than 6 notes cannot place them all), then minimises total |fret-hand_position|
    plus a span penalty, via small backtracking search (n<=~8, branching<=6 -- exhaustive
    is cheap) ordered smallest-candidate-set-first to prune early."""
    n = len(candidates)
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    best = {"n_assigned": -1, "cost": float("inf"), "assign": [None] * n}
    assign: List[Tuple[int, int]] = [None] * n
    used = set()

    def backtrack(k: int, n_assigned: int, cost_so_far: float, frets: List[int]):
        best_possible_n = n_assigned + (n - k)
        if best_possible_n < best["n_assigned"]:
            return
        if best_possible_n == best["n_assigned"] and cost_so_far >= best["cost"]:
            return
        if k == n:
            span = (max(frets) - min(frets)) if len(frets) > 1 else 0
            total_cost = cost_so_far + spread_weight * span
            if n_assigned > best["n_assigned"] or (n_assigned == best["n_assigned"] and total_cost < best["cost"]):
                best["n_assigned"] = n_assigned
                best["cost"] = total_cost
                best["assign"] = assign.copy()
            return
        i = order[k]
        for s, fret in candidates[i]:
            if s in used:
                continue
            used.add(s)
            assign[i] = (s, fret)
            backtrack(k + 1, n_assigned + 1, cost_so_far + abs(fret - hand_position), frets + [fret])
            used.discard(s)
        assign[i] = None
        backtrack(k + 1, n_assigned, cost_so_far, frets)

    backtrack(0, 0, 0.0, [])
    return best["assign"]


def assign_strings(notes: List[Note], tuning: List[int] = STANDARD_TUNING_MIDI, max_fret: int = MAX_FRET,
                    chord_eps_s: float = CHORD_EPS_S, hand_alpha: float = HAND_EMA_ALPHA,
                    spread_weight: float = SPREAD_WEIGHT, tie_eps_s: float = TIE_EPS_S):
    """Flat (possibly polyphonic) note list -> {string_idx: [Note, ...]}, plus a fallback-
    count dict for reporting (see module docstring for what each counter means)."""
    n_strings = len(tuning)
    string_free_at = [float("-inf")] * n_strings
    hand_position = 0.0
    raw: Dict[int, List[Note]] = defaultdict(list)
    stats = {"forced_overlap": 0, "clipped_out_of_range": 0, "dropped_unplayable": 0}

    for group in _group_chords(notes, chord_eps_s):
        candidates = []
        for onset, _dur, pitch in group:
            candidates.append([
                (s, int(round(pitch - tuning[s]))) for s in range(n_strings)
                if 0 <= pitch - tuning[s] <= max_fret and onset >= string_free_at[s] - tie_eps_s
            ])
        for i, (_onset, _dur, pitch) in enumerate(group):
            if candidates[i]:
                continue
            relaxed = [(s, int(round(pitch - tuning[s]))) for s in range(n_strings)
                       if 0 <= pitch - tuning[s] <= max_fret]
            if relaxed:
                candidates[i] = relaxed
                stats["forced_overlap"] += 1
            else:
                s = 0 if pitch < tuning[0] else n_strings - 1
                fret = 0 if pitch < tuning[0] else max_fret
                candidates[i] = [(s, fret)]
                stats["clipped_out_of_range"] += 1

        assign = _solve_chord_assignment(candidates, hand_position, spread_weight)
        assigned_frets = []
        for i, (onset, dur, pitch) in enumerate(group):
            if assign[i] is None:
                stats["dropped_unplayable"] += 1
                continue
            s, fret = assign[i]
            raw[s].append((onset, max(dur, MIN_NOTE_S), pitch))
            string_free_at[s] = onset + dur
            assigned_frets.append(fret)
        if assigned_frets:
            hand_position = (1 - hand_alpha) * hand_position + hand_alpha * (sum(assigned_frets) / len(assigned_frets))

    notes_by_string = {s: sorted(v, key=lambda n: n[0]) for s, v in raw.items()}
    return notes_by_string, stats


def load_guitar_parts(path: str) -> List[GuitarPart]:
    """Every guitar (acoustic or electric) instrument in a plain MIDI file, grouped by
    name like gp_score.load_guitar_parts, with per-string notes reconstructed by
    assign_strings instead of read from an exact tab field."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    grouped: Dict[str, List[Note]] = defaultdict(list)
    program_by_name: Dict[str, int] = {}
    for i, inst in enumerate(pm.instruments):
        if inst.is_drum or inst.program not in GUITAR_PROGRAMS:
            continue
        name = (inst.name or "").strip() or f"track_{i}"
        grouped[name].extend((n.start, n.end - n.start, float(n.pitch)) for n in inst.notes)
        program_by_name.setdefault(name, int(inst.program))

    parts = []
    for name, notes in grouped.items():
        if not notes:
            continue
        t0 = min(onset for onset, _dur, _pitch in notes)
        shifted = [(onset - t0, dur, pitch) for onset, dur, pitch in notes]
        notes_by_string, stats = assign_strings(shifted)
        if any(stats.values()):
            print(f"midi_score: {name!r} string-assignment fallbacks -- {stats}", flush=True)
        if len(notes_by_string) > MAX_STRINGS:
            # tuning has exactly MAX_STRINGS entries, so this cannot happen; guard anyway
            # rather than silently feeding the synth an unsupported string count.
            raise ValueError(f"{name!r}: assigned {len(notes_by_string)} strings, expected <= {MAX_STRINGS}")
        parts.append(GuitarPart(
            name=name, program=program_by_name[name], instrument_name=name,
            notes_by_string=notes_by_string,
        ))
    return parts
