"""Beat-level chord recognition via dynamic programming.

Adapted from music-x-lab/midi-chord-recognition (Jiang, 2025), which implements
the rule-based MIDI chord recognition system described in:

    - Wang et al., "Pop909," ISMIR 2020.
    - Wang et al., "Learning Interpretable Representation," ISMIR 2020.
    - Dai et al., "CSCMC-MuMe," 2020.

The algorithm:
    1. Extract beat/downbeat positions from MIDI tempo map.
    2. Quantise notes to the beat grid → per-beat 12-dim treble chroma + bass chroma.
    3. Channel-weighted aggregation (thickness + bass reweighting).
    4. Score each chord template per beat (with bass bonus).
    5. Dynamic-programming decode with span-length reward and transition penalty.
    6. Output interval-level chord labels: ``[(start, end, label), ...]``.

References:
    - GitHub: https://github.com/music-x-lab/midi-chord-recognition
    - FGG (Zhu et al., ICML 2025): Direct Chord Accuracy uses this method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

__all__ = [
    "recognize_chords",
    "recognize_chords_beat",
    "ChordRecognitionDP",
]

# ── Constants ─────────────────────────────────────────────────────

_PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_SUBBEAT_COUNT = 8  # sub-beat resolution for bass detection
_MAX_PREV = 12      # max lookback in DP decoder (beats)

# ── Chord template library ────────────────────────────────────────

# Chroma templates from music-x-lab chord_class.py
# Each value is a 12-element binary vector: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
# positions:                                C  C# D  Eb E  F  F# G  Ab A  Bb B
_QUALITIES: dict[str, list[int]] = {
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
}

# Qualities that support inversions (bass note != root)
_INVERSIONS: dict[str, list[str]] = {
    'maj':  ['3', '5'],
    'min':  ['b3', '5'],
    '7':    ['3', '5', 'b7'],
    'maj7': ['3', '5', '7'],
    'min7': ['5', 'b7'],
}

# Inversion bass-note offsets (scale degrees from root)
_INVERSION_OFFSETS: dict[str, int] = {
    '1': 0, 'b2': 1, '2': 2, 'b3': 3, '3': 4, '4': 5,
    'b5': 6, '5': 7, '#5': 8, '6': 9, 'b7': 10, '7': 11,
}


@dataclass(frozen=True, slots=True)
class ChordInterval:
    """A chord label with time interval."""
    start: float   # start time in seconds
    end: float     # end time in seconds
    label: str     # e.g. 'C:maj', 'A:min7', 'N'


class _ChordTemplates:
    """Pre-built chord template library with bass templates.

    Generates all chord candidates:
        - N (no chord)
        - 12 roots × |qualities| (root-position chords)
        - 12 roots × |inversions| (inverted chords for selected qualities)
    """

    def __init__(self) -> None:
        bass_template = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        empty_template = np.zeros(12)

        self.chord_list: list[str] = ['N']
        self.chroma_templates: list[np.ndarray] = [empty_template.copy()]
        self.bass_templates: list[np.ndarray] = [empty_template.copy()]

        for root in range(12):
            for qname, intervals in _QUALITIES.items():
                template = np.roll(np.array(intervals, dtype=np.float64), root)
                bass = np.roll(bass_template.copy(), root)
                label = f'{_PITCH_NAMES[root]}:{qname}'
                self.chord_list.append(label)
                self.chroma_templates.append(template)
                self.bass_templates.append(bass)

                # Add inversions for supported qualities
                if qname in _INVERSIONS:
                    for inv_name in _INVERSIONS[qname]:
                        inv_offset = _INVERSION_OFFSETS.get(inv_name, 0)
                        inv_label = f'{_PITCH_NAMES[root]}:{qname}/{inv_name}'
                        inv_bass = np.roll(bass_template.copy(), (root + inv_offset) % 12)
                        self.chord_list.append(inv_label)
                        self.chroma_templates.append(template.copy())
                        self.bass_templates.append(inv_bass)

        self.chroma_arr = np.array(self.chroma_templates)
        self.bass_arr = np.array(self.bass_templates)
        self.n_chords = len(self.chord_list)

    def get_length(self) -> int:
        return self.n_chords

    def score(self, chroma: np.ndarray, bass_chroma: np.ndarray) -> np.ndarray:
        """Score all chord candidates for one beat.

        Scoring formula (from music-x-lab chord_class.py):
            score = (chroma[template>0].sum() - chroma[template==0].sum()) / n_template_notes
                  + 0.5 * bass_chroma[bass_template>0].sum()
                  - n_template_notes * 0.1
                  - 0.05 if inversion

        Args:
            chroma: 12-dim treble chroma vector.
            bass_chroma: 12-dim bass chroma vector.

        Returns:
            Array of shape ``(n_chords,)`` with per-chord scores.
        """
        result = np.zeros(self.n_chords, dtype=np.float64)

        for i, label in enumerate(self.chord_list):
            if label == 'N':
                result[i] = 0.2
                continue

            ref = self.chroma_arr[i]
            ref_bass = self.bass_arr[i]
            n_pos = int(ref.sum())
            if n_pos == 0:
                continue

            # Treble match: reward template hits, penalise non-template energy
            treble_match = float(chroma[ref > 0].sum() - chroma[ref == 0].sum()) / n_pos
            # Bass bonus
            bass_match = 0.5 * float(bass_chroma[ref_bass > 0].sum())
            # Complexity penalty
            penalty = n_pos * 0.1 + (0.05 if '/' in label else 0.0)

            result[i] = treble_match + bass_match - penalty

        return result

    def batch_score(self, chromas: np.ndarray, bass_chromas: np.ndarray) -> np.ndarray:
        """Batch-score multiple beats at once.

        Args:
            chromas: ``(n_beats, 12)`` treble chroma matrix.
            bass_chromas: ``(n_beats, 12)`` bass chroma matrix.

        Returns:
            ``(n_beats, n_chords)`` score matrix.
        """
        n = chromas.shape[0]
        result = np.zeros((n, self.n_chords), dtype=np.float64)

        for i, label in enumerate(self.chord_list):
            if label == 'N':
                result[:, i] = 0.2
                continue

            ref = self.chroma_arr[i]
            ref_bass = self.bass_arr[i]
            n_pos = int(ref.sum())
            if n_pos == 0:
                continue

            treble = (chromas[:, ref > 0].sum(axis=1) - chromas[:, ref == 0].sum(axis=1)) / n_pos
            bass = 0.5 * bass_chromas[:, ref_bass > 0].sum(axis=1)
            penalty = n_pos * 0.1 + (0.05 if '/' in label else 0.0)

            result[:, i] = treble + bass - penalty

        return result


# ── MIDI → beat-aligned features ─────────────────────────────────

def _get_beats_and_downbeats(midi_path) -> tuple[np.ndarray, np.ndarray]:
    """Extract beat times and downbeat times using pretty_midi."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    beats = np.array(pm.get_beats())
    downbeats = np.array(pm.get_downbeats())
    return beats, downbeats


def _quantize_time(time: float, beat_onsets: np.ndarray, beat_offsets: np.ndarray,
                   n_frame: int) -> float:
    """Quantize a time value to fractional beat position."""
    if time <= beat_onsets[0]:
        return 0.0
    if time >= beat_offsets[-1]:
        return float(n_frame)
    idx = int(np.searchsorted(beat_onsets, time, side='right') - 1)
    idx = max(0, min(idx, n_frame - 1))
    length = beat_offsets[idx] - beat_onsets[idx]
    if length <= 0:
        return float(idx)
    return idx + (time - beat_onsets[idx]) / length


def _extract_beat_features(midi_path, extra_division: int = 2) -> tuple:
    """Extract beat-aligned treble chroma and bass chroma features.

    This is a Python port of music-x-lab/midi-chord-recognition's
    ``ChordRecognition.process_feature()`` method.

    Args:
        midi_path: Path to a MIDI file.
        extra_division: Sub-beat divisions (2 = half-beat, default).

    Returns:
        ``(beat_chroma, beat_bass, qt_beat_onset, qt_beat_offset, beat_array)``
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    raw_beats = np.asarray(pm.get_beats())
    downbeats = np.asarray(pm.get_downbeats())

    if len(raw_beats) < 2:
        return np.zeros((0, 12)), np.zeros((0, 12)), np.array([]), np.array([]), np.array([])

    # Subdivide beats if extra_division > 1
    if extra_division > 1:
        beat_interp = np.linspace(raw_beats[:-1], raw_beats[1:], extra_division + 1).T
        last_beat = beat_interp[-1, -1]
        beats = np.append(beat_interp[:, :-1].reshape(-1), last_beat)
    else:
        beats = raw_beats

    # Build beat array with position info (1=downbeat, 2=beat2, etc.)
    j = 0
    beat_pos = -2
    beat_arr = []
    for i in range(len(beats)):
        if j < len(downbeats) and abs(beats[i] - downbeats[j]) < 1e-6:
            beat_pos = 1
            j += 1
        else:
            beat_pos = beat_pos + 1
        beat_arr.append([beats[i], beat_pos])
    beat_arr = np.array(beat_arr)

    n_frame = len(beat_arr)

    # Compute quantised beat onset/offset/length
    qt_beat_onset = np.zeros(n_frame)
    qt_beat_offset = np.zeros(n_frame)
    qt_beat_length = np.zeros(n_frame)

    for i in range(n_frame):
        qt_beat_onset[i] = beat_arr[i, 0]
        if i == 0:
            qt_beat_offset[i] = beat_arr[i, 0]
        elif i == n_frame - 1:
            qt_beat_offset[i] = beat_arr[i, 0] + (beat_arr[i, 0] - beat_arr[i - 1, 0])
        else:
            qt_beat_offset[i] = beat_arr[i + 1, 0]
        qt_beat_length[i] = (
            beat_arr[i + 1, 0] - beat_arr[i, 0]
            if i < n_frame - 1
            else qt_beat_length[i - 1] if i > 0 else 0.5
        )

    beat_chroma = np.zeros((n_frame, 12), dtype=np.float64)
    beat_bass = np.zeros((n_frame, 12), dtype=np.float64)
    min_subbeat_bass = np.full(n_frame * _SUBBEAT_COUNT, 259, dtype=np.int32)

    def quantize(time: float) -> float:
        return _quantize_time(time, qt_beat_onset, qt_beat_offset, n_frame)

    def clamp(qstart: float, qend: float, bstart: int, bend: int) -> float:
        return min(float(bend), qend) - max(float(bstart), qstart)

    # Process all non-drum instruments with uniform channel weights
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            beat_start = quantize(note.start)
            beat_end = quantize(note.end)
            left_beat = int(math.floor(beat_start + 0.2))
            right_beat = int(math.ceil(beat_end - 0.2))
            if right_beat < left_beat:
                right_beat = left_beat
            left_beat = max(0, left_beat)
            right_beat = min(n_frame, right_beat)

            # Sub-beat bass tracking
            left_subbeat = int(math.floor(beat_start * _SUBBEAT_COUNT + 0.2))
            right_subbeat = int(math.floor(beat_end * _SUBBEAT_COUNT + 0.2))
            left_subbeat = max(0, left_subbeat)
            right_subbeat = min(n_frame * _SUBBEAT_COUNT, right_subbeat)

            for sb in range(left_subbeat, right_subbeat):
                min_subbeat_bass[sb] = min(min_subbeat_bass[sb], note.pitch)

            # Beat chroma accumulation
            for b in range(left_beat, right_beat):
                overlap = clamp(beat_start, beat_end, b, b + 1)
                if overlap > 0:
                    beat_chroma[b, note.pitch % 12] = max(
                        beat_chroma[b, note.pitch % 12], overlap
                    )

    # Convert sub-beat bass to beat bass
    for i in range(n_frame):
        update_terms = min_subbeat_bass[i * _SUBBEAT_COUNT: (i + 1) * _SUBBEAT_COUNT]
        valid = update_terms < 259
        if valid.any():
            for v in update_terms[valid]:
                beat_bass[i, int(v) % 12] += 1.0 / _SUBBEAT_COUNT

    return beat_chroma, beat_bass, qt_beat_onset, qt_beat_offset, beat_arr


# ── DP Decoder ────────────────────────────────────────────────────

class ChordRecognitionDP:
    """Beat-level chord recognition engine using dynamic programming.

    Port of music-x-lab/midi-chord-recognition's ``ChordRecognition`` class.

    Args:
        extra_division: Sub-beat divisions for feature extraction (default 2).
        use_transition: Whether to apply transition penalties in DP (default True).
    """

    def __init__(self, extra_division: int = 2, use_transition: bool = True) -> None:
        self.extra_division = extra_division
        self.use_transition = use_transition
        self.templates = _ChordTemplates()

    def recognize(self, midi_path) -> list[ChordInterval]:
        """Run chord recognition on a MIDI file.

        Args:
            midi_path: Path to a MIDI file (string or Path).

        Returns:
            List of :class:`ChordInterval` with ``(start, end, label)`` in seconds.
        """
        # Step 1: Extract features
        beat_chroma, beat_bass, qt_onset, qt_offset, beat_arr = _extract_beat_features(
            midi_path, self.extra_division
        )

        n_frame = len(beat_arr)
        if n_frame == 0:
            return []

        n_class = self.templates.n_chords
        is_downbeat = beat_arr[:, 1] == 1

        # Determine max beat position for half-downbeat detection
        max_beat_pos = int(beat_arr[:, 1].max())
        is_halfdownbeat = beat_arr[:, 1] * 2 - 2 == max_beat_pos
        is_even_beat = beat_arr[:, 1] % 2 == 1

        # Step 2: Batch scoring
        batch_chroma = np.zeros((n_frame, _MAX_PREV, 12), dtype=np.float64)
        batch_bass = np.zeros((n_frame, _MAX_PREV, 12), dtype=np.float64)

        for i in range(n_frame):
            for j in range(_MAX_PREV):
                if i - j < 0:
                    continue
                batch_chroma[i, j] = beat_chroma[i - j: i + 1].sum(axis=0)
                batch_bass[i, j] = beat_bass[i - j: i + 1].sum(axis=0)

        batch_scores = self.templates.batch_score(
            batch_chroma.reshape(-1, 12),
            batch_bass.reshape(-1, 12),
        ).reshape(n_frame, _MAX_PREV, n_class)

        # Step 3: Build observation matrix with span bonuses
        obs = np.full((n_frame, _MAX_PREV, n_class), -np.inf, dtype=np.float64)
        for i in range(n_frame):
            for j in range(_MAX_PREV):
                if i - j < 0:
                    continue
                if self.use_transition:
                    obs[i, j] = (
                        batch_scores[i, j]
                        + j * 0.7
                        + is_halfdownbeat[i - j] * 0.15
                        + is_even_beat[i - j] * 0.2
                    )
                else:
                    obs[i, j] = batch_scores[i, j]

        # Step 4: DP decode
        dp = np.full(n_frame, -np.inf, dtype=np.float64)
        prec = np.zeros(n_frame, dtype=np.int32)   # best chord at each frame
        prei = np.zeros(n_frame, dtype=np.int32)   # best predecessor frame

        for i in range(n_frame):
            for j in range(_MAX_PREV):
                if i - j < 0:
                    continue
                best_c = int(np.argmax(obs[i, j]))
                prev_score = 0.0 if (i - j) == 0 else dp[i - j - 1]
                candidate = prev_score + obs[i, j, best_c]
                if candidate > dp[i]:
                    dp[i] = candidate
                    prec[i] = best_c
                    prei[i] = i - j - 1
                # Stop extending span across downbeat boundary
                if j > 0 and i - j + 1 < n_frame and is_downbeat[i - j + 1]:
                    break

        # Step 5: Backtrack
        current_i = n_frame - 1
        result: list[ChordInterval] = []
        while current_i >= 0:
            prev_i = int(prei[current_i])
            prev_c = int(prec[current_i])

            start_frame = prev_i + 1
            end_frame = current_i

            if start_frame < n_frame and end_frame < n_frame:
                start_time = float(qt_onset[start_frame])
                end_time = float(qt_offset[end_frame])
                label = self.templates.chord_list[prev_c]
                result.append(ChordInterval(start=start_time, end=end_time, label=label))

            current_i = prev_i

        result.reverse()
        return result


# ── Convenience functions ─────────────────────────────────────────

def recognize_chords(
    midi_path: Union[str, Path],
    extra_division: int = 2,
    use_transition: bool = True,
) -> list[ChordInterval]:
    """Recognize chords from a MIDI file using beat-level DP.

    This is the primary entry point for chord recognition, implementing
    the algorithm from music-x-lab/midi-chord-recognition.

    Args:
        midi_path: Path to a MIDI file.
        extra_division: Sub-beat divisions (2 = half-beat resolution).
        use_transition: Apply transition penalties in DP decode.

    Returns:
        List of :class:`ChordInterval` with ``(start, end, label)`` in seconds.

    Reference:
        Wang et al., "Pop909," ISMIR 2020.
        Jiang, "MIDI Chord Recognition via Bar-Level Modeling," 2025.
        https://github.com/music-x-lab/midi-chord-recognition
    """
    import miditoolkit
    midi = miditoolkit.MidiFile(str(midi_path))
    engine = ChordRecognitionDP(extra_division=extra_division, use_transition=use_transition)
    return engine.recognize(midi_path)


def recognize_chords_beat(
    midi_path: Union[str, Path],
    extra_division: int = 2,
    use_transition: bool = True,
) -> list[tuple[float, float, str]]:
    """Recognize chords, returning raw ``(start, end, label)`` tuples.

    Convenience wrapper around :func:`recognize_chords`.

    Args:
        midi_path: Path to a MIDI file.
        extra_division: Sub-beat divisions.
        use_transition: Apply transition penalties.

    Returns:
        List of ``(start_sec, end_sec, chord_label)`` tuples.
    """
    intervals = recognize_chords(midi_path, extra_division, use_transition)
    return [(iv.start, iv.end, iv.label) for iv in intervals]


def _simplify_chord_label(label: str) -> str:
    """Simplify a chord label to root:quality format for comparison.

    Maps detailed labels (e.g. 'C:maj/3', 'A:min7') to simplified
    root:major_or_minor format used in DCA evaluation.

    Args:
        label: Full chord label from recognition.

    Returns:
        Simplified label like 'C:maj', 'A:min', or 'N'.
    """
    if label == 'N' or not label:
        return 'N'

    # Strip inversion
    base = label.split('/')[0]

    if ':' in base:
        root, quality = base.split(':', 1)
    else:
        root = base
        quality = 'maj'

    # Classify into major/minor family
    q = quality.lower()
    if q in ('m', 'min', 'min7', 'minmaj7', 'min6', 'min9', 'dim', 'dim7', 'hdim7'):
        family = 'min'
    else:
        family = 'maj'

    return f'{root}:{family}'
