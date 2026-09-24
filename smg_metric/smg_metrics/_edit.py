"""Shared sequence-editing helpers for smg_metrics.

Centralises Levenshtein edit distance and melody extraction used by
multiple metric modules.

References:
    - Levenshtein: Mongeau & Sankoff, "Comparison of Musical Sequences,"
      Computers and the Humanities, 1990.
    - Melody extraction heuristic: highest average-pitch track.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import miditoolkit

__all__ = [
    "levenshtein",
    "normalised_edit_distance",
    "extract_melody",
]


def levenshtein(seq_a: list[int], seq_b: list[int]) -> int:
    """Compute Levenshtein edit distance between two integer sequences.

    Uses a space-optimised DP algorithm (O(min(m, n)) space).

    Reference:
        Mongeau & Sankoff, "Comparison of Musical Sequences,"
        Computers and the Humanities, 1990.

    Args:
        seq_a: First integer sequence.
        seq_b: Second integer sequence.

    Returns:
        The edit distance (non-negative integer).
    """
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m

    # Ensure seq_a is the longer one for space efficiency
    if m < n:
        seq_a, seq_b = seq_b, seq_a
        m, n = n, m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr
    return prev[n]


def normalised_edit_distance(seq_a: list[int], seq_b: list[int]) -> float:
    """Compute 1 − Levenshtein / max(len(a), len(b)).

    Args:
        seq_a: First integer sequence.
        seq_b: Second integer sequence.

    Returns:
        Similarity in [0, 1] (1 = identical).
    """
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    dist = levenshtein(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b))
    return 1.0 - dist / max_len if max_len > 0 else 0.0


def extract_melody(midi_path: Union[str, Path]) -> list[int]:
    """Extract melody pitch sequence from MIDI.

    Heuristic: the track with the highest average pitch is the melody.
    Returns a list of pitches sorted by onset time.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        A list of MIDI pitches (integers), or an empty list if no
        non-drum tracks are found.
    """
    midi = miditoolkit.MidiFile(str(midi_path))
    best_track = None
    best_avg = -1.0

    for inst in midi.instruments:
        if inst.is_drum or not inst.notes:
            continue
        avg_pitch = sum(n.pitch for n in inst.notes) / len(inst.notes)
        if avg_pitch > best_avg:
            best_avg = avg_pitch
            best_track = inst

    if best_track is None:
        return []

    notes_sorted = sorted(best_track.notes, key=lambda n: n.start)
    return [n.pitch for n in notes_sorted]
