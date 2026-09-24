"""Shared MIDI I/O helpers for smg_metrics.

Centralises note extraction, quantisation, and MIDI metadata access
so that individual metric modules never re-implement parsing logic.

References:
    - MIDI loading: miditoolkit (Böck et al., ISMIR 2016).
    - Quantisation: 16th-note grid convention from NeurIPS 2025, Appendix C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import miditoolkit

__all__ = [
    "Note3",
    "Note4",
    "extract_notes3",
    "extract_notes4",
    "load_midi",
    "quantise_to_pc_sequence",
    "ticks_per_16th",
]

# ── Lightweight note containers ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class Note3:
    """Note without program info: (pitch, duration_ticks, start_tick)."""
    pitch: int
    dur: int
    start: int


@dataclass(frozen=True, slots=True)
class Note4:
    """Note with program info: (pitch, duration_ticks, start_tick, program)."""
    pitch: int
    dur: int
    start: int
    program: int


# ── MIDI helpers ──────────────────────────────────────────────────


def load_midi(midi_path: Union[str, Path]) -> miditoolkit.MidiFile:
    """Load a MIDI file via miditoolkit.

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        A :class:`miditoolkit.MidiFile` object.

    Raises:
        FileNotFoundError: If *midi_path* does not exist.
    """
    p = Path(midi_path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {p}")
    return miditoolkit.MidiFile(str(p))


def ticks_per_16th(midi: miditoolkit.MidiFile) -> int:
    """Return the number of ticks per 16th-note (minimum 1).

    Args:
        midi: An opened :class:`miditoolkit.MidiFile`.

    Returns:
        ``max(1, midi.ticks_per_beat // 4)``.
    """
    return max(1, midi.ticks_per_beat // 4)


# ── Note extraction ───────────────────────────────────────────────


def extract_notes3(midi_path: Union[str, Path]) -> list[Note3]:
    """Extract non-drum notes as 3-tuples (pitch, dur, start).

    Skips drum tracks.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        A list of :class:`Note3`.
    """
    midi = load_midi(midi_path)
    notes: list[Note3] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(Note3(n.pitch, n.end - n.start, n.start))
    return notes


def extract_notes4(midi_path: Union[str, Path]) -> list[Note4]:
    """Extract non-drum notes as 4-tuples (pitch, dur, start, program).

    Skips drum tracks.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        A list of :class:`Note4`.
    """
    midi = load_midi(midi_path)
    notes: list[Note4] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        prog = inst.program
        for n in inst.notes:
            notes.append(Note4(n.pitch, n.end - n.start, n.start, prog))
    return notes


# ── Quantisation ──────────────────────────────────────────────────


def quantise_to_pc_sequence(
    notes: list[Note4],
    tp16: int,
) -> list[int]:
    """Convert notes to a quantised pitch-class sequence on a 16th-note grid.

    Each 16th-note step receives a value:
        - 0 = rest (no note sounding)
        - 1–12 = pitch class + 1

    Args:
        notes: Notes extracted by :func:`extract_notes4`.
        tp16:  Ticks per 16th-note (from :func:`ticks_per_16th`).

    Returns:
        A list of integers representing the pitch-class sequence.
    """
    if not notes:
        return []
    max_tick = max(n.start + n.dur for n in notes)
    n_steps = max_tick // tp16 + 1
    seq = [0] * n_steps
    for n in notes:
        s = n.start // tp16
        e = min((n.start + n.dur) // tp16 + 1, n_steps)
        for step in range(s, e):
            seq[step] = n.pitch % 12 + 1
    return seq
