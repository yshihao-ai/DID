"""Note F1 / Notei F1 / Mel F1 / I-IoU / VER metrics.

Implements five pairwise comparison metrics from:

    Ou et al., "Unifying Symbolic Music Arrangement," NeurIPS 2025,
    arXiv:2408.15176, Appendix C.1.

Note events are quantised to a 16th-note grid before matching.
The matching is greedy one-to-one (provably optimal for exact-match keys).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import miditoolkit

__all__ = ["NoteF1Result", "compute_all"]

_TP16_DIVISOR = 4  # 16th-note = quarter / 4


# ── Data structures ────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class _Note:
    onset: int   # quantised 16th-note step
    pitch: int   # MIDI pitch


@dataclass(frozen=True, slots=True)
class _NoteI(_Note):
    instrument: int  # MIDI program (128 for drums)


@dataclass(frozen=True, slots=True)
class NoteF1Result:
    """Container for five pairwise note-level metrics.

    Attributes:
        note_f1:   Note F1 (onset + pitch) — [0, 1].
        notei_f1:  Notei F1 (onset + pitch + instrument) — [0, 1].
        mel_f1:    Melody F1 (melody track only) — [0, 1].
        i_iou:     Instrument IoU — [0, 1].
        ver:       Voice Error Rate — [0, inf).
    """
    note_f1: float
    notei_f1: float
    mel_f1: float
    i_iou: float
    ver: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a plain dict."""
        from dataclasses import asdict
        return asdict(self)


# ── MIDI loading ───────────────────────────────────────────────────

def _load_notes(midi_path: str | Path) -> list[_NoteI]:
    """Extract quantised note events from *midi_path*.

    Reference:
        Ou et al., "Unifying Symbolic Music Arrangement," NeurIPS 2025.
    """
    midi = miditoolkit.MidiFile(str(midi_path))
    tp16 = max(1, midi.ticks_per_beat // _TP16_DIVISOR)
    notes: list[_NoteI] = []
    for track in midi.instruments:
        program = 128 if track.is_drum else track.program
        for n in track.notes:
            notes.append(_NoteI(
                onset=n.start // tp16,
                pitch=n.pitch,
                instrument=program,
            ))
    return notes


# ── Greedy one-to-one matching ────────────────────────────────────

def _count_matches(pred: list[_Note], ref: list[_Note]) -> int:
    """Count greedy one-to-one (onset, pitch) matches."""
    pool: dict[tuple[int, int], int] = defaultdict(int)
    for n in ref:
        pool[(n.onset, n.pitch)] += 1
    matched = 0
    for n in pred:
        key = (n.onset, n.pitch)
        if pool.get(key, 0) > 0:
            pool[key] -= 1
            matched += 1
    return matched


def _count_matches_i(pred: list[_NoteI], ref: list[_NoteI]) -> int:
    """Count greedy one-to-one (onset, pitch, instrument) matches."""
    pool: dict[tuple[int, int, int], int] = defaultdict(int)
    for n in ref:
        pool[(n.onset, n.pitch, n.instrument)] += 1
    matched = 0
    for n in pred:
        key = (n.onset, n.pitch, n.instrument)
        if pool.get(key, 0) > 0:
            pool[key] -= 1
            matched += 1
    return matched


def _f1(pred_len: int, ref_len: int, matched: int) -> float:
    """Precision / Recall / F1 from set sizes and match count."""
    if pred_len == 0 or ref_len == 0:
        return 0.0
    p = matched / pred_len
    r = matched / ref_len
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ── Melody identification ─────────────────────────────────────────

def _melody_program(notes: list[_NoteI]) -> int:
    """Return the instrument program with the highest average pitch."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for n in notes:
        sums[n.instrument] += n.pitch
        counts[n.instrument] += 1
    if not sums:
        return 0
    return max(sums, key=lambda k: sums[k] / counts[k])


# ── Voice Error Rate ──────────────────────────────────────────────

def _voice_seq(notes: list[_NoteI]) -> list[int]:
    """Instruments sorted by descending average pitch."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for n in notes:
        sums[n.instrument] += n.pitch
        counts[n.instrument] += 1
    return sorted(sums, key=lambda k: sums[k] / counts[k], reverse=True)


def _levenshtein(a: list, b: list) -> int:
    """Standard DP edit distance."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + cost)
    return int(dp[m, n])


# ── Public API ────────────────────────────────────────────────────

def compute_all(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> NoteF1Result:
    """Compute all five note-level metrics between *pred* and *ref*.

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.

    Returns:
        A :class:`NoteF1Result`.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    pred_path, ref_path = Path(pred_path), Path(ref_path)
    for p in (pred_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"MIDI file not found: {p}")

    pred_raw = _load_notes(pred_path)
    ref_raw  = _load_notes(ref_path)

    pred_plain = [_Note(n.onset, n.pitch) for n in pred_raw]
    ref_plain  = [_Note(n.onset, n.pitch) for n in ref_raw]

    # Note F1
    matched = _count_matches(pred_plain, ref_plain)
    nf1 = _f1(len(pred_plain), len(ref_plain), matched)

    # Notei F1
    matched_i = _count_matches_i(pred_raw, ref_raw)
    ni_f1 = _f1(len(pred_raw), len(ref_raw), matched_i)

    # Mel F1
    pred_mel_prog = _melody_program(pred_raw)
    ref_mel_prog  = _melody_program(ref_raw)
    pred_mel = [_Note(n.onset, n.pitch) for n in pred_raw if n.instrument == pred_mel_prog]
    ref_mel  = [_Note(n.onset, n.pitch) for n in ref_raw  if n.instrument == ref_mel_prog]
    mel_match = _count_matches(pred_mel, ref_mel)
    mf1 = _f1(len(pred_mel), len(ref_mel), mel_match)

    # I-IoU
    pred_instrs = {n.instrument for n in pred_raw}
    ref_instrs  = {n.instrument for n in ref_raw}
    union = pred_instrs | ref_instrs
    iou = len(pred_instrs & ref_instrs) / len(union) if union else 1.0

    # VER
    pred_v = _voice_seq(pred_raw)
    ref_v  = _voice_seq(ref_raw)
    ver = _levenshtein(pred_v, ref_v) / len(ref_v) if ref_v else (
        0.0 if not pred_v else 1.0)

    return NoteF1Result(
        note_f1=nf1,
        notei_f1=ni_f1,
        mel_f1=mf1,
        i_iou=iou,
        ver=ver,
    )
