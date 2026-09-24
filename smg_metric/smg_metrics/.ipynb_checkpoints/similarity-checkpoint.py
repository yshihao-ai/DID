"""Chroma Similarity (simChr) and Groove Similarity (simgrv).

Original algorithm: MuseMorphose (Wu & Yang, IEEE/ACM TASLP 2023).
Used by: MuseTok (ICASSP 2026), Rule Guided Diffusion (ICML 2024).

Key implementation notes (verified against MuseTok ``test_evaluation.py``):
    - Chroma: 12-bin pitch-class histogram, L2-normalised, count-based (not velocity).
    - Groove: 48-bin onset-position histogram (``pos_per_bar=48``), L2-normalised.
    - Aggregation: for each generated bar, take max cosine similarity over all
      reference bars, then average across all generated bars.
    - Empty bars use MuseTok's uniform fallback:
      1/12 for chroma and 1/pos_per_bar for groove.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pretty_midi

__all__ = ["SimilarityResult", "compute_all"]

_CHROMA_DIM = 12
_DEFAULT_POS_PER_BAR = 48


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """Container for two bar-level similarity metrics.

    Attributes:
        sim_chr:  Chroma (pitch-class) similarity — [0, 1].
        sim_grv:  Groove (onset-pattern) similarity — [0, 1].
    """
    sim_chr: float
    sim_grv: float

    def to_dict(self) -> dict[str, float]:
        from dataclasses import asdict
        return asdict(self)


# ── Helpers ────────────────────────────────────────────────────────

# def _l2(vec: np.ndarray, dim: int) -> np.ndarray:
#     """L2-normalise *vec*; fall back to uniform if zero."""
#     norm = np.linalg.norm(vec)
#     return vec / norm if norm > 1e-12 else np.ones(dim) / np.sqrt(dim)

def _l2(vec: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalise non-empty vectors; use MuseTok uniform fallback for empty bars."""
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        return vec / norm
    print("Empty Bar!")    
    return np.ones(dim, dtype=np.float64) / dim

def _beats_per_measure(pm: pretty_midi.PrettyMIDI) -> int:
    """Return beats-per-measure from the first time signature (default 4/4).

    Reference:
        Wu & Yang, "MuseMorphose," IEEE/ACM TASLP 2023.
    """
    return 4
    
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        return max(1, ts.numerator * 4 // ts.denominator)
    


def _bar_vectors(
    pm: pretty_midi.PrettyMIDI,
    beats: list[float],
    bpm: int,
    pos_per_bar: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract per-bar chroma and groove vectors from a PrettyMIDI object."""
    n_bars = max(1, len(beats) // bpm) if bpm > 0 else 1
    sub_per_beat = pos_per_bar // bpm if bpm > 0 else _DEFAULT_POS_PER_BAR // 4

    chromas: list[np.ndarray] = []
    grooves: list[np.ndarray] = []

    for m in range(n_bars):
        i0 = m * bpm
        i1 = min((m + 1) * bpm, len(beats))
        if i0 >= len(beats):
            break
        t0 = beats[i0]
        t1 = beats[i1] if i1 < len(beats) else pm.get_end_time()
        dur = t1 - t0
        if dur <= 0:
            dur = 1.0  # guard

        c = np.zeros(_CHROMA_DIM, dtype=np.float64)
        g = np.zeros(pos_per_bar, dtype=np.float64)

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.start < t0 or note.start >= t1:
                    continue
                c[note.pitch % _CHROMA_DIM] += 1.0
                pos = int((note.start - t0) / dur * bpm * sub_per_beat)
                g[min(pos, pos_per_bar - 1)] += 1.0

        chromas.append(_l2(c, _CHROMA_DIM))
        grooves.append(_l2(g, pos_per_bar))

    return chromas, grooves


def _bar_sim(gen: list[np.ndarray], ref: list[np.ndarray]) -> float:
    """max-over-reference, mean-over-generated cosine similarity."""
    if not gen or not ref:
        return float("nan")
    return float(np.mean([max(np.dot(g, r) for r in ref) for g in gen]))


# ── Public API ────────────────────────────────────────────────────

def compute_all(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    pos_per_bar: int = _DEFAULT_POS_PER_BAR,
) -> SimilarityResult:
    """Compute simChr and simgrv between *pred* and *ref*.

    Args:
        pred_path:  Path to the predicted / generated MIDI file.
        ref_path:   Path to the reference / ground-truth MIDI file.
        pos_per_bar: Groove grid resolution (default 48 = 4 beats x 12 sub-divisions).

    Returns:
        A :class:`SimilarityResult`.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    pred_path, ref_path = Path(pred_path), Path(ref_path)
    for p in (pred_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"MIDI file not found: {p}")

    pred_pm = pretty_midi.PrettyMIDI(str(pred_path))
    ref_pm  = pretty_midi.PrettyMIDI(str(ref_path))

    pred_bpm = _beats_per_measure(pred_pm)
    ref_bpm  = _beats_per_measure(ref_pm)

    pred_chroma, pred_groove = _bar_vectors(
        pred_pm, list(pred_pm.get_beats()), pred_bpm, pos_per_bar)
    ref_chroma, ref_groove = _bar_vectors(
        ref_pm, list(ref_pm.get_beats()), ref_bpm, pos_per_bar)

    return SimilarityResult(
        sim_chr=_bar_sim(pred_chroma, ref_chroma),
        sim_grv=_bar_sim(pred_groove, ref_groove),
    )
