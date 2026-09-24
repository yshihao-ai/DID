"""Structural & textural evaluation metrics for symbolic music generation.

Implements four metrics that capture musical structure and texture:

    CHE   — Chord Histogram Entropy (single-file)
    Ngram — N-gram Note Diversity (single-file)
    MM    — Melody Matchness (pairwise)
    TD    — Tonal Distance (pairwise)

References:
    - CHE: Papadopoulos & Peeters, "Large-scale Study of Chord Estimation
      Algorithms Based on Chroma," ISMIR 2012.
    - Ngram: Yang & Lerch, "On the Evaluation of Generative Models in Music,"
      Neural Computing and Applications, 2018.
    - MM: Mongeau & Sankoff, "Comparison of Musical Sequences," Computers and
      the Humanities, 1990.
    - TD: Harte, Sandler & Gasser, "Detecting Harmonic Change in Musical
      Audio," ACM MM 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import numpy as np

from smg_metrics._io import Note3, extract_notes3, load_midi
from smg_metrics._edit import normalised_edit_distance, extract_melody

__all__ = [
    "chord_histogram_entropy",
    "ngram_diversity",
    "melody_matchness",
    "tonal_distance",
    "StructuralSingleResult",
    "StructuralPairResult",
    "compute_single",
    "compute_pair",
]

# ── Data containers ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StructuralSingleResult:
    """Container for single-file structural metrics.

    Attributes:
        che:       Chord Histogram Entropy — [0, log₂(C)].
                   Reference: Papadopoulos & Peeters, ISMIR 2012.
        ngram_div: N-gram (4-gram) pitch-class diversity — [0, 1].
                   Reference: Yang & Lerch, NCA 2018.
    """
    che: float
    ngram_div: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StructuralPairResult:
    """Container for pairwise structural metrics.

    Attributes:
        melody_match: Melody Matchness (edit-distance) — [0, 1].
                      Reference: Mongeau & Sankoff, CH 1990.
        tonal_dist:   Tonal Distance (Euclidean) — [0, inf).
                      Reference: Harte et al., ACM MM 2006.
    """
    melody_match: float
    tonal_dist: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ── 1. Chord Histogram Entropy (CHE) ──────────────────────────────

_CHORD_TEMPLATES: dict[str, list[int]] = {
    "maj":    [0, 4, 7],
    "min":    [0, 3, 7],
    "dim":    [0, 3, 6],
    "aug":    [0, 4, 8],
    "7":      [0, 4, 7, 10],
    "maj7":   [0, 4, 7, 11],
    "min7":   [0, 3, 7, 10],
    "dim7":   [0, 3, 6, 9],
    "sus4":   [0, 5, 7],
    "sus2":   [0, 2, 7],
}


def _classify_chord(chroma: np.ndarray) -> str:
    """Classify a 12-dim chroma vector into a chord type via template matching.

    Reference:
        Papadopoulos & Peeters, ISMIR 2012.

    Args:
        chroma: 12-dimensional pitch-class histogram for one bar.

    Returns:
        Chord label like ``'C:maj'``, ``'A:min'``, or ``'N'`` (no chord).
    """
    total = chroma.sum()
    if total < 1:
        return "N"

    normed = chroma / (np.linalg.norm(chroma) + 1e-12)
    best_score = 0.0
    best_label = "N"

    for root in range(12):
        for kind, intervals in _CHORD_TEMPLATES.items():
            template = np.zeros(12)
            for iv in intervals:
                template[(root + iv) % 12] = 1.0
            template /= np.linalg.norm(template) + 1e-12
            score = float(np.dot(normed, template))
            if score > best_score:
                best_score = score
                best_label = f"{root}:{kind}"

    return best_label if best_score >= 0.5 else "N"


def chord_histogram_entropy(midi_path: Union[str, Path]) -> float:
    """Compute the Shannon entropy of the chord-type histogram.

    Extracts chords per bar using chroma template matching, builds a
    histogram of chord types, and computes its Shannon entropy.

    Reference:
        Papadopoulos & Peeters, "Large-scale Study of Chord Estimation
        Algorithms Based on Chroma," ISMIR 2012.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        Chord Histogram Entropy in bits, or NaN if no chords found.
    """
    notes = extract_notes3(midi_path)
    if not notes:
        return float("nan")

    midi = load_midi(midi_path)
    tp_bar = max(1, midi.ticks_per_beat * 4)  # 4/4 bar

    max_tick = max(n.start + n.dur for n in notes)
    n_bars = max(1, max_tick // tp_bar + 1)
    bar_chroma = np.zeros((n_bars, 12), dtype=np.float64)

    for n in notes:
        bar = n.start // tp_bar
        if 0 <= bar < n_bars:
            bar_chroma[bar, n.pitch % 12] += 1.0

    chord_counts: dict[str, int] = {}
    for b in range(n_bars):
        label = _classify_chord(bar_chroma[b])
        chord_counts[label] = chord_counts.get(label, 0) + 1

    chord_counts.pop("N", None)
    total = sum(chord_counts.values())
    if total == 0:
        return float("nan")

    entropy = 0.0
    for count in chord_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ── 2. N-gram Note Diversity ──────────────────────────────────────


def ngram_diversity(midi_path: Union[str, Path], n: int = 4) -> float:
    """Compute N-gram diversity of pitch-class sequences.

    Diversity = (# unique n-grams) / (# total n-grams).

    Reference:
        Yang & Lerch, "On the Evaluation of Generative Models in Music,"
        Neural Computing and Applications, 2018.

    Args:
        midi_path: Path to a MIDI file.
        n:         N-gram size (default 4).

    Returns:
        N-gram diversity in [0, 1], or NaN if too few notes.
    """
    notes = extract_notes3(midi_path)
    if len(notes) < n:
        return float("nan")

    notes_sorted = sorted(notes, key=lambda x: x.start)
    pc_seq = [n.pitch % 12 for n in notes_sorted]

    if len(pc_seq) < n:
        return float("nan")

    ngrams: set[tuple[int, ...]] = set()
    for i in range(len(pc_seq) - n + 1):
        ngrams.add(tuple(pc_seq[i : i + n]))

    total = len(pc_seq) - n + 1
    return len(ngrams) / total if total > 0 else float("nan")


# ── 3. Melody Matchness (pairwise) ────────────────────────────────


def melody_matchness(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> float:
    """Compute melody similarity via normalised edit distance.

    Extracts the highest-pitch track as melody, converts to pitch
    sequences, and computes:
        matchness = 1 − edit_distance(pred, ref) / max(len(pred), len(ref))

    Reference:
        Mongeau & Sankoff, "Comparison of Musical Sequences,"
        Computers and the Humanities, 1990.

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.

    Returns:
        Melody Matchness in [0, 1] (1 = identical melodies).
    """
    pred_mel = extract_melody(pred_path)
    ref_mel = extract_melody(ref_path)
    return normalised_edit_distance(pred_mel, ref_mel)


# ── 4. Tonal Distance (pairwise) ──────────────────────────────────


def _tonal_matrix(r1: float = 1.0, r2: float = 1.0, r3: float = 0.5) -> np.ndarray:
    """Compute the 6×12 tonal matrix for tonal centroid computation.

    Reference:
        Harte, Sandler & Gasser, "Detecting Harmonic Change in Musical
        Audio," ACM MM 2006.

    Args:
        r1: Weight for fifths dimension.
        r2: Weight for minor-third dimension.
        r3: Weight for major-third dimension.

    Returns:
        6×12 numpy array.
    """
    tm = np.empty((6, 12), dtype=np.float64)
    n = np.arange(12, dtype=np.float64)
    tm[0, :] = r1 * np.sin(n * (7.0 / 6.0) * np.pi)
    tm[1, :] = r1 * np.cos(n * (7.0 / 6.0) * np.pi)
    tm[2, :] = r2 * np.sin(n * (3.0 / 2.0) * np.pi)
    tm[3, :] = r2 * np.cos(n * (3.0 / 2.0) * np.pi)
    tm[4, :] = r3 * np.sin(n * (2.0 / 3.0) * np.pi)
    tm[5, :] = r3 * np.cos(n * (2.0 / 3.0) * np.pi)
    return tm


_TONAL_MATRIX = _tonal_matrix()


def _pitch_class_profile(notes: list[Note3]) -> np.ndarray:
    """Compute a 12-dim pitch-class histogram weighted by duration.

    Args:
        notes: Notes extracted by :func:`~smg_metrics._io.extract_notes3`.

    Returns:
        L1-normalised 12-dim numpy array.
    """
    pcp = np.zeros(12, dtype=np.float64)
    for n in notes:
        pcp[n.pitch % 12] += n.dur
    total = pcp.sum()
    if total > 0:
        pcp /= total
    return pcp


def tonal_distance(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> float:
    """Compute the tonal distance between two MIDI files.

    Projects pitch-class profiles onto a 6-dimensional tonal space
    using the tonal matrix from Harte et al. (2006), then computes
    the Euclidean distance between the two tonal centroids.

    Reference:
        Harte, Sandler & Gasser, "Detecting Harmonic Change in Musical
        Audio," ACM MM 2006.
        Implementation verified against MuseGAN v1 (salu133445/musegan).

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.

    Returns:
        Tonal distance >= 0 (0 = identical tonal content).
    """
    pred_notes = extract_notes3(pred_path)
    ref_notes = extract_notes3(ref_path)

    pcp_pred = _pitch_class_profile(pred_notes)
    pcp_ref = _pitch_class_profile(ref_notes)

    centroid_pred = _TONAL_MATRIX @ pcp_pred
    centroid_ref = _TONAL_MATRIX @ pcp_ref

    return float(np.linalg.norm(centroid_pred - centroid_ref))


# ── High-level wrappers ───────────────────────────────────────────


def compute_single(midi_path: Union[str, Path]) -> StructuralSingleResult:
    """Compute all single-file structural metrics for *midi_path*.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        A :class:`StructuralSingleResult` with CHE and N-gram diversity.
    """
    return StructuralSingleResult(
        che=chord_histogram_entropy(midi_path),
        ngram_div=ngram_diversity(midi_path),
    )


def compute_pair(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> StructuralPairResult:
    """Compute all pairwise structural metrics.

    Args:
        pred_path: Path to the predicted MIDI file.
        ref_path:  Path to the reference MIDI file.

    Returns:
        A :class:`StructuralPairResult` with melody matchness and tonal distance.
    """
    return StructuralPairResult(
        melody_match=melody_matchness(pred_path, ref_path),
        tonal_dist=tonal_distance(pred_path, ref_path),
    )
