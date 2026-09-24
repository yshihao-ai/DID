"""High-level single-file evaluation entry point.

References:
    - 13 MusPy metrics: Dong et al., "MusPy," ISMIR 2020.
    - CHE: Papadopoulos & Peeters, ISMIR 2012.
    - N-gram: Yang & Lerch, Neural Computing and Applications, 2018.
    - Rhythmic/temporal features: Choi et al., "D3PIA," ICASSP 2026.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from smg_metrics.muspy_ext import SingleFileResult, compute_all as _muspy_all
from smg_metrics.structural import (
    StructuralSingleResult,
    compute_single as _structural_single,
)
from smg_metrics.rhythmic import RhythmicResult, compute_single as _rhythmic_single

__all__ = [
    "single_file",
    "SingleFileResult",
    "single_file_structural",
    "StructuralSingleResult",
    "single_file_rhythmic",
    "RhythmicResult",
]


def single_file(
    midi_path: Union[str, Path],
    root: int = 0,
    mode: str = "major",
) -> SingleFileResult:
    """Evaluate a single MIDI file with 13 quality metrics.

    This is a convenience wrapper around :func:`smg_metrics.muspy_ext.compute_all`.

    Args:
        midi_path: Path to a MIDI file.
        root: Root pitch class for pitch-in-scale rate (0=C).
        mode: ``"major"`` or ``"minor"``.

    Returns:
        A :class:`SingleFileResult` dataclass with all 13 metrics.
    """
    return _muspy_all(midi_path, root=root, mode=mode)


def single_file_structural(
    midi_path: Union[str, Path],
) -> StructuralSingleResult:
    """Evaluate a single MIDI file with structural metrics (CHE, N-gram diversity).

    Reference:
        - CHE: Papadopoulos & Peeters, "Large-scale Study of Chord Estimation
          Algorithms Based on Chroma," ISMIR 2012.
        - N-gram: Yang & Lerch, "On the Evaluation of Generative Models in
          Music," Neural Computing and Applications, 2018.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        A :class:`StructuralSingleResult` with CHE and ngram_div.
    """
    return _structural_single(midi_path)


def single_file_rhythmic(
    midi_path: Union[str, Path],
) -> RhythmicResult:
    """Evaluate a single MIDI file with rhythmic metrics.

    Computes mean IOI, rhythmic intensity, rhythmic density, voice number,
    and grooving pattern similarity (4 D3PIA-style metrics + GS).

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        A :class:`RhythmicResult` with five rhythmic metrics.
    """
    return _rhythmic_single(midi_path)
