"""Distribution-level evaluation metrics for symbolic music generation.

Implements pairwise metrics that compare feature distributions between
predicted and reference MIDI files, following the evaluation methodology
from Microsoft Research (muzic) and related works.

Metrics:
    PD  — Pitch Distribution similarity (overlap area)
    DD  — Duration Distribution similarity (overlap area)
    SC_sim — Scale Consistency Similarity (|SC_pred - SC_ref|)
    PCE_sim — Pitch Class Entropy Similarity (|PCE_pred - PCE_ref|)
    GSC  — Groove Pattern Similarity Consistency (|GC_pred - GC_ref|)

References:
    - PD/DD: SongMASS (Ren et al., ACM-MM 2020) and TeleMelody
      (microsoft/muzic, telemelody/evaluation/cal_similarity.py).
    - SC/PCE/GSC: Variants of MusPy single-file metrics,
      converted to pairwise similarity by taking absolute difference.
      PCE/GSC reference: Wu & Yang, "The Jazz Transformer," ISMIR 2020.
      https://archives.ismir.net/ismir2020/paper/000339.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import numpy as np
import muspy

from smg_metrics._io import Note3, extract_notes3
from smg_metrics._stats import histogram_overlap
from smg_metrics.rhythmic import grooving_pattern_similarity

__all__ = ["DistributionResult", "compute_all"]

# ── Data container ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DistributionResult:
    """Container for distribution-level pairwise metrics.

    Attributes:
        pd:     Pitch Distribution similarity — [0, 1].
                Reference: Ren et al., "SongMASS," ACM-MM 2020.
        dd:     Duration Distribution similarity — [0, 1].
                Reference: Ren et al., "SongMASS," ACM-MM 2020.
        sc_sim: Scale Consistency Similarity — [0, 1].
                Reference: MusPy, Dong et al., ISMIR 2020.
        pce_sim: Pitch Class Entropy Similarity — [0, 1].
                Reference: MusPy, Dong et al., ISMIR 2020.
        gsc:    Groove Pattern Similarity Consistency — [0, 1].
                Reference: Wu & Yang, "Jazz Transformer," ISMIR 2020.
    """
    pd: float
    dd: float
    sc_sim: float
    pce_sim: float
    gsc: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a plain dict."""
        return asdict(self)


# ── Histogram helpers ─────────────────────────────────────────────


def _pitch_distribution(notes: list[Note3]) -> np.ndarray:
    """128-bin pitch histogram from note list."""
    hist = np.zeros(128, dtype=np.float64)
    for n in notes:
        if 0 <= n.pitch < 128:
            hist[n.pitch] += 1.0
    return hist


def _duration_distribution(notes: list[Note3], n_bins: int = 64) -> np.ndarray:
    """Duration histogram (quantised to ticks, mapped to bins)."""
    if not notes:
        return np.zeros(n_bins, dtype=np.float64)
    durs = [n.dur for n in notes]
    min_dur = max(1, min(durs))
    hist = np.zeros(n_bins, dtype=np.float64)
    for n in notes:
        idx = min(int(n.dur / min_dur) - 1, n_bins - 1)
        hist[idx] += 1.0
    return hist


# ── Single-file similarity variants ──────────────────────────────


def _sc_sim(pred_path: str | Path, ref_path: str | Path) -> float:
    """Scale Consistency Similarity: 1 − |SC_pred − SC_ref|.

    Reference: MusPy, Dong et al., ISMIR 2020.
    """
    p = muspy.read_midi(str(pred_path))
    r = muspy.read_midi(str(ref_path))
    sc_p = muspy.scale_consistency(p)
    sc_r = muspy.scale_consistency(r)
    if sc_p != sc_p or sc_r != sc_r:
        return float("nan")
    return 1.0 - abs(sc_p - sc_r)


def _pce_sim(pred_path: str | Path, ref_path: str | Path) -> float:
    """Pitch Class Entropy Similarity: 1 − |PCE_pred − PCE_ref| / log₂(12).

    Reference: MusPy, Dong et al., ISMIR 2020.
    """
    p = muspy.read_midi(str(pred_path))
    r = muspy.read_midi(str(ref_path))
    pce_p = muspy.pitch_class_entropy(p)
    pce_r = muspy.pitch_class_entropy(r)
    if pce_p != pce_p or pce_r != pce_r:
        return float("nan")
    return 1.0 - abs(pce_p - pce_r) / np.log2(12)


def _gsc(pred_path: str | Path, ref_path: str | Path) -> float:
    """Groove Pattern Similarity Consistency (GSC): 1 − |GS_pred − GS_ref|.

    Uses the corrected GS implementation with 64-dimensional binary onset
    vectors per bar, computing normalized Hamming similarity between all
    bar pairs as defined in the original paper.

    Reference: Wu & Yang, "The Jazz Transformer," ISMIR 2020.
    https://archives.ismir.net/ismir2020/paper/000339.pdf
    """
    gs_p = grooving_pattern_similarity(pred_path)
    gs_r = grooving_pattern_similarity(ref_path)
    if gs_p != gs_p or gs_r != gs_r:
        return float("nan")
    return 1.0 - abs(gs_p - gs_r)


# ── Public API ────────────────────────────────────────────────────


def compute_all(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> DistributionResult:
    """Compute all distribution-level metrics between *pred* and *ref*.

    Metrics: PD, DD, SC_sim, PCE_sim, GSC.

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.

    Returns:
        A :class:`DistributionResult`.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    pred_path, ref_path = Path(pred_path), Path(ref_path)
    for p in (pred_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"MIDI file not found: {p}")

    pred_notes = extract_notes3(pred_path)
    ref_notes = extract_notes3(ref_path)

    pd_val = histogram_overlap(_pitch_distribution(ref_notes), _pitch_distribution(pred_notes))
    dd_val = histogram_overlap(_duration_distribution(ref_notes), _duration_distribution(pred_notes))

    return DistributionResult(
        pd=pd_val,
        dd=dd_val,
        sc_sim=_sc_sim(pred_path, ref_path),
        pce_sim=_pce_sim(pred_path, ref_path),
        gsc=_gsc(pred_path, ref_path),
    )
