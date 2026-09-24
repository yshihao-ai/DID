"""High-level pairwise evaluation entry point.

References:
    - Note F1 / Notei F1 / Mel F1 / I-IoU / VER:
      Ou et al., "Unifying Symbolic Music Arrangement," NeurIPS 2025.
    - simChr / simgrv: Wu & Yang, "MuseMorphose," IEEE/ACM TASLP 2023.
    - CA: FGG (Zhu et al., ICML 2025) using music-x-lab DP chord recognition.
    - Onset XOR / Note overlap: Choi et al., "D3PIA," ICASSP 2026;
      Raffel et al., "mir_eval," ISMIR 2014.
    - MM: Mongeau & Sankoff, Computers and the Humanities, 1990.
    - TD: Harte, Sandler & Gasser, ACM MM 2006.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

from smg_metrics.note_f1 import NoteF1Result, compute_all as _note_all
from smg_metrics.similarity import SimilarityResult, compute_all as _sim_all
from smg_metrics.chord_accuracy import compute_ca
from smg_metrics.chord_similarity import compute_cs
from smg_metrics.structural import (
    StructuralPairResult,
    compute_pair as _structural_pair,
)
from smg_metrics.rhythmic import onset_xor_distance, note_overlap

__all__ = ["PairResult", "pair_eval", "pair_eval_structural"]


@dataclass(frozen=True, slots=True)
class PairResult:
    """Container for all 11 pairwise comparison metrics.

    Attributes:
        note_f1:      Note F1 (onset + pitch) — [0, 1].
        notei_f1:     Notei F1 (+ instrument) — [0, 1].
        mel_f1:       Melody F1 — [0, 1].
        i_iou:        Instrument IoU — [0, 1].
        ver:          Voice Error Rate — [0, inf).
        sim_chr:      Chroma similarity — [0, 1].
        sim_grv:      Groove similarity — [0, 1].
        ca:           Chord Accuracy — [0, 1].
        cs:           Chord Similarity (deep embedding) — [0, 1].
        onset_xor:    Onset-pattern XOR distance — [0, 1].
        note_overlap: mir_eval transcription average overlap — [0, 1].
    """
    note_f1: float
    notei_f1: float
    mel_f1: float
    i_iou: float
    ver: float
    sim_chr: float
    sim_grv: float
    ca: float
    cs: float
    onset_xor: float
    note_overlap: float

    def to_dict(self) -> dict[str, float]:
        """Return all metrics as a plain dict."""
        return asdict(self)


def pair_eval(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    enable_cs: bool = True,
    similarity_ref_path: Union[str, Path, None] = None,
) -> PairResult:
    """Evaluate *pred* against *ref* with all 11 pairwise metrics.

    Computes:
        - Note F1 / Notei F1 / Mel F1 / I-IoU / VER (note-level, NeurIPS 2025)
        - simChr / simgrv (bar-level, MuseMorphose)
        - CA (chord-level, FGG using DP chord recognition)
        - CS (chord-level deep embedding, Wang et al. ISMIR 2020)
        - onset_xor / note_overlap (rhythmic, D3PIA / mir_eval)

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.
        enable_cs: If True (default), compute CS metric. Requires model weights.
                   Set to False to skip CS if model is unavailable.
        similarity_ref_path: Optional alternate reference MIDI used only for
                   simChr / simgrv. When ``None`` (default), reuse ``ref_path``.

    Returns:
        A :class:`PairResult` dataclass.

    Raises:
        FileNotFoundError: If either file does not exist or CS model not found.
    """
    pred_path, ref_path = Path(pred_path), Path(ref_path)
    sim_ref_path = Path(similarity_ref_path) if similarity_ref_path is not None else ref_path
    for p in (pred_path, ref_path, sim_ref_path):
        if not p.exists():
            raise FileNotFoundError(f"MIDI file not found: {p}")

    note_res: NoteF1Result = _note_all(pred_path, ref_path)
    sim_res: SimilarityResult = _sim_all(pred_path, sim_ref_path)
    ca_val: float = compute_ca(pred_path, ref_path)
    
    # CS metric (optional, requires PyTorch and model weights)
    if enable_cs:
        try:
            cs_val: float = compute_cs(pred_path, ref_path)
        except ImportError as e:
            # PyTorch not installed
            import warnings
            warnings.warn(
                f"CS metric skipped: {e}\n"
                "Install PyTorch with: pip install torch>=2.0.0 or pip install smg-metrics[torch]",
                UserWarning
            )
            cs_val = 0.0
        except FileNotFoundError as e:
            # Model weights not found
            import warnings
            warnings.warn(
                f"CS metric skipped: {e}\n"
                "Download model weights from: "
                "https://github.com/OlyMarco/smg_metric/blob/main/smg_metrics/model_weights/polydis-v1-chd_encoder_only.pt",
                UserWarning
            )
            cs_val = 0.0
    else:
        cs_val = 0.0
    
    onset_xor_val: float = onset_xor_distance(pred_path, ref_path)
    note_overlap_val: float = note_overlap(pred_path, ref_path)

    return PairResult(
        note_f1=note_res.note_f1,
        notei_f1=note_res.notei_f1,
        mel_f1=note_res.mel_f1,
        i_iou=note_res.i_iou,
        ver=note_res.ver,
        sim_chr=sim_res.sim_chr,
        sim_grv=sim_res.sim_grv,
        ca=ca_val,
        cs=cs_val,
        onset_xor=onset_xor_val,
        note_overlap=note_overlap_val,
    )


def pair_eval_structural(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> StructuralPairResult:
    """Evaluate *pred* against *ref* with structural pairwise metrics.

    Computes:
        - Melody Matchness (edit-distance, Mongeau & Sankoff 1990)
        - Tonal Distance (Harte et al., ACM MM 2006)

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.

    Returns:
        A :class:`StructuralPairResult` dataclass.
    """
    return _structural_pair(pred_path, ref_path)
