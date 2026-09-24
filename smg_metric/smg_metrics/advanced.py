"""Advanced evaluation metrics for symbolic music generation.

Implements metrics with external dependencies or more complex algorithms:

    KL Duration / KL IOI / KL Pitch  — Distribution divergence (KDE-based)
    OA  — Overlapping Area across 4 musical attributes (KDE-based)
    CI  — Instrument Coverage F1
    CTS — Correct Time Signature
    CR  — Compression Ratio (COSIATEC-inspired)
    ReconAcc — Reconstruction Accuracy (edit distance)

References:
    - KL metrics: yjhuangcd/rule-guided-music, music_evaluation/mgeval/utils.py
    - OA: Rule Guided Diffusion (ICML 2024), yjhuangcd/rule-guided-music
    - CI: Text2midi (AAAI 2025), yjhuangcd/rule-guided-music
    - CTS: Text2midi (AAAI 2025)
    - CR: Text2midi (AAAI 2025), COSIATEC algorithm
    - ReconAcc: MuseTok (ICASSP 2026)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

from smg_metrics._io import Note4, extract_notes4, load_midi, ticks_per_16th, quantise_to_pc_sequence
from smg_metrics._stats import kl_divergence, overlap_normal
from smg_metrics._edit import levenshtein

__all__ = ["AdvancedResult", "compute_all"]

# ── Data container ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AdvancedResult:
    """Container for advanced evaluation metrics.

    Attributes:
        kl_duration: KL divergence of duration distributions — [0, inf).
        kl_ioi:      KL divergence of IOI distributions — [0, inf).
        kl_pitch:    KL divergence of pitch distributions — [0, inf).
        oa_duration: Overlapping Area of mean duration — [0, 1].
        oa_ioi:      Overlapping Area of mean IOI — [0, 1].
        oa_pitch_range: Overlapping Area of pitch range — [0, 1].
        oa_density:  Overlapping Area of note density — [0, 1].
        ci_precision: Instrument coverage precision — [0, 1].
        ci_recall:   Instrument coverage recall — [0, 1].
        ci_f1:       Instrument coverage F1 — [0, 1].
        cts:         Correct Time Signature — 0 or 1.
        cr_pred:     Compression ratio of predicted file — [0, inf).
        cr_ref:      Compression ratio of reference file — [0, inf).
        recon_acc:   Reconstruction accuracy (edit distance) — [0, 1].
    """
    kl_duration: float
    kl_ioi: float
    kl_pitch: float
    oa_duration: float
    oa_ioi: float
    oa_pitch_range: float
    oa_density: float
    ci_precision: float
    ci_recall: float
    ci_f1: float
    cts: float
    cr_pred: float
    cr_ref: float
    recon_acc: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# ── Helpers ────────────────────────────────────────────────────────


def _instrument_f1(
    pred_programs: set[int],
    ref_programs: set[int],
) -> tuple[float, float, float]:
    """Compute instrument coverage precision, recall, F1.

    Reference:
        Yadav et al., "Text2midi," AAAI 2025.
        yjhuangcd/rule-guided-music, figaro/evaluate.py.

    Args:
        pred_programs: Set of MIDI programs in the predicted file.
        ref_programs:  Set of MIDI programs in the reference file.

    Returns:
        ``(precision, recall, f1)`` tuple, each in [0, 1].
    """
    if not ref_programs and not pred_programs:
        return 1.0, 1.0, 1.0
    if not ref_programs or not pred_programs:
        return 0.0, 0.0, 0.0

    tp = len(pred_programs & ref_programs)
    precision = tp / len(pred_programs) if pred_programs else 0.0
    recall = tp / len(ref_programs) if ref_programs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _correct_time_signature(
    pred_path: str | Path,
    ref_path: str | Path,
) -> float:
    """Check if predicted and reference have the same time signature.

    Reference:
        Yadav et al., "Text2midi," AAAI 2025.

    Args:
        pred_path: Path to the predicted MIDI file.
        ref_path:  Path to the reference MIDI file.

    Returns:
        1.0 if time signatures match, 0.0 if not, NaN if either is missing.
    """
    pred_midi = load_midi(pred_path)
    ref_midi = load_midi(ref_path)

    pred_ts = pred_midi.time_signature_changes
    ref_ts = ref_midi.time_signature_changes

    if not pred_ts or not ref_ts:
        return float("nan")

    p = (pred_ts[0].numerator, pred_ts[0].denominator)
    r = (ref_ts[0].numerator, ref_ts[0].denominator)
    return 1.0 if p == r else 0.0


def _compression_ratio(midi_path: str | Path) -> float:
    """Estimate compression ratio using a simplified 4-gram metric.

    Reference:
        Yadav et al., "Text2midi," AAAI 2025.
        Sturm, "COSIATEC," ISMIR 2013.

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        Compression ratio >= 0 (higher = more repetitive).
    """
    notes = extract_notes4(midi_path)
    if len(notes) < 2:
        return 0.0

    midi = load_midi(midi_path)
    tp16 = ticks_per_16th(midi)

    sequence = [n.pitch % 12 for n in notes]

    if len(sequence) < 4:
        return 0.0

    n = len(sequence)
    unique_4grams: set[tuple[int, ...]] = set()
    for i in range(n - 3):
        unique_4grams.add(tuple(sequence[i:i + 4]))

    total_4grams = n - 3
    if total_4grams <= 0:
        return 0.0

    return total_4grams / len(unique_4grams) if unique_4grams else 0.0


def _reconstruction_accuracy(
    pred_notes: list[Note4],
    ref_notes: list[Note4],
    midi_path: str | Path,
) -> float:
    """Compute reconstruction accuracy using edit distance on pitch-class sequences.

    Reference:
        Zeng et al., "MuseTok," ICASSP 2026.
        Yuer867/MuseTok, test_evaluation.py.

    Args:
        pred_notes: Notes from the predicted file.
        ref_notes:  Notes from the reference file.
        midi_path:  Path to either MIDI file (for ticks_per_beat).

    Returns:
        Reconstruction accuracy in [0, 1].
    """
    if not pred_notes or not ref_notes:
        return 0.0

    midi = load_midi(midi_path)
    tp16 = ticks_per_16th(midi)

    pred_seq = quantise_to_pc_sequence(pred_notes, tp16)
    ref_seq = quantise_to_pc_sequence(ref_notes, tp16)

    if not pred_seq or not ref_seq:
        return 0.0

    # Sub-sample very long sequences to avoid OOM
    m, n = len(pred_seq), len(ref_seq)
    if m * n > 10_000_000:
        pred_seq = pred_seq[::4]
        ref_seq = ref_seq[::4]

    edit_dist = levenshtein(pred_seq, ref_seq)
    max_len = max(len(pred_seq), len(ref_seq))
    return 1.0 - edit_dist / max_len if max_len > 0 else 0.0


# ── Public API ────────────────────────────────────────────────────


def compute_all(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
) -> AdvancedResult:
    """Compute all advanced metrics between *pred* and *ref*.

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.

    Returns:
        An :class:`AdvancedResult`.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    pred_path, ref_path = Path(pred_path), Path(ref_path)
    for p in (pred_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"MIDI file not found: {p}")

    pred_notes = extract_notes4(pred_path)
    ref_notes = extract_notes4(ref_path)

    # ── KL Divergence metrics ──
    pred_durs = [float(n.dur) for n in pred_notes]
    ref_durs = [float(n.dur) for n in ref_notes]
    pred_pitches = [float(n.pitch) for n in pred_notes]
    ref_pitches = [float(n.pitch) for n in ref_notes]

    # IOI (Inter-Onset Interval)
    def _ioi(notes: list[Note4]) -> list[float]:
        starts = sorted(set(n.start for n in notes))
        return [float(starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]

    pred_ioi = _ioi(pred_notes)
    ref_ioi = _ioi(ref_notes)

    kl_dur = kl_divergence(pred_durs, ref_durs)
    kl_ioi = kl_divergence(pred_ioi, ref_ioi)
    kl_pitch = kl_divergence(pred_pitches, ref_pitches)

    # ── Overlapping Area (4 attributes) ──
    oa_dur = overlap_normal(
        np.mean(pred_durs), np.std(pred_durs) if len(pred_durs) > 1 else 1.0,
        np.mean(ref_durs), np.std(ref_durs) if len(ref_durs) > 1 else 1.0,
    )
    oa_ioi = overlap_normal(
        np.mean(pred_ioi), np.std(pred_ioi) if len(pred_ioi) > 1 else 1.0,
        np.mean(ref_ioi), np.std(ref_ioi) if len(ref_ioi) > 1 else 1.0,
    )

    pred_range = max(pred_pitches) - min(pred_pitches) if pred_pitches else 0
    ref_range = max(ref_pitches) - min(ref_pitches) if ref_pitches else 0
    oa_pr = 1.0 - min(1.0, abs(pred_range - ref_range) / 127.0)

    midi_p = load_midi(pred_path)
    midi_r = load_midi(ref_path)
    pred_beats = max(1, midi_p.ticks_per_beat)
    ref_beats = max(1, midi_r.ticks_per_beat)
    pred_max = max(n.start + n.dur for n in pred_notes) if pred_notes else 1
    ref_max = max(n.start + n.dur for n in ref_notes) if ref_notes else 1
    pred_density = len(pred_notes) / max(1, pred_max / pred_beats)
    ref_density = len(ref_notes) / max(1, ref_max / ref_beats)
    oa_density = 1.0 - min(1.0, abs(pred_density - ref_density) / max(pred_density, ref_density, 1.0))

    # ── Instrument Coverage ──
    pred_progs = {n.program for n in pred_notes}
    ref_progs = {n.program for n in ref_notes}
    ci_p, ci_r, ci_f = _instrument_f1(pred_progs, ref_progs)

    # ── Correct Time Signature ──
    cts = _correct_time_signature(pred_path, ref_path)

    # ── Compression Ratio ──
    cr_pred = _compression_ratio(pred_path)
    cr_ref = _compression_ratio(ref_path)

    # ── Reconstruction Accuracy ──
    recon = _reconstruction_accuracy(pred_notes, ref_notes, pred_path)

    return AdvancedResult(
        kl_duration=kl_dur,
        kl_ioi=kl_ioi,
        kl_pitch=kl_pitch,
        oa_duration=oa_dur,
        oa_ioi=oa_ioi,
        oa_pitch_range=oa_pr,
        oa_density=oa_density,
        ci_precision=ci_p,
        ci_recall=ci_r,
        ci_f1=ci_f,
        cts=cts,
        cr_pred=cr_pred,
        cr_ref=cr_ref,
        recon_acc=recon,
    )
