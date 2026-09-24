"""Chord Accuracy (CA) via rule-based chord recognition.

Implements two chord recognition backends and a beat-level comparison metric:

1. **DP method** (default, ``method='dp'``):
   Beat-level chroma features + dynamic-programming decoder, adapted from
   music-x-lab/midi-chord-recognition (Jiang, 2025).  Used by FGG
   (Zhu et al., ICML 2025) for Direct Chord Accuracy (DCA).

2. **Viterbi method** (``method='viterbi'``):
   Bar-level HMM Viterbi decoder from GETMusic's ``magenta_chord_recognition.py``
   (microsoft/muzic).  Used by GETMusic (Lv et al., IJCAI 2025) Eq. 6.

The CA formula (both methods):
    CA = (1 / N) * sum_{j=1}^{N} 1(C'_j == C_j)
where N = number of comparison frames.

References:
    - Wang et al., "Pop909," ISMIR 2020 (DP method origin).
    - Jiang, "MIDI Chord Recognition via Bar-Level Modeling," 2025.
      https://github.com/music-x-lab/midi-chord-recognition
    - Zhu et al., "FGG," ICML 2025, arXiv:2410.08435 (DCA definition).
    - Lv et al., "GETMusic," IJCAI 2025, arXiv:2305.10841, Eq. 6.
    - Ren et al., "PopMAG," ACM-MM 2020 (CA definition).
"""

from __future__ import annotations

import bisect
import itertools
import math
from pathlib import Path
from typing import Union

import numpy as np
import miditoolkit

__all__ = ["compute_ca", "midi_to_chords", "midi_to_chords_dp"]

# ── Constants (identical to GETMusic) ─────────────────────────────

_PITCH_NAMES = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
_KEY_PITCHES = [0, 2, 4, 5, 7, 9, 11]
_KIND_PITCHES: dict[str, list[int]] = {
    '': [0,4,7], 'm': [0,3,7], '+': [0,4,8], 'dim': [0,3,6],
    '7': [0,4,7,10], 'maj7': [0,4,7,11], 'm7': [0,3,7,10], 'm7b5': [0,3,6,10],
}
_KINDS = list(_KIND_PITCHES)
_NC = 'N.C.'
_CHORDS: list = [_NC] + list(itertools.product(range(12), _KINDS))
_KC: list = list(itertools.product(range(12), _CHORDS))
_N_CHORDS = len(_CHORDS)

# Hyper-parameters (GETMusic defaults)
_P_OUT   = 0.01
_P_K     = 0.001
_P_C     = 0.5
_CONC    = 100.0

# ── Data container ────────────────────────────────────────────────

from dataclasses import dataclass as _dataclass

@_dataclass(frozen=True, slots=True)
class _SimpleNote:
    """Minimal note object for Viterbi chord inference."""
    start: int
    end: int
    pitch: int

# ── HMM distributions (pre-computed once) ─────────────────────────

def _kcd(p_out: float) -> np.ndarray:
    """Key-chord emission distribution (12 x 97)."""
    in_k  = np.zeros([12, _N_CHORDS], np.float64)
    out_k = np.zeros([12, _N_CHORDS], np.float64)
    for key in range(12):
        kp = {(key + o) % 12 for o in _KEY_PITCHES}
        for i, ch in enumerate(_CHORDS[1:]):
            cp = {(ch[0] + o) % 12 for o in _KIND_PITCHES[ch[1]]}
            in_k[key, i+1]  = len(cp & kp)
            out_k[key, i+1] = len(cp - kp)
    mat = ((1 - p_out) ** in_k) * (p_out ** out_k)
    mat /= mat.sum(axis=1, keepdims=True)
    return mat


def _kctd(kcd: np.ndarray, p_k: float, p_c: float) -> np.ndarray:
    """Key-chord transition distribution (97*12 x 97*12)."""
    mat = np.zeros([len(_KC), len(_KC)], np.float64)
    for i, kc1 in enumerate(_KC):
        k1, c1 = kc1
        ci1 = i % _N_CHORDS
        for j, kc2 in enumerate(_KC):
            k2, c2 = kc2
            ci2 = j % _N_CHORDS
            if k1 != k2:
                mat[i, j] = (p_k / 11) * kcd[k2, ci2]
            else:
                mat[i, j] = 1 - p_k
                mat[i, j] *= (1 - p_c) if c1 == c2 else (
                    p_c * (kcd[k2, ci2] + kcd[k2, ci1] / (_N_CHORDS - 1)))
    return mat


def _chord_vecs() -> np.ndarray:
    """Unit chord-pitch vectors (97 x 12)."""
    x = np.zeros([_N_CHORDS, 12], np.float64)
    for i, ch in enumerate(_CHORDS[1:]):
        for o in _KIND_PITCHES[ch[1]]:
            x[i+1, (ch[0]+o) % 12] = 1
    x[1:] /= np.linalg.norm(x[1:], axis=1, keepdims=True)
    return x


# Pre-compute HMM matrices once at module load
_KCD_MAT   = _kcd(_P_OUT)
_KCD_LOG   = np.log(_KCD_MAT)
_KCTD_MAT  = _kctd(_KCD_MAT, _P_K, _P_C)
_KCTD_LOG  = np.log(_KCTD_MAT)
_CVEC      = _chord_vecs()
_VEC_T     = _CVEC.T

# ── Pitch-vector extraction ──────────────────────────────────────

def _pitch_vectors(notes: list[_SimpleNote], boundaries: list[int]) -> np.ndarray:
    """Build per-frame 12-dim pitch-class vectors from a note list."""
    nf = len(boundaries) + 1
    x = np.zeros([nf, 12], np.float64)
    for n in notes:
        sf = bisect.bisect_right(boundaries, n.start)
        ef = bisect.bisect_left(boundaries, n.end)
        pc = n.pitch % 12
        if sf >= ef:
            x[sf, pc] += n.end - n.start
        else:
            x[sf, pc] += boundaries[sf] - n.start
            for f in range(sf + 1, ef):
                x[f, pc] += boundaries[f] - boundaries[f - 1]
            x[ef, pc] += n.end - boundaries[ef - 1]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    mask = norms.ravel() > 0
    x[mask] /= norms[mask]
    return x

# ── Viterbi decoder ──────────────────────────────────────────────

def _viterbi(frame_ll: np.ndarray) -> list:
    """Decode key-chord sequence via Viterbi."""
    nf, nc = frame_ll.shape
    nkc = len(_KC)
    loglik = np.zeros([nf, nkc])
    path   = np.zeros([nf, nkc], np.int32)
    for i, (k, _) in enumerate(_KC):
        loglik[0, i] = -math.log(12) + _KCD_LOG[k, i % nc] + frame_ll[0, i % nc]
    for f in range(1, nf):
        mat = loglik[f-1][:, None] + _KCTD_LOG
        path[f] = mat.argmax(axis=0)
        loglik[f] = mat[path[f], range(nkc)] + np.tile(frame_ll[f], 12)
    idx = [int(np.argmax(loglik[-1]))]
    for f in range(nf - 1, 0, -1):
        idx.append(path[f, idx[-1]])
    return [_CHORDS[i % nc] for i in idx[::-1]]


def _format_chord(chord) -> str:
    """Convert an internal chord state to a stable string label."""
    if chord == _NC:
        return _NC
    root, kind = chord
    quality = "maj" if kind == "" else str(kind)
    return f"{_PITCH_NAMES[int(root)]}:{quality}"

# ── MIDI → chord sequence ────────────────────────────────────────

def midi_to_chords(midi_path: Union[str, Path]) -> list[str]:
    """Infer a chord label per measure from *midi_path*.

    Reference:
        Lv et al., "GETMusic," arXiv:2305.10841, 2023.
        Magenta chord_inference (Apache 2.0).

    Args:
        midi_path: Path to a MIDI file.

    Returns:
        List of chord strings, e.g. ``['C:maj7', 'G:7', 'N.C.']``.
    """
    midi = miditoolkit.MidiFile(str(midi_path))
    notes: list[_SimpleNote] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append(_SimpleNote(start=n.start, end=n.end, pitch=n.pitch))
    if not notes:
        return []
    notes.sort(key=lambda n: (n.start, -n.end))
    ts = midi.time_signature_changes[0] if midi.time_signature_changes else None
    num = ts.numerator if ts else 4
    den = ts.denominator if ts else 4
    bar_len = int(num * midi.ticks_per_beat * 4 / den)
    n_bars = max(1, math.ceil(max(n.end for n in notes) / bar_len))
    boundaries = [bar_len * i for i in range(1, n_bars)]
    pv = _pitch_vectors(notes, boundaries)
    fl = _CONC * pv @ _VEC_T
    return [_format_chord(chord) for chord in _viterbi(fl)]


def midi_to_chords_dp(
    midi_path: Union[str, Path],
    extra_division: int = 2,
    use_transition: bool = True,
) -> list[str]:
    """Infer per-beat chord labels using the DP method.

    Uses beat-level chroma features and dynamic-programming decode from
    music-x-lab/midi-chord-recognition.  Returns one label per beat.

    Reference:
        Wang et al., "Pop909," ISMIR 2020.
        Jiang, "MIDI Chord Recognition," 2025.
        https://github.com/music-x-lab/midi-chord-recognition

    Args:
        midi_path: Path to a MIDI file.
        extra_division: Sub-beat divisions (2 = half-beat).
        use_transition: Apply transition penalties in DP.

    Returns:
        List of chord label strings, one per beat.
    """
    from smg_metrics.chord_recognition import recognize_chords

    intervals = recognize_chords(midi_path, extra_division, use_transition)

    # Expand interval labels to per-beat
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    beats = pm.get_beats()
    if len(beats) < 2 or not intervals:
        return [iv.label for iv in intervals]

    beat_labels: list[str] = []
    iv_idx = 0
    for bi, bt in enumerate(beats):
        # Find which interval this beat falls in
        while iv_idx < len(intervals) - 1 and intervals[iv_idx].end <= bt:
            iv_idx += 1
        if iv_idx < len(intervals) and intervals[iv_idx].start <= bt < intervals[iv_idx].end:
            beat_labels.append(intervals[iv_idx].label)
        else:
            beat_labels.append('N')

    return beat_labels


def _normalize_label(label: str) -> str:
    """Normalize a chord label to root:major_or_minor for comparison.

    Maps detailed labels to simplified format for DCA evaluation.
    Both DP and Viterbi outputs are normalised to the same space.

    Args:
        label: Chord label string.

    Returns:
        Normalized label like 'C:maj', 'A:min', or 'N'.
    """
    if label in ('N.C.', 'N', ''):
        return 'N'

    # Strip inversion and bass note
    base = label.split('/')[0]

    if ':' in base:
        root, quality = base.split(':', 1)
    else:
        root = base
        quality = 'maj'

    q = quality.lower()
    if q in ('m', 'min', 'min7', 'minmaj7', 'min6', 'min9', 'dim', 'dim7', 'hdim7'):
        family = 'min'
    else:
        family = 'maj'

    return f'{root}:{family}'


def compute_ca(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    method: str = "dp",
) -> float:
    """Compute Chord Accuracy between *pred* and *ref*.

    Two methods are available:

    - ``'dp'`` (default): Beat-level DP decoder from music-x-lab.
      Used by FGG (Zhu et al., ICML 2025) for Direct Chord Accuracy.
    - ``'viterbi'``: Bar-level HMM Viterbi decoder from GETMusic.
      Used by GETMusic (Lv et al., IJCAI 2025) Eq. 6.

    Reference:
        Zhu et al., "FGG," ICML 2025, arXiv:2410.08435.
        Lv et al., "GETMusic," IJCAI 2025, arXiv:2305.10841, Eq. 6.

    Args:
        pred_path: Path to the predicted / generated MIDI file.
        ref_path:  Path to the reference / ground-truth MIDI file.
        method:    ``'dp'`` or ``'viterbi'``.

    Returns:
        CA value in [0, 1].
    """
    if method == 'dp':
        sp = midi_to_chords_dp(pred_path)
        sr = midi_to_chords_dp(ref_path)
    elif method == 'viterbi':
        sp = midi_to_chords(pred_path)
        sr = midi_to_chords(ref_path)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'dp' or 'viterbi'.")

    n = min(len(sp), len(sr))
    if n == 0:
        return 0.0

    matches = 0
    for i in range(n):
        if _normalize_label(sp[i]) == _normalize_label(sr[i]):
            matches += 1
    return matches / n


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m smg_metrics.chord_accuracy <pred.mid> <ref.mid> [method]")
        print("  method: dp (default) or viterbi")
        sys.exit(1)
    method = sys.argv[3] if len(sys.argv) > 3 else 'dp'
    ca = compute_ca(sys.argv[1], sys.argv[2], method=method)
    print(f"CA ({method}) = {ca:.4f}")
