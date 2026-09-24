"""Rhythmic and temporal metrics for symbolic music evaluation.

Single-file metrics (IOI, RI, RD, VN) follow the MIDISym/D3PIA feature
extraction conventions: note onsets are quantised to a 16-position-per-bar
grid, rhythmic intensity is averaged over one-second bins, rhythmic density
is the mean binary onset occupancy, and voice number is the average number
of active voices at onset positions.

Grooving Pattern Similarity (GS) measures rhythmic pattern consistency within
a piece using normalized Hamming similarity of bar-level onset patterns.

Pairwise metrics (XOR, NOvlp) compare onset patterns and note transcriptions
between predicted and reference files.

References:
    - IOI/RI/RD/VN: Standard rhythmic features in Music Information Retrieval.
      Implementation conventions follow MIDISym/D3PIA feature extraction.
    - XOR: Onset pattern comparison metric (D3PIA implementation).
    - GS: Wu & Yang, "The Jazz Transformer," ISMIR 2020.
    - NOvlp: Raffel et al., "mir_eval," ISMIR 2014.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Union

import numpy as np

from smg_metrics._io import Note3, extract_notes3, load_midi

__all__ = [
    "RhythmicResult",
    "mean_ioi",
    "rhythmic_intensity",
    "rhythmic_density",
    "voice_number",
    "onset_xor_distance",
    "note_overlap",
    "grooving_pattern_similarity",
    "compute_single",
]


@dataclass(frozen=True, slots=True)
class RhythmicResult:
    """Container for five single-file rhythmic metrics.

    Attributes:
        mean_ioi: Mean inter-onset interval in seconds.
        rhythmic_intensity: Average number of note onsets per occupied second.
        rhythmic_density: Binary onset occupancy over a quantised grid.
        voice_number: Average number of simultaneous voices at active grid steps.
        gs: Grooving Pattern Similarity (groove consistency within the piece).

    References:
        Standard MIR rhythmic features (mean_ioi, rhythmic_intensity, rhythmic_density, voice_number).
        Implementation follows MIDISym/D3PIA conventions.
        Wu & Yang, "The Jazz Transformer," ISMIR 2020 (gs).
    """

    mean_ioi: float
    rhythmic_intensity: float
    rhythmic_density: float
    voice_number: float
    gs: float

    def to_dict(self) -> dict[str, float]:
        """Return all metrics as a plain dict."""
        return asdict(self)


# ── Timing helpers ────────────────────────────────────────────────


def _tick_to_second_mapper(midi) -> Callable[[int | float], float]:
    """Return a tick-to-second converter that respects tempo changes.

    miditoolkit stores tempo changes in BPM.  The converter integrates each
    tempo segment, mirroring MIDISym's tick-to-second grid without materialising
    a potentially huge per-tick array.
    """
    ticks_per_beat = max(1, int(midi.ticks_per_beat))
    tempo_changes = sorted(midi.tempo_changes, key=lambda t: t.time)
    if not tempo_changes or tempo_changes[0].time > 0:
        from miditoolkit.midi.containers import TempoChange

        tempo_changes.insert(0, TempoChange(tempo=120.0, time=0))

    starts: list[int] = []
    bpms: list[float] = []
    elapsed: list[float] = []
    current_seconds = 0.0
    prev_tick = int(tempo_changes[0].time)
    prev_bpm = float(tempo_changes[0].tempo) or 120.0

    starts.append(prev_tick)
    bpms.append(prev_bpm)
    elapsed.append(0.0)

    for change in tempo_changes[1:]:
        tick = int(change.time)
        if tick < prev_tick:
            continue
        current_seconds += (tick - prev_tick) * 60.0 / (ticks_per_beat * prev_bpm)
        starts.append(tick)
        bpms.append(float(change.tempo) or prev_bpm)
        elapsed.append(current_seconds)
        prev_tick = tick
        prev_bpm = float(change.tempo) or prev_bpm

    starts_arr = np.asarray(starts, dtype=np.int64)
    elapsed_arr = np.asarray(elapsed, dtype=np.float64)
    bpms_arr = np.asarray(bpms, dtype=np.float64)

    def tick_to_second(tick: int | float) -> float:
        t = max(0.0, float(tick))
        idx = int(np.searchsorted(starts_arr, t, side="right") - 1)
        idx = max(0, min(idx, len(starts_arr) - 1))
        return float(elapsed_arr[idx] + (t - starts_arr[idx]) * 60.0 / (ticks_per_beat * bpms_arr[idx]))

    return tick_to_second


def _ticks_per_bar(midi) -> int:
    """Return first-time-signature bar length in ticks, defaulting to 4/4."""
    ts = midi.time_signature_changes[0] if midi.time_signature_changes else None
    numerator = int(ts.numerator) if ts else 4
    denominator = int(ts.denominator) if ts else 4
    return max(1, int(round(numerator * midi.ticks_per_beat * 4 / denominator)))


def _grid_ticks(midi, positions_per_bar: int) -> int:
    """Return quantisation grid size in ticks."""
    return max(1, int(round(_ticks_per_bar(midi) / max(1, positions_per_bar))))


def _onset_voice_vector(
    midi_path: Union[str, Path],
    positions_per_bar: int = 16,
) -> np.ndarray:
    """Return per-grid onset voice counts.

    Each grid bin contains the number of non-drum note onsets quantised to that
    position.  This is the representation used by the D3PIA-style rhythmic
    density and onset-XOR metrics.
    """
    midi = load_midi(midi_path)
    notes = extract_notes3(midi_path)
    if not notes:
        return np.zeros(0, dtype=np.int64)

    grid = _grid_ticks(midi, positions_per_bar)
    max_tick = max(n.start + n.dur for n in notes)
    n_steps = max(1, int(np.ceil(max_tick / grid)) + 1)
    counts = np.zeros(n_steps, dtype=np.int64)
    for note in notes:
        idx = min(n_steps - 1, int(round(note.start / grid)))
        counts[idx] += 1
    return counts


def _binary_onset_bar_matrix(
    midi_path: Union[str, Path],
    positions_per_bar: int = 16,
) -> np.ndarray:
    """Return a padded ``(n_bars, positions_per_bar)`` binary onset matrix."""
    counts = _onset_voice_vector(midi_path, positions_per_bar=positions_per_bar)
    if counts.size == 0:
        return np.zeros((1, positions_per_bar), dtype=np.int8)
    n_bars = max(1, int(np.ceil(counts.size / positions_per_bar)))
    padded = np.zeros(n_bars * positions_per_bar, dtype=np.int8)
    padded[: counts.size] = (counts > 0).astype(np.int8)
    return padded.reshape(n_bars, positions_per_bar)


# ── Single-file metrics ───────────────────────────────────────────


def mean_ioi(midi_path: Union[str, Path]) -> float:
    """Compute mean inter-onset interval (IOI) in seconds.

    Consecutive notes with the same onset contribute zero IOI, matching the
    MIDISym/D3PIA implementation.
    """
    midi = load_midi(midi_path)
    notes = sorted(extract_notes3(midi_path), key=lambda n: (n.start, n.pitch))
    if len(notes) < 2:
        return 0.0
    tick_to_second = _tick_to_second_mapper(midi)
    onsets = np.asarray([tick_to_second(n.start) for n in notes], dtype=np.float64)
    return float(np.mean(np.diff(onsets)))


def rhythmic_intensity(midi_path: Union[str, Path]) -> float:
    """Compute D3PIA rhythmic intensity as average onsets per occupied second."""
    midi = load_midi(midi_path)
    notes = sorted(extract_notes3(midi_path), key=lambda n: (n.start, n.pitch))
    if not notes:
        return 0.0
    tick_to_second = _tick_to_second_mapper(midi)
    bins: dict[int, int] = {}
    for note in notes:
        second = int(tick_to_second(note.start))
        bins[second] = bins.get(second, 0) + 1
    return float(np.mean(list(bins.values()))) if bins else 0.0


def rhythmic_density(
    midi_path: Union[str, Path],
    positions_per_bar: int = 16,
) -> float:
    """Compute binary onset occupancy over a quantised bar grid.

    Args:
        midi_path: Path to a MIDI file.
        positions_per_bar: Grid positions per bar.  The D3PIA convention is 16
            positions per 4/4 bar, i.e. 16th-note resolution.
    """
    counts = _onset_voice_vector(midi_path, positions_per_bar=positions_per_bar)
    if counts.size == 0:
        return 0.0
    return float(np.mean(counts > 0))


def voice_number(
    midi_path: Union[str, Path],
    positions_per_bar: int = 16,
    count_only_onset_position: bool = True,
) -> float:
    """Compute D3PIA voice number from quantised onset voice counts.

    By default this is the average number of simultaneous voices over active
    onset positions.  If ``count_only_onset_position`` is false, empty grid
    positions are included in the average.
    """
    counts = _onset_voice_vector(midi_path, positions_per_bar=positions_per_bar)
    if counts.size == 0:
        return 0.0
    if count_only_onset_position:
        active = counts[counts > 0]
        return float(np.mean(active)) if active.size else 0.0
    return float(np.mean(counts))


def compute_single(midi_path: Union[str, Path]) -> RhythmicResult:
    """Compute all five single-file rhythmic metrics."""
    return RhythmicResult(
        mean_ioi=mean_ioi(midi_path),
        rhythmic_intensity=rhythmic_intensity(midi_path),
        rhythmic_density=rhythmic_density(midi_path),
        voice_number=voice_number(midi_path),
        gs=grooving_pattern_similarity(midi_path),
    )


# ── Pairwise metrics ──────────────────────────────────────────────


def onset_xor_distance(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    positions_per_bar: int = 16,
) -> float:
    """Compute mean XOR distance between aligned binary onset bar matrices.

    The two files are quantised independently to ``positions_per_bar`` grid
    positions per bar, padded to the same number of bars, then compared over the
    full matrix.  Identical onset patterns score 0; completely opposite patterns
    score 1.
    """
    pred = _binary_onset_bar_matrix(pred_path, positions_per_bar=positions_per_bar)
    ref = _binary_onset_bar_matrix(ref_path, positions_per_bar=positions_per_bar)
    n_bars = max(pred.shape[0], ref.shape[0])

    def _pad(mat: np.ndarray) -> np.ndarray:
        if mat.shape[0] == n_bars:
            return mat
        out = np.zeros((n_bars, positions_per_bar), dtype=np.int8)
        out[: mat.shape[0], :] = mat
        return out

    pred_p = _pad(pred)
    ref_p = _pad(ref)
    return float(np.mean(np.abs(pred_p - ref_p)))


def _midi_to_intervals_pitches(
    midi_path: Union[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MIDI notes to mir_eval intervals and MIDI pitches."""
    midi = load_midi(midi_path)
    notes: list[Note3] = sorted(extract_notes3(midi_path), key=lambda n: (n.start, n.pitch, n.dur))
    if not notes:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    tick_to_second = _tick_to_second_mapper(midi)
    intervals = np.asarray(
        [[tick_to_second(n.start), tick_to_second(n.start + n.dur)] for n in notes],
        dtype=np.float64,
    )
    pitches = np.asarray([n.pitch for n in notes], dtype=np.float64)

    # mir_eval expects strictly positive durations.
    valid = intervals[:, 1] > intervals[:, 0]
    return intervals[valid], pitches[valid]


def note_overlap(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
) -> float:
    """Compute mir_eval transcription average overlap score.

    Args:
        pred_path: Predicted MIDI file.
        ref_path: Reference MIDI file.
        onset_tolerance: Onset matching tolerance in seconds.
        offset_ratio: Offset tolerance as a note-duration ratio.
    """
    try:
        import mir_eval
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise ImportError("note_overlap requires the 'mir-eval' package") from exc

    ref_intervals, ref_pitches = _midi_to_intervals_pitches(ref_path)
    pred_intervals, pred_pitches = _midi_to_intervals_pitches(pred_path)
    if len(ref_pitches) == 0 or len(pred_pitches) == 0:
        return 0.0

    ref_hz = mir_eval.util.midi_to_hz(ref_pitches)
    pred_hz = mir_eval.util.midi_to_hz(pred_pitches)
    _, _, _, overlap = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_hz,
        pred_intervals,
        pred_hz,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
    )
    return float(overlap)


# ── Grooving Pattern Similarity (single-file) ────────────────────

def grooving_pattern_similarity(midi_path: Union[str, Path]) -> float:
    """Compute Grooving Pattern Similarity (GS) for a single MIDI file.

    GS measures rhythmic pattern consistency within a piece. Each bar is
    represented as a 64-dimensional binary onset vector (64 positions per bar),
    and GS is computed as the normalized Hamming similarity between all pairs
    of bars.

    Formula: GS(bar_a, bar_b) = 1 - (1/64) · Σ XOR(bar_a[i], bar_b[i])

    The function returns the average GS across all bar pairs in the piece.
    Higher values (closer to 1) indicate more consistent groove patterns;
    lower values indicate more rhythmic variation between bars.

    Reference:
        Wu & Yang, "The Jazz Transformer," ISMIR 2020.
        https://archives.ismir.net/ismir2020/paper/000339.pdf

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        GS value in [0, 1]. Returns 0.0 if the file has fewer than 2 bars.
    """
    # Get binary onset matrix with 64 positions per bar (as per paper)
    bar_matrix = _binary_onset_bar_matrix(midi_path, positions_per_bar=64)

    n_bars = bar_matrix.shape[0]
    if n_bars < 2:
        return 0.0

    # Compute pairwise normalized Hamming similarity for all bar pairs
    similarities = []
    for i in range(n_bars):
        for j in range(i + 1, n_bars):
            # XOR gives 1 where bits differ, 0 where they match
            xor_result = np.bitwise_xor(bar_matrix[i], bar_matrix[j])
            # Normalized Hamming distance (proportion of differing bits)
            hamming_dist = np.mean(xor_result)
            # GS = 1 - distance (convert to similarity)
            gs_pair = 1.0 - hamming_dist
            similarities.append(gs_pair)

    return float(np.mean(similarities))


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        res = compute_single(sys.argv[1])
        for key, value in res.to_dict().items():
            print(f"{key}: {value:.4f}")
    elif len(sys.argv) == 3:
        print(f"onset_xor:    {onset_xor_distance(sys.argv[1], sys.argv[2]):.4f}")
        print(f"note_overlap: {note_overlap(sys.argv[1], sys.argv[2]):.4f}")
    else:
        print("Usage: python -m smg_metrics.rhythmic <midi> [ref.mid]")
        raise SystemExit(1)
