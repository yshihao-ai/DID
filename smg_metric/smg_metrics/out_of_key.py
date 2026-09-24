"""Out-of-Key (OOK) metric for evaluating key consistency.

Computes the percentage of steps containing at least one out-of-key note, where
each step corresponds to a 16th note. This metric quantifies dissonance caused by
notes that fall outside the detected key signature.

Reference:
    Zhu et al., "Flexible Music Generation with Flexible Global Guidance," ICML 2025.
    arXiv:2410.08435v1 [cs.SD] 11 Oct 2024.
    
    "We present the frequency of out-of-key notes by computing the percentage of
    steps in the generated sequences containing at least one out-of-key note,
    where each step corresponds to a 16th note. The frequency of out-of-key notes
    in the baselines is roughly 2%–4%, equating to about 1–3 occurrences in a
    4-measure piece. In contrast, our sampling control method effectively
    eliminates such dissonant notes in the generated samples."
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pretty_midi
import miditoolkit

__all__ = ["compute_ook", "compute_ook_percentage"]

# Major and minor key scale degrees (pitch classes)
_MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11])  # C major: C D E F G A B
_MINOR_SCALE = np.array([0, 2, 3, 5, 7, 8, 10])  # C minor: C D Eb F G Ab Bb

# Pitch class names for key detection
_PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _detect_key(notes: list) -> tuple[int, str]:
    """Detect the key signature using Krumhansl-Schmuckler algorithm.
    
    Args:
        notes: List of miditoolkit Note objects.
        
    Returns:
        (tonic_pc, mode): Tonic pitch class (0-11) and mode ('major' or 'minor').
    """
    if not notes:
        return (0, 'major')  # Default to C major
    
    # Krumhansl-Kessler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Build pitch class histogram (weighted by duration)
    pc_histogram = np.zeros(12)
    for note in notes:
        duration = note.end - note.start
        pc_histogram[note.pitch % 12] += duration
    
    # Normalize
    if pc_histogram.sum() > 0:
        pc_histogram = pc_histogram / pc_histogram.sum()
    
    # Correlate with all 24 keys
    best_corr = -1
    best_tonic = 0
    best_mode = 'major'
    
    for tonic in range(12):
        # Major key
        rotated_hist = np.roll(pc_histogram, -tonic)
        corr_major = np.corrcoef(rotated_hist, major_profile)[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            best_tonic = tonic
            best_mode = 'major'
        
        # Minor key
        corr_minor = np.corrcoef(rotated_hist, minor_profile)[0, 1]
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_tonic = tonic
            best_mode = 'minor'
    
    return (best_tonic, best_mode)


def compute_ook(
    midi_path: Union[str, Path],
    step_resolution: int = 4,
    return_details: bool = False,
) -> Union[float, tuple[float, dict]]:
    """Compute Out-of-Key (OOK) fraction for a MIDI file.
    
    Quantizes the MIDI file into steps (default: 16th notes) and computes the
    fraction of steps containing at least one out-of-key note.
    
    Args:
        midi_path: Path to a MIDI file.
        step_resolution: Steps per quarter note (4 = 16th notes, 8 = 32nd notes).
        return_details: If True, return (fraction, details_dict).
        
    Returns:
        OOK fraction (0-1), or tuple (fraction, details) if return_details=True.
        
    Details dict contains:
        - 'key': Detected key string (e.g., "C:major", "A:minor")
        - 'total_steps': Total number of steps
        - 'ook_steps': Number of steps with out-of-key notes
        - 'ook_notes': Total number of out-of-key notes
        
    Example:
        >>> ook = compute_ook("generated.mid")
        >>> print(f"OOK: {ook:.4f}")
        OOK: 0.0234
        
        >>> ook, details = compute_ook("generated.mid", return_details=True)
        >>> print(f"Key: {details['key']}, OOK steps: {details['ook_steps']}/{details['total_steps']}")
    """
    # Load MIDI
    midi = miditoolkit.MidiFile(str(midi_path))
    
    # Collect all non-drum notes
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    if not notes:
        result = 0.0
        details = {'key': 'N/A', 'total_steps': 0, 'ook_steps': 0, 'ook_notes': 0}
        return (result, details) if return_details else result
    
    # Detect key
    tonic_pc, mode = _detect_key(notes)
    key_scale = _MINOR_SCALE if mode == 'minor' else _MAJOR_SCALE
    in_key_pcs = set((tonic_pc + degree) % 12 for degree in key_scale)
    key_name = f"{_PITCH_NAMES[tonic_pc]}:{mode}"
    
    # Determine time range
    ticks_per_beat = midi.ticks_per_beat
    ticks_per_step = ticks_per_beat // step_resolution
    
    start_tick = min(n.start for n in notes)
    end_tick = max(n.end for n in notes)
    total_steps = (end_tick - start_tick + ticks_per_step - 1) // ticks_per_step
    
    if total_steps == 0:
        result = 0.0
        details = {'key': key_name, 'total_steps': 0, 'ook_steps': 0, 'ook_notes': 0}
        return (result, details) if return_details else result
    
    # Build step-level presence matrix
    step_has_ook = np.zeros(total_steps, dtype=bool)
    ook_note_count = 0
    
    for note in notes:
        pc = note.pitch % 12
        is_out_of_key = pc not in in_key_pcs
        
        if is_out_of_key:
            ook_note_count += 1
            # Mark all steps this note spans
            note_start_step = (note.start - start_tick) // ticks_per_step
            note_end_step = (note.end - start_tick) // ticks_per_step
            note_start_step = max(0, min(note_start_step, total_steps - 1))
            note_end_step = max(0, min(note_end_step, total_steps - 1))
            
            for step in range(note_start_step, note_end_step + 1):
                if 0 <= step < total_steps:
                    step_has_ook[step] = True
    
    ook_steps = int(step_has_ook.sum())
    ook_percentage = ook_steps / total_steps
    
    if return_details:
        details = {
            'key': key_name,
            'total_steps': int(total_steps),
            'ook_steps': ook_steps,
            'ook_notes': ook_note_count,
        }
        return (ook_percentage, details)
    
    return ook_percentage


def compute_ook_percentage(midi_path: Union[str, Path]) -> float:
    """Convenience function: compute OOK fraction only.
    
    Args:
        midi_path: Path to a MIDI file.
        
    Returns:
        OOK fraction (0-1).
    """
    return compute_ook(midi_path, return_details=False)
