"""Chord Similarity (CS) metric using deep chord embeddings.

Computes pairwise chord similarity by encoding 2-bar chord progressions into
a 256-dimensional latent space using a lightweight GRU-based encoder, then
measuring cosine similarity between latent vectors.

The chord encoder is trained on the EC2-VAE model from Wang et al. (2020),
with a pruned version containing only the chord encoder (29 MB vs 104 MB).

**Note**: This metric requires PyTorch (torch >= 2.0.0) to load the pretrained
encoder. Install with: ``pip install torch`` or ``pip install smg-metrics[torch]``

References:
    Wang, Z., Wang, D., Zhang, Y., and Xia, G. "Learning interpretable
    representation for controllable polyphonic music generation."
    Proceedings of 21st International Society for Music Information Retrieval
    Conference (ISMIR), 2020.
    https://arxiv.org/abs/2008.07122

Architecture:
    Input: (batch, 8, 36) - 8 beats × 36-dim chord vectors
           [Root (12) | Chroma (12) | Bass (12)]
    Encoder: Bidirectional GRU (36 → 1024) + Linear (2048 → 256)
    Output: (batch, 256) - Chord progression embedding

The 36-dimensional chord vector captures:
    - Root (0-11): One-hot encoded root note
    - Chroma (12-23): Pitch class energy distribution
    - Bass (24-35): One-hot bass note (relative to root)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union, TYPE_CHECKING

import numpy as np

# Optional PyTorch dependency for CS metric
try:
    import torch
    from torch import nn
    from torch.distributions import Normal
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy objects to allow class definitions (won't be instantiated without torch)
    class _DummyModule:
        pass
    class _DummyNN:
        Module = _DummyModule
        GRU = None
        Linear = None
    nn = _DummyNN()  # type: ignore
    Normal = None  # type: ignore
    F = None  # type: ignore
    torch = None  # type: ignore

__all__ = [
    "compute_cs",
    "ChordEncoder",
    "LightweightChordModel",
    "extract_chord_vectors",
    "clear_cs_model_cache",
]

# ── Model cache for batch evaluation ─────────────────────────────
# Stores loaded models keyed by (weights_path, device) to avoid
# repeated loading during multi-file evaluation.
if TYPE_CHECKING:
    _MODEL_CACHE: dict[tuple[str, str], 'LightweightChordModel'] = {}
else:
    _MODEL_CACHE = {}

# Pitch class names (consistent with chord_recognition.py)
_PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# Inversion offsets (consistent with chord_recognition.py)
_INVERSION_OFFSETS = {
    '1': 0, 'b2': 1, '2': 2, 'b3': 3, '3': 4, '4': 5,
    'b5': 6, '5': 7, '#5': 8, '6': 9, 'b7': 10, '7': 11,
}


class ChordEncoder(nn.Module):
    """GRU-based chord encoder (36 → 1024 → 256)."""
    
    def __init__(self, input_dim: int = 36, hidden_dim: int = 1024, z_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.linear_var = nn.Linear(hidden_dim * 2, z_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

    def forward(self, x):
        """Encode chord sequence to latent distribution.
        
        Args:
            x: (batch, 8, 36) chord vectors
            
        Returns:
            Normal distribution with mean and variance
        """
        x = self.gru(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x).exp_()
        return Normal(mu, var)


class LightweightChordModel(nn.Module):
    """Lightweight chord similarity model (29 MB, pruned from 104 MB)."""
    
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.encoder = ChordEncoder(36, 1024, 256)
        
    def encode(self, c: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Encode chord vectors to 256-dim latent space.
        
        Args:
            c: (batch, 8, 36) chord vectors
            sample: If True, sample from distribution; else use mean
            
        Returns:
            (batch, 256) latent vectors
        """
        self.eval()
        with torch.no_grad():
            dist = self.encoder(c)
            z = dist.sample() if sample else dist.mean
        return z
    
    def load_weights(self, weights_path: Union[str, Path]):
        """Load model weights (supports both pruned and full models)."""
        state_dict = torch.load(str(weights_path), map_location=self.device)
        
        # Handle both pruned and full model formats
        if any(k.startswith('chd_encoder.') for k in state_dict.keys()):
            # Full model: extract chd_encoder
            encoder_state = {
                k.replace('chd_encoder.', ''): v
                for k, v in state_dict.items()
                if k.startswith('chd_encoder.')
            }
        else:
            # Pruned model: use directly
            encoder_state = state_dict
        
        self.encoder.load_state_dict(encoder_state)
        self.to(self.device)
        
    @classmethod
    def from_pretrained(cls, weights_path: Union[str, Path], device=None):
        """Load pretrained model."""
        model = cls(device=device)
        model.load_weights(weights_path)
        return model


def _parse_chord_label(label: str) -> tuple[int, str, int]:
    """Parse chord label to (root_pc, quality, bass_pc).
    
    Examples:
        'C:maj' -> (0, 'maj', 0)
        'C:maj/3' -> (0, 'maj', 4)  # First inversion
        'A:min7' -> (9, 'min7', 9)
        'N' -> (-1, 'N', -1)
    """
    if label == 'N' or not label:
        return (-1, 'N', -1)
    
    # Split inversion
    if '/' in label:
        base, inversion = label.split('/', 1)
    else:
        base = label
        inversion = None
    
    # Split root and quality
    if ':' in base:
        root_name, quality = base.split(':', 1)
    else:
        root_name = base
        quality = 'maj'
    
    # Convert root name to pitch class
    try:
        root_pc = _PITCH_NAMES.index(root_name)
    except ValueError:
        return (-1, 'N', -1)
    
    # Calculate bass pitch class
    if inversion is None:
        bass_pc = root_pc
    else:
        offset = _INVERSION_OFFSETS.get(inversion, 0)
        bass_pc = (root_pc + offset) % 12
    
    return (root_pc, quality, bass_pc)


def _build_chord_vector(label: str, chroma: np.ndarray) -> np.ndarray:
    """Build 36-dim chord vector from label and chroma.
    
    Args:
        label: Chord label (e.g., 'C:maj', 'A:min7/5')
        chroma: 12-dim chroma vector
        
    Returns:
        36-dim vector: [Root (12) | Chroma (12) | Bass (12)]
    """
    root_pc, quality, bass_pc = _parse_chord_label(label)
    
    # Root: one-hot
    root_vec = np.zeros(12)
    if root_pc >= 0:
        root_vec[root_pc] = 1
    
    # Chroma: use provided chroma
    chroma_vec = chroma.copy()
    
    # Bass: one-hot relative to root
    bass_vec = np.zeros(12)
    if root_pc >= 0 and bass_pc >= 0:
        bass_relative = (bass_pc - root_pc) % 12
        bass_vec[bass_relative] = 1
    
    return np.concatenate([root_vec, chroma_vec, bass_vec])


def extract_chord_vectors(midi_path: Union[str, Path]) -> np.ndarray:
    """Extract 2-bar chord vector segments from MIDI file.
    
    Uses smg_metrics.chord_recognition for high-quality chord detection
    (17 chord qualities + inversions, dynamic programming decoder).
    
    Args:
        midi_path: Path to MIDI file
        
    Returns:
        (n_segments, 8, 36) array of chord vectors
        Each segment = 2 bars = 8 beats
    """
    from smg_metrics.chord_recognition import recognize_chords_beat
    import pretty_midi
    
    # Recognize chords
    chord_intervals = recognize_chords_beat(midi_path)
    
    # Load MIDI for beat and note info
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    beats = midi.get_beats()
    
    # Collect notes
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    segments = []
    
    # Extract 2-bar segments (8 beats each)
    for i in range(0, len(beats) - 8, 8):
        segment_vecs = []
        
        for j in range(8):
            beat_start = beats[i + j]
            beat_end = beats[i + j + 1] if (i + j + 1) < len(beats) else beat_start + 0.5
            beat_mid = (beat_start + beat_end) / 2
            
            # Find chord label for this beat
            chord_label = 'N'
            for start, end, label in chord_intervals:
                if start <= beat_mid < end:
                    chord_label = label
                    break
            
            # Compute chroma for this beat
            active_notes = [n for n in notes 
                          if n.start >= beat_start and n.start < beat_end]
            if not active_notes:
                active_notes = [n for n in notes 
                              if n.start <= beat_end and n.end >= beat_start]
            
            chroma = np.zeros(12)
            if active_notes:
                for n in active_notes:
                    duration = min(n.end, beat_end) - max(n.start, beat_start)
                    chroma[n.pitch % 12] += duration
                if chroma.sum() > 0:
                    chroma = chroma / chroma.sum()
            
            # Build 36-dim vector
            vec = _build_chord_vector(chord_label, chroma)
            segment_vecs.append(vec)
        
        segments.append(np.array(segment_vecs))
    
    return np.array(segments) if segments else np.zeros((0, 8, 36))

def _get_cached_model(
    weights_path: Union[str, Path],
    device: str,
) -> LightweightChordModel:
    """Get or create cached model instance.
    
    This function implements a simple caching mechanism to avoid reloading
    the model weights (28.90 MB) on every compute_cs() call. The model is
    cached by (weights_path, device) and reused across multiple evaluations.
    
    Args:
        weights_path: Path to model weights file
        device: 'cuda' or 'cpu'
        
    Returns:
        Cached or newly loaded LightweightChordModel
    """
    global _MODEL_CACHE
    cache_key = (str(weights_path), device)
    
    if cache_key not in _MODEL_CACHE:
        torch_device = torch.device(device)
        model = LightweightChordModel.from_pretrained(weights_path, device=torch_device)
        _MODEL_CACHE[cache_key] = model
    
    return _MODEL_CACHE[cache_key]

def clear_cs_model_cache() -> int:
    """Clear the CS model cache to free GPU/CPU memory.
    
    Use this after batch evaluation to release model memory, or when
    switching between different model files or devices.
    
    Returns:
        Number of cached models cleared
        
    Example:
        >>> # Batch evaluation
        >>> for pred, ref in file_pairs:
        ...     cs = compute_cs(pred, ref)  # Model loaded once
        >>> clear_cs_model_cache()  # Free memory after batch
        1
    """
    global _MODEL_CACHE
    count = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    return count


def compute_cs(
    midi1_path: Union[str, Path],
    midi2_path: Union[str, Path],
    weights_path: Union[str, Path, None] = None,
    device: str = 'auto',
) -> float:
    """Compute Chord Similarity between two MIDI files.
    
    Extracts 2-bar chord progressions, encodes them to 256-dim latent vectors
    using a pretrained GRU encoder, and computes average cosine similarity.
    
    Args:
        midi1_path: Path to first MIDI file
        midi2_path: Path to second MIDI file
        weights_path: Path to model weights. If None, searches for:
            1. polydis-v1-chd_encoder_only.pt (29 MB, recommended)
            2. polydis-v1.pt (104 MB, full model)
        device: 'auto', 'cuda', or 'cpu'
        
    Returns:
        Average cosine similarity across all 2-bar segments (0.0-1.0)
        
    Example:
        >>> cs = compute_cs("generated.mid", "reference.mid")
        >>> print(f"Chord Similarity: {cs:.4f}")
        Chord Similarity: 0.9408
        
    Reference:
        Wang et al., "Learning interpretable representation for controllable
        polyphonic music generation," ISMIR 2020.
        https://arxiv.org/abs/2008.07122
        
    Raises:
        ImportError: If PyTorch is not installed
    """
    # Check torch availability
    if not HAS_TORCH:
        raise ImportError(
            "Chord Similarity (CS) metric requires PyTorch. "
            "Install with: pip install torch>=2.0.0 or pip install smg-metrics[torch]"
        )
    
    # Auto-locate weights
    if weights_path is None:
        search_paths = [
            Path(__file__).parent / "model_weights" / "polydis-v1-chd_encoder_only.pt",
            Path(__file__).parent / "model_weights" / "polydis-v1.pt",
        ]
        for path in search_paths:
            if path.exists():
                weights_path = path
                break
        
        if weights_path is None:
            raise FileNotFoundError(
                "No model weights found. Please download polydis-v1-chd_encoder_only.pt "
                "or polydis-v1.pt and place in smg_metrics/model_weights/. "
                "See README for download instructions."
            )
    
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get cached model (or load if first call)
    model = _get_cached_model(weights_path, device)
    
    # Extract chord vectors
    segs1 = extract_chord_vectors(midi1_path)
    segs2 = extract_chord_vectors(midi2_path)
    
    # Check lengths
    min_segs = min(len(segs1), len(segs2))
    if min_segs == 0:
        raise ValueError("At least one MIDI file is too short (< 2 bars)")
    
    # Compute cosine similarities
    cos_sims = []
    for i in range(min_segs):
        c1 = torch.from_numpy(segs1[i]).float().unsqueeze(0).to(model.device)
        c2 = torch.from_numpy(segs2[i]).float().unsqueeze(0).to(model.device)
        
        z1 = model.encode(c1)
        z2 = model.encode(c2)
        
        sim = F.cosine_similarity(z1, z2).item()
        cos_sims.append(sim)
    
    return float(np.mean(cos_sims))
