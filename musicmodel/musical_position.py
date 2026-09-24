"""Utilities for DID musical-time position IDs.

This module is intentionally small and model-agnostic.  It does two things:

1) Build Note-Decoder ``position_ids`` from the existing DID event features.
2) Patch the Qwen2 rotary embedding used by the Note Decoder so arbitrary,
   non-monotonic musical position IDs can index the full precomputed RoPE cache.

V1 protocol
-----------
* Prefix/query tokens: positions ``0 .. prefix_len-1``.
* CHORD / BEAT / NOTE: ``prefix_len + BeatPosition``.
* INST / STO / EOS / SOS / other structural tokens: ``prefix_len``.
* Padding: ``0`` (masked by attention_mask).

Causal order is NOT encoded here.  It remains entirely determined by the
serialized track-major token order and the causal attention mask.
"""

from __future__ import annotations

from types import MethodType
from typing import Iterable, Sequence, Tuple

import torch


def _full_range_rotary_forward(self, x: torch.Tensor, seq_len: int | None = None):
    """Qwen2RotaryEmbedding.forward replacement for transformers==4.37.x.

    Qwen2 4.37 normally returns ``cos_cached[:seq_len]`` and
    ``sin_cached[:seq_len]`` where ``seq_len`` is based on current KV length.
    With musical-time positions, ``max(position_ids)`` can be larger than the
    current KV length (e.g. prefix 0..4 followed by Beat_12 => position 17).

    We therefore expose the whole already-precomputed cache.  If the actual KV
    length itself grows beyond the current cache we preserve the original
    dynamic cache extension behavior.

    IMPORTANT: production code should still assert that all musical position
    IDs are < config.max_position_embeddings (or otherwise explicitly grow the
    cache).  The DID v1 design is safely far below that limit.
    """

    if seq_len is None:
        seq_len = int(x.shape[-2])
    else:
        seq_len = int(seq_len)

    if seq_len > int(self.max_seq_len_cached):
        self._set_cos_sin_cache(
            seq_len=seq_len,
            device=x.device,
            dtype=x.dtype,
        )

    return (
        self.cos_cached.to(device=x.device, dtype=x.dtype),
        self.sin_cached.to(device=x.device, dtype=x.dtype),
    )


def enable_full_range_qwen2_rope(qwen2_model) -> int:
    """Patch every self-attention RoPE module of a ``Qwen2Model`` in place.

    Returns the number of layers patched.  Calling this function repeatedly is
    safe; already patched rotary modules are skipped.

    This function deliberately avoids reconstructing ``Qwen2RotaryEmbedding``
    objects so it does not depend on constructor details.  It only replaces
    ``forward`` on the existing per-layer rotary modules.
    """

    layers = getattr(qwen2_model, "layers", None)
    if layers is None:
        raise TypeError("Expected a Qwen2Model-like object with a .layers attribute.")

    patched = 0
    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None or not hasattr(self_attn, "rotary_emb"):
            raise TypeError(f"Layer {layer_idx} has no self_attn.rotary_emb.")

        rotary = self_attn.rotary_emb
        if getattr(rotary, "_did_full_range_rope_enabled", False):
            continue

        # Keep the original callable for debugging/reversibility.  It is not a
        # parameter/buffer, so this has no checkpoint/state_dict consequence.
        rotary._did_original_forward = rotary.forward
        rotary.forward = MethodType(_full_range_rotary_forward, rotary)
        rotary._did_full_range_rope_enabled = True
        patched += 1

    return patched


def disable_full_range_qwen2_rope(qwen2_model) -> int:
    """Restore original RoPE ``forward`` methods when available."""

    layers = getattr(qwen2_model, "layers", None)
    if layers is None:
        raise TypeError("Expected a Qwen2Model-like object with a .layers attribute.")

    restored = 0
    for layer in layers:
        rotary = layer.self_attn.rotary_emb
        original = getattr(rotary, "_did_original_forward", None)
        if original is None:
            continue
        rotary.forward = original
        rotary._did_full_range_rope_enabled = False
        delattr(rotary, "_did_original_forward")
        restored += 1
    return restored


def _as_lookup_tensor(
    beat_id_to_position: Sequence[int] | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    lookup = torch.as_tensor(beat_id_to_position, dtype=torch.long, device=device)
    if lookup.ndim != 1:
        raise ValueError("beat_id_to_position must be a 1-D lookup table.")
    if lookup.numel() == 0:
        raise ValueError("beat_id_to_position must not be empty.")
    return lookup


def _temporal_type_mask(type_ids: torch.Tensor, temporal_type_ids: Iterable[int]) -> torch.Tensor:
    mask = torch.zeros_like(type_ids, dtype=torch.bool)
    for type_id in temporal_type_ids:
        mask |= type_ids.eq(int(type_id))
    return mask


def build_event_musical_position_ids(
    note_ids: torch.Tensor,
    note_mask: torch.Tensor,
    *,
    prefix_len: int,
    note_type_idx: int,
    note_beat_idx: int,
    temporal_type_ids: Sequence[int],
    beat_id_to_position: Sequence[int] | torch.Tensor,
    max_position_embeddings: int | None = None,
) -> torch.LongTensor:
    """Build musical positions for DID event tokens (without prefix tokens).

    Parameters
    ----------
    note_ids:
        Shape ``[..., N, F]``.
    note_mask:
        Shape ``[..., N]``; 1/True means an active event token.
    prefix_len:
        Number of non-event prefix tokens placed before this bar in the Note
        Decoder (4 continuation; 5 conditioned tasks in current DID).
    temporal_type_ids:
        Type IDs whose tokens carry a real onset: CHORD, BEAT and NOTE.
    beat_id_to_position:
        Existing DID lookup mapping beat token ID -> integer musical position;
        invalid/non-Beat IDs must contain a negative value.

    Returns
    -------
    LongTensor with shape ``[..., N]``.
    """

    if note_ids.ndim < 2:
        raise ValueError("note_ids must have shape [..., N, F].")
    if note_mask.shape != note_ids.shape[:-1]:
        raise ValueError(
            f"note_mask shape {tuple(note_mask.shape)} must equal "
            f"note_ids.shape[:-1] {tuple(note_ids.shape[:-1])}."
        )
    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive.")
    if note_type_idx < 0 or note_type_idx >= note_ids.shape[-1]:
        raise IndexError("note_type_idx is out of range.")
    if note_beat_idx < 0 or note_beat_idx >= note_ids.shape[-1]:
        raise IndexError("note_beat_idx is out of range.")

    device = note_ids.device
    active = note_mask.bool()
    type_ids = note_ids[..., note_type_idx].long()
    temporal = active & _temporal_type_mask(type_ids, temporal_type_ids)

    # V1: all active structural tokens live at bar-start coordinate; padding=0.
    positions = torch.full(
        note_ids.shape[:-1],
        fill_value=int(prefix_len),
        dtype=torch.long,
        device=device,
    )
    positions = positions.masked_fill(~active, 0)

    if temporal.any():
        beat_ids = note_ids[..., note_beat_idx].long()
        lookup = _as_lookup_tensor(beat_id_to_position, device=device)

        temporal_beat_ids = beat_ids[temporal]
        if (temporal_beat_ids < 0).any() or (temporal_beat_ids >= lookup.numel()).any():
            bad = temporal_beat_ids[
                (temporal_beat_ids < 0) | (temporal_beat_ids >= lookup.numel())
            ][:10].detach().cpu().tolist()
            raise ValueError(f"Temporal token contains beat id outside lookup range: {bad}")

        musical_offsets = lookup.index_select(0, temporal_beat_ids)
        if (musical_offsets < 0).any():
            bad_ids = temporal_beat_ids[musical_offsets < 0][:10].detach().cpu().tolist()
            raise ValueError(
                "CHORD/BEAT/NOTE token uses an invalid/non-Beat beat token id: "
                f"{bad_ids}"
            )

        positions[temporal] = int(prefix_len) + musical_offsets

    if max_position_embeddings is not None and active.any():
        active_pos = positions[active]
        max_pos = int(active_pos.max().item())
        if max_pos >= int(max_position_embeddings):
            raise ValueError(
                f"Musical position id {max_pos} exceeds/equal max_position_embeddings="
                f"{int(max_position_embeddings)}. Refuse silent clipping."
            )

    return positions


def build_note_decoder_musical_position_ids(
    note_ids: torch.Tensor,
    note_mask: torch.Tensor,
    *,
    prefix_len: int,
    note_type_idx: int,
    note_beat_idx: int,
    temporal_type_ids: Sequence[int],
    beat_id_to_position: Sequence[int] | torch.Tensor,
    max_position_embeddings: int | None = None,
) -> torch.LongTensor:
    """Build full Note-Decoder positions = prefix positions + event positions.

    Input shape ``[..., N, F]`` -> output shape ``[..., prefix_len + N]``.
    """

    event_pos = build_event_musical_position_ids(
        note_ids,
        note_mask,
        prefix_len=prefix_len,
        note_type_idx=note_type_idx,
        note_beat_idx=note_beat_idx,
        temporal_type_ids=temporal_type_ids,
        beat_id_to_position=beat_id_to_position,
        max_position_embeddings=max_position_embeddings,
    )

    leading_shape = event_pos.shape[:-1]
    prefix = torch.arange(prefix_len, dtype=torch.long, device=note_ids.device)
    prefix = prefix.view(*([1] * len(leading_shape)), prefix_len)
    prefix = prefix.expand(*leading_shape, prefix_len)

    full = torch.cat([prefix, event_pos], dim=-1)
    if max_position_embeddings is not None and full.numel() > 0:
        max_pos = int(full.max().item())
        if max_pos >= int(max_position_embeddings):
            raise ValueError(
                f"Musical position id {max_pos} exceeds/equal max_position_embeddings="
                f"{int(max_position_embeddings)}."
            )
    return full


def build_single_event_musical_position_id(
    final_event_ids: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    prefix_len: int,
    note_type_idx: int,
    note_beat_idx: int,
    temporal_type_ids: Sequence[int],
    beat_id_to_position: Sequence[int] | torch.Tensor,
    max_position_embeddings: int | None = None,
) -> torch.LongTensor:
    """Generation-step builder for *finalized* event IDs.

    ``final_event_ids`` must already include all grammar fixes and condition
    forcing.  This ordering is intentional: force/finalize token -> calculate
    position -> append to Qwen2 KV cache.

    Shape ``[..., F]`` -> ``[..., 1]``.
    """

    if final_event_ids.ndim < 1:
        raise ValueError("final_event_ids must have shape [..., F].")
    if active_mask.shape != final_event_ids.shape[:-1]:
        raise ValueError("active_mask must match final_event_ids.shape[:-1].")

    event_pos = build_event_musical_position_ids(
        final_event_ids.unsqueeze(-2),
        active_mask.unsqueeze(-1),
        prefix_len=prefix_len,
        note_type_idx=note_type_idx,
        note_beat_idx=note_beat_idx,
        temporal_type_ids=temporal_type_ids,
        beat_id_to_position=beat_id_to_position,
        max_position_embeddings=max_position_embeddings,
    )
    return event_pos
