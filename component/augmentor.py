import random
from typing import Optional, Union

import torch


class Augmentor:
    def __init__(
        self,
        vocab,
        music_config,
        aug_type: Union[str, None],
        input_length: int,
    ):
        self.vocab = vocab
        self.aug_type = aug_type
        self.input_length = input_length
        self.feature_list = vocab.feature_list
        self.num_features = len(self.feature_list)
        self.encoding_scheme = vocab.encoding_scheme

        self.type_idx = music_config.note_features.index("type")
        self.pitch_idx = music_config.note_features.index("pitch")
        self.chord_idx = music_config.note_features.index("chord") if "chord" in music_config.note_features else None
        self.chord_type_id = int(getattr(music_config, "chord_note_id", [0])[0])

        # DID chord protocol:
        # 0=PAD, 1=Chord_N_N, 2+=normal transposable chord classes.
        self.did_chord_shift_start = 2
        chord_vocab = getattr(vocab, "event2idx", {}).get("chord", None)
        if isinstance(chord_vocab, dict):
            transposable_ids = []
            for token_name, token_id in chord_vocab.items():
                if isinstance(token_name, str) and token_name.startswith("Chord_") and token_name != "Chord_N_N":
                    transposable_ids.append(int(token_id))
            if len(transposable_ids) > 0:
                self.did_chord_shift_start = min(transposable_ids)
            elif "Chord_N_N" in chord_vocab:
                self.did_chord_shift_start = int(chord_vocab["Chord_N_N"]) + 1

    def _get_shift(self, segment: torch.Tensor) -> int:
        # Sample one feasible semitone shift based on current pitch range.
        if self.encoding_scheme in ("cp", "nb"):
            pitch_mask = segment != 0
            pitch_segment = segment[pitch_mask[:, self.pitch_idx], self.pitch_idx]
            if pitch_segment.numel() == 0:
                return 0
            lowest_pitch = max(12, torch.min(pitch_segment))
            highest_pitch = min(119, torch.max(pitch_segment))
            lower_shift_bound = torch.where(lowest_pitch - torch.arange(6) > 11)[0][-1].item()
            upper_shift_bound = torch.where(highest_pitch + torch.arange(7) < 120)[0][-1].item()
            return random.randint(-lower_shift_bound, upper_shift_bound)

        if self.encoding_scheme == "did":
            pitch_plane = segment[:, :, self.pitch_idx]
            valid_mask = pitch_plane != 0
            pitch_segment = pitch_plane[valid_mask]
            if pitch_segment.numel() == 0:
                return 0
            lowest_pitch = max(12, torch.min(pitch_segment))
            highest_pitch = min(119, torch.max(pitch_segment))
            lower_shift_bound = torch.where(lowest_pitch - torch.arange(6) > 11)[0][-1].item()
            upper_shift_bound = torch.where(highest_pitch + torch.arange(7) < 120)[0][-1].item()
            return random.randint(-lower_shift_bound, upper_shift_bound)

        # remi
        mask_for_pitch = self.vocab.total_mask["pitch"].to(segment.device)
        segment_pitch_mask = mask_for_pitch[segment]
        segment_pitch = segment * segment_pitch_mask
        segment_pitch = segment_pitch[segment_pitch != 0]
        if segment_pitch.numel() == 0:
            return 0
        lower_bound = torch.argwhere(mask_for_pitch == 1)[0].item()
        upper_bound = torch.argwhere(mask_for_pitch == 1)[-1].item()
        lowest_pitch = max(lower_bound, torch.min(segment_pitch))
        highest_pitch = min(upper_bound, torch.max(segment_pitch))
        lower_shift_bound = torch.where(lowest_pitch - torch.arange(6) >= lower_bound)[0][-1].item()
        upper_shift_bound = torch.where(highest_pitch + torch.arange(7) <= upper_bound)[0][-1].item()
        return random.randint(-lower_shift_bound, upper_shift_bound)

    def apply_shift(self, segment: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return segment

        if self.encoding_scheme in ("cp", "nb"):
            segment_pitch_mask = segment != 0
            new_segment = segment.clone()
            new_segment[segment_pitch_mask[:, self.pitch_idx], self.pitch_idx] += shift
            if self.chord_idx is not None and "chord" in self.feature_list:
                segment_chord_mask = (segment[:, self.chord_idx] != 0) & (segment[:, self.chord_idx] != 1)
                new_segment[segment_chord_mask, self.chord_idx] = (
                    (((new_segment[segment_chord_mask, self.chord_idx] - 2) % 12) + shift) % 12
                    + ((new_segment[segment_chord_mask, self.chord_idx] - 2) // 12) * 12
                    + 2
                )
            return new_segment

        if self.encoding_scheme == "did":
            new_segment = segment.clone()

            # Shift pitch values.
            pitch_plane = new_segment[:, :, self.pitch_idx]
            pitch_mask = pitch_plane != 0
            pitch_plane[pitch_mask] += shift

            # Shift all valid chord values, regardless of token type.
            # This keeps DID attribute consistency when chord is meaningful on BEAT/CHORD tokens.
            if self.chord_idx is not None:
                chord_plane = new_segment[:, :, self.chord_idx]
                base = self.did_chord_shift_start
                chord_mask = chord_plane >= base
                if chord_mask.any():
                    target = chord_plane[chord_mask]
                    chord_plane[chord_mask] = (
                        (((target - base) % 12) + shift) % 12
                        + ((target - base) // 12) * 12
                        + base
                    )
            return new_segment

        # remi
        mask_for_pitch = self.vocab.total_mask["pitch"].to(segment.device)
        segment_pitch_mask = mask_for_pitch[segment]
        new_segment = segment.clone()
        new_segment_valid = (new_segment + shift) * segment_pitch_mask
        new_segment = new_segment * (1 - segment_pitch_mask) + new_segment_valid

        if self.chord_idx is not None and "chord" in self.feature_list:
            mask_for_chord = self.vocab.total_mask["chord"].clone().to(segment.device)
            chord_n_n_idx = torch.argwhere(mask_for_chord == 1)[-1].item()
            mask_for_chord[chord_n_n_idx] = 0
            start_idx_chord = self.vocab.remi_vocab_boundaries_by_key["chord"][0]
            segment_chord_mask = mask_for_chord[segment]
            new_segment_valid = (
                ((((new_segment - start_idx_chord) % 12 + shift) % 12) + ((new_segment - start_idx_chord) // 12) * 12 + start_idx_chord)
                * segment_chord_mask
            )
            new_segment = new_segment * (1 - segment_chord_mask) + new_segment_valid

        return new_segment

    def __call__(
        self,
        segment: torch.Tensor,
        shift: Optional[int] = None,
        return_shift: bool = False,
    ):
        if self.aug_type != "random":
            if return_shift:
                return segment, 0
            return segment

        applied_shift = self._get_shift(segment) if shift is None else int(shift)
        out = self.apply_shift(segment, applied_shift)
        if return_shift:
            return out, applied_shift
        return out
