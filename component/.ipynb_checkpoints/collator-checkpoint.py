from copy import deepcopy
from typing import Any, Dict, List, Tuple, Optional

import torch


class DidDataCollator(object):
    def __init__(self, config, max_seq_length: int):
        self.max_seq_length = int(max_seq_length)

        self.pos_pad_token_id = list(config.pad_pos_id)
        self.note_pad_token_id = list(config.sto_note_id)
        self.eos_note_id = list(config.eos_note_id)
        self.instrument_pad_token_id = 0
        self.task_mode = config.task_mode
        self.note_features = list(config.note_features)

        self.type_idx = self.note_features.index("type")
        self.beat_idx = self.note_features.index("beat")
        self.chord_idx = self.note_features.index("chord")
        self.instrument_idx = self.note_features.index("instrument")

        self.chord_type_id = config.chord_note_id[self.type_idx]
        self.inst_type_id = config.inst_note_id[self.type_idx]
        self.beat_type_id = config.beat_note_id[self.type_idx]
        self.sto_type_id = config.sto_note_id[self.type_idx]
        self.eos_type_id = config.eos_note_id[self.type_idx]

        self.beat_0_id = int(config.beat_0_id)
        self.beat_id_to_position = tuple(
            int(x) for x in config.beat_id_to_position
        )
        self.condition_instrument_prefix_ids = tuple(config.condition_instrument_prefix_ids)

        if self.task_mode == "chord":
            if self.chord_idx is None:
                raise ValueError(
                    "Chord task requires 'chord' "
                    "in note_features."
                )
        if self.task_mode == "accompaniment":
            if self.instrument_idx is None:
                raise ValueError(
                    "Accompaniment task requires 'instrument' "
                    "in note_features."
                )
            if len(self.condition_instrument_prefix_ids) == 0:
                raise ValueError(
                    "Accompaniment task requires a non-empty "
                    "condition_instrument_prefix_ids."
                )

    def _beat_position(self, beat_id: int) -> int:
        beat_id = int(beat_id)
        if beat_id < 0 or beat_id >= len(self.beat_id_to_position):
            raise ValueError(f"Beat id out of range: {beat_id}")
        position = int(self.beat_id_to_position[beat_id])
        if position < 0:
            raise ValueError(f"Token id {beat_id} is not a valid Beat_x token.")
        return position
    
    def _validate_bar_chord_protocol(
        self,
        bar: torch.Tensor,
        chord_prefix_len: int,
        batch_idx: int,
        time_idx: int,
    ):
        chord_changes = []
        previous_position = None

        for n in range(chord_prefix_len):
            token = bar[n]
            beat_id = int(token[self.beat_idx].item())
            chord_id = int(token[self.chord_idx].item())
            beat_position = self._beat_position(beat_id)
            if n == 0:
                if beat_id != self.beat_0_id or beat_position != 0:
                    raise ValueError("Every bar must start with CHORD(..., Beat_0).")
            elif beat_position <= previous_position:
                raise ValueError("CHORD positions must be strictly increasing.")
            chord_changes.append((beat_position, chord_id))
            previous_position = beat_position

        # 验证所有 BEAT.chord 与时间线一致
        for n in range(chord_prefix_len, bar.shape[0]):
            token = bar[n]
            token_type = int(token[self.type_idx].item())
            if token_type != self.beat_type_id:
                continue

            beat_position = self._beat_position(int(token[self.beat_idx].item()))

            expected_chord_id = chord_changes[0][1]

            for change_position, change_chord_id in chord_changes:
                if change_position > beat_position:
                    break
                expected_chord_id = change_chord_id

            actual_chord_id = int(token[self.chord_idx].item())

            if actual_chord_id != expected_chord_id:
                raise ValueError("BEAT.chord is inconsistent with the CHORD timeline." )

    @staticmethod
    def _safe_get_time_len(sample: Dict[str, Any]) -> int:
        return len(sample.get("pos_input_ids", []))

    def _allocate_condition_buffers(
        self,
        source_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source_ids.dim() != 4:
            raise ValueError(
                "source_ids must be [B,T,N,F], "
                f"got {tuple(source_ids.shape)}."
            )
        bsz, target_steps, note_nums, feat_dim = source_ids.shape
        pad_vec = torch.tensor(
            self.note_pad_token_id,
            dtype=source_ids.dtype,
            device=source_ids.device,
        )
        if pad_vec.numel() != feat_dim:
            raise ValueError(
                "note_pad_token_id feature count does not match "
                f"source_ids: {pad_vec.numel()} vs {feat_dim}."
            )
        condition_ids = pad_vec.view(1, 1, 1, feat_dim).expand(bsz, target_steps, note_nums, feat_dim).clone()
        condition_mask = torch.zeros(bsz, target_steps, note_nums, dtype=torch.long, device=source_ids.device)
        return condition_ids, condition_mask
    
    def _get_valid_bar_tokens(
        self,
        bar_ids: torch.Tensor,
        bar_mask: torch.Tensor,
        batch_idx: int,
        time_idx: int,
    ) -> torch.Tensor:
        if bar_ids.dim() != 2:
            raise ValueError(
                f"bar_ids must be [N,F], got {tuple(bar_ids.shape)}."
            )
        if bar_mask.dim() != 1:
            raise ValueError(
                f"bar_mask must be [N], got {tuple(bar_mask.shape)}."
            )
        mask_bool = bar_mask.bool()
        valid_len = int(mask_bool.long().sum().item())
        if valid_len <= 0:
            raise ValueError(
                "Active target bar has no valid source note tokens: "
                f"batch={batch_idx}, target_bar={time_idx}."
            )
        # 检查 note mask 必须是连续的 1...1,0...0
        expected_mask = (torch.arange(mask_bool.numel(), device=mask_bool.device) < valid_len)
        if not torch.equal(mask_bool, expected_mask):
            raise ValueError(
                "note_attention_mask is not prefix-contiguous at "
                f"batch={batch_idx}, target_bar={time_idx}. "
                f"mask={bar_mask.detach().cpu().tolist()}"
            )
        return bar_ids[:valid_len]

    def _extract_chord_condition(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        target_time_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        source_ids:
            [B,T-1,N,F]
            已经完成 shift：
            source_ids[:, t] 对应 target bar t。

        source_mask:
            [B,T-1,N]

        target_time_mask:
            [B,T-1]
        """
        if source_ids.dim() != 4:
            raise ValueError(
                "source_ids must be [B,T,N,F]."
            )

        if source_mask.shape != source_ids.shape[:3]:
            raise ValueError(
                "source_mask shape does not match source_ids: "
                f"{tuple(source_mask.shape)} vs "
                f"{tuple(source_ids.shape[:3])}."
            )

        if target_time_mask.shape != source_ids.shape[:2]:
            raise ValueError(
                "target_time_mask shape does not match source_ids: "
                f"{tuple(target_time_mask.shape)} vs "
                f"{tuple(source_ids.shape[:2])}."
            )

        condition_ids, condition_mask = self._allocate_condition_buffers(source_ids)
        bsz, target_steps, _, _ = source_ids.shape

        for b in range(bsz):
            for t in range(target_steps):
                if int(target_time_mask[b, t].item()) == 0:
                    continue
                bar = self._get_valid_bar_tokens(source_ids[b, t], source_mask[b, t], batch_idx=b, time_idx=t)
                bar_types, prefix_len = bar[:, self.type_idx], 0
                while (
                    (prefix_len < bar.shape[0])
                    and 
                    (int(bar_types[prefix_len].item()) == self.chord_type_id)
                ):
                    prefix_len += 1

                # 根据编码协议，每一个有效小节都必须有至少一个 CHORD-prefix，包括 Chord_N_N。
                if prefix_len == 0:
                    raise ValueError(
                        "Active bar has no CHORD-prefix. "
                        "This violates the DID encoding protocol: "
                        f"batch={b}, target_bar={t}."
                    )

                # CHORD 只能位于连续前缀。
                if (bar_types[prefix_len:] == self.chord_type_id).any():
                    late_positions = (
                        bar_types[prefix_len:] == self.chord_type_id
                    ).nonzero(as_tuple=False).flatten()

                    raise ValueError(
                        "CHORD token found outside the chord prefix: "
                        f"batch={b}, target_bar={t}, "
                        f"late_positions="
                        f"{(late_positions + prefix_len).tolist()}."
                    )
                self._validate_bar_chord_protocol(
                    bar=bar,
                    chord_prefix_len=prefix_len,
                    batch_idx=b,
                    time_idx=t,
                )
                condition_ids[b, t, :prefix_len, :] = bar[:prefix_len, :]
                condition_mask[b, t, :prefix_len] = 1

        
        return condition_ids, condition_mask

    def _extract_accompaniment_condition(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        target_time_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if source_mask.shape != source_ids.shape[:3]:
            raise ValueError("source_mask shape does not match source_ids.")
        if target_time_mask.shape != source_ids.shape[:2]:
            raise ValueError("target_time_mask shape does not match source_ids.")

        selected_ids = list(self.condition_instrument_prefix_ids)
        selected_set = set(selected_ids)
        condition_ids, condition_mask = self._allocate_condition_buffers(source_ids)

        bsz, target_steps, _, _ = source_ids.shape
        for b in range(bsz):
            for t in range(target_steps):
                if int(target_time_mask[b, t].item()) == 0:
                    continue
                bar = self._get_valid_bar_tokens(
                    source_ids[b, t],
                    source_mask[b, t],
                    batch_idx=b,
                    time_idx=t,
                )
                bar_types = bar[:, self.type_idx]
                # 1. 找真实 STO/EOS
                stop_mask = (
                    (bar_types == self.sto_type_id)
                    | (bar_types == self.eos_type_id)
                )
                stop_positions = stop_mask.nonzero(as_tuple=False).flatten()

                if stop_positions.numel() == 0:
                    raise ValueError(
                        "Active condition bar contains no STO/EOS: "
                        f"batch={b}, target_bar={t}."
                    )

                stop_pos = int(stop_positions[0].item())

                # 按当前 Dataset mask 设计，STO/EOS 应该是最后一个有效 token。
                if stop_pos != bar.shape[0] - 1:
                    raise ValueError(
                        "STO/EOS is not the last valid token: "
                        f"batch={b}, target_bar={t}, "
                        f"stop_pos={stop_pos}, "
                        f"valid_len={bar.shape[0]}."
                    )

                # 2. CHORD-prefix
                chord_prefix_end = 0
                while (
                    (chord_prefix_end < stop_pos)
                    and (int(bar_types[chord_prefix_end].item()) == self.chord_type_id)
                ):
                    chord_prefix_end += 1

                if chord_prefix_end == 0:
                    raise ValueError(
                        "Accompaniment condition bar has no "
                        "CHORD-prefix: "
                        f"batch={b}, target_bar={t}."
                    )

                if (bar_types[chord_prefix_end:stop_pos] == self.chord_type_id).any():
                    raise ValueError(
                        "CHORD token found outside chord prefix: "
                        f"batch={b}, target_bar={t}."
                    )
                self._validate_bar_chord_protocol(
                    bar=bar,
                    chord_prefix_len=chord_prefix_end,
                    batch_idx=b,
                    time_idx=t,
                )
                # 3. Selected condition instruments must form the leading INST prefix.
                inst_positions: List[int] = []
                inst_ids: List[int] = []
                for n in range(chord_prefix_end, stop_pos):
                    if int(bar_types[n].item()) == self.inst_type_id:
                        inst_positions.append(n)
                        inst_ids.append(int(bar[n, self.instrument_idx].item()))

                if len(inst_ids) != len(set(inst_ids)):
                    raise ValueError(
                        "Duplicate INST blocks found in one bar: "
                        f"batch={b}, target_bar={t}, INST order={inst_ids}."
                    )

                present_selected_ids = [
                    instrument_id for instrument_id in selected_ids if instrument_id in inst_ids
                ]
                actual_prefix = inst_ids[:len(present_selected_ids)]
                if actual_prefix != present_selected_ids:
                    raise ValueError(
                        "Selected condition instruments are not a continuous prefix of this bar. "
                        f"batch={b}, target_bar={t}, INST order={inst_ids}, "
                        f"configured prefix={selected_ids}, present selected={present_selected_ids}."
                    )

                prefix_inst_count = len(present_selected_ids)
                if prefix_inst_count < len(inst_positions):
                    cut_pos = inst_positions[prefix_inst_count]
                else:
                    cut_pos = stop_pos

                if cut_pos <= 0:
                    raise ValueError(
                        "Computed empty accompaniment condition: "
                        f"batch={b}, target_bar={t}."
                    )
                condition_ids[b, t, :cut_pos, :] = bar[:cut_pos, :]
                condition_mask[b, t, :cut_pos] = 1

        return condition_ids, condition_mask
    
    def _build_condition_inputs(
        self,
        note_input_ids: torch.Tensor,
        note_attention_mask: torch.Tensor,
        pos_attention_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor],]:
        if self.task_mode == "continuation":
            return None, None

        if note_input_ids.dim() != 4:
            raise ValueError(
                "note_input_ids must be [B,T,N,F], "
                f"got {tuple(note_input_ids.shape)}."
            )

        if note_attention_mask.shape != (
            note_input_ids.shape[0],
            note_input_ids.shape[1],
            note_input_ids.shape[2],
        ):
            raise ValueError("note_attention_mask shape mismatch.")

        if pos_attention_mask.shape != (
            note_input_ids.shape[0],
            note_input_ids.shape[1],
        ):
            raise ValueError("pos_attention_mask shape mismatch.")

        if note_input_ids.shape[1] < 2:
            raise ValueError("Conditioned training requires at least SOS plus one target bar.")

        # 唯一一次 shift。
        # input:
        #   note_input_ids[:,0] = SOS
        #   note_input_ids[:,1] = target bar 0
        #
        # output:
        #   source_ids[:,0] = target bar 0 condition
        source_ids = note_input_ids[:, 1:, :, :]
        source_mask = note_attention_mask[:, 1:, :]
        target_time_mask = pos_attention_mask[:, 1:]

        if source_ids.shape[:2] != target_time_mask.shape:
            raise RuntimeError(
                "Internal condition alignment error: "
                f"source time shape={source_ids.shape[:2]}, "
                f"target mask shape={target_time_mask.shape}."
            )

        if self.task_mode == "chord":
            return self._extract_chord_condition(
                source_ids=source_ids,
                source_mask=source_mask,
                target_time_mask=target_time_mask,
            )

        if self.task_mode == "accompaniment":
            return self._extract_accompaniment_condition(
                source_ids=source_ids,
                source_mask=source_mask,
                target_time_mask=target_time_mask,
            )

        raise RuntimeError(
            f"Unexpected task_mode={self.task_mode!r}."
        )
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            raise ValueError("All samples in this batch are None. Please check dataset loading logs.")

        pos_lengths = [self._safe_get_time_len(x) for x in batch]
        batch_pos_max_len = min(max(pos_lengths), self.max_seq_length)

        note_lengths = []
        instrument_lengths = []
        for x in batch:
            note_lengths.extend(len(ts) for ts in x["note_input_ids"])
            instrument_lengths.extend(len(ts) for ts in x["instrument_input_ids"])

        batch_note_max_len = max(note_lengths)
        batch_instrument_max_len = max(instrument_lengths)

        pos_input_ids_batch, pos_attention_mask_batch = [], []
        note_input_ids_batch, note_attention_mask_batch = [], []
        instrument_input_ids_batch, instrument_attention_mask_batch = [], []
        labels_batch = []

        for sample in batch:
            pos_input_ids = deepcopy(sample["pos_input_ids"])
            pos_attention_mask = deepcopy(sample["pos_attention_mask"])
            note_input_ids = deepcopy(sample["note_input_ids"])
            note_attention_mask = deepcopy(sample["note_attention_mask"])
            instrument_input_ids = deepcopy(sample["instrument_input_ids"])
            instrument_attention_mask = deepcopy(sample["instrument_attention_mask"])
            target_mask = deepcopy(sample["target_mask"])

            original_seq_len = len(pos_input_ids)

            pos_padding_len = batch_pos_max_len - len(pos_input_ids)
            if pos_padding_len > 0:
                pos_input_ids += [self.pos_pad_token_id[:] for _ in range(pos_padding_len)]
                pos_attention_mask += [0] * pos_padding_len

                note_input_ids += [
                    [self.note_pad_token_id[:] for _ in range(batch_note_max_len)]
                    for _ in range(pos_padding_len)
                ]
                note_attention_mask += [[0] * batch_note_max_len for _ in range(pos_padding_len)]
                target_mask += [[0] * batch_note_max_len for _ in range(pos_padding_len)]

                instrument_input_ids += [
                    [self.instrument_pad_token_id] * batch_instrument_max_len
                    for _ in range(pos_padding_len)
                ]
                instrument_attention_mask += [[0] * batch_instrument_max_len for _ in range(pos_padding_len)]

            pos_input_ids = pos_input_ids[:batch_pos_max_len]
            pos_attention_mask = pos_attention_mask[:batch_pos_max_len]
            note_input_ids = note_input_ids[:batch_pos_max_len]
            note_attention_mask = note_attention_mask[:batch_pos_max_len]
            target_mask = target_mask[:batch_pos_max_len]
            instrument_input_ids = instrument_input_ids[:batch_pos_max_len]
            instrument_attention_mask = instrument_attention_mask[:batch_pos_max_len]

            for ts in range(batch_pos_max_len):
                note_padding_len = batch_note_max_len - len(note_input_ids[ts])
                if note_padding_len > 0:
                    note_input_ids[ts] += [self.note_pad_token_id[:] for _ in range(note_padding_len)]
                    note_attention_mask[ts] += [0] * note_padding_len
                    target_mask[ts] += [0] * note_padding_len

                instrument_padding_len = batch_instrument_max_len - len(instrument_input_ids[ts])
                if instrument_padding_len > 0:
                    instrument_input_ids[ts] += [self.instrument_pad_token_id] * instrument_padding_len
                    instrument_attention_mask[ts] += [0] * instrument_padding_len

            # If truncated in time dimension, force EOS in the last kept bar.
            if original_seq_len > self.max_seq_length:
                found_sto = False
                for idx, note in enumerate(note_input_ids[-1]):
                    if note == self.note_pad_token_id:
                        note_input_ids[-1][idx] = self.eos_note_id[:]
                        note_attention_mask[-1][idx] = 1
                        target_mask[-1][idx] = 1
                        found_sto = True
                        break
                if not found_sto:
                    last_idx = len(note_input_ids[-1]) - 1
                    note_input_ids[-1][last_idx] = self.eos_note_id[:]
                    note_attention_mask[-1][last_idx] = 1
                    target_mask[-1][last_idx] = 1

            labels = []
            for ts in range(batch_pos_max_len):
                labels_per_ts = []
                for idx, note in enumerate(note_input_ids[ts]):
                    if target_mask[ts][idx] == 1:
                        labels_per_ts.append(note[:])
                    else:
                        labels_per_ts.append([-100] * len(note))
                labels.append(labels_per_ts)

            pos_input_ids_batch.append(pos_input_ids)
            pos_attention_mask_batch.append(pos_attention_mask)
            note_input_ids_batch.append(note_input_ids)
            note_attention_mask_batch.append(note_attention_mask)
            instrument_input_ids_batch.append(instrument_input_ids)
            instrument_attention_mask_batch.append(instrument_attention_mask)
            labels_batch.append(labels)
        
        output = {
            "pos_input_ids": torch.tensor(pos_input_ids_batch, dtype=torch.long),
            "pos_attention_mask": torch.tensor(pos_attention_mask_batch, dtype=torch.long),
            "note_input_ids": torch.tensor(note_input_ids_batch, dtype=torch.long),
            "note_attention_mask": torch.tensor(note_attention_mask_batch, dtype=torch.long),
            "instrument_input_ids": torch.tensor(instrument_input_ids_batch, dtype=torch.long),
            "instrument_attention_mask": torch.tensor(instrument_attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
        
        if self.task_mode != "continuation":
            condition_ids, condition_mask =  self._build_condition_inputs(
                    note_input_ids=output["note_input_ids"],
                    note_attention_mask=output["note_attention_mask"],
                    pos_attention_mask=output["pos_attention_mask"],
            )

            output["condition_note_ids"] = condition_ids
            output["condition_note_mask"] = condition_mask

        return output
