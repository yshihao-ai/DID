import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from .augmentor import Augmentor
from data_representation.vocab_utils import MusicTokenVocabDID


class QuarterPlanMeterError(ValueError):
    """Raised when V9 quarter-PLAN sees data outside its fixed 4/4, resolution-4 contract."""


class DidDataset(Dataset):
    def __init__(
        self,
        music_config,
        dataset_path,
        max_seq_length,
        split="train",
        metadata_path=None,
        max_data_nums=None,
        aug_type="random",
    ):
        self.music_config = music_config
        self.max_seq_length = max_seq_length
        self.split = split
        self.task_mode = str(self.music_config.task_mode)
        self.use_harmonic_plan = bool(getattr(self.music_config, "use_harmonic_plan", False))
        self.bridge_instrument_id = None

        # Runtime schema keeps the legacy DID8 fields first. PLAN adds two numeric
        # fields only after augmentation/filtering, so old tuneidx/vocab remain valid.
        self.note_features = list(self.music_config.note_features)
        self.base_note_feature_count = int(getattr(self.music_config, "base_note_feature_count", len(self.note_features)))
        self.type_idx = self.note_features.index("type")
        self.chord_idx = self.note_features.index("chord")
        self.beat_idx = self.note_features.index("beat")
        self.instrument_idx = self.note_features.index("instrument")
        self.pitch_idx = self.note_features.index("pitch")
        self.note_type_id = int(self.music_config.note_note_id[self.type_idx])
        self.inst_type_id = int(self.music_config.inst_note_id[self.type_idx])
        self.sto_type_id = int(self.music_config.sto_note_id[self.type_idx])
        self.eos_type_id = int(self.music_config.eos_note_id[self.type_idx])
        self.sos_type_id = int(self.music_config.sos_note_id[self.type_idx])
        self.chord_type_id = int(self.music_config.chord_note_id[self.type_idx])
        self.plan_type_id = int(getattr(self.music_config, "plan_type_id", -1))
        self.plan_bass_idx = self.note_features.index("plan_bass") if self.use_harmonic_plan else None
        self.plan_set_idx = self.note_features.index("plan_set") if self.use_harmonic_plan else None
        self.condition_instrument_prefix_ids = tuple(self.music_config.condition_instrument_prefix_ids)

        vocab_data = None
        if self.use_harmonic_plan or (self.task_mode in {"chord", "accompaniment"} and not music_config.use_bridge):
            with open(self.music_config.vocab_file_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)

        if self.use_harmonic_plan:
            if self.base_note_feature_count != 8:
                raise ValueError(
                    "The current PLAN implementation expects the uploaded DID8 base schema "
                    "and appends [plan_bass, plan_set] at runtime."
                )
            if self.plan_type_id < 0:
                raise ValueError("PLAN is enabled but plan_type_id was not resolved.")
            self._init_plan_runtime_tables(vocab_data)

        if self.task_mode in {"chord", "accompaniment"} and not music_config.use_bridge:
            logger.info("Remove Bridge Track before building PLAN.")
            instrument_to_id = {
                str(instrument_name): int(instrument_id)
                for instrument_id, instrument_name in vocab_data["instrument"].items()
            }

            if "Instrument_24" not in instrument_to_id:
                raise ValueError("Instrument_24 (BRIDGE) is missing from the instrument vocabulary.")

            self.bridge_instrument_id = instrument_to_id["Instrument_24"]
            logger.info(
                f"Chord and Accompaniment mode: BRIDGE track will be removed in Dataset "
                f"(Instrument_24, encoded id={self.bridge_instrument_id})."
            )
        
        logger.info(f"Loading data: {dataset_path}")
        all_npz_files = glob.glob(os.path.join(dataset_path, "*.npz"))
        logger.info(f"Scanned {len(all_npz_files)} usable files in directory.")

        if metadata_path:
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found at {metadata_path}. Generating new split (80/10/10)...")
                self._generate_and_save_metadata(all_npz_files, metadata_path)
            else:
                logger.info("Metadata file is found.")
            valid_files = self._filter_files_by_metadata(all_npz_files, metadata_path, split)
        else:
            logger.warning("No metadata path provided. Using all scanned files (no split control).")
            valid_files = all_npz_files

        self.npz_file_paths = valid_files
        if max_data_nums:
            self.npz_file_paths = self.npz_file_paths[:max_data_nums]

        # Only continuation training uses full-coverage segmentation.
        # All other task/split combinations preserve the original one-file-per-sample behavior.
        self.use_segmentation = self.split == "train" and self.task_mode == "continuation"
        if self.use_segmentation:
            self.target_bars_per_segment = self.max_seq_length - 1
            if self.target_bars_per_segment <= 0:
                raise ValueError(f"max_seq_length must be at least 2, got {self.max_seq_length}")
            self.segment_specs = self._build_segment_specs(self.npz_file_paths)
            logger.info(
                f"Continuation segmentation: ON | "
                f"{len(self.npz_file_paths)} songs -> {len(self.segment_specs)} segments | "
                f"target_bars_per_segment={self.target_bars_per_segment}"
            )
        else:
            logger.info(
                f"Continuation segmentation: OFF | split={self.split}, task_mode={self.task_mode}"
            )

        self.augmentor = None
        use_aug = bool(getattr(self.music_config, "use_augmentation", False))
        if self.split == "train" and use_aug:
            vocab = MusicTokenVocabDID(
                in_vocab_file_path=Path(self.music_config.vocab_file_path),
                event_data=None,
                encoding_scheme="did",
                num_features=self.base_note_feature_count,
            )
            self.augmentor = Augmentor(
                vocab=vocab,
                music_config=music_config,
                aug_type=aug_type,
                input_length=max_seq_length,
            )
            logger.info("Online augmentation: ON (Augmentor)")
        else:
            logger.info("Online augmentation: OFF")

        if self.use_segmentation:
            logger.info(
                f"Final {split} dataset size: {len(self.npz_file_paths)} files, "
                f"{len(self.segment_specs)} samples"
            )
        else:
            logger.info(f"Final {split} dataset size: {len(self.npz_file_paths)} files")

    def _init_plan_runtime_tables(self, vocab_data):
        if vocab_data is None:
            raise ValueError("PLAN runtime tables require vocab_data.")

        self.pitch_id_to_midi = {}
        for token_id, token_name in vocab_data["pitch"].items():
            if isinstance(token_name, str) and token_name.startswith("Note_Pitch_"):
                self.pitch_id_to_midi[int(token_id)] = int(token_name.rsplit("_", 1)[-1])

        runtime_pos_features = [f for f in self.music_config.pos_features if f != "all_instruments"]
        self.time_signature_pos_idx = (
            runtime_pos_features.index("time_signature")
            if "time_signature" in runtime_pos_features else None
        )
        self.time_signature_id_to_bar_steps = {}
        for token_id, token_name in vocab_data.get("time_signature", {}).items():
            if not (isinstance(token_name, str) and token_name.startswith("time_signature_")):
                continue
            frac = token_name[len("time_signature_"):]
            try:
                numerator, denominator = (int(x) for x in frac.split("/", 1))
            except Exception as exc:
                raise ValueError(f"Invalid time-signature token: {token_name}") from exc
            steps = round(numerator * (4.0 / denominator) * int(self.music_config.in_beat_resolution))
            self.time_signature_id_to_bar_steps[int(token_id)] = max(int(steps), 1)
        self.default_bar_steps = 4 * int(self.music_config.in_beat_resolution)

    @staticmethod
    def _special_token_prefix(token, special):
        return list(special[:len(token)])

    def _expand_base_token(self, token):
        token = [int(x) for x in token[:self.base_note_feature_count]]
        return token + [0] * (len(self.note_features) - len(token))

    def _bar_steps_from_pos(self, pos_token):
        if self.time_signature_pos_idx is None or self.time_signature_pos_idx >= len(pos_token):
            return self.default_bar_steps
        ts_id = int(pos_token[self.time_signature_pos_idx])
        return int(self.time_signature_id_to_bar_steps.get(ts_id, self.default_bar_steps))

    def _split_bar_prefix_body_stop(self, valid_tokens):
        if not valid_tokens:
            return [], [], []
        types = [int(tok[self.type_idx]) for tok in valid_tokens]
        if types[0] in (self.sos_type_id, self.eos_type_id):
            return [], [], valid_tokens
        chord_end = 0
        while chord_end < len(valid_tokens) and types[chord_end] == self.chord_type_id:
            chord_end += 1
        if chord_end == 0:
            raise ValueError("Real music bar has no CHORD prefix before PLAN insertion.")
        stop_pos = None
        for i in range(chord_end, len(valid_tokens)):
            if types[i] in (self.sto_type_id, self.eos_type_id):
                stop_pos = i
                break
        if stop_pos is None:
            stop_pos = len(valid_tokens)
        return valid_tokens[:chord_end], valid_tokens[chord_end:stop_pos], valid_tokens[stop_pos:]

    def _condition_prefix_body_cut(self, body):
        """Return token index after existing condition-track blocks in track-major body."""
        if self.task_mode != "accompaniment" or not self.condition_instrument_prefix_ids:
            return 0
        inst_starts, inst_ids = [], []
        for idx, token in enumerate(body):
            if int(token[self.type_idx]) == self.inst_type_id:
                inst_starts.append(idx)
                inst_ids.append(int(token[self.instrument_idx]))
        selected = list(self.condition_instrument_prefix_ids)
        present_selected = [inst for inst in selected if inst in inst_ids]
        actual_prefix = inst_ids[:len(present_selected)]
        if actual_prefix != present_selected:
            raise ValueError(
                "Selected accompaniment condition instruments are not a continuous track prefix: "
                f"INST order={inst_ids}, configured={selected}, present={present_selected}."
            )
        prefix_count = len(present_selected)
        if prefix_count == 0:
            return 0
        if prefix_count < len(inst_starts):
            return inst_starts[prefix_count]
        return len(body)

    def _insert_harmonic_plan(self, notes_data, note_pos_arr):
        """Insert the onset-consistent quarter-level harmonic PLAN.

        Fixed contract (4/4, four quantization positions per quarter):
          PLAN_q = (quarter Beat, QuarterOnsetBassPC, QuarterOnsetIntervalSet)

        * quarter Beat is one of Beat_0 / Beat_4 / Beat_8 / Beat_12;
        * QuarterOnsetBassPC is the pitch class of the lowest MIDI-pitch NOTE
          onset anywhere inside [q, q+4);
        * QuarterOnsetIntervalSet is the union of those same NOTE-onset pitch
          classes, expressed relative to QuarterOnsetBassPC.

        PLAN is emitted only for quarters containing at least one NOTE onset, so
        every real 4/4 bar has at most four PLAN events. Sustained notes from an
        earlier onset never affect a later quarter's PLAN. Because the bass onset
        belongs to the same onset set, every emitted Set mask contains interval 0.
        """
        if int(self.music_config.in_beat_resolution) != 4:
            raise QuarterPlanMeterError(
                "DID v9 quarter-level PLAN requires in_beat_resolution=4."
            )

        quarter_positions = tuple(int(x) for x in self.music_config.plan_quarter_positions)
        quarter_beat_ids = tuple(int(x) for x in self.music_config.plan_quarter_beat_ids)
        if quarter_positions != (0, 4, 8, 12) or len(quarter_beat_ids) != 4:
            raise QuarterPlanMeterError(
                "DID v9 quarter-level PLAN requires quarter positions [0,4,8,12]."
            )
        quarter_position_to_beat_id = dict(zip(quarter_positions, quarter_beat_ids))

        output = []

        for row_idx, row in enumerate(notes_data):
            row_list = row.tolist() if hasattr(row, "tolist") else row
            mask = self._build_note_mask(row_list)
            valid_len = int(sum(mask))
            valid_base = [
                list(map(int, tok[:self.base_note_feature_count]))
                for tok in row_list[:valid_len]
            ]

            if not valid_base:
                output.append([])
                continue

            first_type = int(valid_base[0][self.type_idx])
            if first_type in (self.sos_type_id, self.eos_type_id):
                output.append([self._expand_base_token(tok) for tok in valid_base])
                continue

            pos_token = note_pos_arr[row_idx] if row_idx < len(note_pos_arr) else []
            bar_steps = self._bar_steps_from_pos(pos_token)
            if int(bar_steps) != 16:
                raise QuarterPlanMeterError(
                    "DID v9 quarter-level PLAN only supports 4/4 bars at resolution=4 "
                    f"(expected 16 positions/bar, got {bar_steps}; row={row_idx})."
                )

            prefix, body, stop = self._split_bar_prefix_body_stop(valid_base)

            attacks_by_position = {}
            for tok in body:
                if int(tok[self.type_idx]) != self.note_type_id:
                    continue

                beat_id = int(tok[self.beat_idx])
                if beat_id < 0 or beat_id >= len(self.music_config.beat_id_to_position):
                    raise ValueError(f"Invalid NOTE beat id={beat_id}.")
                beat_pos = int(self.music_config.beat_id_to_position[beat_id])
                if not (0 <= beat_pos < 16):
                    raise QuarterPlanMeterError(
                        "DID v9 quarter-level PLAN expects NOTE beat positions in [0,15], "
                        f"got position={beat_pos} (beat_id={beat_id})."
                    )

                pitch_id = int(tok[self.pitch_idx])
                if pitch_id not in self.pitch_id_to_midi:
                    raise ValueError(f"Unknown pitch token id={pitch_id}.")

                attacks_by_position.setdefault(beat_pos, []).append(
                    int(self.pitch_id_to_midi[pitch_id])
                )

            plan_tokens = []
            for quarter_start in quarter_positions:
                quarter_end = quarter_start + 4
                quarter_attack_positions = [
                    pos for pos in sorted(attacks_by_position)
                    if quarter_start <= pos < quarter_end
                ]

                if quarter_attack_positions:
                    quarter_attack_pitches = [
                        int(pitch)
                        for attack_pos in quarter_attack_positions
                        for pitch in attacks_by_position[attack_pos]
                    ]
                    quarter_bass_midi = min(quarter_attack_pitches)

                    quarter_bass_pc = int(quarter_bass_midi) % 12
                    attack_mask = 0
                    for pitch in quarter_attack_pitches:
                        interval = ((int(pitch) % 12) - quarter_bass_pc) % 12
                        attack_mask |= 1 << interval

                    if (attack_mask & 1) == 0:
                        raise RuntimeError(
                            "Onset-consistent Quarter PLAN must contain interval 0."
                        )

                    plan = [0] * len(self.note_features)
                    plan[self.type_idx] = self.plan_type_id
                    plan[self.beat_idx] = quarter_position_to_beat_id[quarter_start]
                    plan[self.plan_bass_idx] = quarter_bass_pc + 1
                    plan[self.plan_set_idx] = int(attack_mask)
                    plan_tokens.append(plan)

            prefix_rt = [self._expand_base_token(tok) for tok in prefix]
            body_rt = [self._expand_base_token(tok) for tok in body]
            stop_rt = [self._expand_base_token(tok) for tok in stop]

            if self.task_mode == "accompaniment":
                cut = self._condition_prefix_body_cut(body)
                final_bar = prefix_rt + body_rt[:cut] + plan_tokens + body_rt[cut:] + stop_rt
            else:
                final_bar = prefix_rt + plan_tokens + body_rt + stop_rt
            output.append(final_bar)

        return output

    def _generate_and_save_metadata(self, all_files, output_path):
        core_ids = set()
        for f_path in all_files:
            file_name = os.path.basename(f_path)
            file_stem = os.path.splitext(file_name)[0]
            core_ids.add(file_stem)

        sorted_ids = sorted(list(core_ids))
        total_songs = len(sorted_ids)
        if total_songs == 0:
            logger.error("No songs found to split!")
            return

        seed = 42
        random.seed(seed)
        random.shuffle(sorted_ids)

        train_end = int(total_songs * 0.8)
        valid_end = int(total_songs * 0.9)
        train_ids = sorted_ids[:train_end]
        valid_ids = sorted_ids[train_end:valid_end]
        test_ids = sorted_ids[valid_end:]
        logger.info(f"Generated Split - Train: {len(train_ids)}, Valid: {len(valid_ids)}, Test: {len(test_ids)}")

        meta_dict = {
            "shuffle_seed": seed,
            "shuffled_names": sorted_ids,
            "train": train_ids,
            "valid": valid_ids,
            "test": test_ids,
        }

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=4, ensure_ascii=False)
            logger.info(f"Metadata saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _filter_files_by_metadata(self, all_files, metadata_path, split):
        logger.info(f"Filtering files using metadata: {metadata_path} [{split}]")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                
            irr_ids = None
            if "Pop909" in metadata_path:
                irr_meta_path = Path(metadata_path).parent / "Pop909_irregular_tunes.json"
                logger.info(f"Pop909 irregular tune filter enabled: {irr_meta_path}")
                with open(irr_meta_path, "r", encoding="utf-8") as f:
                    irr_meta = json.load(f)
                irr_ids = set(irr_meta)
                
            if "LakhClean" in metadata_path:
                irr_meta_path = Path(metadata_path).parent / "LakhClean_irregular_tunes.json"
                logger.info(f"LakhClean irregular tune filter enabled: {irr_meta_path}")
                with open(irr_meta_path, "r", encoding="utf-8") as f:
                    irr_meta = json.load(f)
                irr_ids = set(irr_meta)
            
            if "SOD" in metadata_path:
                irr_meta_path = Path(metadata_path).parent / "SOD_irregular_tunes.json"
                logger.info(f"SOD irregular tune filter enabled: {irr_meta_path}")
                with open(irr_meta_path, "r", encoding="utf-8") as f:
                    irr_meta = json.load(f)
                irr_ids = set(irr_meta)

            if split not in meta:
                raise ValueError(f"Split '{split}' not found in metadata file.")
            target_ids = set(meta[split])
            filtered_files = []
            for file_path in all_files:
                file_name = os.path.basename(file_path)
                file_stem = os.path.splitext(file_name)[0]
                core_id = file_stem
                if core_id in target_ids:
                    if irr_ids is None:
                        filtered_files.append(file_path)
                    elif core_id not in irr_ids:
                        filtered_files.append(file_path)
            return filtered_files
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise e

    def _build_segment_specs(self, file_paths):
        segment_specs = []
        for file_path in file_paths:
            try:
                with np.load(file_path) as data:
                    if not all(key in data for key in ["pos", "notes", "instruments"]):
                        logger.warning(f"Skip malformed file: {file_path}")
                        continue
                    total_raw_steps = int(data["pos"].shape[0])

                # DID layout:
                # raw index 0      = SOS
                # raw index 1..B   = B real bars
                # raw index B + 1  = EOS
                num_bars = total_raw_steps - 2
                if num_bars <= 0:
                    logger.warning(f"No target bars in file: {file_path}")
                    continue

                # Keep the segment width unchanged, but randomize the hop so
                # adjacent continuation-training segments overlap. For integer
                # bar indices, [w/2, w] is implemented as [ceil(w/2), w].
                segment_width = self.target_bars_per_segment
                min_hop = (segment_width + 1) // 2
                bar_start = 0

                while bar_start < num_bars:
                    bar_end = min(bar_start + segment_width, num_bars)
                    segment_specs.append((file_path, bar_start, bar_end))

                    # The current segment already reaches the end of the song,
                    # so coverage is complete and no extra tail segment is needed.
                    if bar_end >= num_bars:
                        break

                    hop = random.randint(min_hop, segment_width)
                    bar_start += hop
            except Exception as exc:
                logger.warning(f"Failed to index {file_path}: {exc}")
        return segment_specs

    def __len__(self):
        if self.use_segmentation:
            return len(self.segment_specs)
        return len(self.npz_file_paths)

    def get_npz_file_path(self, idx):
        if self.use_segmentation:
            file_path, _, _ = self.segment_specs[idx]
            return file_path
        return self.npz_file_paths[idx]

    def _build_note_mask(self, note_list):
        if not note_list:
            return []
        width = len(note_list[0])
        note_padding_token = self._special_token_prefix(note_list[0], self.music_config.sto_note_id)
        sos_note_id = self._special_token_prefix(note_list[0], self.music_config.sos_note_id)
        eos_note_id = self._special_token_prefix(note_list[0], self.music_config.eos_note_id)

        last_note = None
        last_note_pos = -1
        note_mask = []
        for note in note_list:
            note_cmp = list(note[:width])
            if note_cmp != note_padding_token:
                note_mask.append(1)
                last_note_pos = len(note_mask) - 1
                last_note = note_cmp
            else:
                note_mask.append(0)

        if len(note_mask) == 0:
            return note_mask
        if last_note_pos < 0:
            note_mask[0] = 1
        elif last_note_pos + 1 < len(note_mask) and last_note != sos_note_id and last_note != eos_note_id:
            note_mask[last_note_pos + 1] = 1
        return note_mask

    def _remove_bridge_track(self, notes_arr, instruments_arr):
        bridge_id = self.bridge_instrument_id
        note_pad = np.asarray(self.music_config.sto_note_id[:notes_arr.shape[-1]], dtype=np.int64)

        filtered_notes = np.broadcast_to(note_pad, notes_arr.shape).copy()
        filtered_instruments = np.zeros_like(instruments_arr)

        for time_idx, note_row in enumerate(notes_arr):
            note_mask = self._build_note_mask(note_row.tolist())
            valid_len = int(sum(note_mask))
            valid_tokens = note_row[:valid_len]

            kept_tokens = []
            skipping_bridge = False

            for token in valid_tokens:
                token_type = int(token[self.type_idx])

                if token_type == self.inst_type_id:
                    skipping_bridge = int(token[self.instrument_idx]) == bridge_id
                    if not skipping_bridge:
                        kept_tokens.append(token)
                    continue

                if token_type in (self.sos_type_id, self.sto_type_id, self.eos_type_id):
                    skipping_bridge = False
                    kept_tokens.append(token)
                    continue

                if not skipping_bridge:
                    kept_tokens.append(token)

            if kept_tokens:
                filtered_notes[time_idx, :len(kept_tokens)] = np.asarray(kept_tokens, dtype=np.int64)

        for time_idx, instrument_row in enumerate(instruments_arr):
            kept_instruments = [
                int(instrument_id)
                for instrument_id in instrument_row
                if int(instrument_id) > 0 and int(instrument_id) != bridge_id
            ]

            if kept_instruments:
                filtered_instruments[time_idx, :len(kept_instruments)] = kept_instruments

        return filtered_notes, filtered_instruments

    def __getitem__(self, index):
        if self.use_segmentation:
            npz_file_path, bar_start, bar_end = self.segment_specs[index]
        else:
            npz_file_path = self.npz_file_paths[index]
            bar_start = None
            bar_end = None

        try:
            data = np.load(npz_file_path)
            if not all(key in data for key in ["pos", "notes", "instruments"]):
                logger.warning(f"Missing required tensors in npz file: {npz_file_path}")
                return None

            pos_arr = np.asarray(data["pos"], dtype=np.int64)
            notes_arr = np.asarray(data["notes"], dtype=np.int64)
            instruments_arr = np.asarray(data["instruments"], dtype=np.int64)

            if pos_arr.ndim != 2:
                raise ValueError(f"pos must be 2D [T,P], got {pos_arr.shape}")
            if notes_arr.ndim != 3:
                raise ValueError(f"notes must be 3D [T,N,F], got {notes_arr.shape}")
            if instruments_arr.ndim != 2:
                raise ValueError(f"instruments must be 2D [T,I], got {instruments_arr.shape}")
            if not (pos_arr.shape[0] == notes_arr.shape[0] == instruments_arr.shape[0]):
                raise ValueError(
                    "Time-step mismatch among pos/notes/instruments: "
                    f"pos={pos_arr.shape[0]}, notes={notes_arr.shape[0]}, instruments={instruments_arr.shape[0]}"
                )

            expected_note_dim = self.base_note_feature_count if self.use_harmonic_plan else len(self.music_config.note_features)
            if int(notes_arr.shape[-1]) != expected_note_dim:
                raise ValueError(
                    "Unsupported source note feature layout. "
                    f"Expected dim={expected_note_dim}, got {int(notes_arr.shape[-1])}."
                )
            if self.task_mode in {"chord", "accompaniment"} and not self.music_config.use_bridge:
                notes_arr, instruments_arr = self._remove_bridge_track(
                    notes_arr=notes_arr,
                    instruments_arr=instruments_arr,
                )

            if self.use_segmentation:
                raw_start = bar_start
                raw_stop = bar_end + 2
                pos_arr = pos_arr[raw_start:raw_stop]
                notes_arr = notes_arr[raw_start:raw_stop]
                instruments_arr = instruments_arr[raw_start:raw_stop]

            # Core DID shift: pos_t + note_{t-1} -> note_t. note_pos_arr stays
            # aligned with notes_data only for deterministic PLAN reconstruction.
            note_pos_arr = pos_arr[:-1]
            pos_data = pos_arr[1:]
            notes_data = notes_arr[:-1]
            instruments_data = instruments_arr[1:]

            if len(pos_data) <= 1:
                # Need at least two shifted steps so forward(labels[:,1:]) has supervision.
                return None

            if self.augmentor is not None:
                notes_tensor = torch.tensor(notes_data, dtype=torch.long)
                notes_tensor = self.augmentor(notes_tensor)
                notes_data = notes_tensor

            if self.use_harmonic_plan:
                notes_data = self._insert_harmonic_plan(notes_data, note_pos_arr)
            elif hasattr(notes_data, "tolist"):
                notes_data = notes_data.tolist()

            pos_data = pos_data.tolist()
            instruments_data = instruments_data.tolist()

            if self.use_segmentation and len(pos_data) > self.max_seq_length:
                raise RuntimeError(
                    f"Segment construction produced an oversized sample: "
                    f"{len(pos_data)} > {self.max_seq_length}"
                )

        except QuarterPlanMeterError:
            # This is a project-level data contract, not a corrupt-sample case.
            # Re-raise so a non-4/4/resolution-4 training set cannot be silently used.
            raise
        except Exception as e:
            import traceback

            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.error(f"Error loading {npz_file_path}: {e}")
            return None

        pos_input_ids, pos_attention_mask = [], []
        note_input_ids, note_attention_mask = [], []
        instrument_input_ids, instrument_attention_mask = [], []
        target_mask = []

        time_steps = len(pos_data)
        for ts in range(time_steps):
            pos_input_ids.append(pos_data[ts])
            pos_attention_mask.append(1)

            cur_notes = notes_data[ts]
            cur_note_mask = self._build_note_mask(cur_notes)
            note_input_ids.append(cur_notes)
            note_attention_mask.append(cur_note_mask)
            target_mask.append(cur_note_mask)

            cur_instruments = instruments_data[ts]
            instrument_input_ids.append(cur_instruments)
            instrument_attention_mask.append([1 if int(inst) > 0 else 0 for inst in cur_instruments])

        return {
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "note_input_ids": note_input_ids,
            "note_attention_mask": note_attention_mask,
            "instrument_input_ids": instrument_input_ids,
            "instrument_attention_mask": instrument_attention_mask,
            "target_mask": target_mask,
        }
