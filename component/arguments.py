import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger


@dataclass
class CustomizedArguments:
    """Training entry arguments."""

    dataset_path: str = field(metadata={"help": "Path to tuneidx dataset."})
    metadata_path: str = field(metadata={"help": "Path to dataset split metadata.json."})
    llm_model_name_or_path: str = field(metadata={"help": "Base LLM config/model path."})
    max_seq_length: int = field(default=500, metadata={"help": "Max bar-level sequence length for training."})
    gpu_devices: Optional[str] = field(default=None, metadata={"help": "Visible GPUs, e.g. '0,1'."})


@dataclass
class DidMusicArguments:
    """DID model arguments (single-stage unified protocol)."""

    # Core schema
    vocab_file_path: str = field(metadata={"help": "Path to DID vocab json."})
    pos_features: List[str] = field(metadata={"help": "Position feature list."})
    note_features: List[str] = field(metadata={"help": "Note feature list."})

    # Embedding dimensions
    bar_embedding_dim: int = field(default=128, metadata={"help": "bar embedding dim."})
    type_embedding_dim: int = field(default=32, metadata={"help": "type embedding dim."})
    beat_embedding_dim: int = field(default=64, metadata={"help": "beat embedding dim."})
    time_signature_embedding_dim: int = field(default=32, metadata={"help": "time signature embedding dim."})
    tempo_embedding_dim: int = field(default=32, metadata={"help": "tempo embedding dim."})
    chord_embedding_dim: int = field(default=64, metadata={"help": "chord embedding dim."})
    instrument_embedding_dim: int = field(default=64, metadata={"help": "instrument embedding dim."})
    pitch_embedding_dim: int = field(default=512, metadata={"help": "pitch embedding dim."})
    duration_embedding_dim: int = field(default=32, metadata={"help": "duration embedding dim."})
    velocity_embedding_dim: int = field(default=32, metadata={"help": "velocity embedding dim."})
    plan_bass_embedding_dim: int = field(default=32, metadata={"help": "PLAN bass-PC embedding dim."})
    plan_set_embedding_dim: int = field(default=32, metadata={"help": "PLAN 12-bit relative attack-set embedding dim."})

    # Projection dimensions
    pos_proj_dim: int = field(default=128, metadata={"help": "pos projection dim."})
    note_proj_dim: int = field(default=128, metadata={"help": "note projection dim."})

    # Runtime filled from vocab
    sto_note_id: List[int] = field(
        default_factory=lambda: [3, 0, 0, 0, 0, 0, 0, 0],
        metadata={"help": "Runtime-resolved STO note token id vector."},
    )

    # Time decoder
    time_decoder_layers: int = field(default=32, metadata={"help": "Time decoder layers."})
    time_decoder_hidden_size: int = field(default=1024, metadata={"help": "Time decoder hidden size."})
    time_decoder_attention_heads: int = field(default=16, metadata={"help": "Time decoder attention heads."})
    time_decoder_intermediate_size: int = field(default=5632, metadata={"help": "Time decoder FFN size."})

    # Note decoder
    note_decoder_layers: int = field(default=6, metadata={"help": "Note decoder layers."})
    note_decoder_hidden_size: int = field(default=768, metadata={"help": "Note decoder hidden size."})
    note_decoder_attention_heads: int = field(default=12, metadata={"help": "Note decoder attention heads."})
    note_decoder_intermediate_size: int = field(default=3072, metadata={"help": "Note decoder FFN size."})

    # Runtime inferred field
    pitch_offset: int = field(default=0, init=False, metadata={"help": "Inferred pitch offset from vocab."})

    # Data augmentation
    use_augmentation: bool = field(
        default=False,
        metadata={"help": "Enable online augmentation for training split."},
    )
    focal_loss_alpha: float = field(default=1.0, metadata={"help": "FocalLoss alpha."})
    focal_loss_gamma: float = field(default=2.0, metadata={"help": "FocalLoss gamma."})

    # Bar encoder / generation controls
    bar_encoder_layers: int = field(default=4, metadata={"help": "Bar encoder layers."})
    bar_encoder_heads: int = field(default=8, metadata={"help": "Bar encoder attention heads."})
    bar_encoder_dropout: float = field(default=0.1, metadata={"help": "Bar encoder dropout."})
    use_bar_note_pos_encoding: bool = field(
        default=True,
        metadata={"help": "Enable learned absolute position embedding in BarNoteEncoder."},
    )
    bar_note_pos_max_len: int = field(
        default=512,
        metadata={"help": "Maximum bar-note token length supported by BarNoteEncoder position embedding."},
    )
    raw_note_history_bars: int = field(
        default=0,
        metadata={"help": "Recent raw-history bars K for did_model_recentk_raw (0 disables history branch)."},
    )
    raw_history_order: str = field(
        default="oldest_first",
        metadata={"help": "Order of K history bars: oldest_first or newest_first."},
    )
    history_note_budget: int = field(
        default=0,
        metadata={
            "help": (
                "Fixed packed-token budget M for recent raw history. "
                "0 means fallback to K * max_note_num."
            )
        },
    )
    # V9 quarter-level harmonic PLAN. Base tuneidx/vocab remain DID8; PLAN is
    # inserted only at runtime after filtering and augmentation.
    use_harmonic_plan: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable V9 quarter-level PLAN: at most one PLAN per quarter, "
                "with Beat in {0,4,8,12}, quarter-start sounding BassPC, and "
                "the quarter's onset-PC union relative to that BassPC."
            )
        },
    )
    in_beat_resolution: int = field(
        default=4,
        metadata={
            "help": "Quantization steps per quarter note. V9 PLAN requires exactly 4."
        },
    )
    plan_type_id: int = field(default=-1, init=False)
    plan_note_id: List[int] = field(default_factory=list, init=False)
    base_note_feature_count: int = field(default=0, init=False)
    plan_quarter_positions: List[int] = field(default_factory=list, init=False)
    plan_quarter_beat_ids: List[int] = field(default_factory=list, init=False)

    # multi tasks
    task_mode: str = field(
        default="continuation",
        metadata={
            "help": "Task mode: continuation, chord, or accompaniment."
        },
    )

    condition_instrument_prefix: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "Prefix instruments used only in accompaniment mode, "
                "e.g. ['Instrument_0', 'Instrument_24']."
            )
        },
    )
    condition_instrument_prefix_ids: List[int] = field(
        default_factory=list,
        init=False,
        metadata={
            "help": "Runtime-resolved instrument token IDs."
        },
    )
    use_bridge: bool = field(
        default=False,
        metadata={
            "help": "Using Bridge track."
        },
    )
    condition_encoder_layers: int = field(default=2)
    condition_encoder_heads: int = field(default=8)
    condition_encoder_dropout: float = field(default=0.1)
    condition_cross_attention_heads: int = field(default=8)

    # Placeholder for Qwen2 config construction
    dummy_vocab_size: int = field(default=1, metadata={"help": "Placeholder vocab_size for Qwen2Config."})

    beat_0_id: int = field(
        default=0,
        init=False,
    )

    beat_id_to_position: List[int] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self):
        vocab_file = Path(self.vocab_file_path)
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {self.vocab_file_path}")
        if "chord" not in self.note_features:
            raise ValueError("Unified chord-prefix baseline requires 'chord' in note_features.")
        if "tempo" not in self.note_features:
            raise ValueError("Unified INST+BEAT+NOTE protocol requires 'tempo' in note_features.")
        if len(self.note_features) < 3 or self.note_features[:3] != ["type", "chord", "beat"]:
            raise ValueError(
                "Only chord-first note schema is supported: note_features must start with "
                "['type', 'chord', 'beat']."
            )
        plan_features = ["plan_bass", "plan_set"]
        if self.use_harmonic_plan:
            if self.note_features[-2:] != plan_features:
                raise ValueError(
                    "use_harmonic_plan=True requires note_features to end with "
                    "['plan_bass', 'plan_set']; the original DID8 features must remain first."
                )
            self.base_note_feature_count = len(self.note_features) - 2
        else:
            if any(name in self.note_features for name in plan_features):
                raise ValueError("PLAN features require use_harmonic_plan=True.")
            self.base_note_feature_count = len(self.note_features)
        if int(self.in_beat_resolution) <= 0:
            raise ValueError("in_beat_resolution must be positive.")
        if self.use_harmonic_plan and int(self.in_beat_resolution) != 4:
            raise ValueError(
                "DID v9 quarter-level PLAN requires in_beat_resolution=4 "
                "(four 16th-note positions per quarter)."
            )

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # POS vocab sizes
        for feature in self.pos_features:
            key = "instrument" if feature == "all_instruments" else feature
            if key not in vocab_data:
                raise ValueError(f"Missing POS feature in vocab: {feature}")
            setattr(self, f"{key}_vocab_size", len(vocab_data[key]))

        # NOTE vocab sizes. PLAN fields are runtime numeric features and deliberately
        # do not enlarge the stored DID8 vocabulary.
        for feature in self.note_features:
            if feature == "plan_bass":
                # 0=PAD/non-PLAN, 1..12 = pitch classes C..B.
                setattr(self, "plan_bass_vocab_size", 13)
                continue
            if feature == "plan_set":
                # Raw tensor stores a 12-bit integer mask, but this is not a vocab.
                # The model uses 12 interval embeddings and 12 binary decisions.
                self.plan_set_num_intervals = 12
                continue
            if feature not in vocab_data:
                raise ValueError(f"Missing NOTE feature in vocab: {feature}")
            setattr(self, f"{feature}_vocab_size", len(vocab_data[feature]))

        if "chord" not in vocab_data:
            raise ValueError("Missing chord vocab.")
        self.chord_vocab_size = len(vocab_data["chord"])

        type_to_id = {v: int(k) for k, v in vocab_data["type"].items()}
        if self.use_harmonic_plan:
            # PLAN is a virtual runtime type appended after the legacy DID8 type vocab,
            # so every existing token id remains bit-for-bit compatible.
            if "PLAN" in type_to_id:
                self.plan_type_id = int(type_to_id["PLAN"])
            else:
                self.plan_type_id = len(type_to_id)
                type_to_id["PLAN"] = self.plan_type_id
                self.type_vocab_size = max(int(self.type_vocab_size), self.plan_type_id + 1)
        chord_to_id = {v: int(k) for k, v in vocab_data["chord"].items()}
        beat_to_id = {beat_name: int(beat_id)
            for beat_id, beat_name in vocab_data["beat"].items()
            if isinstance(beat_name, str)
        }
        if "Beat_0" not in beat_to_id:
            raise ValueError("Beat vocab missing Beat_0.")
        self.beat_0_id = beat_to_id["Beat_0"]
        max_beat_id = max(
            int(beat_id)
            for beat_id in vocab_data["beat"].keys()
        )
        self.beat_id_to_position = [-1] * (max_beat_id + 1)
        for beat_id, beat_name in vocab_data["beat"].items():
            if not isinstance(beat_name, str):
                continue
            if not beat_name.startswith("Beat_"):
                continue
            position = int(beat_name.split("_")[-1])
            self.beat_id_to_position[int(beat_id)] = position

        if self.use_harmonic_plan:
            self.plan_quarter_positions = [0, 4, 8, 12]
            position_to_beat_id = {
                int(position): int(beat_id)
                for beat_id, position in enumerate(self.beat_id_to_position)
                if int(position) >= 0
            }
            missing = [p for p in self.plan_quarter_positions if p not in position_to_beat_id]
            if missing:
                raise ValueError(
                    "DID v9 quarter-level PLAN requires Beat_0/4/8/12 in the beat vocab; "
                    f"missing positions={missing}."
                )
            self.plan_quarter_beat_ids = [position_to_beat_id[p] for p in self.plan_quarter_positions]
            
        def _mk_note_id_primary(token_name: str) -> List[int]:
            if token_name not in type_to_id:
                raise ValueError(f"Missing required type token in vocab: {token_name}")
            tid = type_to_id[token_name]
            return [int(tid)] + [0] * (len(self.note_features) - 1)

        self.sos_note_id = _mk_note_id_primary("SOS")
        self.eos_note_id = _mk_note_id_primary("EOS")
        self.note_note_id = _mk_note_id_primary("NOTE")
        self.con_note_id = list(self.note_note_id)
        self.inst_note_id = _mk_note_id_primary("INST")
        self.beat_note_id = _mk_note_id_primary("BEAT")
        self.sto_note_id = _mk_note_id_primary("STO")

        # Unified protocol IDs
        self.chord_note_id = _mk_note_id_primary("CHORD")
        self.plan_note_id = (
            _mk_note_id_primary("PLAN")
            if self.use_harmonic_plan
            else []
        )

        if "Chord_N_N" not in chord_to_id:
            raise ValueError("Chord vocab missing Chord_N_N.")
        self.chord_n_n_id = chord_to_id["Chord_N_N"]

        # all_instruments is pooled at runtime and does not occupy one pos token slot
        pos_token_len = len(self.pos_features) - (1 if "all_instruments" in self.pos_features else 0)
        self.pad_pos_id = [0] * pos_token_len

        # Infer pitch offset: token id "1" -> midi pitch from token name
        if "pitch" in vocab_data:
            pitch_vocab = vocab_data["pitch"]
            found_offset = False
            target_key = "1"
            if target_key in pitch_vocab:
                token_name = str(pitch_vocab[target_key])
                if "Note_Pitch_" in token_name:
                    try:
                        midi_pitch_val = int(token_name.split("_")[-1])
                        token_id = int(target_key)
                        self.pitch_offset = midi_pitch_val - token_id
                        logger.info(
                            f"Auto-detected pitch offset: {self.pitch_offset} "
                            f"(token {token_id} -> pitch {midi_pitch_val})"
                        )
                        found_offset = True
                    except ValueError:
                        found_offset = False
            if not found_offset:
                logger.warning("Failed to infer pitch offset automatically, fallback to 0.")
                self.pitch_offset = 0
        else:
            logger.warning("Pitch vocab missing, pitch_offset fallback to 0.")
            self.pitch_offset = 0

        self.task_mode = self.task_mode.strip().lower()

        valid_modes = {
            "continuation",
            "chord",
            "accompaniment",
        }

        if self.task_mode not in valid_modes:
            raise ValueError(
                f"Unknown task_mode={self.task_mode!r}; "
                f"expected one of {sorted(valid_modes)}."
            )

        if self.task_mode == "accompaniment":
            if len(self.condition_instrument_prefix) == 0:
                raise ValueError(
                    "condition_instrument_prefix must be non-empty "
                    "when task_mode='accompaniment'."
                )
        elif len(self.condition_instrument_prefix) > 0:
            raise ValueError(
                "condition_instrument_prefix is only valid when "
                "task_mode='accompaniment'."
            )
        if self.task_mode != "continuation":
            if self.time_decoder_hidden_size % self.condition_encoder_heads != 0:
                raise ValueError(
                    "time_decoder_hidden_size must be divisible by "
                    "condition_encoder_heads."
                )

            if self.time_decoder_hidden_size % self.condition_cross_attention_heads != 0:
                raise ValueError(
                    "time_decoder_hidden_size must be divisible by "
                    "condition_cross_attention_heads."
                )
        if "instrument" not in vocab_data:
            raise ValueError("Missing instrument vocab.")

        instrument_to_id = {
            str(instrument_name): int(instrument_id)
            for instrument_id, instrument_name
            in vocab_data["instrument"].items()
        }

        if self.task_mode == "accompaniment":
            if len(self.condition_instrument_prefix) == 0:
                raise ValueError("condition_instrument_prefix must be non-empty when task_mode='accompaniment'.")

            resolved_ids = []
            for instrument_name in self.condition_instrument_prefix:
                instrument_name = str(instrument_name).strip()
                if instrument_name not in instrument_to_id:
                    raise ValueError(f"Unknown condition instrument {instrument_name!r}.")

                instrument_id = int(instrument_to_id[instrument_name])
                if instrument_id == 0:
                    raise ValueError("Instrument id 0 is reserved for PAD.")

                resolved_ids.append(instrument_id)

            if len(resolved_ids) != len(set(resolved_ids)):
                raise ValueError(f"condition_instrument_prefix contains duplicates: {self.condition_instrument_prefix}")

            if resolved_ids != sorted(resolved_ids):
                raise ValueError(
                    f"condition_instrument_prefix must follow the encoded instrument order. Resolved ids={resolved_ids}."
                )

            self.condition_instrument_prefix_ids = resolved_ids

        else:
            if len(self.condition_instrument_prefix) > 0:
                raise ValueError(
                    "condition_instrument_prefix is only valid for accompaniment mode."
                )
            self.condition_instrument_prefix_ids = []

    def __str__(self):
        class_name = self.__class__.__name__
        attributes = ",\n\t".join([f"{attr}={getattr(self, attr)}" for attr in dir(self) if not attr.startswith("__")])
        return f"{class_name}" + "{" + "\n\t" + f"{attributes}" + "\n}"


@dataclass
class EvaluateArguments:
    """Inference and evaluation entry arguments."""

    dataset_name: str = field(metadata={"help": "Dataset name."})
    dataset_path: str = field(metadata={"help": "Inference dataset path."})
    metadata_path: str = field(metadata={"help": "Split metadata path."})
    output_dir: str = field(metadata={"help": "Inference output directory."})
    llm_model_name_or_path: str = field(metadata={"help": "Base LLM config path."})
    model_name_or_path: str = field(metadata={"help": "Checkpoint path."})

    max_time_steps: int = field(default=1024, metadata={"help": "Max generated bars."})
    max_note_nums: int = field(default=64, metadata={"help": "Max notes per bar."})
    batch_size: int = field(default=96, metadata={"help": "Inference batch size."})
    decode_strategy: str = field(default="greedy", metadata={"help": "Decode strategy: greedy/sampling."})

    dump_generated_events: bool = field(
        default=False,
        metadata={"help": "Dump generated events (human-readable) to text file during inference."},
    )
    dump_generated_event_max_tokens: int = field(
        default=64,
        metadata={"help": "Max generated note tokens to print per bar when dumping events."},
    )
    dump_generated_event_file: str = field(
        default="generated_events.txt",
        metadata={"help": "Relative output filename for generated-event dump."},
    )


    sampling_temperature: float = field(default=1.0)
    sampling_top_p: float = field(default=0.9)
    type_temperature: float = field(default=0.1)
    type_top_p: float = field(default=0.95)
    chord_temperature: float = field(default=0.2)
    chord_top_p: float = field(default=0.90)
    beat_temperature: float = field(default=1.0)
    beat_top_p: float = field(default=0.90)
    tempo_temperature: float = field(default=1.0)
    tempo_top_p: float = field(default=0.90)
    pitch_temperature: float = field(default=1.0)
    pitch_top_p: float = field(default=0.90)
    duration_temperature: float = field(default=1.0)
    duration_top_p: float = field(default=0.90)
    velocity_temperature: float = field(default=1.0)
    velocity_top_p: float = field(default=1.0)
    instrument_temperature: float = field(default=0.3)
    instrument_top_p: float = field(default=0.90)
    plan_bass_temperature: float = field(default=1.0)
    plan_bass_top_p: float = field(default=0.0)
    plan_set_temperature: float = field(default=1.0)
    plan_set_top_p: float = field(default=0.0)

    prompt_measures: int = field(default=0, metadata={"help": "Prompt bars for continuation. 0 means from scratch."})
    eval_script_path: str = field(default="inference/music_evaluation/music_evaluator.py")
    gt_dataset_dir: str = field(default="dataset/MIDI_test_dataset/Pop909/")
    eval_num_sample: int = field(default=1000)
    eval_num_runs: int = field(default=3)
    eval_crop_bars: int = field(default=32)

    def __post_init__(self):
        
        self.sampling_params = {
            "default": {"t": self.sampling_temperature, "p": self.sampling_top_p},
            "type": {"t": self.type_temperature, "p": self.type_top_p},
            "chord": {"t": self.chord_temperature, "p": self.chord_top_p},
            "beat": {"t": self.beat_temperature, "p": self.beat_top_p},
            "tempo": {"t": self.tempo_temperature, "p": self.tempo_top_p},
            "pitch": {"t": self.pitch_temperature, "p": self.pitch_top_p},
            "duration": {"t": self.duration_temperature, "p": self.duration_top_p},
            "velocity": {"t": self.velocity_temperature, "p": self.velocity_top_p},
            "instrument": {"t": self.instrument_temperature, "p": self.instrument_top_p},
            "plan_bass": {"t": self.plan_bass_temperature, "p": self.plan_bass_top_p},
            "plan_set": {"t": self.plan_set_temperature, "p": self.plan_set_top_p},
        }

    def __str__(self):
        class_name = self.__class__.__name__
        attributes = ",\n\t".join([f"{attr}={getattr(self, attr)}" for attr in dir(self) if not attr.startswith("__")])
        return f"{class_name}" + "{" + "\n\t" + f"{attributes}" + "\n}"
