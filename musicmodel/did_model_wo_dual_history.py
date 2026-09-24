import json
import os
import re
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import PreTrainedModel, Qwen2Config, Qwen2Model
from transformers.modeling_outputs import CausalLMOutputWithPast

class TimeToNoteMLPConnector(nn.Module):
    def __init__(self, time_hidden_size: int, note_proj_dim: int, num_queries: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.note_proj_dim = note_proj_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_hidden_size, time_hidden_size * 2),
            nn.GELU(),
            nn.Linear(time_hidden_size * 2, num_queries * note_proj_dim),
        )

    def forward(self, time_hidden_states: torch.Tensor) -> torch.Tensor:
        b, t, _ = time_hidden_states.shape
        out = self.mlp(time_hidden_states)
        return out.view(b, t, self.num_queries, self.note_proj_dim)

    def forward_step(self, current_time_hidden: torch.Tensor) -> torch.Tensor:
        b, _ = current_time_hidden.shape
        out = self.mlp(current_time_hidden)
        return out.view(b, self.num_queries, self.note_proj_dim)

class BarNoteEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        max_positions: int = 512,
        enable_condition_summary: bool = False,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        self.bar_sum = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.enable_condition_summary = enable_condition_summary
        if self.enable_condition_summary:
            self.con_sum = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        self.use_pos_encoding = bool(use_pos_encoding)
        self.max_positions = int(max_positions)
        if self.use_pos_encoding:
            self.note_pos_embedding = nn.Embedding(self.max_positions, hidden_size)
            nn.init.normal_(self.note_pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, note_tokens: torch.Tensor, note_mask: torch.Tensor, summary_type: str = "bar") -> torch.Tensor:
        # note_tokens: [B,T,N,H], note_mask: [B,T,N]
        b, t, n, h = note_tokens.shape
        x = note_tokens.reshape(b * t, n, h)
        m = note_mask.reshape(b * t, n)

        if self.use_pos_encoding:
            if n > self.max_positions:
                raise ValueError(
                    f"BarNoteEncoder note length {n} exceeds bar_note_pos_max_len={self.max_positions}."
                )
            pos_ids = torch.arange(n, device=x.device).unsqueeze(0)
            pos_emb = self.note_pos_embedding(pos_ids)
            if pos_emb.dtype != x.dtype:
                pos_emb = pos_emb.to(dtype=x.dtype)
            x = x + pos_emb

        if summary_type == "bar":
            summary_token = self.bar_sum
        elif summary_type == "condition":
            if not self.enable_condition_summary:
                raise RuntimeError(
                    "Condition summary was not enabled for this model."
                )
            summary_token = self.con_sum
        else:
            raise ValueError(
                f"Unknown summary_type={summary_type!r}."
            )
        
        summary_token = summary_token.expand(b * t, -1, -1)
        x = torch.cat([summary_token, x], dim=1)
        key_padding = torch.cat(
            [torch.zeros((b * t, 1), dtype=torch.bool, device=x.device), m == 0], dim=1
        )
        y = self.encoder(x, src_key_padding_mask=key_padding)
        return y[:, 0, :].view(b, t, h)


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    if 0.0 <= top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = 0
        remove = remove.scatter(1, sorted_indices, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss

        if self.reduction == "mean":
            valid_mask = (targets != self.ignore_index).to(dtype=loss.dtype)
            return loss.sum() / (valid_mask.sum() + 1e-8)
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PairwiseSonoritySetHead(nn.Module):
    """V18 exact joint Set head with shared unary + pairwise sonority energies.

    The model still defines one categorical distribution over all 4096 complete
    12-bit Bass-relative pitch-class Sets.  Unlike a free 4096-way Linear head,
    the logit of a Set is assembled from context-dependent musical factors:

        score(S) = sum_{i in S} u_i + sum_{i<j, i,j in S} v_ij

    where u_i are 12 interval tendencies and v_ij are the 66 pairwise interval
    compatibilities.  This shares statistical strength across related complete
    Sets without factorizing the output into independent sigmoid decisions and
    without predicting cardinality K.
    """

    num_intervals = 12
    num_unary = 12
    num_pairwise = 66
    num_factors = 78
    num_sets = 1 << 12

    def __init__(self, input_dim: int):
        super().__init__()
        self.factor_head = nn.Linear(input_dim, self.num_factors, bias=False)

        # Fixed incidence matrix A[S, factor].  The first 12 columns indicate
        # interval membership.  The remaining 66 columns indicate whether each
        # unordered interval pair is jointly present in the complete Set.
        masks = torch.arange(self.num_sets, dtype=torch.long)
        bit_ids = torch.arange(self.num_intervals, dtype=torch.long)
        bits = ((masks.unsqueeze(1) >> bit_ids.unsqueeze(0)) & 1).float()

        pair_i = []
        pair_j = []
        for i in range(self.num_intervals):
            for j in range(i + 1, self.num_intervals):
                pair_i.append(i)
                pair_j.append(j)
        pair_i = torch.tensor(pair_i, dtype=torch.long)
        pair_j = torch.tensor(pair_j, dtype=torch.long)
        pair_features = bits[:, pair_i] * bits[:, pair_j]

        incidence = torch.cat([bits, pair_features], dim=1)
        if incidence.shape != (self.num_sets, self.num_factors):
            raise RuntimeError(
                "Unexpected V18 Set-factor incidence shape: "
                f"{tuple(incidence.shape)}"
            )

        self.register_buffer("set_factor_incidence", incidence, persistent=False)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)

    def factor_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return [..., 78] = [12 unary interval, 66 pairwise] scores."""
        return self.factor_head(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        factors = self.factor_scores(hidden_states)
        incidence = self.set_factor_incidence
        if incidence.dtype != factors.dtype:
            incidence = incidence.to(dtype=factors.dtype)
        # F.linear treats incidence as [4096, 78] fixed output weights:
        # [...,78] @ [78,4096] -> [...,4096].
        return F.linear(factors, incidence)


class DidForMusicLM(PreTrainedModel):
    config_class = Qwen2Config
    def __init__(self, time_config: Qwen2Config, note_config: Qwen2Config, music_config):
        super().__init__(time_config)
        self.music_config = music_config
        self.time_config = time_config
        self.note_config = note_config

        self.task_mode = music_config.task_mode
        self.use_global_condition = (self.task_mode != "continuation")

        self.pos_features = list(music_config.pos_features)
        self.note_features = list(music_config.note_features)
        self.num_queries = 4

        # 

        self.pos_features_wo_inst = [f for f in self.pos_features if f != "all_instruments"]
        self.use_pos_all_instruments = "all_instruments" in self.pos_features

        # pos embeddings
        self.sum_pos_embedding_dim = 0
        for feat in self.pos_features_wo_inst:
            vocab_size = getattr(music_config, f"{feat}_vocab_size")
            emb_dim = getattr(music_config, f"{feat}_embedding_dim")
            setattr(self, f"{feat}_embedding", nn.Embedding(vocab_size, emb_dim))
            self.sum_pos_embedding_dim += emb_dim

        if self.use_pos_all_instruments:
            inst_vocab = getattr(music_config, "instrument_vocab_size")
            inst_dim = getattr(music_config, "instrument_embedding_dim")
            self.instrument_embedding = nn.Embedding(inst_vocab, inst_dim)
            self.sum_pos_embedding_dim += inst_dim

        # note embeddings. plan_set is compositional: sum 12 interval embeddings
        # selected by the raw 12-bit mask; never allocate a 4096-entry table.
        self.use_harmonic_plan = bool(getattr(music_config, "use_harmonic_plan", False))
        self.sum_note_embedding_dim = 0
        for feat in self.note_features:
            emb_dim = getattr(music_config, f"{feat}_embedding_dim")
            if feat == "plan_set":
                self.plan_interval_embedding = nn.Embedding(12, emb_dim)
            else:
                vocab_size = getattr(music_config, f"{feat}_vocab_size")
                setattr(self, f"{feat}_embedding", nn.Embedding(vocab_size, emb_dim))
            self.sum_note_embedding_dim += emb_dim

        self.pos_embed_proj = nn.Linear(self.sum_pos_embedding_dim, music_config.pos_proj_dim)
        self.note_embed_proj = nn.Linear(self.sum_note_embedding_dim, music_config.note_proj_dim)

        # bar summary + time model
        self.bar_note_in_proj = nn.Linear(music_config.note_proj_dim, time_config.hidden_size)
        self.bar_note_encoder = BarNoteEncoder(
            hidden_size=time_config.hidden_size,
            num_heads=getattr(music_config, "bar_encoder_heads", 8),
            num_layers=getattr(music_config, "bar_encoder_layers", 4),
            dropout=getattr(music_config, "bar_encoder_dropout", 0.1),
            use_pos_encoding=bool(getattr(music_config, "use_bar_note_pos_encoding", True)),
            max_positions=int(getattr(music_config, "bar_note_pos_max_len", 512)),
            enable_condition_summary=self.use_global_condition,
        )
        
        self.time_input_proj = nn.Linear(music_config.pos_proj_dim + time_config.hidden_size, time_config.hidden_size)
        self.time_model = Qwen2Model(time_config)
        self.qformer_connector = TimeToNoteMLPConnector(time_config.hidden_size, music_config.note_proj_dim, self.num_queries)

        if self.use_global_condition:
            hidden_size = time_config.hidden_size
            
            self.condition_input_proj = nn.Linear(music_config.pos_proj_dim + hidden_size, hidden_size)
            condition_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=music_config.condition_encoder_heads,
                dim_feedforward=hidden_size * 4,
                dropout=music_config.condition_encoder_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.condition_encoder = nn.TransformerEncoder(
                condition_layer,
                num_layers=music_config.condition_encoder_layers,
                norm=nn.LayerNorm(hidden_size),
            )
            self.condition_to_note = nn.Linear(
                time_config.hidden_size,
                music_config.note_proj_dim,
                bias=False
            )

        # note decoder
        self.note_emb_tokens_proj = nn.Linear(music_config.note_proj_dim, note_config.hidden_size)
        self.note_model = Qwen2Model(note_config)
        for feat in self.note_features:
            if feat == "plan_set":
                # V18 Pairwise Sonority Energy Head:
                # keep ONE exact 4096-way categorical distribution over the
                # complete Bass-relative 12-bit Set, but replace 4096 unrelated
                # class weights with 12 unary interval tendencies + 66 pairwise
                # interval compatibilities.  The fixed Set incidence matrix then
                # composes those 78 factors into all 4096 complete-Set logits.
                #
                # This is NOT 12 independent sigmoids and does NOT predict K.
                # Training remains plain exact-mask categorical CE; inference
                # remains ordinary 4096-way MAP / nucleus sampling.
                plan_bass_dim = getattr(music_config, "plan_bass_embedding_dim")
                setattr(
                    self,
                    "plan_set_head",
                    PairwiseSonoritySetHead(
                        note_config.hidden_size + plan_bass_dim
                    ),
                )
            else:
                vocab_size = getattr(music_config, f"{feat}_vocab_size")
                setattr(self, f"{feat}_head", nn.Linear(note_config.hidden_size, vocab_size, bias=False))

        self.type_cond_proj = nn.Linear(
            note_config.hidden_size + getattr(music_config, "type_embedding_dim"),
            note_config.hidden_size,
        )

        self.eval_feature_loss_sum = {}
        self.eval_feature_count = {}
        self.using_type_mask = True
        focal_alpha = float(getattr(music_config, "focal_loss_alpha", 1.0))
        focal_gamma = float(getattr(music_config, "focal_loss_gamma", 2.0))

        # V18 keeps v16.1's ONE plain categorical CE over the complete 4096
        # Set masks.  Only the logit parameterization changes; there is no
        # auxiliary bit loss, no K/cardinality loss, and no structured-margin
        # loss.  Preserve the original v16.1 semantic-feature normalization.
        self.plan_joint_set_loss_scale = focal_alpha * (0.5 ** focal_gamma) / 12.0

        loss_reduction = "none" if self.using_type_mask else "mean"
        self.note_focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=-100,
            reduction=loss_reduction,
        )
        self.note_focal_loss_eval = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=-100, reduction="none")

        

        if self.using_type_mask:
            logger.info("Using type mask in Focal Loss.")
        else:
            logger.info("Only using valid mask in Focal Loss")

        try:
            with open(music_config.vocab_file_path, "r", encoding="utf-8") as f:
                self.vocab_data = json.load(f)
        except Exception:
            self.vocab_data = None

        logger.info(f"Position Feature: {self.pos_features}")
        logger.info(f"Note Feature: {self.note_features}")
        self.note_feature_to_idx = {name: i for i, name in enumerate(self.note_features)}
        self.note_type_idx = self.note_feature_to_idx["type"]
        self.note_chord_idx = self.note_feature_to_idx.get("chord", None)
        self.note_beat_idx = self.note_feature_to_idx.get("beat", None)
        self.note_tempo_idx = self.note_feature_to_idx.get("tempo", None)
        self.note_inst_idx = self.note_feature_to_idx.get("instrument", None)
        self.note_pitch_idx = self.note_feature_to_idx.get("pitch", None)
        self.note_dur_idx = self.note_feature_to_idx.get("duration", None)
        self.note_vel_idx = self.note_feature_to_idx.get("velocity", None)
        self.note_plan_bass_idx = self.note_feature_to_idx.get("plan_bass", None)
        self.note_plan_set_idx = self.note_feature_to_idx.get("plan_set", None)

        # ``use_harmonic_plan`` is the single semantic switch for PLAN.
        # MusicConfig already owns the runtime schema, but keep a model-level
        # invariant here so an inconsistent checkpoint/config fails immediately
        # instead of silently mixing PLAN and no-PLAN representations.
        if self.use_harmonic_plan:
            if self.note_plan_bass_idx is None or self.note_plan_set_idx is None:
                raise RuntimeError(
                    "use_harmonic_plan=True requires both 'plan_bass' and "
                    "'plan_set' in note_features."
                )
        else:
            if self.note_plan_bass_idx is not None or self.note_plan_set_idx is not None:
                raise RuntimeError(
                    "use_harmonic_plan=False requires 'plan_bass' and 'plan_set' "
                    "to be absent from note_features."
                )

        if self.note_inst_idx is None:
            raise RuntimeError(
                "V18-D5.1 requires the existing NOTE instrument feature so the "
                "soft chord prior can be applied to PIANO only."
            )

        self.sos_type_id = int(getattr(music_config, "sos_note_id")[0])
        self.eos_type_id = int(getattr(music_config, "eos_note_id")[0])
        self.con_type_id = int(getattr(music_config, "con_note_id")[0])
        self.note_type_id = int(getattr(music_config, "note_note_id")[0])
        self.inst_type_id = int(getattr(music_config, "inst_note_id")[0])
        self.beat_type_id = int(getattr(music_config, "beat_note_id")[0])
        self.sto_type_id = int(getattr(music_config, "sto_note_id")[0])
        self.chord_type_id = int(getattr(music_config, "chord_note_id")[0])
        self.plan_type_id = int(getattr(music_config, "plan_type_id", -1))
        self.plan_quarter_positions = tuple(int(x) for x in getattr(music_config, "plan_quarter_positions", []))
        self.plan_quarter_beat_ids = tuple(int(x) for x in getattr(music_config, "plan_quarter_beat_ids", []))
        self.chord_n_n_id = int(music_config.chord_n_n_id)
        self.beat_id_to_position = tuple(int(x)for x in music_config.beat_id_to_position)

        # V18-D5 decode-only music-theory lookups. These are fixed buffers derived
        # from the existing vocabulary, so the checkpoint/state dict is unchanged.
        if self.task_mode in {"chord", "accompaniment"}:
            self.register_buffer(
                "pitch_token_to_pc",
                self._build_pitch_token_to_pc_lookup_d5(),
                persistent=False,
            )
            self.register_buffer(
                "chord_pc_membership",
                self._build_chord_pc_membership_lookup_d5(),
                persistent=False,
            )
            self.register_buffer(
                "d51_piano_instrument_mask",
                self._build_piano_instrument_mask_d51(),
                persistent=False,
            )
        else:
            self.register_buffer(
                "pitch_token_to_pc",
                None,
                persistent=False,
            )
            self.register_buffer(
                "chord_pc_membership",
                None,
                persistent=False,
            )
            self.register_buffer(
                "d51_piano_instrument_mask",
                None,
                persistent=False,
            )

        logger.info("Initialized DID model with unified chord-prefix note autoregression.")
        if self.use_harmonic_plan:
            logger.info(
                "V18 PLAN: one Bass-conditioned exact 4096-way Set distribution "
                "parameterized by 12 unary + 66 pairwise sonority energies; "
                "plain exact-mask CE, no sigmoid/K/auxiliary loss/reranking."
            )
        
        self.post_init()
        logger.info(
            "V18 Partitioned Global-Local Context: retains V16.1 distant-history "
            "TimeModel + immediate previous-bar raw NOTE memory unchanged; only "
            "the PLAN Set logit parameterization is replaced."
        )
        logger.info(
            "V18-D5.1 decode: for chord/accompaniment tasks only, D5's finite +ln(2) "
            "active-chord NOTE-pitch bonus is applied to PIANO instrument events. "
            "Continuation uses the original V18 pitch distribution without this "
            "chord-tone prior. BRIDGE and all other instruments keep the original "
            "V18 pitch distribution. No new hard pitch mask, no PLAN reranking, "
            "and no trainable parameter."
        )

    def _build_piano_instrument_mask_d51(self) -> torch.Tensor:
        """Return [instrument_vocab] bool mask selecting PIANO-like instrument tokens.

        D5.1 is intentionally instrument-aware rather than track-separating:
        PIANO receives the D5 chord-tone prior; BRIDGE/other instruments remain
        exactly on the original V18 NOTE-pitch distribution.

        Instrument ids are resolved from the existing vocabulary instead of being
        hard-coded, so checkpoint/state-dict semantics are unchanged.
        """
        vocab_size = int(getattr(self.music_config, "instrument_vocab_size"))
        mask = torch.zeros((vocab_size,), dtype=torch.bool)
        vocab_data = self.vocab_data if isinstance(self.vocab_data, dict) else {}
        mapping = vocab_data.get("instrument", {}) if isinstance(vocab_data, dict) else {}

        matched = []
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                token_id = None
                event_name = None
                if isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit()):
                    token_id = int(k)
                    event_name = str(v)
                elif isinstance(v, int) or (isinstance(v, str) and str(v).lstrip("-").isdigit()):
                    token_id = int(v)
                    event_name = str(k)
                if token_id is None or not (0 <= token_id < vocab_size):
                    continue

                # Pop909 role convention used by this project:
                #   Instrument_0  -> MELODY (condition)
                #   Instrument_24 -> BRIDGE
                #   Instrument_49 -> PIANO
                # Resolve the encoded token id from the vocabulary; never assume
                # that the encoded id itself is 49.
                m = re.fullmatch(r"Instrument_(-?\d+)", event_name, flags=re.IGNORECASE)
                if m is not None and int(m.group(1)) == 49:
                    mask[token_id] = True
                    matched.append((token_id, event_name))

        if len(matched) != 1:
            raise RuntimeError(
                "V18-D5.1 expected exactly one Pop909 PIANO role token "
                "(Instrument_49), but resolved "
                f"{matched}. Instrument vocabulary={mapping}"
            )

        logger.info(
            "V18-D5.1 Pop909 PIANO role resolved from Instrument_49: "
            f"{matched}"
        )
        return mask

    def _build_pitch_token_to_pc_lookup_d5(self) -> torch.Tensor:
        """Return [pitch_vocab] token -> absolute pitch class; -1 for PAD/unknown."""
        vocab_size = int(getattr(self.music_config, "pitch_vocab_size"))
        token_to_pc = torch.full((vocab_size,), -1, dtype=torch.long)
        vocab_data = self.vocab_data if isinstance(self.vocab_data, dict) else {}
        mapping = vocab_data.get("pitch", {}) if isinstance(vocab_data, dict) else {}

        parsed = 0
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                token_id = None
                event_name = None
                if isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit()):
                    token_id = int(k)
                    event_name = str(v)
                elif isinstance(v, int) or (isinstance(v, str) and str(v).lstrip("-").isdigit()):
                    token_id = int(v)
                    event_name = str(k)
                if token_id is None or token_id == 0 or not (0 <= token_id < vocab_size):
                    continue
                m = re.search(r"(?:Note_)?Pitch_(-?\d+)$", event_name)
                if m is not None:
                    midi_pitch = int(m.group(1))
                else:
                    nums = re.findall(r"-?\d+", event_name)
                    midi_pitch = int(nums[-1]) if nums else None
                if midi_pitch is None or not (0 <= midi_pitch <= 127):
                    continue
                token_to_pc[token_id] = midi_pitch % 12
                parsed += 1

        expected = max(vocab_size - 1, 1)
        coverage = parsed / expected
        if coverage < 0.99:
            raise RuntimeError(
                "V18-D5 cannot safely build NOTE pitch-class lookup: parsed only "
                f"{parsed}/{expected} non-PAD pitch tokens."
            )
        logger.info(
            f"V18-D5 pitch-token PC lookup: {parsed}/{expected} "
            f"(coverage={coverage:.4f})."
        )
        return token_to_pc

    def _build_chord_pc_membership_lookup_d5(self) -> torch.Tensor:
        """Build [chord_vocab,12] active-chord pitch-class membership.

        Supported Pop909 qualities:
          M, m, 7, M7, m7, +, o, o7, /o7, sus2, sus4.
        Inversion suffixes do not change pitch-class membership. Chord_N_N and
        PAD remain all-zero (neutral D5 bias).
        """
        chord_vocab_size = int(getattr(self.music_config, "chord_vocab_size"))
        table = torch.zeros((chord_vocab_size, 12), dtype=torch.bool)
        vocab_data = self.vocab_data if isinstance(self.vocab_data, dict) else {}
        mapping = vocab_data.get("chord", {}) if isinstance(vocab_data, dict) else {}

        root_pc = {
            "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
            "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
            "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
        }
        quality_intervals = {
            "M": (0, 4, 7),
            "m": (0, 3, 7),
            "7": (0, 4, 7, 10),
            "M7": (0, 4, 7, 11),
            "m7": (0, 3, 7, 10),
            "+": (0, 4, 8),
            "o": (0, 3, 6),
            "o7": (0, 3, 6, 9),
            "/o7": (0, 3, 6, 10),
            "sus2": (0, 2, 7),
            "sus4": (0, 5, 7),
        }

        parsed = 0
        expected = 0
        unparsed = []
        if isinstance(mapping, dict):
            for k, v in mapping.items():
                chord_id = None
                name = None
                if isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit()):
                    chord_id = int(k)
                    name = str(v)
                elif isinstance(v, int) or (isinstance(v, str) and str(v).lstrip("-").isdigit()):
                    chord_id = int(v)
                    name = str(k)
                if chord_id is None or not (0 <= chord_id < chord_vocab_size):
                    continue
                if chord_id == 0 or chord_id == self.chord_n_n_id or name.upper() in {"0", "CHORD_N_N", "N_N", "N:N"}:
                    continue
                expected += 1
                # m = re.match(r"^Chord_([^_]+)_(.+)_(-?\d+)$", name)
                m = re.match(r"^Chord_([^_]+)_(.+?)(?:_(-?\d+))?$", name)
                if m is None:
                    unparsed.append(name)
                    continue
                root_name = m.group(1).upper()
                quality = m.group(2)
                if root_name not in root_pc or quality not in quality_intervals:
                    unparsed.append(name)
                    continue
                r = root_pc[root_name]
                for interval in quality_intervals[quality]:
                    table[chord_id, (r + interval) % 12] = True
                parsed += 1

        coverage = parsed / max(expected, 1)
        if coverage < 0.99:
            raise RuntimeError(
                "V18-D5 cannot safely parse active-chord templates: "
                f"{parsed}/{expected}, examples={unparsed[:10]}"
            )
        logger.info(
            f"V18-D5 chord-template lookup: {parsed}/{expected} parsed "
            f"(coverage={coverage:.4f}); N_N is neutral."
        )
        return table

    def _soft_bias_pitch_logits_with_active_chord_d51(
        self,
        pitch_logits: torch.Tensor,
        active_chord_ids: torch.Tensor,
        instrument_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply D5's soft chord-tone bonus only to PIANO NOTE events.

        PIANO: chord-tone odds x2 (finite +ln2), non-chord odds x1.
        BRIDGE/other instruments: exact original V18 pitch logits.

        This is not a hard pitch filter: no candidate receives -inf here, and
        every instrument continues through the same NOTE decoder and sampler.
        """
        if pitch_logits.numel() == 0:
            return pitch_logits

        token_pc = self.pitch_token_to_pc.to(device=pitch_logits.device)
        known = token_pc >= 0
        if not known.any():
            return pitch_logits

        inst_ids = instrument_ids.long().clamp(
            0, self.d51_piano_instrument_mask.size(0) - 1
        )
        piano_rows = self.d51_piano_instrument_mask[inst_ids].to(
            device=pitch_logits.device
        )
        if not piano_rows.any():
            return pitch_logits

        chord_ids = active_chord_ids.long().clamp(
            0, self.chord_pc_membership.size(0) - 1
        )
        membership = self.chord_pc_membership[chord_ids].to(
            device=pitch_logits.device
        )
        is_chord_tone = membership[:, token_pc[known]]

        ln2 = torch.log(
            torch.tensor(
                1.0,
                device=pitch_logits.device,
                dtype=pitch_logits.dtype,
            )
        )
        print(f"{ln2}")

        # Row gating only changes PIANO logits. BRIDGE/other rows get exactly
        # zero added bias and therefore preserve the original V18 distribution.
        row_gate = piano_rows.to(dtype=pitch_logits.dtype).unsqueeze(1)
        selected_bias = (
            is_chord_tone.to(dtype=pitch_logits.dtype)
            * row_gate
            * ln2
        )

        out = pitch_logits.clone()
        out[:, known] = out[:, known] + selected_bias
        return out

    @staticmethod
    def _build_compact_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """Build causal positions while ignoring masked local-memory slots."""
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.clamp_min(0)

    def reset_eval_trackers(self):
        self.eval_feature_loss_sum.clear()
        self.eval_feature_count.clear()

    def _compute_type_masked_tokenwise_loss(self, logits, labels_shifted, valid_mask, type_ids):
        """Average semantically active feature losses inside each valid event."""
        type_is_chord = type_ids == self.chord_type_id
        type_is_beat = type_ids == self.beat_type_id
        type_is_plan = (
            (type_ids == self.plan_type_id)
            if self.use_harmonic_plan and self.plan_type_id >= 0
            else torch.zeros_like(type_ids, dtype=torch.bool)
        )
        type_is_chord_or_beat = type_is_chord | type_is_beat
        type_is_inst_or_note = (type_ids == self.inst_type_id) | (type_ids == self.note_type_id)
        type_is_note = type_ids == self.note_type_id

        token_loss_sum = None
        token_feature_count = None
        for feat_idx, feat_name in enumerate(self.note_features):
            if feat_name == "type":
                feat_valid = valid_mask
            elif feat_name == "chord":
                feat_valid = valid_mask & type_is_chord_or_beat
            elif feat_name == "instrument":
                feat_valid = valid_mask & type_is_inst_or_note
            elif feat_name == "beat":
                feat_valid = valid_mask & (type_is_chord_or_beat | type_is_plan)
            elif feat_name == "tempo":
                feat_valid = valid_mask & type_is_beat
            elif feat_name in {"pitch", "duration", "velocity"}:
                feat_valid = valid_mask & type_is_note
            elif feat_name in {"plan_bass", "plan_set"}:
                feat_valid = valid_mask & type_is_plan
            else:
                raise ValueError(f"Unknown NOTE feature in type-masked loss: {feat_name}")

            if feat_name == "plan_set":
                # V18: the stored scalar is already the 12-bit mask id [0,4095].
                # Learn the complete Set jointly with the same plain categorical CE.
                # Class 0 is used as padding/non-PLAN in the sequence, but PLAN
                # events are structurally non-empty and are selected by feat_valid.
                mask_values = labels_shifted[:, :, :, feat_idx].clamp(0, (1 << 12) - 1).long()
                flat_logits = logits[feat_idx].reshape(-1, 1 << 12)
                flat_targets = mask_values.reshape(-1)
                feat_valid_flat = feat_valid.reshape(-1)
                flat_targets = flat_targets.masked_fill(~feat_valid_flat, -100)
                raw_loss = F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    reduction="none",
                    ignore_index=-100,
                ).view_as(feat_valid)
                raw_loss = raw_loss * self.plan_joint_set_loss_scale
            else:
                feat_logits = logits[feat_idx].reshape(-1, logits[feat_idx].size(-1))
                feat_target_flat = labels_shifted[:, :, :, feat_idx].reshape(-1)
                feat_valid_flat = feat_valid.reshape(-1)
                feat_label_flat = feat_target_flat.masked_fill(~feat_valid_flat, -100)
                raw_loss = self.note_focal_loss(feat_logits, feat_label_flat).view_as(valid_mask)

            feat_valid_f = feat_valid.to(dtype=raw_loss.dtype)
            if token_loss_sum is None:
                token_loss_sum = raw_loss * feat_valid_f
                token_feature_count = feat_valid_f
            else:
                token_loss_sum = token_loss_sum + raw_loss * feat_valid_f
                token_feature_count = token_feature_count + feat_valid_f

            if not self.training and feat_name != "type":
                valid_sum_tensor = (raw_loss * feat_valid_f).sum()
                valid_count_tensor = feat_valid_f.sum()
                valid_count = int(valid_count_tensor.item())
                if feat_name not in self.eval_feature_loss_sum:
                    self.eval_feature_loss_sum[feat_name] = 0.0
                    self.eval_feature_count[feat_name] = 0
                self.eval_feature_loss_sum[feat_name] += float(valid_sum_tensor.item())
                self.eval_feature_count[feat_name] += valid_count

        token_feature_count = token_feature_count.clamp_min(1e-8)
        token_loss = token_loss_sum / token_feature_count
        valid_token_f = valid_mask.to(dtype=token_loss.dtype)
        return (token_loss * valid_token_f).sum() / (valid_token_f.sum() + 1e-8)

    def _compute_pos_embeddings(self, pos_input_ids, instrument_input_ids=None, instrument_attention_mask=None):
        emb = []
        for idx, feat in enumerate(self.pos_features_wo_inst):
            layer = getattr(self, f"{feat}_embedding")
            emb.append(layer(pos_input_ids[:, :, idx]))
        if self.use_pos_all_instruments:
            if instrument_input_ids is None or instrument_attention_mask is None:
                raise ValueError("all_instruments requires instrument_input_ids/instrument_attention_mask.")
            inst = self.instrument_embedding(instrument_input_ids)
            mask = instrument_attention_mask.unsqueeze(-1).to(dtype=inst.dtype)
            pooled = (inst * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-6)
            emb.append(pooled)
        pos_emb = torch.cat(emb, dim=-1)
        proj_dtype = self.pos_embed_proj.weight.dtype
        if pos_emb.dtype != proj_dtype:
            pos_emb = pos_emb.to(dtype=proj_dtype)
        return self.pos_embed_proj(pos_emb)

    def _compute_note_embeddings(self, note_input_ids):
        # [B,T,N,F] or [B,N,F]
        squeeze_t = False
        if note_input_ids.dim() == 3:
            note_input_ids = note_input_ids.unsqueeze(1)
            squeeze_t = True
        emb = []
        for idx, feat in enumerate(self.note_features):
            values = note_input_ids[:, :, :, idx]
            if feat == "plan_set":
                bit_ids = torch.arange(12, device=values.device, dtype=torch.long)
                bits = ((values.unsqueeze(-1) >> bit_ids) & 1).to(
                    dtype=self.plan_interval_embedding.weight.dtype
                )
                # [...,12] @ [12,D] -> [...,D]. A zero/non-PLAN mask maps
                # exactly to the zero vector rather than a learned PAD set.
                emb.append(torch.matmul(bits, self.plan_interval_embedding.weight))
            else:
                layer = getattr(self, f"{feat}_embedding")
                emb.append(layer(values))
        x = self.note_embed_proj(torch.cat(emb, dim=-1))
        return x[:, 0] if squeeze_t else x

    def _compute_bar_summary(self, note_input_ids, note_attention_mask):
        note_proj = self._compute_note_embeddings(note_input_ids)
        return self._compute_bar_summary_from_note_proj(note_proj, note_attention_mask)

    def _compute_bar_summary_from_note_proj(self, note_proj, note_attention_mask, summary_type: str = "bar"):
        bar_note_tokens = self.bar_note_in_proj(note_proj)
        return self.bar_note_encoder(bar_note_tokens, note_attention_mask, summary_type=summary_type)

    def _condition_hidden_on_type(self, hidden_states: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        type_emb = self.type_embedding(type_ids)
        if type_emb.dtype != hidden_states.dtype:
            type_emb = type_emb.to(dtype=hidden_states.dtype)
        return self.type_cond_proj(torch.cat([hidden_states, type_emb], dim=-1))

    def _condition_plan_set_on_bass(
        self,
        hidden_states: torch.Tensor,
        bass_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build P(Set | Bass, history) input for the Bass-relative PLAN Set."""
        bass_emb = self.plan_bass_embedding(bass_ids)
        if bass_emb.dtype != hidden_states.dtype:
            bass_emb = bass_emb.to(dtype=hidden_states.dtype)
        return torch.cat([hidden_states, bass_emb], dim=-1)
    
    # condition
    def _encode_condition_memory(
        self,
        condition_note_ids: torch.Tensor,
        condition_note_mask: torch.Tensor,
        condition_pos_proj: torch.Tensor,
        condition_time_mask: torch.Tensor,
    ):
        if not self.use_global_condition:
            raise RuntimeError("_encode_condition_memory called in continuation mode.")

        if condition_note_ids.dim() != 4:
            raise ValueError("condition_note_ids must be [B,T,N,F].")

        if condition_note_mask.dim() != 3:
            raise ValueError("condition_note_mask must be [B,T,N].")

        if condition_note_ids.shape[:2] != condition_pos_proj.shape[:2]:
            raise ValueError("Condition note sequence and condition position sequence are not time-aligned.")

        condition_note_proj = self._compute_note_embeddings(condition_note_ids)
        condition_bar_summary = self._compute_bar_summary_from_note_proj(
                condition_note_proj,
                condition_note_mask,
                summary_type="condition",
            )
        condition_input = self.condition_input_proj(torch.cat([condition_pos_proj, condition_bar_summary,], dim=-1))
        condition_bar_mask = (condition_time_mask.bool() & condition_note_mask.bool().any(dim=-1))

        if (~condition_bar_mask.any(dim=1)).any():
            bad_samples = (~condition_bar_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Some samples contain no valid condition bars: {bad_samples[:10]}")

        condition_memory = self.condition_encoder(condition_input, src_key_padding_mask=~condition_bar_mask)

        return condition_memory, condition_bar_mask

    def _build_conditioned_prefix(self, time_hidden: torch.Tensor, local_condition: Optional[torch.Tensor] = None):
        history_prefix = self.qformer_connector(time_hidden)
        condition_token = self.condition_to_note(local_condition).unsqueeze(2)
        return torch.cat([condition_token, history_prefix], dim=2)
    
    def _validate_forward_mode_inputs(self, condition_note_ids, condition_note_mask):
        if self.task_mode == "continuation":
            if (condition_note_ids is not None or condition_note_mask is not None):
                raise ValueError("Continuation forward must not receive condition inputs.")
            return

        if self.task_mode in {"chord", "accompaniment"}:
            if (condition_note_ids is None or condition_note_mask is None):
                raise ValueError(
                    f"task_mode={self.task_mode!r} requires condition_note_ids and condition_note_mask."
                )
            return

        raise RuntimeError(f"Unexpected task_mode={self.task_mode!r}.")
    
    def forward(
        self,
        pos_input_ids: torch.LongTensor = None,
        pos_attention_mask: Optional[torch.Tensor] = None,
        note_input_ids: torch.LongTensor = None,
        note_attention_mask: Optional[torch.Tensor] = None,
        instrument_input_ids: Optional[torch.Tensor] = None,
        instrument_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        condition_note_ids: Optional[torch.LongTensor] = None,
        condition_note_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if note_input_ids is None:
            raise ValueError("note_input_ids is required.")
        if pos_input_ids is None:
            raise ValueError("pos_input_ids is required.")
        if pos_attention_mask is None:
            raise ValueError("pos_attention_mask is required and must be provided by data pipeline.")
        if note_attention_mask is None:
            raise ValueError("note_attention_mask is required and must be provided by data pipeline.")
        if note_input_ids.size(1) < 2:
            raise ValueError("Need at least 2 shifted bars in a batch sample for time-shift supervision.")
        self._validate_forward_mode_inputs(condition_note_ids, condition_note_mask)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        note_tokens_full = self._compute_note_embeddings(note_input_ids)
        bar_summary = self._compute_bar_summary_from_note_proj(note_tokens_full, note_attention_mask)
        if instrument_input_ids is not None and instrument_attention_mask is not None:
            pos_proj = self._compute_pos_embeddings(
                pos_input_ids,
                instrument_input_ids,
                instrument_attention_mask,
            )
        else:
            pos_proj = self._compute_pos_embeddings(pos_input_ids, None, None)

        # All-Global ablation: compress the complete available history.
        #
        # For target bar B_{t+1}, bar_summary[t] summarizes B_t.  Feeding it
        # into the same causal TimeModel step makes the Global path contain
        # history through the immediate previous bar:
        #
        #   Global path = compressed history through B_t
        #   Local path  = removed
        #
        # This is the only architectural ablation relative to the partitioned
        # G(t-1)+L(t) design; target/label alignment and within-bar AR shifting
        # remain unchanged.
        time_in = self.time_input_proj(
            torch.cat([pos_proj, bar_summary], dim=-1)
        )
        time_out = self.time_model(
            inputs_embeds=time_in,
            attention_mask=pos_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        time_hidden = time_out.last_hidden_state

        if self.task_mode == "continuation":
            note_time_hidden = time_hidden[:, :-1, :,]
            prefix = self.qformer_connector(note_time_hidden)
            condition_loss_mask = None
        else:
            if condition_note_ids is None or condition_note_mask is None:
                raise ValueError(
                    f"task_mode={self.task_mode!r} requires condition_note_ids and condition_note_mask."
                )

            # time_hidden[t] 预测 note_input_ids[t+1]
            note_time_hidden = time_hidden[:, :-1, :]

            # pos_input_ids[t] 对应 target bar t+1，
            # 因此这里也是取 pos_proj[:, :-1]
            condition_pos_proj = pos_proj[:, :-1, :]
            condition_time_mask = pos_attention_mask[:, 1:]

            if condition_note_ids.shape[1] != note_time_hidden.shape[1]:
                raise ValueError(
                    "Condition time length does not match target time length: "
                    f"{condition_note_ids.shape[1]} vs {note_time_hidden.shape[1]}."
                )

            condition_memory, condition_bar_mask = self._encode_condition_memory(
                    condition_note_ids=condition_note_ids,
                    condition_note_mask=condition_note_mask,
                    condition_pos_proj=condition_pos_proj,
                    condition_time_mask=condition_time_mask,
            )

            prefix = self._build_conditioned_prefix(
                time_hidden=note_time_hidden,
                local_condition=condition_memory,
            )
            condition_loss_mask = condition_note_mask.bool()
        

        # All-Global ablation.
        #
        # The immediate previous bar is already compressed into the current
        # Global Prefix by the TimeModel above, so the full-resolution Local
        # NOTE Memory is intentionally removed.  The Note decoder receives:
        #
        #   [Global Prefix][Current Bar]
        #
        # Current-bar teacher forcing and its one-token autoregressive shift
        # are unchanged.
        note_tokens = note_tokens_full[:, 1:, :, :]

        combined = torch.cat([prefix, note_tokens], dim=2)
        b, t, l, d = combined.shape
        prefix_len = prefix.shape[2]
        current_start = prefix_len

        prefix_mask = torch.ones(
            (b, t, prefix_len),
            dtype=torch.long,
            device=combined.device,
        )
        note_dec_mask = torch.cat(
            [
                prefix_mask,
                note_attention_mask[:, 1:, :].long(),
            ],
            dim=-1,
        )
        flat_note_dec_mask = note_dec_mask.reshape(b * t, l)
        note_position_ids = self._build_compact_position_ids(flat_note_dec_mask)

        note_inputs = self.note_emb_tokens_proj(combined.reshape(b * t, l, d))
        note_out = self.note_model(
            inputs_embeds=note_inputs,
            attention_mask=flat_note_dec_mask,
            position_ids=note_position_ids,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = note_out.last_hidden_state.view(b, t, l, -1)

        # With no full-resolution Local memory, the first current-bar event is
        # predicted directly from the final Global Prefix state.
        first_pred_hidden = hidden[:, :, prefix_len - 1 : prefix_len, :]

        # Subsequent current-bar events preserve the original one-token
        # autoregressive shift: event j is predicted from current event j-1.
        n = note_tokens_full.size(2)
        if n > 1:
            previous_current_positions = (
                current_start
                + torch.arange(n - 1, device=hidden.device, dtype=torch.long)
            )
            later_pred_hidden = hidden[:, :, previous_current_positions, :]
            note_pred_hidden = torch.cat(
                [first_pred_hidden, later_pred_hidden],
                dim=2,
            )
        else:
            note_pred_hidden = first_pred_hidden
        type_logits = self.type_head(note_pred_hidden)
        if labels is not None:
            type_target = labels[:, 1:, :, self.note_type_idx].clamp_min(0)
        else:
            type_target = note_input_ids[:, 1:, :, self.note_type_idx]
        type_conditioned_hidden = self._condition_hidden_on_type(note_pred_hidden, type_target)

        # PLAN-only teacher forcing.  With use_harmonic_plan=False the runtime
        # schema has no PLAN fields, so training never reads a PLAN Bass/Set and
        # no PLAN-conditioned hidden state is constructed.
        plan_set_hidden = None
        if self.use_harmonic_plan:
            if labels is not None:
                # Teacher forcing: the joint Set mask is expressed in the coordinate
                # system defined by the GT QuarterBass, so condition on that GT Bass.
                plan_set_bass_ids = labels[:, 1:, :, self.note_plan_bass_idx].clamp_min(0)
            else:
                plan_set_bass_ids = note_input_ids[:, 1:, :, self.note_plan_bass_idx].clamp_min(0)
            plan_set_hidden = self._condition_plan_set_on_bass(
                type_conditioned_hidden,
                plan_set_bass_ids,
            )
        
        logits = []
        for feat in self.note_features:
            if feat == "type":
                logits.append(type_logits)
            elif feat == "plan_set":
                if not self.use_harmonic_plan or plan_set_hidden is None:
                    raise RuntimeError(
                        "plan_set logits requested while use_harmonic_plan=False."
                    )
                logits.append(self.plan_set_head(plan_set_hidden))
            else:
                head = getattr(self, f"{feat}_head")
                logits.append(head(type_conditioned_hidden))

        # Compute Logits and Loss
        loss = None
        if labels is not None:
            labels_shifted = labels[:, 1:, :, :]
            valid_mask = labels_shifted[..., 0] != -100
            
            if condition_loss_mask is not None:
                if condition_loss_mask.shape != valid_mask.shape:
                    raise ValueError(
                        "condition_note_mask must align with shifted labels."
                    )

                valid_mask = (
                    valid_mask
                    & ~condition_loss_mask
                )

            type_ids = type_target

            if self.using_type_mask:
                loss = self._compute_type_masked_tokenwise_loss(
                    logits=logits,
                    labels_shifted=labels_shifted,
                    valid_mask=valid_mask,
                    type_ids=type_ids,
                )
            else:
                note_loss = None
                for feat_idx, feat_name in enumerate(self.note_features):
                    feat_logits = logits[feat_idx].reshape(-1, logits[feat_idx].size(-1))
                    feat_target_flat = labels_shifted[:, :, :, feat_idx].reshape(-1)

                    feat_valid = valid_mask
                    feat_valid_flat = feat_valid.reshape(-1)
                    feat_label_flat = feat_target_flat.masked_fill(~feat_valid_flat, -100)

                    if feat_name == "plan_set":
                        raw = F.cross_entropy(
                            feat_logits,
                            feat_label_flat.long(),
                            reduction="none",
                            ignore_index=-100,
                        ) * self.plan_joint_set_loss_scale
                        valid_flat = feat_label_flat != -100
                        valid_flat_f = valid_flat.to(dtype=raw.dtype)
                        valid_sum_tensor = (raw * valid_flat_f).sum()
                        valid_count_tensor = valid_flat_f.sum()
                        current_feature_loss = valid_sum_tensor / (valid_count_tensor + 1e-8)
                    elif self.training:
                        current_feature_loss = self.note_focal_loss(
                            feat_logits,
                            feat_label_flat,
                        )
                    else:
                        raw = self.note_focal_loss_eval(
                            feat_logits,
                            feat_label_flat,
                        )
                        valid_flat = feat_label_flat != -100
                        valid_flat_f = valid_flat.to(dtype=raw.dtype)
                        valid_sum_tensor = (raw * valid_flat_f).sum()
                        valid_count_tensor = valid_flat_f.sum()
                        current_feature_loss = valid_sum_tensor / (valid_count_tensor + 1e-8)

                    if not self.training:
                        # For plan_set the tensors above were already computed by
                        # joint CE; for all other features they come from focal loss.
                        if feat_name == "plan_set":
                            valid_flat = feat_label_flat != -100
                            valid_flat_f = valid_flat.to(dtype=raw.dtype)
                            valid_sum_tensor = (raw * valid_flat_f).sum()
                            valid_count_tensor = valid_flat_f.sum()
                        valid_count = int(valid_count_tensor.item())
                        if feat_name != "type":
                            if feat_name not in self.eval_feature_loss_sum:
                                self.eval_feature_loss_sum[feat_name] = 0.0
                                self.eval_feature_count[feat_name] = 0
                            self.eval_feature_loss_sum[feat_name] += float(valid_sum_tensor.item())
                            self.eval_feature_count[feat_name] += valid_count

                    if note_loss is None:
                        note_loss = current_feature_loss
                    else:
                        note_loss = note_loss + current_feature_loss

                if note_loss is not None:
                    loss = note_loss / max(len(self.note_features), 1)
                
        if not return_dict:
            return (loss, logits[0], None)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits[0],
            past_key_values=None,
            hidden_states=note_out.hidden_states,
            attentions=note_out.attentions,
        )

    def _validate_generation_mode_inputs(
        self,
        prompt_note_ids: Optional[torch.Tensor],
        prompt_lens: Optional[torch.Tensor],
        condition_note_ids: Optional[torch.Tensor],
        condition_note_mask: Optional[torch.Tensor],
    ) -> None:
        if self.task_mode == "continuation":
            if condition_note_ids is not None or condition_note_mask is not None:
                raise ValueError(
                    "task_mode='continuation' must not receive "
                    "condition_note_ids/condition_note_mask."
                )
            if prompt_note_ids is not None and prompt_lens is None:
                raise ValueError("prompt_lens is required when prompt_note_ids is provided.")
            return

        if self.task_mode in {"chord", "accompaniment"}:
            if condition_note_ids is None or condition_note_mask is None:
                raise ValueError(
                    f"task_mode={self.task_mode!r} requires "
                    "condition_note_ids and condition_note_mask."
                )

            if prompt_note_ids is not None or prompt_lens is not None:
                raise ValueError(
                    f"task_mode={self.task_mode!r} must not use "
                    "prompt_note_ids/prompt_lens."
                )

            return

        raise RuntimeError(f"Unexpected task_mode={self.task_mode!r}.")
    
    
    def _prepare_generation_condition(
        self,
        condition_note_ids: torch.Tensor,
        condition_note_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        instrument_input_ids: Optional[torch.Tensor],
        instrument_attention_mask: Optional[torch.Tensor],
        target_steps: torch.Tensor,
        max_time: int,
        max_note_nums: int,
        loop_device: torch.device,
    ):
        if not self.use_global_condition:
            raise RuntimeError(
                "_prepare_generation_condition was called "
                "when use_global_condition=False."
            )

        condition_note_ids = condition_note_ids.to(loop_device)
        condition_note_mask = condition_note_mask.to(loop_device)
        
        if condition_note_ids.dim() != 4:
            raise ValueError(
                "condition_note_ids must be [B,T,N,F], "
                f"got {tuple(condition_note_ids.shape)}."
            )

        if condition_note_mask.dim() != 3:
            raise ValueError(
                "condition_note_mask must be [B,T,N], "
                f"got {tuple(condition_note_mask.shape)}."
            )

        batch_size = pos_input_ids.shape[0]

        if condition_note_ids.shape[0] != batch_size:
            raise ValueError(
                "condition_note_ids batch size does not match "
                f"pos_input_ids: {condition_note_ids.shape[0]} "
                f"vs {batch_size}."
            )

        if condition_note_mask.shape != condition_note_ids.shape[:3]:
            raise ValueError(
                "condition_note_mask shape does not match "
                "condition_note_ids: "
                f"{tuple(condition_note_mask.shape)} vs "
                f"{tuple(condition_note_ids.shape[:3])}."
            )

        if condition_note_ids.shape[-1] != len(self.note_features):
            raise ValueError(
                "condition_note_ids feature dimension mismatch: "
                f"{condition_note_ids.shape[-1]} vs "
                f"{len(self.note_features)}."
            )

        if condition_note_ids.shape[1] < max_time:
            raise ValueError(
                "Condition time sequence is shorter than generation length: "
                f"{condition_note_ids.shape[1]} vs {max_time}."
            )

        # Collator 已经完成唯一一次 shift。
        # 这里仅按实际生成长度裁剪，不能再做 [:, 1:]。
        condition_note_ids = condition_note_ids[:, :max_time, :, :]
        condition_note_mask = condition_note_mask[:, :max_time, :].bool()

        time_indices = torch.arange(max_time, device=loop_device,).unsqueeze(0)
        target_step_mask = (time_indices < target_steps.unsqueeze(1))

        # padded target bar 的 condition 必须关闭。
        condition_note_mask = (condition_note_mask & target_step_mask.unsqueeze(-1))
        active_without_condition = (target_step_mask & ~condition_note_mask.any(dim=-1))
        if active_without_condition.any():
            examples = (
                active_without_condition
                .nonzero(as_tuple=False)[:10].detach().cpu().tolist()
            )
            raise ValueError(
                "Some active target bars contain no condition tokens. "
                f"Examples [batch, target_bar]: {examples}"
            )

        # 检查 forcing token 是否超出 max_note_nums。
        note_positions = torch.arange(condition_note_mask.shape[2],device=loop_device,).view(1, 1, -1) + 1
        condition_width = torch.where(condition_note_mask,note_positions,torch.zeros_like(note_positions),).amax(dim=-1)
        too_long = (target_step_mask & (condition_width > max_note_nums))

        if too_long.any():
            examples = (too_long.nonzero(as_tuple=False)[:10].detach().cpu().tolist())
            widths = condition_width[too_long][:10].detach().cpu().tolist()
            raise ValueError(
                "Condition prefix exceeds max_note_nums. "
                f"Examples [batch, bar]={examples}, "
                f"condition widths={widths}, "
                f"max_note_nums={max_note_nums}."
            )

        condition_pos_proj = self._compute_pos_embeddings(
            pos_input_ids[:, :max_time, :],
            (
                instrument_input_ids[:, :max_time, :]
                if instrument_input_ids is not None
                else None
            ),
            (
                instrument_attention_mask[:, :max_time, :]
                if instrument_attention_mask is not None
                else None
            ),
        )

        condition_memory, condition_bar_mask = (
            self._encode_condition_memory(
                condition_note_ids=condition_note_ids,
                condition_note_mask=condition_note_mask.long(),
                condition_pos_proj=condition_pos_proj,
                condition_time_mask=target_step_mask.long(),
            )
        )

        sto_vec = torch.tensor(
            self.music_config.sto_note_id,
            dtype=torch.long,
            device=loop_device,
        )

        feature_dim = len(self.note_features)

        force_ids = (sto_vec.view(1, 1, 1, feature_dim)
            .repeat(batch_size, max_time, max_note_nums, 1,)
        )

        force_mask = torch.zeros(
            batch_size,
            max_time,
            max_note_nums,
            dtype=torch.long,
            device=loop_device,
        )
        copy_note_num = min(condition_note_ids.shape[2], max_note_nums,)
        force_ids[:, :, :copy_note_num, :] = condition_note_ids[:, :, :copy_note_num, :]
        force_mask[:, :, :copy_note_num] = condition_note_mask[:, :, :copy_note_num].long()
        condition_chord_lookup = None
        if self.task_mode in {"chord", "accompaniment"}:
            condition_chord_lookup = (
                self._build_condition_chord_lookup(
                    condition_note_ids,
                    condition_note_mask,
                )
            )
        return (
            condition_memory,
            condition_bar_mask,
            force_ids,
            force_mask,
            condition_chord_lookup
        )

    def _build_condition_chord_lookup(self, condition_note_ids, condition_note_mask):
        """
        output:
        lookup[b, t, beat_token_id]
        -> active chord token id
        """
        device = condition_note_ids.device
        bsz, time_steps = condition_note_ids.shape[:2]
        beat_vocab_size = int(self.music_config.beat_vocab_size)

        lookup = torch.full(
            (bsz, time_steps, beat_vocab_size),
            fill_value=self.chord_n_n_id,
            dtype=torch.long,
            device=device,
        )
        beat_positions = torch.tensor(
            self.beat_id_to_position,
            dtype=torch.long,
            device=device,
        )

        valid_beat_ids = beat_positions >= 0
        for b in range(bsz):
            for t in range(time_steps):
                valid = condition_note_mask[b, t].bool()
                if not valid.any():
                    continue
                tokens = condition_note_ids[ b, t, valid, :]
                chord_tokens = tokens[tokens[:, self.note_type_idx] == self.chord_type_id]

                if chord_tokens.shape[0] == 0:
                    raise ValueError("Conditioned bar contains no CHORD tokens.")

                change_beat_ids = chord_tokens[:, self.note_beat_idx]
                change_chord_ids = chord_tokens[:, self.note_chord_idx]
                change_positions = beat_positions[change_beat_ids]

                if int(change_positions[0]) != 0:
                    raise ValueError("CHORD timeline must start at Beat_0.")
                if (change_positions.shape[0] > 1 and (change_positions[1:] <= change_positions[:-1]).any()):
                    raise ValueError("CHORD positions must be strictly increasing.")

                eligible = (change_positions.unsqueeze(1) <= beat_positions.unsqueeze(0))
                last_change_idx = (eligible.long().sum(dim=0) - 1)
                valid_lookup = (valid_beat_ids & (last_change_idx >= 0))
                lookup[b, t, valid_lookup] = (change_chord_ids[last_change_idx[valid_lookup]])
        return lookup

    def _mask_invalid_beat_logits(
        self,
        beat_logits: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = torch.tensor(
            [
                position >= 0
                for position
                in self.beat_id_to_position
            ],
            dtype=torch.bool,
            device=beat_logits.device,
        )

        return beat_logits.masked_fill(
            ~valid_mask.unsqueeze(0),
            torch.finfo(beat_logits.dtype).min,
        )

    @torch.no_grad()
    def generate(
        self,
        pos_input_ids: torch.LongTensor,
        pos_attention_mask: Optional[torch.LongTensor] = None,
        prompt_note_ids: Optional[torch.LongTensor] = None,
        prompt_lens: Optional[torch.LongTensor] = None,
        condition_note_ids: Optional[torch.LongTensor] = None,
        condition_note_mask: Optional[torch.LongTensor] = None,
        instrument_input_ids: Optional[torch.Tensor] = None,
        instrument_attention_mask: Optional[torch.Tensor] = None,
        max_time_steps: Optional[int] = None,
        max_note_nums: Optional[int] = None,
        device: Optional[torch.device] = None,
        decode_strategy: str = "greedy",
        sampling_parameters: Optional[dict] = None,
    ):
        self.eval()
        loop_device = device or pos_input_ids.device
        pos_input_ids = pos_input_ids.to(loop_device)
        if pos_attention_mask is not None:
            pos_attention_mask = pos_attention_mask.to(loop_device)
        if instrument_input_ids is not None:
            if instrument_attention_mask is None:
                raise ValueError(
                    "instrument_attention_mask is required when "
                    "instrument_input_ids is provided.")
            instrument_input_ids = instrument_input_ids.to(loop_device)
            instrument_attention_mask = instrument_attention_mask.to(loop_device)
        
        if condition_note_ids is not None:
            condition_note_ids = condition_note_ids.to(loop_device)
        if condition_note_mask is not None:
            condition_note_mask = condition_note_mask.to(loop_device)
        
        self._validate_generation_mode_inputs(
            prompt_note_ids=prompt_note_ids,
            prompt_lens=prompt_lens,
            condition_note_ids=condition_note_ids,
            condition_note_mask=condition_note_mask,
        )

        b, t_total = pos_input_ids.shape[:2]
        if pos_attention_mask is None:
            raise ValueError(
                "pos_attention_mask is required."
            )
        valid_time_steps = (pos_attention_mask.long().sum(dim=1).clamp(min=0, max=t_total))

        target_steps = (valid_time_steps - 1).clamp(min=0, max=max(t_total - 1, 0),)

        if int(target_steps.max().item()) <= 0:
            raise ValueError("No valid target bars are available for generation.")
        max_time_limit = (int(max_time_steps) if max_time_steps is not None else t_total - 1)
        max_time = min( max_time_limit, t_total - 1, int(target_steps.max().item()),)
        max_note_nums = (int(max_note_nums) if max_note_nums is not None else 64)
        
        is_continuation = self.task_mode == "continuation"
        beat_positions = torch.tensor(self.beat_id_to_position, dtype=torch.long,device=loop_device)
        valid_beat_ids = beat_positions >= 0
        beat_0_candidates = torch.nonzero(beat_positions == 0, as_tuple=False).flatten()
        if beat_0_candidates.numel() != 1:
            raise ValueError("Beat vocabulary must contain exactly one Beat_0 token.")

        beat_0_id = int(beat_0_candidates.item())
        max_beat_position = int(beat_positions[valid_beat_ids].max().item())

        if self.use_harmonic_plan:
            if self.plan_quarter_positions != (0, 4, 8, 12) or len(self.plan_quarter_beat_ids) != 4:
                raise RuntimeError(
                    "DID v9 quarter-level PLAN requires resolved Beat_0/4/8/12 IDs."
                )
            plan_valid_beat_ids = torch.zeros_like(valid_beat_ids, dtype=torch.bool)
            for beat_id in self.plan_quarter_beat_ids:
                plan_valid_beat_ids[int(beat_id)] = True
            max_plan_position = 12
        else:
            plan_valid_beat_ids = torch.zeros_like(valid_beat_ids, dtype=torch.bool)
            max_plan_position = -1

        sto_vec = torch.tensor(self.music_config.sto_note_id, dtype=torch.long, device=loop_device)
        sos_vec = torch.tensor(self.music_config.sos_note_id, dtype=torch.long, device=loop_device)

        generated_notes = sto_vec.view(1, 1, 1, -1).repeat(b, max_time, max_note_nums, 1)
        note_feature_nums = len(self.note_features)

        condition_memory = None
        condition_bar_mask = None
        force_ids = None
        force_mask = None
        condition_chord_lookup = None
        if self.use_global_condition:
            condition_memory, condition_bar_mask, force_ids, force_mask, condition_chord_lookup = self._prepare_generation_condition(
                condition_note_ids=condition_note_ids,
                condition_note_mask=condition_note_mask,
                pos_input_ids=pos_input_ids,
                instrument_input_ids=instrument_input_ids,
                instrument_attention_mask=instrument_attention_mask,
                target_steps=target_steps,
                max_time=max_time,
                max_note_nums=max_note_nums,
                loop_device=loop_device,
            )

        prev_notes = sto_vec.view(1, 1, -1).repeat(b, max_note_nums, 1)
        prev_notes[:, 0, :] = sos_vec
        prev_mask = torch.zeros((b, max_note_nums), dtype=torch.long, device=loop_device)
        prev_mask[:, 0] = 1

        if prompt_note_ids is not None:
            prompt_note_ids = prompt_note_ids.to(loop_device)
            if prompt_lens is None:
                raise ValueError(
                    "prompt_lens is required when prompt_note_ids is provided. "
                )
            prompt_lens = prompt_lens.to(loop_device)
            prompt_lens = torch.minimum(prompt_lens, target_steps)
        else:
            prompt_lens = torch.zeros((b,), dtype=torch.long, device=loop_device)

        # All-Global ablation keeps no separate Local NOTE memory.  At every
        # generation step, prev_notes is summarized and immediately supplied
        # to the current causal TimeModel step, matching training exactly.
        time_past = None
        for t in range(max_time):
            active_bar = (t < target_steps)
            if not active_bar.any():
                break

            # Compress the FINAL immediate previous bar (generated, forced, or
            # prompt-replaced) and use it in this same TimeModel step.  Thus
            # the current Global state contains history through B_{t-1}.
            prev_bar_sum = self._compute_bar_summary(
                prev_notes.unsqueeze(1),
                prev_mask.unsqueeze(1),
            )

            pos_t = pos_input_ids[:, t : t + 1, :]
            inst_t = instrument_input_ids[:, t : t + 1, :] if instrument_input_ids is not None else None
            inst_m_t = instrument_attention_mask[:, t : t + 1, :] if instrument_attention_mask is not None else None
            pos_proj = self._compute_pos_embeddings(pos_t, inst_t, inst_m_t)
            time_in = self.time_input_proj(
                torch.cat([pos_proj, prev_bar_sum], dim=-1)
            )

            time_out = self.time_model(inputs_embeds=time_in, past_key_values=time_past, use_cache=True, return_dict=True)
            time_past = time_out.past_key_values
            cur_time_hidden = time_out.last_hidden_state

            if self.use_global_condition:
                local_condition = condition_memory[:, t : t + 1, :]
                local_condition = (
                    local_condition * active_bar.view(b, 1, 1).to(dtype=local_condition.dtype)
                )
                prefix = self._build_conditioned_prefix(
                    time_hidden=cur_time_hidden,
                    local_condition=local_condition,
                )[:, 0, :, :]

            else:
                prefix = self.qformer_connector.forward_step(cur_time_hidden[:, 0, :])

            # All-Global inference mirrors training: the previous bar has
            # already entered the Global state as prev_bar_sum, so NoteModel is
            # primed by the Global Prefix only.
            prime = self.note_emb_tokens_proj(prefix)
            prefix_len = prefix.size(1)
            note_running_mask = torch.ones(
                (b, prefix_len), dtype=torch.long, device=loop_device
            )
            prime_position_ids = self._build_compact_position_ids(note_running_mask)
            note_out = self.note_model(
                inputs_embeds=prime,
                attention_mask=note_running_mask,
                position_ids=prime_position_ids,
                use_cache=True,
                return_dict=True,
            )
            note_past = note_out.past_key_values

            # First generated event is predicted from the final Global Prefix
            # state; there is intentionally no full-resolution Local state.
            cur_hidden = note_out.last_hidden_state[:, -1, :]

            cur_notes = sto_vec.view(1, 1, -1).repeat(b, max_note_nums, 1)
            finished = torch.zeros((b,), dtype=torch.bool, device=loop_device)
            inst_ctx = torch.zeros((b,), dtype=torch.long, device=loop_device)
            beat_ctx = torch.zeros((b,), dtype=torch.long, device=loop_device)
            prefix_open = torch.ones((b,), dtype=torch.bool, device=loop_device)
            generated_chord_lookup = torch.full(
                (b, int(self.music_config.beat_vocab_size)),
                fill_value=self.chord_n_n_id,
                dtype=torch.long,
                device=loop_device,
            )
            generated_last_chord_position = torch.full(
                (b,),
                fill_value=-1,
                dtype=torch.long,
                device=loop_device,
            )
            generated_has_chord = torch.zeros(
                (b,),
                dtype=torch.bool,
                device=loop_device,
            )
            plan_started = torch.zeros((b,), dtype=torch.bool, device=loop_device)
            target_track_started = torch.zeros((b,), dtype=torch.bool, device=loop_device)
            last_plan_position = torch.full((b,), -1, dtype=torch.long, device=loop_device)
            if force_mask is not None:
                condition_width_t = force_mask[:, t, :].long().sum(dim=-1)
            else:
                condition_width_t = torch.zeros((b,), dtype=torch.long, device=loop_device)

            for n_idx in range(max_note_nums):
                # After the forced CHORD/condition-track prefix, PLAN starts from a
                # clean realization context rather than inheriting the final forced
                # condition-track INST/BEAT state.
                if force_mask is not None:
                    boundary = active_bar & (condition_width_t == n_idx) & (~finished)
                    if boundary.any():
                        inst_ctx[boundary] = 0
                        beat_ctx[boundary] = 0
                        prefix_open[boundary] = False
                already_finished = finished.clone()
                force_here = torch.zeros((b,), dtype=torch.bool, device=loop_device)
                ids = torch.zeros((b, len(self.note_features)), dtype=torch.long, device=loop_device)
                ids[:] = sto_vec.view(1, -1)

                free_mask = ~already_finished
                if free_mask.any():
                    ids_free = ids[free_mask].clone()
                    hidden_free = cur_hidden[free_mask]

                    type_logits = getattr(self, "type_head")(hidden_free)
                    free_batch_indices = free_mask.nonzero(as_tuple=True)[0]
                    condition_done = (n_idx >= condition_width_t[free_batch_indices])

                    allow_chord = prefix_open[free_mask]
                    if force_ids is not None:
                        allow_chord = torch.zeros_like(allow_chord, dtype=torch.bool)
                    if is_continuation:
                        allow_chord = (
                            allow_chord
                            & ~plan_started[free_batch_indices]
                            & ~target_track_started[free_batch_indices]
                            & (generated_last_chord_position[free_batch_indices] < max_beat_position)
                        )

                    if self.use_harmonic_plan:
                        chord_ready = (
                            generated_has_chord[free_batch_indices]
                            if is_continuation
                            else torch.ones_like(condition_done, dtype=torch.bool)
                        )
                        allow_plan = (
                            condition_done
                            & chord_ready
                            & ~target_track_started[free_batch_indices]
                            & (last_plan_position[free_batch_indices] < max_plan_position)
                        )
                        allow_inst = condition_done & plan_started[free_batch_indices]
                    else:
                        allow_plan = torch.zeros_like(condition_done, dtype=torch.bool)
                        allow_inst = condition_done if force_ids is not None else torch.ones_like(condition_done, dtype=torch.bool)

                    has_inst = inst_ctx[free_mask] > 0
                    has_beat = beat_ctx[free_mask] > 0
                    type_logits = self._apply_generation_type_constraints(
                        type_logits=type_logits,
                        allow_chord=allow_chord,
                        allow_plan=allow_plan,
                        allow_inst=allow_inst,
                        has_inst=has_inst,
                        has_beat=has_beat,
                    )

                    # 每个 continuation 小节必须首先生成一条 CHORD。
                    if is_continuation:
                        need_first_chord = (prefix_open[free_batch_indices] & ~generated_has_chord[free_batch_indices])
                        if need_first_chord.any():
                            min_value = torch.finfo(type_logits.dtype).min
                            type_logits[need_first_chord, :] = min_value
                            type_logits[need_first_chord, self.chord_type_id] = 0.0
                    # Continuation evaluation requires 32 complete bars.
                    # Suppress EOS throughout bars 1..32 (t=0..31); EOS becomes
                    # available again from bar 33 (t=32). Non-continuation modes
                    # preserve the existing behavior of always suppressing EOS.
                    if (not is_continuation) or t < 32:
                        type_logits[:, self.eos_type_id] = torch.finfo(type_logits.dtype).min
                    sampled_type = self._sample_generate_token(type_logits, decode_strategy, sampling_parameters, "type")
                    ids_free[:, self.note_type_idx] = sampled_type
                    hidden_free = self._condition_hidden_on_type(hidden_free, sampled_type)

                    is_chord = sampled_type == self.chord_type_id
                    is_plan = (sampled_type == self.plan_type_id) if self.use_harmonic_plan else torch.zeros_like(sampled_type, dtype=torch.bool)
                    is_inst = sampled_type == self.inst_type_id
                    is_beat = sampled_type == self.beat_type_id
                    is_note = sampled_type == self.note_type_id

                    # if self.note_chord_idx is not None and is_chord.any():
                    #     chord_logits = getattr(self, "chord_head")(hidden_free)
                    #     sampled_chord = self._sample_generate_token(chord_logits, decode_strategy, sampling_parameters, "chord")
                    #     sampled_chord_ids = sampled_chord[is_chord]
                    #     ids_free[is_chord, self.note_chord_idx] = sampled_chord_ids
                    #     chord_beat_logits = getattr(self, "beat_head")(hidden_free)
                    #     sampled_chord_beat = self._sample_generate_token(
                    #         chord_beat_logits,
                    #         decode_strategy,
                    #         sampling_parameters,
                    #         "beat",
                    #     )
                    #     ids_free[is_chord, self.note_beat_idx] = sampled_chord_beat[is_chord]
                    if self.note_chord_idx is not None and is_chord.any():
                        chord_logits = getattr(self, "chord_head")(hidden_free)
                        sampled_chord = self._sample_generate_token(chord_logits, decode_strategy, sampling_parameters, "chord")
                        sampled_chord_ids = sampled_chord[is_chord]
                        ids_free[is_chord,  self.note_chord_idx] = sampled_chord_ids

                        if is_continuation:
                            chord_batch_indices = free_batch_indices[is_chord]
                            chord_beat_logits = getattr(self, "beat_head")(hidden_free[is_chord])
                            last_positions = (generated_last_chord_position[chord_batch_indices])
                            has_chord = generated_has_chord[chord_batch_indices]
                            # 第一条只能是 Beat_0；
                            # 后续 CHORD 的位置必须严格递增。
                            allowed_chord_beats = (
                                valid_beat_ids.unsqueeze(0)
                                & (beat_positions.unsqueeze(0) > last_positions.unsqueeze(1))
                            )
                            first_chord_rows = ~has_chord
                            if first_chord_rows.any():
                                allowed_chord_beats[first_chord_rows,:] = False
                                allowed_chord_beats[first_chord_rows,beat_0_id] = True
                            min_value = torch.finfo(chord_beat_logits.dtype).min
                            chord_beat_logits = (chord_beat_logits.masked_fill(~allowed_chord_beats,min_value))
                            sampled_chord_beat = (
                                self._sample_generate_token(chord_beat_logits, decode_strategy, sampling_parameters, "beat")
                            )
                            ids_free[is_chord, self.note_beat_idx] = sampled_chord_beat
                            sampled_positions = beat_positions[sampled_chord_beat]

                            # 更新本小节的动态和弦 lookup。
                            for row_idx in range(chord_batch_indices.numel()):
                                batch_idx = int( chord_batch_indices[row_idx].item())
                                change_position = int(sampled_positions[row_idx].item())
                                chord_id = sampled_chord_ids[row_idx]
                                active_from_here = (
                                    valid_beat_ids
                                    & (beat_positions >= change_position)
                                )
                                generated_chord_lookup[batch_idx, active_from_here] = chord_id
                                generated_last_chord_position[batch_idx] = change_position
                                generated_has_chord[batch_idx] = True
                        else:
                            # conditioned task 正常情况下不会自由生成 CHORD；
                            # 保留原逻辑以避免改变其他模式。
                            chord_beat_logits = getattr(self,"beat_head")(hidden_free)
                            chord_beat_logits = (self._mask_invalid_beat_logits(chord_beat_logits))
                            sampled_chord_beat = (
                                self._sample_generate_token(chord_beat_logits,decode_strategy,sampling_parameters,"beat")
                            )
                            ids_free[is_chord, self.note_beat_idx] = sampled_chord_beat[is_chord]

                    if self.use_harmonic_plan and is_plan.any():
                        plan_hidden = hidden_free[is_plan]
                        plan_batch_indices = free_batch_indices[is_plan]

                        plan_beat_logits = self.beat_head(plan_hidden)
                        allowed_plan_beats = (
                            plan_valid_beat_ids.unsqueeze(0)
                            & (beat_positions.unsqueeze(0) > last_plan_position[plan_batch_indices].unsqueeze(1))
                        )
                        min_value = torch.finfo(plan_beat_logits.dtype).min
                        plan_beat_logits = plan_beat_logits.masked_fill(~allowed_plan_beats, min_value)
                        sampled_plan_beat = self._sample_generate_token(
                            plan_beat_logits, decode_strategy, sampling_parameters, "beat"
                        )
                        ids_free[is_plan, self.note_beat_idx] = sampled_plan_beat

                        bass_logits = self.plan_bass_head(plan_hidden).clone()
                        bass_logits[:, 0] = torch.finfo(bass_logits.dtype).min
                        sampled_bass = self._sample_generate_token(
                            bass_logits, decode_strategy, sampling_parameters, "plan_bass"
                        )
                        ids_free[is_plan, self.note_plan_bass_idx] = sampled_bass

                        # Autoregressive factorization inside PLAN:
                        # P(Bass | history) then P(Set | sampled Bass, history).
                        plan_set_hidden = self._condition_plan_set_on_bass(
                            plan_hidden,
                            sampled_bass,
                        )
                        # V18 still samples one complete Set mask.  Mask 0 is
                        # structurally impossible for an existing Quarter PLAN,
                        # so remove it before greedy / nucleus sampling.
                        set_logits = self.plan_set_head(plan_set_hidden).clone()
                        set_logits[:, 0] = torch.finfo(set_logits.dtype).min
                        set_masks = self._sample_generate_token(
                            set_logits,
                            decode_strategy,
                            sampling_parameters,
                            "plan_set",
                        ).long()
                        ids_free[is_plan, self.note_plan_set_idx] = set_masks

                    if self.note_inst_idx is not None and is_inst.any():
                        inst_logits = getattr(self, "instrument_head")(hidden_free)
                        sampled_inst = self._sample_generate_token(inst_logits, decode_strategy, sampling_parameters, "instrument")
                        ids_free[is_inst, self.note_inst_idx] = sampled_inst[is_inst]

                    if self.note_beat_idx is not None and is_beat.any():
                        beat_logits = getattr(self, "beat_head")(hidden_free)
                        beat_logits = self._mask_invalid_beat_logits(beat_logits)
                        sampled_beat = self._sample_generate_token(beat_logits, decode_strategy, sampling_parameters, "beat")
                        ids_free[is_beat, self.note_beat_idx] = sampled_beat[is_beat]
                    if self.note_tempo_idx is not None and is_beat.any():
                        tempo_logits = getattr(self, "tempo_head")(hidden_free)
                        sampled_tempo = self._sample_generate_token(tempo_logits, decode_strategy, sampling_parameters, "tempo")
                        ids_free[is_beat, self.note_tempo_idx] = sampled_tempo[is_beat]

                    if self.note_chord_idx is not None and is_beat.any():
                        beat_batch_indices = free_batch_indices[is_beat]
                        generated_beat_ids = ids_free[is_beat, self.note_beat_idx]
                        if condition_chord_lookup is not None:
                            # Chord / accompaniment task：
                            # 使用外部条件和弦时间线。
                            ids_free[is_beat, self.note_chord_idx] = condition_chord_lookup[
                                beat_batch_indices,
                                t,
                                generated_beat_ids,
                            ]
                        else:
                            # Continuation task：使用本小节刚刚生成的 CHORD 时间线。
                            if (~generated_has_chord[beat_batch_indices]).any():
                                raise RuntimeError("Continuation generated BEAT before the required Beat_0 CHORD.")
                            ids_free[is_beat,self.note_chord_idx] = generated_chord_lookup[beat_batch_indices, generated_beat_ids]
                        

                    if is_note.any():
                        if self.note_beat_idx is not None:
                            # NOTE inherits musical time from the latest BEAT
                            # in the current instrument block.
                            beat_ctx_free = beat_ctx[free_mask]
                            ids_free[is_note, self.note_beat_idx] = beat_ctx_free[is_note]
                        if self.note_inst_idx is not None:
                            # NOTE token inherits instrument from the latest INST token in the same bar.
                            inst_ctx_free = inst_ctx[free_mask]
                            ids_free[is_note, self.note_inst_idx] = inst_ctx_free[is_note]
                        
                        if self.note_pitch_idx is not None:
                            note_hidden = hidden_free[is_note]
                            pitch_logits = self.pitch_head(note_hidden)

                            # V18-D5.1: keep D5's finite chord-tone prior only
                            # for PIANO. BRIDGE/other instruments keep the exact
                            # original V18 pitch distribution. No track is removed.
                            note_batch_indices = free_batch_indices[is_note]
                            note_beat_ids = ids_free[is_note, self.note_beat_idx]
                            note_instrument_ids = ids_free[
                                is_note, self.note_inst_idx
                            ]
                            if condition_chord_lookup is not None:
                                active_chord_ids = condition_chord_lookup[
                                    note_batch_indices,
                                    t,
                                    note_beat_ids,
                                ]
                            else:
                                active_chord_ids = generated_chord_lookup[
                                    note_batch_indices,
                                    note_beat_ids,
                                ]
                            # D5.1 is a decode-only prior for conditioned tasks.
                            # Continuation must keep the original V18 pitch distribution.
                            if self.task_mode in {"chord", "accompaniment"}:
                                pitch_logits = self._soft_bias_pitch_logits_with_active_chord_d51(
                                    pitch_logits,
                                    active_chord_ids,
                                    note_instrument_ids,
                                )
                            sampled_pitch = self._sample_generate_token(
                                pitch_logits,
                                decode_strategy,
                                sampling_parameters,
                                "pitch",
                            )
                            ids_free[is_note, self.note_pitch_idx] = sampled_pitch
                        if self.note_dur_idx is not None:
                            dur_logits = getattr(self, "duration_head")(hidden_free)
                            sampled_dur = self._sample_generate_token(dur_logits, decode_strategy, sampling_parameters, "duration")
                            ids_free[is_note, self.note_dur_idx] = sampled_dur[is_note]
                        if self.note_vel_idx is not None:
                            vel_logits = getattr(self, "velocity_head")(hidden_free)
                            sampled_vel = self._sample_generate_token(vel_logits, decode_strategy, sampling_parameters, "velocity")
                            ids_free[is_note, self.note_vel_idx] = sampled_vel[is_note]

                    ids[free_mask] = ids_free

                if already_finished.any():
                    ids[already_finished] = sto_vec

                # Apply forcing during decoding so the forced token participates in subsequent autoregression.
                if (
                    force_ids is not None
                    and force_mask is not None
                    and t < force_ids.shape[1]
                    and n_idx < force_ids.shape[2]
                ):
                    force_here = (force_mask[:, t, n_idx] > 0) & active_bar & (~already_finished)
                    if force_here.any():
                        forced_vec = force_ids[:, t, n_idx, :]
                        f_copy = min(forced_vec.shape[1], ids.shape[1])
                        ids[force_here, :f_copy] = forced_vec[force_here, :f_copy]

                free_indices = free_mask.nonzero(as_tuple=True)[0]
                if free_indices.numel() > 0:
                    final_type = ids[free_indices, self.note_type_idx]
                    is_plan_final = (final_type == self.plan_type_id) if self.use_harmonic_plan else torch.zeros_like(final_type, dtype=torch.bool)
                    is_inst_final = final_type == self.inst_type_id
                    is_beat_final = final_type == self.beat_type_id
                    is_note_final = final_type == self.note_type_id
                    is_stop_final = (final_type == self.sto_type_id) | (final_type == self.eos_type_id)
                    free_not_forced = ~force_here[free_indices]
                    if is_plan_final.any():
                        plan_indices = free_indices[is_plan_final & free_not_forced]
                        if plan_indices.numel() > 0:
                            plan_started[plan_indices] = True
                            last_plan_position[plan_indices] = beat_positions[ids[plan_indices, self.note_beat_idx]]
                            prefix_open[plan_indices] = False
                    if is_inst_final.any() and self.note_inst_idx is not None:
                        inst_indices = free_indices[is_inst_final]
                        inst_ctx[inst_indices] = ids[inst_indices, self.note_inst_idx]
                        beat_ctx[inst_indices] = 0
                        prefix_open[inst_indices] = False
                        target_inst_indices = free_indices[is_inst_final & free_not_forced]
                        if target_inst_indices.numel() > 0:
                            target_track_started[target_inst_indices] = True
                    if is_beat_final.any() and self.note_beat_idx is not None:
                        beat_indices = free_indices[is_beat_final]
                        beat_ctx[beat_indices] = ids[beat_indices, self.note_beat_idx]
                        prefix_open[beat_indices] = False
                    if is_note_final.any():
                        note_indices = free_indices[is_note_final]
                        prefix_open[note_indices] = False
                    if is_stop_final.any():
                        stop_indices = free_indices[is_stop_final]
                        prefix_open[stop_indices] = False

                typ = ids[:, self.note_type_idx]
                stop = (typ == self.sto_type_id) | (typ == self.eos_type_id)
                for feat_idx in range(len(self.note_features)):
                    if feat_idx == self.note_type_idx:
                        continue
                    ids[:, feat_idx] = torch.where(stop, torch.zeros_like(ids[:, feat_idx]), ids[:, feat_idx])
                cur_notes[:, n_idx, :] = ids
                finished = finished | stop
                if finished.all() and n_idx >= 1:
                    break

                next_in = self._compute_note_embeddings(ids.unsqueeze(1).unsqueeze(1))
                next_in = self.note_emb_tokens_proj(next_in[:, 0, :, :])
                note_running_mask = torch.cat(
                    [note_running_mask, torch.ones((b, 1), dtype=torch.long, device=loop_device)],
                    dim=1,
                )
                next_position_ids = note_running_mask.long().sum(dim=-1, keepdim=True) - 1
                note_out = self.note_model(
                    inputs_embeds=next_in,
                    attention_mask=note_running_mask,
                    position_ids=next_position_ids,
                    past_key_values=note_past,
                    use_cache=True,
                    return_dict=True,
                )
                note_past = note_out.past_key_values
                cur_hidden = note_out.last_hidden_state[:, -1, :]

            # Ensure every bar has an explicit stop token.
            cur_types = cur_notes[:, :, self.note_type_idx]
            has_stop = ((cur_types == self.sto_type_id) | (cur_types == self.eos_type_id)).any(dim=1)
            if (~has_stop).any():
                idx = (~has_stop).nonzero(as_tuple=True)[0]
                cur_notes[idx, -1, :] = sto_vec.view(1, -1).expand(idx.numel(), -1)

            is_prompt = (t < prompt_lens)
            if prompt_note_ids is not None and t < prompt_note_ids.shape[1] and is_prompt.any():
                p = prompt_note_ids[:, t, :, :]
                m = is_prompt.view(b, 1, 1).expand_as(cur_notes)
                cur_notes = torch.where(m, p, cur_notes)

            if (~active_bar).any():
                inactive_note_mask = (~active_bar).view(b, 1, 1).expand_as(cur_notes)
                cur_notes = torch.where(
                    inactive_note_mask,
                    sto_vec.view(1, 1, -1).expand(b, max_note_nums, -1),
                    cur_notes,
                )

            generated_notes[:, t, :, :] = cur_notes

            # The finalized current bar becomes prev_notes.  On the next loop
            # it is compressed immediately into that step's Global history.
            prev_notes = cur_notes
            typ = prev_notes[:, :, self.note_type_idx]
            stop = (typ == self.sto_type_id) | (typ == self.eos_type_id)
            stop_idx = torch.where(
                stop.any(dim=1),
                stop.int().argmax(dim=1),
                torch.full((b,), max_note_nums - 1, dtype=torch.long, device=loop_device),
            )
            pos = torch.arange(max_note_nums, device=loop_device).unsqueeze(0).expand(b, -1)
            prev_mask = (pos <= stop_idx.unsqueeze(-1)).long()

        return {"note_feats": generated_notes}

    def _build_idx2event_cache(self):
        if hasattr(self, "_idx2event_cache") and self._idx2event_cache is not None:
            return self._idx2event_cache

        idx2event_cache = {}
        vocab_data = self.vocab_data if isinstance(self.vocab_data, dict) else {}
        for feat_name, mapping in vocab_data.items():
            feat_map = {}
            if isinstance(mapping, dict):
                for k, v in mapping.items():
                    k_is_int = isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit())
                    v_is_int = isinstance(v, int) or (isinstance(v, str) and str(v).lstrip("-").isdigit())
                    if k_is_int:
                        feat_map[int(k)] = str(v)
                    elif v_is_int:
                        feat_map[int(v)] = str(k)
            idx2event_cache[str(feat_name)] = feat_map

        self._idx2event_cache = idx2event_cache
        return self._idx2event_cache

    def _event_name(self, feat_name: str, token_id: int) -> str:
        cache = self._build_idx2event_cache()
        feat_map = cache.get(str(feat_name), {})
        token_id_int = int(token_id)
        if feat_name == "type" and self.use_harmonic_plan and token_id_int == self.plan_type_id:
            return "PLAN"
        if token_id_int in feat_map:
            return feat_map[token_id_int]
        return str(token_id_int)

    def _format_note_token_event(self, note_token: torch.Tensor) -> str:
        type_idx = self.note_type_idx
        type_id = int(note_token[type_idx].item())
        type_name = self._event_name("type", type_id)

        if type_name in {"SOS", "EOS", "STO"}:
            return type_name

        if type_name == "CHORD":
            if self.note_chord_idx is None:
                return "CHORD"
            chord_name = self._event_name("chord",
                int(note_token[self.note_chord_idx].item()),
            )
            beat_name = self._event_name("beat",
                int(note_token[self.note_beat_idx].item()),
            )
            return (
                f"CHORD("
                f"chord={chord_name}, "
                f"beat={beat_name}"
                f")"
            )

        if type_name == "PLAN":
            beat_name = self._event_name("beat", int(note_token[self.note_beat_idx].item()))
            bass_raw = int(note_token[self.note_plan_bass_idx].item()) if self.note_plan_bass_idx is not None else 0
            bass_pc = bass_raw - 1 if bass_raw > 0 else None
            set_mask = int(note_token[self.note_plan_set_idx].item()) if self.note_plan_set_idx is not None else 0
            intervals = [i for i in range(12) if (set_mask >> i) & 1]
            return f"PLAN(beat={beat_name}, bass_pc={bass_pc}, intervals={{{','.join(map(str, intervals))}}})"

        if type_name == "INST":
            if self.note_inst_idx is None:
                return "INST"
            inst_name = self._event_name("instrument", int(note_token[self.note_inst_idx].item()))
            return f"INST(instrument={inst_name})"

        if type_name == "BEAT":
            items = []
            if self.note_beat_idx is not None:
                beat_name = self._event_name("beat", int(note_token[self.note_beat_idx].item()))
                items.append(f"beat={beat_name}")
            if self.note_tempo_idx is not None:
                tempo_name = self._event_name("tempo", int(note_token[self.note_tempo_idx].item()))
                items.append(f"tempo={tempo_name}")
            if self.note_chord_idx is not None:
                chord_name = self._event_name("chord", int(note_token[self.note_chord_idx].item()))
                items.append(f"chord={chord_name}")
            content = ", ".join(items) if len(items) > 0 else ""
            return f"BEAT({content})"

        if type_name in {"NOTE", "CON"}:
            items = []
            if self.note_inst_idx is not None:
                inst_name = self._event_name("instrument", int(note_token[self.note_inst_idx].item()))
                items.append(f"instrument={inst_name}")
            if self.note_pitch_idx is not None:
                pitch_name = self._event_name("pitch", int(note_token[self.note_pitch_idx].item()))
                items.append(f"pitch={pitch_name}")
            if self.note_dur_idx is not None:
                dur_name = self._event_name("duration", int(note_token[self.note_dur_idx].item()))
                items.append(f"duration={dur_name}")
            if self.note_vel_idx is not None:
                vel_name = self._event_name("velocity", int(note_token[self.note_vel_idx].item()))
                items.append(f"velocity={vel_name}")
            content = ", ".join(items) if len(items) > 0 else ""
            return f"{type_name}({content})"

        items = []
        feat_count = min(note_token.shape[0], len(self.note_features))
        for feat_idx in range(feat_count):
            feat_name = self.note_features[feat_idx]
            feat_val_name = self._event_name(feat_name, int(note_token[feat_idx].item()))
            items.append(f"{feat_name}={feat_val_name}")
        return f"{type_name}({', '.join(items)})"

    @torch.no_grad()
    def dump_generated_events(
        self,
        pos_input_ids: torch.LongTensor,
        generated_note_ids: torch.LongTensor,
        output_file: str,
        pos_attention_mask: Optional[torch.LongTensor] = None,
        max_tokens: Optional[int] = None,
        sample_id_offset: int = 0,
        append: bool = True,
    ) -> None:
        if pos_input_ids is None or generated_note_ids is None:
            raise ValueError("pos_input_ids and generated_note_ids are required.")

        pos_cpu = pos_input_ids.detach().cpu()
        note_cpu = generated_note_ids.detach().cpu()
        mask_cpu = pos_attention_mask.detach().cpu() if pos_attention_mask is not None else None

        if pos_cpu.dim() != 3:
            raise ValueError(f"pos_input_ids must be [B,T,P], got {tuple(pos_cpu.shape)}")
        if note_cpu.dim() != 4:
            raise ValueError(f"generated_note_ids must be [B,T,N,F], got {tuple(note_cpu.shape)}")

        bsz = min(pos_cpu.shape[0], note_cpu.shape[0])
        t_max = min(pos_cpu.shape[1], note_cpu.shape[1])
        n_cap = int(max_tokens) if max_tokens is not None else int(note_cpu.shape[2])
        n_cap = max(0, min(n_cap, int(note_cpu.shape[2])))

        if len(self.pos_features_wo_inst) == pos_cpu.shape[-1]:
            pos_feat_names = list(self.pos_features_wo_inst)
        else:
            pos_feat_names = list(self.pos_features[: pos_cpu.shape[-1]])

        mode = "a" if append else "w"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, mode, encoding="utf-8") as f:
            for b in range(bsz):
                sample_id = sample_id_offset + b
                f.write(f"\n===== SAMPLE {sample_id} =====\n")

                if mask_cpu is not None:
                    valid_t = int(mask_cpu[b, :t_max].long().sum().item())
                else:
                    valid_t = int(t_max)
                valid_t = max(0, min(valid_t, t_max))

                for t in range(valid_t):
                    pos_parts = []
                    for p_idx, feat_name in enumerate(pos_feat_names):
                        feat_id = int(pos_cpu[b, t, p_idx].item())
                        feat_name_val = self._event_name(feat_name, feat_id)
                        pos_parts.append(f"{feat_name}={feat_name_val}")
                    pos_text = "POS(" + ", ".join(pos_parts) + ")"

                    token_texts = []
                    for n_idx in range(n_cap):
                        token_texts.append(self._format_note_token_event(note_cpu[b, t, n_idx, :]))
                    tokens_text = ", ".join(token_texts)
                    f.write(f"T{t:04d} {pos_text} [{tokens_text}]\n")

    def _apply_generation_type_constraints(
        self,
        type_logits: torch.Tensor,
        allow_chord: torch.Tensor,
        allow_plan: torch.Tensor,
        allow_inst: torch.Tensor,
        has_inst: torch.Tensor,
        has_beat: torch.Tensor,
    ) -> torch.Tensor:
        constrained = type_logits.clone()
        min_value = torch.finfo(constrained.dtype).min

        if 0 <= self.chord_type_id < constrained.size(-1):
            constrained[~allow_chord, self.chord_type_id] = min_value
        if 0 <= self.plan_type_id < constrained.size(-1):
            if self.use_harmonic_plan:
                constrained[~allow_plan, self.plan_type_id] = min_value
            else:
                # Defensive guard for stale/custom type vocabularies: a no-PLAN
                # model must never sample PLAN even if a PLAN id is present.
                constrained[:, self.plan_type_id] = min_value

        if self.note_inst_idx is None and 0 <= self.inst_type_id < constrained.size(-1):
            constrained[:, self.inst_type_id] = min_value
        elif 0 <= self.inst_type_id < constrained.size(-1):
            constrained[~allow_inst, self.inst_type_id] = min_value
        if self.note_beat_idx is None and 0 <= self.beat_type_id < constrained.size(-1):
            constrained[:, self.beat_type_id] = min_value

        if 0 <= self.beat_type_id < constrained.size(-1):
            constrained[~has_inst, self.beat_type_id] = min_value
        if 0 <= self.note_type_id < constrained.size(-1):
            constrained[~has_beat, self.note_type_id] = min_value
        return constrained

    def _sample_generate_token(self, logits, decode_strategy, sampling_parameters, feature_name):
        if decode_strategy == "greedy":
            return torch.argmax(logits, dim=-1)
        cfg = {"t": 1.0, "p": 0.9}
        if sampling_parameters is not None:
            cfg = sampling_parameters.get(feature_name, sampling_parameters.get("default", cfg))
        return sample_from_logits(logits, cfg.get("t", 1.0), cfg.get("p", 0.9))
