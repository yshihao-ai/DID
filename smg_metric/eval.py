from __future__ import annotations

"""
# Pop909
python ./smg_metric/eval.py ./baseline/Pop909_baseline/nmt_cp_cond_7features dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/32_nmt_cp_cond_7features.json --max-bars 32

python ./smg_metric/eval.py ./baseline/Pop909_baseline/nmt_nb_cond_7features ../test1/dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/32_nmt_nb_cond_7features.json --max-bars 32

python ./smg_metric/eval.py ./baseline/Pop909_baseline/nmt_remi_cond_7features ../test1/dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/32_nmt_remi_cond_7features.json --max-bars 32

python ./smg_metric/eval.py ./baseline/Pop909_baseline/musetok_pop909_32bars_cond ../test1/dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/32bars/32_musetok_pop909_32bars_cond.json --max-bars 32

python ./smg_metric/eval.py ./baseline/Pop909_baseline/nmt_cp_cond_7features ../test1/dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/nmt_cp_cond_7features.json
python ./smg_metric/eval.py ./baseline/Pop909_baseline/nmt_nb_cond_7features ../test1/dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/nmt_nb_cond_7features.json
python ./smg_metric/eval.py ./baseline/Pop909_baseline/nmt_remi_cond_7features ../test1/dataset/MIDI_test_dataset/Pop909/ ./baseline_smg_metrics/Pop909_baseline/nmt_remi_cond_7features.json


# Pop1k7
python ./smg_metric/eval.py ./baseline/Pop1k7_baseline/nmt_pre_pop1k7_remi_cond ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./baseline_smg_metrics/Pop1k7_baseline/nmt_remi_cond_7features.json
python ./smg_metric/eval.py ./baseline/Pop1k7_baseline/nmt_pre_pop1k7_cp_cond ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./baseline_smg_metrics/Pop1k7_baseline/nmt_cp_cond_7features.json
python ./smg_metric/eval.py ./baseline/Pop1k7_baseline/nmt_pre_pop1k7_nmt_cond ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./baseline_smg_metrics/Pop1k7_baseline/nmt_nb_cond_7features.json

python ./smg_metric/eval.py ./baseline/Pop1k7_baseline/nmt_pre_pop1k7_remi_cond ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./baseline_smg_metrics/Pop1k7_baseline/32bars/32_nmt_remi_cond_7features.json  --max-bars 32
python ./smg_metric/eval.py ./baseline/Pop1k7_baseline/nmt_pre_pop1k7_cp_cond ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./baseline_smg_metrics/Pop1k7_baseline/32bars/32_nmt_cp_cond_7features.json  --max-bars 32
python ./smg_metric/eval.py ./baseline/Pop1k7_baseline/nmt_pre_pop1k7_nmt_cond ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./baseline_smg_metrics/Pop1k7_baseline/32bars/32_nmt_nb_cond_7features.json  --max-bars 32

# PISAT
python smg_metric/eval.py output/did_PIAST_tiny/PIAST_prompt4_infer_checkpoint-66740/song_predict dataset/MIDI_test_dataset/PIAST/ output/did_PIAST_tiny/PIAST_prompt4_infer_checkpoint-66740-32bars.json  --max-bars 32

# EMOPIA+
python smg_metric/eval.py output/did_EMOPIA+_tiny/EMOPIA+_prompt4_infer_checkpoint-13568/song_predict dataset/MIDI_test_dataset/EMOPIA+ output/did_EMOPIA+_tiny/EMOPIA+_prompt4_infer_checkpoint-13568-32bars.json  --max-bars 32



# Lakh
python ./smg_metric/eval.py ./baseline/LakhClean_baseline/nmt_remi_cond_8features ../test1/dataset/MIDI_test_dataset/LakhClean/ ./baseline_smg_metrics/LakhClean_baseline/nmt_remi_cond_8features.json
python ./smg_metric/eval.py ./baseline/LakhClean_baseline/nmt_cp_cond_8features ../test1/dataset/MIDI_test_dataset/LakhClean/ ./baseline_smg_metrics/LakhClean_baseline/nmt_cp_cond_8features.json
python ./smg_metric/eval.py ./baseline/LakhClean_baseline/nmt_nb_cond_8features ../test1/dataset/MIDI_test_dataset/LakhClean/ ./baseline_smg_metrics/LakhClean_baseline/nmt_nb_cond_8features.json

python ./smg_metric/eval.py ./baseline/LakhClean_baseline/nmt_remi_cond_8features ../test1/dataset/MIDI_test_dataset/LakhClean/ ./baseline_smg_metrics/LakhClean_baseline/32bars/32_nmt_remi_cond_8features.json  --max-bars 32
python ./smg_metric/eval.py ./baseline/LakhClean_baseline/nmt_cp_cond_8features ../test1/dataset/MIDI_test_dataset/LakhClean/ ./baseline_smg_metrics/LakhClean_baseline/32bars/32_nmt_cp_cond_8features.json  --max-bars 32
python ./smg_metric/eval.py ./baseline/LakhClean_baseline/nmt_nb_cond_8features ../test1/dataset/MIDI_test_dataset/LakhClean/ ./baseline_smg_metrics/LakhClean_baseline/32bars/32_nmt_nb_cond_8features.json  --max-bars 32

python ./smg_metric/eval.py baseline/FGG_baseline/generated_samples/test_chord_only_32bars_with_ground_truth_x0edit/prediction baseline/FGG_baseline/generated_samples/test_chord_only_32bars_with_ground_truth_x0edit/answer baseline/FGG_baseline/generated_samples/x0edit/chord_4.json --max-bars 4
python ./smg_metric/eval.py baseline/FGG_baseline/generated_samples/test_chord_melody_null_rhythm_32bars_with_ground_truth_x0edit/prediction_accompaniment_only baseline/FGG_baseline/generated_samples/test_chord_melody_null_rhythm_32bars_with_ground_truth_x0edit/answer_accompaniment_only baseline/FGG_baseline/generated_samples/x0edit/acc_32.json --max-bars 32


----------
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_inst64_dur64_vel48/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_inst64_dur64_vel48/96_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 96
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_htf/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_htf/48_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 48
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny/32_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ./output/did_Pop909_tiny/pop909_infer_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny/pop909_infer_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9984/song_predict_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9984/song_answer_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9984/song_predict_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9984/song_answer_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9984_4bars.json --max-bars 4


python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_A_balanced_bar96_type192_beat96_ts48_tempo48_chord128_inst48_pitch192_dur48_vel32/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_A_balanced_bar96_type192_beat96_ts48_tempo48_chord128_inst48_pitch192_dur48_vel32/32_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_B_rhythm_bar96_type192_beat128_ts48_tempo64_chord96_inst48_pitch176_dur64_vel32/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_B_rhythm_bar96_type192_beat128_ts48_tempo64_chord96_inst48_pitch176_dur64_vel32/32_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_C_harmony_bar96_type192_beat64_ts48_tempo48_chord160_inst48_pitch224_dur48_vel32/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_C_harmony_bar96_type192_beat64_ts48_tempo48_chord160_inst48_pitch224_dur48_vel32/32_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_D_long_bar128_type224_beat96_ts48_tempo48_chord128_inst48_pitch192_dur48_vel32/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_D_long_bar128_type224_beat96_ts48_tempo48_chord128_inst48_pitch192_dur48_vel32/32_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_E_regularized_bar96_type160_beat96_ts48_tempo48_chord128_inst32_pitch176_dur48_vel24/pop909_prompt4_infer_chord75_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_E_regularized_bar96_type160_beat96_ts48_tempo48_chord128_inst32_pitch176_dur48_vel24/32_pop909_prompt4_infer_chord75_checkpoint-9984.json --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop909_tiny_htf/pop909_prompt4_infer_chord75_type50_beat50_tempo50_checkpoint-9984/song_predict ../test1/dataset/MIDI_test_dataset/Pop909/ ./output/did_Pop909_tiny_htf/pop909_prompt4_infer_chord75_type50_beat50_tempo50_checkpoint-9984.json


python ./smg_metric/eval.py ./output/did_Pop1k7_tiny/Pop1k7_prompt4_infer_chord75_checkpoint-22272/song_predict ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./output/did_Pop1k7_tiny/Pop1k7_prompt4_infer_chord75_checkpoint-22272.json --max-bars 32

python ./smg_metric/eval.py ../test1/output/did_Pop1k7_tiny_TDdrop0.1_htf/Pop1k7_prompt4_infer_chord75_checkpoint-22272/song_predict ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./output/did_Pop1k7_tiny_TDdrop0.1_htf/Pop1k7_prompt4_infer_chord75_checkpoint-22272.json
python ./smg_metric/eval.py ../test1/output/did_Pop1k7_tiny_TDdrop0.1_htf/Pop1k7_prompt4_infer_chord75_checkpoint-22272/song_predict ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./output/did_Pop1k7_tiny_TDdrop0.1_htf/32_Pop1k7_prompt4_infer_chord75_checkpoint-22272.json  --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop1k7_tiny/Pop1k7_prompt4_infer_chord75_checkpoint-22272/song_predict ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./output/did_Pop1k7_tiny/32_Pop1k7_prompt4_infer_chord75_checkpoint-22272.json  --max-bars 32
python ./smg_metric/eval.py ../test1/output/did_Pop1k7_tiny_C_harmony_bar96_type192_beat64_ts48_tempo48_chord160_inst48_pitch224_dur48_vel32/Pop1k7_prompt4_infer_chord75_checkpoint-22272/song_predict ../test1/dataset/MIDI_test_dataset/Pop1k7/ ./output/did_Pop1k7_tiny_C_harmony_bar96_type192_beat64_ts48_tempo48_chord160_inst48_pitch224_dur48_vel32/32_Pop1k7_prompt4_infer_chord75_checkpoint-22272.json  --max-bars 32



python ./smg_metric/eval.py ../test1/output/did_LakhClean_tiny_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704/song_predict ../test1/dataset/MIDI_test_dataset/LakhClean/ ./output/did_LakhClean_tiny_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704.json  --max-bars 32
python ./smg_metric/eval.py ../test1/output/ms128_mn128_did_LakhClean_tiny_INSTONLEY_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704/song_predict ../test1/dataset/MIDI_test_dataset/LakhClean/ ./output/ms128_mn128_did_LakhClean_tiny_INSTONLEY_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704.json  --max-bars 32
python ./smg_metric/eval.py ../test1/output/ms64_mn128_did_LakhClean_tiny_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704/song_predict ../test1/dataset/MIDI_test_dataset/LakhClean/ ./output/ms64_mn128_did_LakhClean_tiny_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704.json  --max-bars 32
python ./smg_metric/eval.py ../test1/output/ms64_mn128_did_LakhClean_tiny_MultiHot_INSTONLEY_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704/song_predict ../test1/dataset/MIDI_test_dataset/LakhClean/ ./output/ms64_mn128_did_LakhClean_tiny_MultiHot_INSTONLEY_bar128_type224_beat96_ts64_tempo64_chord128_inst128_pitch192_dur96_vel64/LakhClean_prompt4_infer_chord75_checkpoint-104704.json  --max-bars 32

python ./smg_metric/eval.py ./baseline/FGG_baseline/generated_samples/test_chord_melody_null_rhythm_32bars_with_ground_truth/prediction_accompaniment_only ./baseline/FGG_baseline/generated_samples/test_chord_melody_null_rhythm_32bars_with_ground_truth/answer_accompaniment_only ./did_bar_level_v4/baseline/FGG_baseline/generated_samples/test_chord_melody_null_rhythm_32bars_with_ground_truth/answer_accompaniment_only_32bars.json --max-bars 32

python ./smg_metric/eval.py ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9472/song_predict_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9472/song_answer_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9472_32.json --max-bars 32
python ./smg_metric/eval.py ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9472-pitch0.5/song_predict_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9472-pitch0.5/song_answer_full ./output/did_Pop909_tiny_chord/pop909_infer_checkpoint-9472-pitch0.5_32.json --max-bars 32
python ./smg_metric/eval.py ./output/did_Pop909_tiny_chord_wobr/pop909_infer_checkpoint-9472-pitch0.5/song_predict_full ./output/did_Pop909_tiny_chord_wobr/pop909_infer_checkpoint-9472-pitch0.5/song_answer_full ./output/did_Pop909_tiny_chord_wobr/pop909_infer_checkpoint-9472-pitch0.5_32.json --max-bars 32
python ./smg_metric/eval.py ./output/did_Pop909_tiny_acc/pop909_infer_checkpoint-9472-pitch0.5/song_predict_accompaniment_only ./output/did_Pop909_tiny_acc/pop909_infer_checkpoint-9472-pitch0.5/song_answer_accompaniment_only ./output/did_Pop909_tiny_acc/pop909_infer_checkpoint-9472-pitch0.5_32.json --max-bars 32

"""


import argparse
import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SINGLE_QUALITY_KEYS = [
    "pce",
    "ebr",
    "sc",
    "pisr",
    "polyphony",
    "polyphony_rate",
    "pitch_range",
    "n_pitches_used",
    "n_pitch_classes_used",
    "emr",
    "pe",
    "dpc",
    "ook",
]
NOTE_LEVEL_KEYS = ["note_f1", "notei_f1", "mel_f1", "i_iou", "ver"]
PAIR_RHYTHMIC_KEYS = ["onset_xor", "note_overlap"]
BAR_LEVEL_KEYS = ["sim_chr", "sim_grv"]
CHORD_LEVEL_KEYS = ["ca", "cs"]
DIST_KEYS = ["pd", "dd", "sc_sim", "pce_sim", "gsc"]
ADV_KEYS = [
    "kl_duration",
    "kl_ioi",
    "kl_pitch",
    "oa_duration",
    "oa_ioi",
    "oa_pitch_range",
    "oa_density",
    "ci_precision",
    "ci_recall",
    "ci_f1",
    "cts",
    "cr_pred",
    "cr_ref",
    "recon_acc",
]
STRUCT_SINGLE_KEYS = ["che", "ngram_div"]
STRUCT_PAIR_KEYS = ["melody_match", "tonal_dist"]
RHYTHMIC_SINGLE_KEYS = [
    "mean_ioi",
    "rhythmic_intensity",
    "rhythmic_density",
    "voice_number",
    "gs",
]
_WINDOWS_UNSAFE_CHARS = '<>:"/\\|?*'


def _import_metrics_api() -> dict[str, Any]:
    try:
        from smg_metrics import (
            advanced_eval,
            clear_cs_model_cache,
            distribution_eval,
            pair_eval,
            pair_eval_structural,
            single_file,
            single_file_rhythmic,
            single_file_structural,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency: {exc.name}. Install the project dependencies before running eval.py."
        ) from exc

    return {
        "advanced_eval": advanced_eval,
        "clear_cs_model_cache": clear_cs_model_cache,
        "distribution_eval": distribution_eval,
        "pair_eval": pair_eval,
        "pair_eval_structural": pair_eval_structural,
        "single_file": single_file,
        "single_file_rhythmic": single_file_rhythmic,
        "single_file_structural": single_file_structural,
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _pick(metrics: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: metrics[key] for key in keys if key in metrics}


def _mean_metric_dict(metric_dicts: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in keys:
        values = []
        for item in metric_dicts:
            value = item.get(key)
            if isinstance(value, (int, float)):
                value = float(value)
                if math.isfinite(value):
                    values.append(value)
        summary[key] = sum(values) / len(values) if values else float("nan")
    return summary


def _default_output_path(pred_dir: Path) -> Path:
    return pred_dir.parent / f"{pred_dir.name}_results.json"


def _resolve_output_path(pred_dir: Path, output: str | None) -> Path:
    if output is None:
        return _default_output_path(pred_dir)

    output_path = Path(output)
    if output_path.suffix.lower() == ".json":
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / _default_output_path(pred_dir).name


def _match_key(filename: str) -> str:
    return "".join("_" if ch in _WINDOWS_UNSAFE_CHARS else ch for ch in filename).lower()


def _ticks_per_bar(midi: Any) -> int:
    """Reuse the project's first-time-signature bar-length convention."""
    ts = midi.time_signature_changes[0] if midi.time_signature_changes else None
    # numerator = int(ts.numerator) if ts else 4
    # denominator = int(ts.denominator) if ts else 4
    numerator = 4
    denominator = 4
    return max(1, int(round(numerator * midi.ticks_per_beat * 4 / denominator)))


def _trim_midi_to_bars(src_path: Path, dst_path: Path, max_bars: int) -> None:
    """Trim a MIDI file to the first ``max_bars`` bars and write it to ``dst_path``."""
    try:
        import miditoolkit
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing dependency: {exc.name}. Install the project dependencies before running eval.py."
        ) from exc

    midi = miditoolkit.MidiFile(str(src_path))
    cutoff_tick = _ticks_per_bar(midi) * max_bars

    for inst in midi.instruments:
        kept_notes = []
        for note in inst.notes:
            if note.start >= cutoff_tick:
                continue
            note.end = min(note.end, cutoff_tick)
            if note.end > note.start:
                kept_notes.append(note)
        inst.notes = kept_notes
        inst.pitch_bends = [event for event in inst.pitch_bends if event.time < cutoff_tick]
        inst.control_changes = [event for event in inst.control_changes if event.time < cutoff_tick]

    midi.tempo_changes = [event for event in midi.tempo_changes if event.time < cutoff_tick]
    midi.time_signature_changes = [
        event for event in midi.time_signature_changes if event.time < cutoff_tick
    ]
    if hasattr(midi, "key_signature_changes"):
        midi.key_signature_changes = [
            event for event in midi.key_signature_changes if event.time < cutoff_tick
        ]
    if hasattr(midi, "markers"):
        midi.markers = [event for event in midi.markers if event.time < cutoff_tick]
    if hasattr(midi, "lyrics"):
        midi.lyrics = [event for event in midi.lyrics if event.time < cutoff_tick]

    midi.max_tick = min(getattr(midi, "max_tick", cutoff_tick), cutoff_tick)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(dst_path))


def _collect_pairs(pred_dir: Path, ref_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    preds = sorted(pred_dir.glob("*.mid"))
    refs = sorted(ref_dir.glob("*.mid"))

    if preds is None or len(preds) == 0:
        preds = sorted(pred_dir.glob("*.midi"))
        refs = sorted(ref_dir.glob("*.midi"))


    if not preds:
        raise FileNotFoundError(f"No MIDI files found in prediction directory: {pred_dir}")
    if not refs:
        raise FileNotFoundError(f"No MIDI files found in reference directory: {ref_dir}")

    ref_map = {ref.name: ref for ref in refs}
    normalized_ref_map: dict[str, Path] = {}
    ambiguous_keys: set[str] = set()
    for ref in refs:
        key = _match_key(ref.name)
        if key in normalized_ref_map and normalized_ref_map[key].name != ref.name:
            ambiguous_keys.add(key)
        else:
            normalized_ref_map[key] = ref

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    ambiguous_preds: list[str] = []
    used_ref_names: set[str] = set()
    for pred in preds:
        ref = ref_map.get(pred.name)
        if ref is None:
            key = _match_key(pred.name)
            if key in ambiguous_keys:
                ambiguous_preds.append(pred.name)
                continue
            ref = normalized_ref_map.get(key)
        if ref is None:
            missing.append(pred.name)
            continue
        pairs.append((pred, ref))
        used_ref_names.add(ref.name)

    if ambiguous_preds:
        raise FileNotFoundError(
            "Ambiguous reference MIDI files after normalized matching for predictions: "
            + ", ".join(ambiguous_preds)
        )
    if missing:
        raise FileNotFoundError(
            "Missing reference MIDI files for predictions: " + ", ".join(missing)
        )

    extra_refs = sorted(ref.name for ref in refs if ref.name not in used_ref_names)
    return pairs, extra_refs


def _evaluate_pair(
    pred_path: Path,
    ref_path: Path,
    root: int,
    mode: str,
    api: dict[str, Any],
    similarity_ref_path: Path | None = None,
) -> dict[str, Any]:
    single_quality = api["single_file"](pred_path, root=root, mode=mode).to_dict()
    structural_single = api["single_file_structural"](pred_path).to_dict()
    rhythmic_single = api["single_file_rhythmic"](pred_path).to_dict()
    pairwise = api["pair_eval"](
        pred_path,
        ref_path,
        similarity_ref_path=similarity_ref_path,
    ).to_dict()
    structural_pair = api["pair_eval_structural"](pred_path, ref_path).to_dict()
    distribution = api["distribution_eval"](pred_path, ref_path).to_dict()
    advanced = api["advanced_eval"](pred_path, ref_path).to_dict()

    return {
        "files": {
            "pred": str(pred_path),
            "ref": str(ref_path),
        },
        "A_single_file_quality": _pick(single_quality, SINGLE_QUALITY_KEYS),
        "B_note_level_pairwise": _pick(pairwise, NOTE_LEVEL_KEYS),
        "B2_pairwise_rhythmic": _pick(pairwise, PAIR_RHYTHMIC_KEYS),
        "C_bar_level_pairwise": _pick(pairwise, BAR_LEVEL_KEYS),
        "D_chord_level_pairwise": _pick(pairwise, CHORD_LEVEL_KEYS),
        "E_distribution_level": _pick(distribution, DIST_KEYS),
        "F_advanced": _pick(advanced, ADV_KEYS),
        "G_structural": {
            "single_file": _pick(structural_single, STRUCT_SINGLE_KEYS),
            "pairwise": _pick(structural_pair, STRUCT_PAIR_KEYS),
        },
        "H_rhythmic_temporal_single_file": _pick(rhythmic_single, RHYTHMIC_SINGLE_KEYS),
    }


def _build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_file = list(results.values())
    return {
        "pair_count": len(per_file),
        "A_single_file_quality": _mean_metric_dict(
            [item["A_single_file_quality"] for item in per_file],
            SINGLE_QUALITY_KEYS,
        ),
        "B_note_level_pairwise": _mean_metric_dict(
            [item["B_note_level_pairwise"] for item in per_file],
            NOTE_LEVEL_KEYS,
        ),
        "B2_pairwise_rhythmic": _mean_metric_dict(
            [item["B2_pairwise_rhythmic"] for item in per_file],
            PAIR_RHYTHMIC_KEYS,
        ),
        "C_bar_level_pairwise": _mean_metric_dict(
            [item["C_bar_level_pairwise"] for item in per_file],
            BAR_LEVEL_KEYS,
        ),
        "D_chord_level_pairwise": _mean_metric_dict(
            [item["D_chord_level_pairwise"] for item in per_file],
            CHORD_LEVEL_KEYS,
        ),
        "E_distribution_level": _mean_metric_dict(
            [item["E_distribution_level"] for item in per_file],
            DIST_KEYS,
        ),
        "F_advanced": _mean_metric_dict(
            [item["F_advanced"] for item in per_file],
            ADV_KEYS,
        ),
        "G_structural": {
            "single_file": _mean_metric_dict(
                [item["G_structural"]["single_file"] for item in per_file],
                STRUCT_SINGLE_KEYS,
            ),
            "pairwise": _mean_metric_dict(
                [item["G_structural"]["pairwise"] for item in per_file],
                STRUCT_PAIR_KEYS,
            ),
        },
        "H_rhythmic_temporal_single_file": _mean_metric_dict(
            [item["H_rhythmic_temporal_single_file"] for item in per_file],
            RHYTHMIC_SINGLE_KEYS,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all 52 smg_metrics metrics on pred/ref MIDI directories.",
    )
    parser.add_argument("pred_dir", help="Directory containing predicted/generated .mid files")
    parser.add_argument("ref_dir", help="Directory containing reference/ground-truth .mid files")
    parser.add_argument(
        "output",
        nargs="?",
        help=(
            "Optional output directory or JSON file path. "
            "If omitted, saves to pred_dir's sibling path as <pred_dir_name>_results.json."
        ),
    )
    parser.add_argument(
        "--root",
        type=int,
        default=0,
        help="Root pitch class for PISR (same as smg_metrics CLI, default: 0)",
    )
    parser.add_argument(
        "--mode",
        choices=["major", "minor"],
        default="major",
        help="Scale mode for PISR (same as smg_metrics CLI, default: major)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Optionally trim both pred/ref MIDIs to the first N bars before evaluation.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    ref_dir = Path(args.ref_dir).resolve()
    output_path = _resolve_output_path(pred_dir, args.output).resolve()

    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Prediction directory not found: {pred_dir}")
    if not ref_dir.is_dir():
        raise NotADirectoryError(f"Reference directory not found: {ref_dir}")
    if args.max_bars is not None and args.max_bars <= 0:
        raise ValueError(f"--max-bars must be a positive integer, got: {args.max_bars}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        api = _import_metrics_api()
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    temp_dirs: list[Any] = []
    try:
        t0 = time.time()
        pairs, extra_refs = _collect_pairs(pred_dir, ref_dir)

        print(f"Matched {len(pairs)} MIDI pairs.")
        if extra_refs:
            print(f"Ignoring {len(extra_refs)} unmatched reference files: {', '.join(extra_refs)}")

        skipped_pairs: list[dict[str, str]] = []
        eval_pairs: list[tuple[Path, Path, Path]] = [
            (pred_path, ref_path, ref_path) for pred_path, ref_path in pairs
        ]
        if args.max_bars is not None:
            print(f"Trimming matched MIDI pairs to the first {args.max_bars} bars...")
            pred_tmp = TemporaryDirectory(prefix="smg_eval_pred_")
            ref_tmp = TemporaryDirectory(prefix="smg_eval_ref_")
            temp_dirs.extend([pred_tmp, ref_tmp])
            pred_tmp_dir = Path(pred_tmp.name)
            ref_tmp_dir = Path(ref_tmp.name)
            eval_pairs = []
            for pred_path, ref_path in pairs:
                trimmed_pred = pred_tmp_dir / pred_path.name
                trimmed_ref = ref_tmp_dir / ref_path.name
                try:
                    _trim_midi_to_bars(pred_path, trimmed_pred, args.max_bars)
                    _trim_midi_to_bars(ref_path, trimmed_ref, args.max_bars)
                except Exception as exc:
                    print(f"Skipping pair during trimming due to error: {pred_path.name} | {ref_path.name} | {exc}")
                    skipped_pairs.append(
                        {
                            "pred": pred_path.name,
                            "ref": ref_path.name,
                            "error": str(exc),
                        }
                    )
                    continue
                # Keep the original full reference only for simChr / simgrv.
                eval_pairs.append((trimmed_pred, trimmed_ref, ref_path))

        results: dict[str, dict[str, Any]] = {}
        for idx, (pred_path, ref_path, similarity_ref_path) in enumerate(eval_pairs, start=1):
            print(f"[{idx}/{len(eval_pairs)}] pre: {pred_path.name} | ref: {ref_path.name}")
            try:
                results[pred_path.name] = _evaluate_pair(
                    pred_path,
                    ref_path,
                    args.root,
                    args.mode,
                    api,
                    similarity_ref_path=similarity_ref_path,
                )
            except Exception as exc:
                print(f"Skipping pair due to error: {pred_path.name} | {ref_path.name} | {exc}")
                skipped_pairs.append(
                    {
                        "pred": pred_path.name,
                        "ref": ref_path.name,
                        "error": str(exc),
                    }
                )
                continue

        summary = _build_summary(results)
        elapsed = time.time() - t0

        output = {
            "meta": {
                "pred_dir": str(pred_dir),
                "ref_dir": str(ref_dir),
                "output_path": str(output_path),
                "pair_count": len(results),
                "matched_pair_count": len(pairs),
                "skipped_pair_count": len(skipped_pairs),
                "skipped_pairs": skipped_pairs,
                "extra_reference_files_ignored": extra_refs,
                "root": args.root,
                "mode": args.mode,
                "max_bars": args.max_bars,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(elapsed, 2),
                "total_metrics_per_pair": 52,
            },
            "summary": summary,
        }

        output_path.write_text(
            json.dumps(_json_safe(output), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    finally:
        cleared = api["clear_cs_model_cache"]()
        for temp_dir in temp_dirs:
            temp_dir.cleanup()
    if cleared > 0:
        print(f"Cleared {cleared} cached CS model(s).")
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
