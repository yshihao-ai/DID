import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from os.path import join
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import HfArgumentParser, Qwen2Config

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from component.arguments import DidMusicArguments, EvaluateArguments
from component.collator import DidDataCollator
from component.dataset import DidDataset
from data_representation.vocab_utils import MusicTokenVocabDID
from inference.create_midi_zip import collect_and_zip_midis
from inference.decoding import MidiDecoder4DID
from musicmodel.did_model import DidForMusicLM
# from musicmodel.did_model_wo_condition_encoder import DidForMusicLM
# from musicmodel.did_model_wo_dual_history import DidForMusicLM


from dataclasses import fields

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

def resolve_model_directory(
    model_name_or_path: str,
) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    if model_name_or_path.endswith(".bin"):
        return os.path.dirname(os.path.abspath(model_name_or_path))

    raise ValueError("model_name_or_path must be a checkpoint directory or .bin file.")

def load_music_config_from_checkpoint(
    model_name_or_path: str,
) -> DidMusicArguments:
    model_dir = resolve_model_directory(model_name_or_path)

    candidate_paths = [os.path.join(model_dir,"music_config.json",),
        os.path.join(model_dir, "train_args_backup.json",),
        os.path.join(os.path.dirname(model_dir),"train_args_backup.json",),
    ]

    config_path = next((path for path in candidate_paths
        if os.path.exists(path)
    ), None,)

    if config_path is None:
        raise FileNotFoundError(f"Cannot find music_config.json for checkpoint. Checked: {candidate_paths}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    init_field_names = {item.name for item in fields(DidMusicArguments) if item.init}
    music_kwargs = {key: value for key, value in raw_config.items() if key in init_field_names}
    music_config = DidMusicArguments(**music_kwargs)

    logger.info(f"Loaded music config from checkpoint: {config_path}")
    logger.info(f"Checkpoint task_mode={music_config.task_mode}")

    return music_config

def load_model_and_token_decoder(args, music_config, device=torch.device("cuda")):
    logger.info("Initiating model and decoder ...")

    time_config = Qwen2Config(
        vocab_size=music_config.dummy_vocab_size,
        hidden_size=music_config.time_decoder_hidden_size,
        num_hidden_layers=music_config.time_decoder_layers,
        num_attention_heads=music_config.time_decoder_attention_heads,
        intermediate_size=music_config.time_decoder_intermediate_size,
        num_key_value_heads=music_config.time_decoder_attention_heads,
        max_position_embeddings=512,
        hidden_act="silu",
        rope_theta=1000000.0,
        attention_dropout=0.0,
        use_cache=True,
        tie_word_embeddings=False,
    )

    note_config = Qwen2Config(
        vocab_size=music_config.dummy_vocab_size,
        hidden_size=music_config.note_decoder_hidden_size,
        num_hidden_layers=music_config.note_decoder_layers,
        num_attention_heads=music_config.note_decoder_attention_heads,
        intermediate_size=music_config.note_decoder_intermediate_size,
        num_key_value_heads=music_config.note_decoder_attention_heads,
        max_position_embeddings=1024,
        hidden_act="silu",
        rope_theta=1000000.0,
        attention_dropout=0.0,
        use_cache=True,
        tie_word_embeddings=False,
    )

    model = DidForMusicLM(time_config=time_config, note_config=note_config, music_config=music_config)

    model_path = args.model_name_or_path
    if os.path.isdir(model_path):
        bin_file = os.path.join(model_path, "pytorch_model.bin")
    elif model_path.endswith(".bin"):
        bin_file = model_path
    else:
        raise ValueError("model_name_or_path must be a checkpoint directory or a .bin checkpoint file.")
    logger.info(f"Loading PyTorch checkpoint from: {bin_file}")

    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Cannot find PyTorch checkpoint (.bin): {bin_file}")
    state_dict = torch.load(bin_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(dtype=torch.bfloat16, device=device)

    vocab = MusicTokenVocabDID(
        in_vocab_file_path=Path(music_config.vocab_file_path),
        event_data=None,
        encoding_scheme="did",
        num_features=8,
    )
    in_beat_resolution_dict = {
        "BachChorale": 4,
        "Pop1k7": 4,
        "Pop909": 4,
        "SOD": 12,
        "LakhClean": 4,
        "SymphonyMIDI": 8,
        "EMOPIA+": 4,
        "PIAST": 4
    }
    decoder = MidiDecoder4DID(
        vocab=vocab,
        in_beat_resolution=in_beat_resolution_dict[args.dataset_name],
        dataset_name=args.dataset_name,
        pos_feature_keys=list(music_config.pos_features),
        note_feature_keys=list(music_config.note_features),
    )
    return model, decoder


def load_eval_dataset_and_collator(args, music_config):
    eval_dataset = DidDataset(
        music_config,
        args.dataset_path,
        args.max_time_steps,
        split="test",
        metadata_path=args.metadata_path,
        max_data_nums=2048,
        # segment_mode="head_only"
    )
    collator = DidDataCollator(music_config, args.max_time_steps)
    return eval_dataset, collator


def prepare_jagged_prompt(
    pos_input_ids,
    note_input_ids,
    music_config,
    pos_attention_mask=None,
    target_measures=4,
    max_note_nums=16,
):
    """
    Build per-sample prompt tensor with variable prompt lengths.
    """
    batch_size = pos_input_ids.shape[0]
    cutoff_steps = []
    for b in range(batch_size):
        if pos_attention_mask is not None:
            valid_len = int(pos_attention_mask[b].long().sum().item())
        else:
            valid_len = int(pos_input_ids.shape[1])
        # note_input_ids[0] is shifted previous bar (typically SOS), prompt bars start from index 1.
        step = max(0, min(int(target_measures), max(valid_len - 1, 0)))
        cutoff_steps.append(step)

    max_prompt_len = max(cutoff_steps)
    if max_prompt_len == 0:
        return None, None

    sto_vec = torch.tensor(music_config.sto_note_id, dtype=note_input_ids.dtype, device=note_input_ids.device)
    prompt_tensor = sto_vec.view(1, 1, 1, -1).repeat(batch_size, max_prompt_len, max_note_nums, 1)

    valid_note_width = min(note_input_ids.shape[2], max_note_nums)
    for b, length in enumerate(cutoff_steps):
        if length > 0:
            src = note_input_ids[b, 1 : 1 + length, :valid_note_width, :]
            prompt_tensor[b, :length, :valid_note_width, :] = src

    prompt_lens = torch.tensor(cutoff_steps, dtype=torch.long, device=note_input_ids.device)
    return prompt_tensor, prompt_lens

def parse_condition_instrument_programs(
    music_config,
):
    """
    ['Instrument_0'] -> [0]
    """
    programs = []
    for instrument_name in (music_config.condition_instrument_prefix):
        text = str(instrument_name)
        if not text.startswith("Instrument_"):
            raise ValueError(f"Condition instrument must have the form 'Instrument_<program>', got {text!r}.")
        program = int(text.rsplit("_", 1)[-1])
        programs.append(program)
    return sorted(set(programs))

def build_decode_output_dirs(output_root, task_mode):
    if task_mode == "continuation":
        return {
            "predict": os.path.join(output_root, "song_predict"),
            "answer": os.path.join(output_root, "song_answer"),      
        }

    if task_mode == "chord":
        return {
            "predict_full": os.path.join(output_root, "song_predict_full"),
            "answer_full": os.path.join(output_root, "song_answer_full"),
        }

    if task_mode == "accompaniment":
        return {
            "predict_accompaniment_only": os.path.join(output_root, "song_predict_accompaniment_only"),
            "predict_full": os.path.join(output_root, "song_predict_full"),
            "answer_accompaniment_only": os.path.join(output_root, "song_answer_accompaniment_only"),
            "answer_full": os.path.join(output_root, "song_answer_full"),
        }

    raise ValueError(f"Unknown task_mode={task_mode!r}.")

def build_expected_rel_midi_paths(eval_dataset, dataset_path):
    rel_midi_paths = []
    for idx in range(len(eval_dataset)):
        full_npz_path = (eval_dataset.get_npz_file_path(idx))
        rel_npz_path = os.path.relpath(full_npz_path, dataset_path)
        rel_midi_path = (os.path.splitext(rel_npz_path)[0] + ".mid")
        rel_midi_paths.append(rel_midi_path)
    return rel_midi_paths

def decode_outputs_are_complete(output_dirs, expected_rel_midi_paths):
    if not expected_rel_midi_paths:
        return False
    all_complete = True
    for output_name, output_root in output_dirs.items():
        missing_paths = []
        for rel_midi_path in expected_rel_midi_paths:
            full_midi_path = os.path.join(output_root, rel_midi_path)
            if not os.path.isfile(full_midi_path):
                missing_paths.append(rel_midi_path)
        existing_count = (len(expected_rel_midi_paths) - len(missing_paths))

        logger.info(
            f"Decode output [{output_name}]: "
            f"{existing_count}/"
            f"{len(expected_rel_midi_paths)} complete."
        )

        if missing_paths:
            all_complete = False
            logger.info(
                f"Decode output [{output_name}] "
                f"still misses {len(missing_paths)} files. "
                f"Examples: {missing_paths[:3]}"
            )
    return all_complete

def generate_batch(model, args, music_config, inputs):
    task_mode = music_config.task_mode
    prompt_note_ids = None
    prompt_lens = None
    condition_note_ids = None
    condition_note_mask = None
    if task_mode == "continuation":
        if args.prompt_measures > 0:
            prompt_note_ids, prompt_lens = prepare_jagged_prompt(
                inputs["pos_input_ids"],
                inputs["note_input_ids"],
                music_config,
                pos_attention_mask=inputs.get("pos_attention_mask"),
                target_measures=args.prompt_measures,
                max_note_nums=args.max_note_nums,
            )
        if "condition_note_ids" in inputs:
            raise ValueError("Continuation collator output unexpectedly contains condition_note_ids.")
    elif task_mode in {"chord", "accompaniment"}:
        if args.prompt_measures != 0:
            raise ValueError(f"prompt_measures must be 0 when task_mode={task_mode!r}.")
        if "condition_note_ids" not in inputs:
            raise KeyError(f"Collator did not return condition_note_ids for task_mode={task_mode!r}.")
        if "condition_note_mask" not in inputs:
            raise KeyError(f"Collator did not return condition_note_mask for task_mode={task_mode!r}.")
        condition_note_ids = inputs["condition_note_ids"]
        condition_note_mask = inputs["condition_note_mask"]
    else:
        raise RuntimeError(f"Unexpected task_mode={task_mode!r}.")

    with torch.inference_mode():
        out = model.generate(
            device=model.device,
            pos_input_ids=inputs["pos_input_ids"].to(model.device),
            pos_attention_mask=inputs["pos_attention_mask"].to(model.device),
            instrument_input_ids=inputs["instrument_input_ids"].to(model.device),
            instrument_attention_mask=inputs["instrument_attention_mask"].to(model.device),
            prompt_note_ids=(prompt_note_ids.to(model.device) if prompt_note_ids is not None else None),
            prompt_lens=(prompt_lens.to(model.device) if prompt_lens is not None else None),
            condition_note_ids=(condition_note_ids.to(model.device) if condition_note_ids is not None else None),
            condition_note_mask=(condition_note_mask.to(model.device) if condition_note_mask is not None else None),
            max_time_steps=args.max_time_steps,
            max_note_nums=args.max_note_nums,
            decode_strategy=args.decode_strategy,
            sampling_parameters=args.sampling_params,
        )
    return out["note_feats"]

def dump_generated_events_if_needed(model, args, inputs, batch_notes, sample_id_offset: int):
    if not args.dump_generated_events:
        return
    dump_path = os.path.join(args.output_dir, args.dump_generated_event_file)
    model.dump_generated_events(
        pos_input_ids=inputs["pos_input_ids"],
        generated_note_ids=batch_notes,
        output_file=dump_path,
        pos_attention_mask=inputs.get("pos_attention_mask"),
        max_tokens=args.dump_generated_event_max_tokens,
        sample_id_offset=sample_id_offset,
        append=(sample_id_offset != 0),
    )


def decode_worker(task):
    pos_input_id, pos_attention_mask, notes, output_path, include_instruments, exclude_instruments = task
    if isinstance(pos_input_id, list):
        pos_input_id = torch.tensor(pos_input_id)
    if isinstance(pos_attention_mask, list):
        pos_attention_mask = torch.tensor(pos_attention_mask)
    if isinstance(notes, list):
        notes = torch.tensor(notes)

    return token_decoder(
        pos_input_id,
        notes,
        output_path,
        pos_attention_mask=pos_attention_mask,
        include_instruments=include_instruments,
        exclude_instruments=exclude_instruments
    )


def resolve_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(project_root, path_str))


def copy_midis_flat(src_dir, dst_dir):
    src_root = Path(src_dir)
    dst_root = Path(dst_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    used_names = set()
    for midi_path in src_root.rglob("*"):
        if not midi_path.is_file():
            continue
        if midi_path.suffix.lower() not in {".mid", ".midi"}:
            continue

        rel_name = midi_path.relative_to(src_root).as_posix().replace("/", "__")
        if not rel_name:
            rel_name = midi_path.name

        stem, suffix = os.path.splitext(rel_name)
        candidate = rel_name
        dedup_idx = 1
        while candidate in used_names:
            candidate = f"{stem}__dup{dedup_idx}{suffix}"
            dedup_idx += 1
        used_names.add(candidate)

        shutil.copy2(midi_path, dst_root / candidate)
        copied += 1
    return copied

# ==================== 新增：隔离评测环境配置 ====================
EVAL_ENV_NAME = "did_eval"  # 你新建的 CPU 评测专用 conda 环境名

def get_eval_cmd_prefix() -> List[str]:
    """生成安全切换到 did_eval 环境执行 Python 的命令前缀"""
    conda_exe = shutil.which("conda") or "conda"
    # --no-capture-output 允许评测脚本的 print 和 tqdm 进度条实时打印到屏幕上
    return [conda_exe, "run", "--no-capture-output", "-n", EVAL_ENV_NAME, "python"]

def get_clean_env() -> Dict[str, str]:
    """核心防坑：剥离当前 PyTorch 2.9 环境的动态链接库污染"""
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    return env
# ===============================================================

def run_figaro_evaluation(generated_dir, gt_dir, output_dir, max_samples):
    figaro_workspace = os.path.join(output_dir, "figaro_workspace")
    figaro_gt_dir = os.path.join(figaro_workspace, "gt")
    figaro_out_csv = os.path.join(output_dir, "figaro_metrics.csv")

    if os.path.exists(figaro_workspace):
        shutil.rmtree(figaro_workspace)
    os.makedirs(figaro_gt_dir, exist_ok=True)

    gen_count = copy_midis_flat(generated_dir, figaro_workspace)
    gt_count = copy_midis_flat(gt_dir, figaro_gt_dir)

    if gen_count == 0 or gt_count == 0:
        logger.error(f"Skip Figaro evaluation due to empty data: gen={gen_count}, gt={gt_count}")
        return

    env = os.environ.copy()
    env["SAMPLE_DIR"] = figaro_workspace
    env["OUT_FILE"] = figaro_out_csv
    env["MAX_SAMPLES"] = str(max_samples)

    figaro_runner = (
        "import importlib, runpy, sys; "
        "mod = importlib.import_module('inference.music_evaluation.figaro.input_representation'); "
        "sys.modules['input_representation'] = mod; "
        "runpy.run_module('inference.music_evaluation.figaro.evaluate', run_name='__main__')"
    )
    figaro_cmd = [sys.executable, "-c", figaro_runner]
    logger.info(f"Running Figaro evaluator: {' '.join(figaro_cmd)}")
    try:
        subprocess.run(figaro_cmd, check=True, cwd=project_root, env=env)
        logger.info(f"Figaro evaluation done: {figaro_out_csv}")
    except Exception as e:
        logger.error(f"Figaro evaluation failed: {e}")
        raise


parser = argparse.ArgumentParser()
parser.add_argument("--infer_args_file", type=str, help="inference args file")
cmd = parser.parse_args()
infer_args_file = cmd.infer_args_file

eval_parser = HfArgumentParser(EvaluateArguments)
(args,) = eval_parser.parse_json_file(json_file=infer_args_file)
music_args = load_music_config_from_checkpoint(args.model_name_or_path)
if (music_args.task_mode != "continuation" and args.prompt_measures != 0):
    raise ValueError(
        "prompt_measures must be 0 for conditioned tasks. "
        f"Current task_mode={music_args.task_mode!r}, "
        f"prompt_measures={args.prompt_measures}."
    )
model, token_decoder = load_model_and_token_decoder(args, music_config=music_args,)

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(join(args.output_dir, "eval.log"))
    logger.info(f"eval_args:\n{args}")

    with open(infer_args_file, "r", encoding="utf-8") as f:
        eval_args = json.load(f)
    with open(join(args.output_dir, "eval_args.json"), "w", encoding="utf-8") as f:
        json.dump(eval_args, f, indent=4)

    output_dirs = build_decode_output_dirs(args.output_dir, music_args.task_mode)
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    condition_programs = (
        parse_condition_instrument_programs(music_args)
        if music_args.task_mode == "accompaniment" else []
    )
    
    eval_dataset, data_collator = load_eval_dataset_and_collator(args, music_args)
    batch_size = args.batch_size
    if music_args.task_mode == "continuation":
        midi_output_dir = output_dirs["predict"]
    else:
        midi_output_dir = output_dirs["predict_full"]
    expected_rel_midi_paths = build_expected_rel_midi_paths(eval_dataset, args.dataset_path)
    outputs_complete = decode_outputs_are_complete(output_dirs, expected_rel_midi_paths)
    
    if outputs_complete:
        logger.warning("All required decoded MIDI outputs already exist. Skip generation and decoding.")
    else:
        if args.dump_generated_events:
            dump_path = os.path.join(args.output_dir, args.dump_generated_event_file)
            dump_dir = os.path.dirname(dump_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            
            if os.path.exists(dump_path):
                os.remove(dump_path)
            logger.info(f"Generated event dump enabled: {dump_path}")
    
        for i in tqdm(range(0, len(eval_dataset), batch_size)):
                r = min(len(eval_dataset),i + batch_size)
                batch = [eval_dataset[p] for p in range(i, r)]
                inputs = data_collator(batch)
                batch_notes = generate_batch(model, args, music_args, inputs)
                dump_generated_events_if_needed(model, args, inputs, batch_notes, sample_id_offset=i)
                # batch_notes: [batch, generated_time, note_width, feature]
                batch_generated_time = int(batch_notes.shape[1])
                tasks = []
                for j in range(batch_notes.shape[0]):
                    valid_position_count = int(inputs["pos_attention_mask"][j].long().sum().item())
                    valid_target_len = max(valid_position_count - 1, 0)
                    decode_len = min(valid_target_len, batch_generated_time)

                    if decode_len <= 0:
                        logger.warning(f"Skip sample {i + j}: no valid target bars.")
                        continue

                    # -------------------------------------------------
                    # 位置 P1 ... P_decode_len
                    # -------------------------------------------------
                    sample_pos_ids = (inputs["pos_input_ids"][j, :decode_len, :,].cpu().tolist())
                    sample_pos_mask = (inputs["pos_attention_mask"][j, :decode_len,].cpu().tolist())

                    # -------------------------------------------------
                    # 模型预测：
                    # generated Bar1 ... Bar_decode_len
                    # -------------------------------------------------
                    sample_predict_notes = (batch_notes[j, :decode_len, :, :,].cpu().tolist())

                    # -------------------------------------------------
                    # 答案编码矩阵：
                    # note_input_ids[j, 0] 是 SOS，
                    # 所以答案必须从索引 1 开始。
                    # 得到：Bar1 ... Bar_decode_len
                    # -------------------------------------------------
                    sample_answer_notes = (inputs["note_input_ids"][j, 1 : 1 + decode_len, :, :, ].cpu().tolist())

                    if not (len(sample_pos_ids) == len(sample_pos_mask) == len(sample_predict_notes)  == len(sample_answer_notes) == decode_len):
                        raise RuntimeError(
                            "Decode tensor time alignment "
                            "mismatch: "
                            f"sample={i + j}, decode_len={decode_len}, "
                            f"pos={len(sample_pos_ids)}, mask={len(sample_pos_mask)}, "
                            f"predict={len(sample_predict_notes)}, "
                            f"answer={len(sample_answer_notes)}."
                        )

                    full_npz_path = (eval_dataset.get_npz_file_path(i + j))
                    rel_npz_path = os.path.relpath(full_npz_path, args.dataset_path)
                    rel_midi_path = (os.path.splitext(rel_npz_path)[0] + ".mid")

                    def add_decode_task(output_key, note_matrix, exclude_instruments=None):
                        output_path = os.path.join(output_dirs[output_key], rel_midi_path)
                        output_parent = os.path.dirname(output_path)
                        if output_parent:
                            os.makedirs(output_parent, exist_ok=True)
                        tasks.append(
                            (sample_pos_ids, sample_pos_mask, note_matrix, output_path, None, exclude_instruments)
                        )

                    # =============================================
                    # Continuation
                    # =============================================
                    if music_args.task_mode == "continuation":
                        add_decode_task("predict", sample_predict_notes)
                        add_decode_task("answer", sample_answer_notes)
                    # =============================================
                    # Chord injection
                    # 1. 模型预测完整 MIDI
                    # 2. 答案编码矩阵完整 MIDI
                    # =============================================
                    elif music_args.task_mode == "chord":
                        bridge_exclude = [24] if args.dataset_name == "Pop909" and not music_args.use_bridge else None
                        add_decode_task("predict_full", sample_predict_notes, exclude_instruments=bridge_exclude)
                        add_decode_task("answer_full", sample_answer_notes, exclude_instruments=bridge_exclude)

                    # =============================================
                    # Accompaniment
                    # 1. 模型预测，仅伴奏
                    # 2. 模型预测，条件 + 伴奏
                    # 3. 答案编码，仅伴奏
                    # 4. 答案编码，条件 + 伴奏
                    # =============================================
                    elif music_args.task_mode  == "accompaniment":
                        add_decode_task("predict_full",sample_predict_notes)
                        add_decode_task("answer_full", sample_answer_notes)

                        add_decode_task(
                            "predict_accompaniment_only",
                            sample_predict_notes,
                            exclude_instruments=condition_programs,
                        )

                        add_decode_task(
                            "answer_accompaniment_only", 
                            sample_answer_notes,
                            exclude_instruments= condition_programs
                        )


                    else:
                        raise RuntimeError(f"Unexpected task_mode={music_args.task_mode!r}.")

                if tasks:
                    with ProcessPoolExecutor() as executor:
                        _ = list(executor.map(decode_worker, tasks))

    # collect_and_zip_midis(midi_output_dir)
    # if args.enable_chord_injection:
    #     abs_gt_dir = resolve_path(args.gt_dataset_dir)
    #     if not os.path.exists(abs_gt_dir):
    #         logger.error(f"GT directory not found for Figaro evaluation: {abs_gt_dir}")
    #         return
    #     run_figaro_evaluation(
    #         generated_dir=midi_output_dir,
    #         gt_dir=abs_gt_dir,
    #         output_dir=args.output_dir,
    #         max_samples=args.eval_num_sample,
    #     )
    #     return

    # # default evaluator for prompt continuation / non-injection inference
    # eval_out_dir = os.path.join(args.output_dir, "eval_result")
    # os.makedirs(eval_out_dir, exist_ok=True)
    # abs_eval_script = resolve_path(args.eval_script_path)
    # abs_gt_dir = resolve_path(args.gt_dataset_dir)

    # if not os.path.exists(abs_eval_script):
    #     logger.error(f"Evaluation script not found: {abs_eval_script}")
    #     return

    # eval_cmd = get_eval_cmd_prefix() + [
    #     # sys.executable,
    #     abs_eval_script,
    #     "--set1dir",
    #     abs_gt_dir,
    #     "--set2dir",
    #     midi_output_dir,
    #     "--outdir",
    #     eval_out_dir,
    #     "--num_sample",
    #     str(args.eval_num_sample),
    #     "--num_runs",
    #     str(args.eval_num_runs),
    #     "--crop_bars",
    #     str(args.eval_crop_bars),
    # ]
    # logger.info(f"Running evaluator: {' '.join(eval_cmd)}")
    # try:
    #     subprocess.run(eval_cmd, check=True)
    #     logger.info(f"Evaluation done: {eval_out_dir}")
    # except Exception as e:
    #     logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
