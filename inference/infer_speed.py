import argparse
import csv
import importlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from transformers import HfArgumentParser, Qwen2Config

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from component.arguments import DidMusicArguments, EvaluateArguments
from component.collator import DidDataCollator
from component.dataset import DidDataset


# -----------------------------------------------------------------------------
# D3PIA-style inference-speed benchmark for DID.
#
# Main metric:
#   mean wall-clock seconds to generate ONE 8-bar sample, batch size = 1.
#
# Timing EXCLUDES:
#   - dataset loading
#   - collation / condition extraction
#   - CPU -> GPU transfer
#   - model/checkpoint loading
#   - MIDI decoding
#   - file writing
#
# Timing INCLUDES:
#   - the complete model.generate() call, including the DID TimeModel,
#     global condition encoder (when enabled by the selected model),
#     NoteModel autoregression, sampling, and all GPU-side generation logic.
#
# IMPORTANT DID boundary contract:
#   To generate B target bars, DID needs B+1 shifted time positions:
#       pos = [B0, ..., B(B-1), boundary]
#   because model.generate() computes target_steps = valid_pos - 1.
#   Therefore 8-bar timing MUST preserve 9 valid time positions while calling
#   model.generate(..., max_time_steps=8).
# -----------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_model_directory(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    if model_name_or_path.endswith(".bin"):
        return os.path.dirname(os.path.abspath(model_name_or_path))
    raise ValueError("model_name_or_path must be a checkpoint directory or .bin file.")


def load_music_config_from_checkpoint(model_name_or_path: str) -> DidMusicArguments:
    model_dir = resolve_model_directory(model_name_or_path)
    candidate_paths = [
        os.path.join(model_dir, "music_config.json"),
        os.path.join(model_dir, "train_args_backup.json"),
        os.path.join(os.path.dirname(model_dir), "train_args_backup.json"),
    ]
    config_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Cannot find music_config.json for checkpoint. Checked: {candidate_paths}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    init_field_names = {item.name for item in fields(DidMusicArguments) if item.init}
    music_kwargs = {k: v for k, v in raw_config.items() if k in init_field_names}
    music_config = DidMusicArguments(**music_kwargs)
    logger.info(f"Loaded music config from checkpoint: {config_path}")
    logger.info(f"Checkpoint task_mode={music_config.task_mode}")
    return music_config


def load_model(
    args: EvaluateArguments,
    music_config: DidMusicArguments,
    model_module: str,
    device: torch.device,
):
    module = importlib.import_module(model_module)
    DidForMusicLM = getattr(module, "DidForMusicLM")

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

    model = DidForMusicLM(
        time_config=time_config,
        note_config=note_config,
        music_config=music_config,
    )

    model_path = args.model_name_or_path
    if os.path.isdir(model_path):
        bin_file = os.path.join(model_path, "pytorch_model.bin")
    elif model_path.endswith(".bin"):
        bin_file = model_path
    else:
        raise ValueError("model_name_or_path must be a checkpoint directory or .bin file.")

    if not os.path.exists(bin_file):
        raise FileNotFoundError(f"Cannot find PyTorch checkpoint: {bin_file}")

    logger.info(f"Loading checkpoint: {bin_file}")
    state_dict = torch.load(bin_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(dtype=torch.bfloat16, device=device)
    return model


def _slice_time_aligned_sample(sample: Dict, keep_steps: int) -> Dict:
    """Crop one DidDataset sample before collation.

    We crop before DidDataCollator so the collator does NOT enter its
    truncation branch that replaces the last kept STO with EOS.  This keeps the
    first 8 target bars byte-for-byte identical to normal inference input.
    """
    time_keys = {
        "pos_input_ids",
        "pos_attention_mask",
        "note_input_ids",
        "note_attention_mask",
        "instrument_input_ids",
        "instrument_attention_mask",
        "target_mask",
    }
    out = deepcopy(sample)
    for key in time_keys:
        if key in out:
            out[key] = out[key][:keep_steps]
    return out


def build_benchmark_examples(
    eval_dataset: DidDataset,
    collator: DidDataCollator,
    bars: int,
    num_samples: int,
) -> Tuple[List[Tuple[int, str, Dict[str, torch.Tensor]]], List[Tuple[int, str, int]]]:
    # DID needs one extra shifted position to delimit B target bars.
    required_steps = bars + 1
    selected = []
    skipped_short = []

    for dataset_idx in range(len(eval_dataset)):
        sample = eval_dataset[dataset_idx]
        path = eval_dataset.get_npz_file_path(dataset_idx)
        if sample is None:
            logger.warning(f"Skip unreadable sample: idx={dataset_idx}, path={path}")
            continue

        seq_len = len(sample.get("pos_input_ids", []))
        if seq_len < required_steps:
            skipped_short.append((dataset_idx, path, seq_len))
            continue

        cropped = _slice_time_aligned_sample(sample, required_steps)
        inputs = collator([cropped])

        valid_pos = int(inputs["pos_attention_mask"][0].long().sum().item())
        target_bars = valid_pos - 1
        if valid_pos != required_steps or target_bars != bars:
            raise RuntimeError(
                "8-bar boundary contract failed after collation: "
                f"idx={dataset_idx}, valid_pos={valid_pos}, "
                f"target_bars={target_bars}, expected_pos={required_steps}, "
                f"expected_bars={bars}."
            )

        if "condition_note_ids" in inputs:
            cond_steps = int(inputs["condition_note_ids"].shape[1])
            if cond_steps != bars:
                raise RuntimeError(
                    "Condition time length is not aligned with target bars: "
                    f"idx={dataset_idx}, condition_steps={cond_steps}, bars={bars}."
                )

        selected.append((dataset_idx, path, inputs))
        if len(selected) == num_samples:
            break

    if len(selected) != num_samples:
        raise RuntimeError(
            f"Requested {num_samples} songs with >= {bars} target bars, but only "
            f"found {len(selected)} eligible samples in split. "
            f"Short/unusable samples skipped={len(skipped_short)}."
        )

    return selected, skipped_short


def prepare_device_inputs(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = [
        "pos_input_ids",
        "pos_attention_mask",
        "instrument_input_ids",
        "instrument_attention_mask",
        "condition_note_ids",
        "condition_note_mask",
    ]
    out = {}
    for key in keys:
        if key in inputs and inputs[key] is not None:
            out[key] = inputs[key].to(device, non_blocking=False)
    return out


def generate_once(
    model,
    eval_args: EvaluateArguments,
    device_inputs: Dict[str, torch.Tensor],
    bars: int,
):
    # Accompaniment benchmark: no continuation prompt.
    with torch.inference_mode():
        out = model.generate(
            device=model.device,
            pos_input_ids=device_inputs["pos_input_ids"],
            pos_attention_mask=device_inputs["pos_attention_mask"],
            instrument_input_ids=device_inputs["instrument_input_ids"],
            instrument_attention_mask=device_inputs["instrument_attention_mask"],
            prompt_note_ids=None,
            prompt_lens=None,
            condition_note_ids=device_inputs.get("condition_note_ids"),
            condition_note_mask=device_inputs.get("condition_note_mask"),
            # Distinguish generated bars (8) from required input positions (9).
            max_time_steps=bars,
            max_note_nums=eval_args.max_note_nums,
            decode_strategy=eval_args.decode_strategy,
            sampling_parameters=eval_args.sampling_params,
        )
    note_feats = out["note_feats"]
    if int(note_feats.shape[0]) != 1:
        raise RuntimeError(f"Benchmark requires batch size 1, got {note_feats.shape[0]}.")
    if int(note_feats.shape[1]) != bars:
        raise RuntimeError(
            f"Generation returned {note_feats.shape[1]} bars; expected exactly {bars}."
        )
    return note_feats


def count_autoregressive_steps(model, note_feats: torch.Tensor) -> int:
    """Count tokens actually emitted/forced up to STO/EOS in each bar.

    This is diagnostic only and is computed AFTER the timed region.
    """
    type_idx = int(model.note_type_idx)
    sto_id = int(model.sto_type_id)
    eos_id = int(model.eos_type_id)
    types = note_feats[0, :, :, type_idx].detach().cpu()

    total = 0
    for bar_types in types:
        stop = (bar_types == sto_id) | (bar_types == eos_id)
        if bool(stop.any()):
            first_stop = int(stop.int().argmax().item())
            total += first_stop + 1
        else:
            total += int(bar_types.numel())
    return total


def benchmark(
    model,
    eval_args: EvaluateArguments,
    examples,
    bars: int,
    warmup_runs: int,
    seed: int,
    output_dir: str,
    split: str,
    model_module: str,
):
    device = model.device
    if device.type != "cuda":
        raise RuntimeError("This benchmark is defined for CUDA inference.")

    # Warm up with the first example.  Reset the RNG afterwards so warm-up does
    # not alter the measured sampling sequence.
    if warmup_runs > 0:
        warm_inputs = prepare_device_inputs(examples[0][2], device)
        logger.info(f"CUDA warm-up: {warmup_runs} run(s), excluded from timing.")
        for _ in range(warmup_runs):
            _ = generate_once(model, eval_args, warm_inputs, bars)
        torch.cuda.synchronize(device)

    set_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    rows = []
    times = []
    total_ar_steps = 0

    for run_idx, (dataset_idx, npz_path, inputs) in enumerate(examples):
        # H2D transfer is intentionally OUTSIDE the timed region, matching the
        # generation-only interpretation of D3PIA's reported inference time.
        device_inputs = prepare_device_inputs(inputs, device)
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        note_feats = generate_once(model, eval_args, device_inputs, bars)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        ar_steps = count_autoregressive_steps(model, note_feats)
        total_ar_steps += ar_steps
        times.append(elapsed)
        rows.append(
            {
                "run": run_idx,
                "dataset_idx": dataset_idx,
                "file": os.path.basename(npz_path),
                "bars": bars,
                "seconds": elapsed,
                "seconds_per_bar": elapsed / bars,
                "ar_steps": ar_steps,
                "ar_steps_per_second": (ar_steps / elapsed if elapsed > 0 else float("inf")),
            }
        )
        logger.info(
            f"[{run_idx + 1:03d}/{len(examples):03d}] "
            f"{os.path.basename(npz_path)} | {bars} bars | "
            f"{elapsed:.4f} s | AR steps={ar_steps}"
        )

    peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    avg = mean(times)
    med = median(times)
    std = pstdev(times) if len(times) > 1 else 0.0
    summary = {
        "metric": f"seconds_per_{bars}_bar_sample",
        "timing_scope": "model.generate only; excludes data loading, collation, H2D transfer, MIDI decoding, and file I/O",
        "split": split,
        "num_samples": len(times),
        "bars_per_sample": bars,
        "batch_size": 1,
        "model_module": model_module,
        "checkpoint": eval_args.model_name_or_path,
        "decode_strategy": eval_args.decode_strategy,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(device),
        "warmup_runs": warmup_runs,
        "mean_seconds": avg,
        "median_seconds": med,
        "std_seconds_population": std,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "total_seconds": sum(times),
        "mean_seconds_per_bar": avg / bars,
        "bars_per_second": bars / avg,
        "total_autoregressive_steps": total_ar_steps,
        "mean_autoregressive_steps_per_sample": total_ar_steps / len(times),
        "autoregressive_steps_per_second_overall": total_ar_steps / sum(times),
        "peak_cuda_allocated_gb": peak_alloc_gb,
        "peak_cuda_reserved_gb": peak_reserved_gb,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"inference_speed_{bars}bar_summary.json")
    csv_path = os.path.join(output_dir, f"inference_speed_{bars}bar_per_sample.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=" * 72)
    logger.info("D3PIA-style DID inference-speed benchmark")
    logger.info(f"Samples               : {len(times)}")
    logger.info(f"Bars / sample         : {bars} (verified exactly)")
    logger.info("Batch size            : 1")
    logger.info("MIDI decode / save    : EXCLUDED")
    logger.info("CPU->GPU transfer     : EXCLUDED")
    logger.info(f"Mean Inf. Time        : {avg:.4f} s / {bars} bars")
    logger.info(f"Median Inf. Time      : {med:.4f} s / {bars} bars")
    logger.info(f"Std                   : {std:.4f} s")
    logger.info(f"Range                 : [{min(times):.4f}, {max(times):.4f}] s")
    logger.info(f"Mean time / bar       : {avg / bars:.4f} s")
    logger.info(f"Bars / second         : {bars / avg:.3f}")
    logger.info(f"Mean AR steps/sample  : {total_ar_steps / len(times):.2f}")
    logger.info(f"AR steps / second     : {total_ar_steps / sum(times):.2f}")
    logger.info(f"Peak CUDA allocated   : {peak_alloc_gb:.3f} GB")
    logger.info(f"Peak CUDA reserved    : {peak_reserved_gb:.3f} GB")
    logger.info(f"Summary JSON          : {json_path}")
    logger.info(f"Per-sample CSV        : {csv_path}")
    logger.info("=" * 72)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="D3PIA-style 8-bar inference-speed benchmark for DID."
    )
    parser.add_argument("--infer_args_file", type=str, required=True)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=74)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help=(
            "Dataset split. Default=test. The benchmark takes the first N eligible "
            "songs from this split, preserving DidDataset order."
        ),
    )
    parser.add_argument("--warmup_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_module",
        type=str,
        default="musicmodel.did_model",
        help=(
            "Module containing DidForMusicLM. Examples: musicmodel.did_model, "
            "musicmodel.did_model_wo_condition_encoder, "
            "musicmodel.did_model_wo_dual_history"
        ),
    )
    parser.add_argument(
        "--benchmark_output_dir",
        type=str,
        default=None,
        help="Defaults to <infer output_dir>/speed_benchmark.",
    )
    return parser.parse_args()


def main():
    bench_args = parse_args()
    if bench_args.bars <= 0:
        raise ValueError("--bars must be positive.")
    if bench_args.num_samples <= 0:
        raise ValueError("--num_samples must be positive.")
    if bench_args.warmup_runs < 0:
        raise ValueError("--warmup_runs cannot be negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference-speed benchmark.")

    set_seed(bench_args.seed)

    eval_parser = HfArgumentParser(EvaluateArguments)
    (eval_args,) = eval_parser.parse_json_file(json_file=bench_args.infer_args_file)
    music_args = load_music_config_from_checkpoint(eval_args.model_name_or_path)

    if music_args.task_mode != "accompaniment":
        raise ValueError(
            "This benchmark version is intentionally restricted to accompaniment, "
            f"got task_mode={music_args.task_mode!r}."
        )
    if eval_args.prompt_measures != 0:
        raise ValueError("Accompaniment benchmark requires prompt_measures=0.")

    device = torch.device("cuda")
    model = load_model(
        eval_args,
        music_config=music_args,
        model_module=bench_args.model_module,
        device=device,
    )

    # Dataset itself stays one-song-per-sample for conditioned tasks.  We crop
    # each selected song to B+1 shifted time positions BEFORE collation.
    required_steps = bench_args.bars + 1
    eval_dataset = DidDataset(
        music_args,
        eval_args.dataset_path,
        required_steps,
        split=bench_args.split,
        metadata_path=eval_args.metadata_path,
        max_data_nums=2048,
    )
    collator = DidDataCollator(music_args, required_steps)

    logger.info(
        f"Benchmark split={bench_args.split!r}: dataset has {len(eval_dataset)} songs; "
        f"requesting first {bench_args.num_samples} eligible songs."
    )
    logger.info(
        f"Boundary contract: {bench_args.bars} target bars require "
        f"{required_steps} valid shifted position steps."
    )

    examples, skipped_short = build_benchmark_examples(
        eval_dataset=eval_dataset,
        collator=collator,
        bars=bench_args.bars,
        num_samples=bench_args.num_samples,
    )
    if skipped_short:
        logger.warning(
            f"Skipped {len(skipped_short)} songs shorter than {bench_args.bars} bars "
            "before collecting the requested benchmark set."
        )

    output_dir = (
        bench_args.benchmark_output_dir
        if bench_args.benchmark_output_dir is not None
        else os.path.join(eval_args.output_dir, "speed_benchmark")
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.add(os.path.join(output_dir, "inference_speed.log"))

    benchmark(
        model=model,
        eval_args=eval_args,
        examples=examples,
        bars=bench_args.bars,
        warmup_runs=bench_args.warmup_runs,
        seed=bench_args.seed,
        output_dir=output_dir,
        split=bench_args.split,
        model_module=bench_args.model_module,
    )


if __name__ == "__main__":
    main()
