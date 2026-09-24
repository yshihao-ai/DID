import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Qwen2Config,
    Trainer,
    TrainingArguments,
    set_seed,
)

from component.arguments import CustomizedArguments, DidMusicArguments
from component.callbacks import (
    CheckpointIntegrityCallback,
    GpuMemoryCallback,
    LoggingCallback,
    PPLCallback,
    SaveDidConfigCallback,
)
from component.checking_logs import inspect_dataset_logic
from component.collator import DidDataCollator
from component.dataset import DidDataset
from musicmodel.did_model import DidForMusicLM
# from musicmodel.did_model_wo_dual_history import DidForMusicLM
# from musicmodel.did_model_wo_condition_encoder import DidForMusicLM




def setup_logger(output_dir: str):
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        level="INFO",
    )
    log_path = Path(output_dir) / "train.log"
    logger.add(log_path, rotation="50 MB", encoding="utf-8", level="DEBUG")
    return logger


def _log_table(title: str, header: Tuple[str, str], rows: List[Tuple[str, str]], widths: Tuple[int, int]):
    c1_w, c2_w = widths
    total_w = c1_w + c2_w + 7
    line = "+" + "-" * (c1_w + 2) + "+" + "-" * (c2_w + 2) + "+"

    lines = ["", line, f"| {title:^{total_w - 4}} |", line]
    h1, h2 = header
    lines.append(f"| {h1:<{c1_w}} | {h2:<{c2_w}} |")
    lines.append(line)

    for key, val in rows:
        key_str = str(key)
        val_str = str(val)
        if len(key_str) > c1_w:
            key_str = key_str[: c1_w - 3] + "..."
        if len(val_str) > c2_w:
            val_str = val_str[: c2_w - 3] + "..."
        lines.append(f"| {key_str:<{c1_w}} | {val_str:<{c2_w}} |")

    lines.append(line)
    lines.append("")
    logger.info("\n".join(lines))


def log_environment_info():
    rows = [
        ("Python Version", sys.version.split()[0]),
        ("PyTorch Version", torch.__version__),
        ("Platform", platform.platform().split("-")[0]),
    ]

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        rows.append(("CUDA Available", f"True (Count: {num_gpus})"))
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            rows.append((f"GPU {i}", f"{gpu_name} ({gpu_mem})"))
    else:
        rows.append(("CUDA Available", "False"))

    _log_table(
        title="System Info",
        header=("Key", "Value"),
        rows=rows,
        widths=(22, 56),
    )


def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    rows = [
        ("Total Params", f"{total_params:,} ({total_params / 1e6:.2f}M)"),
        ("Trainable Params", f"{trainable_params:,} ({trainable_params / 1e6:.2f}M)"),
        ("Non-trainable", f"{non_trainable_params:,} ({non_trainable_params / 1e6:.2f}M)"),
    ]
    _log_table(
        title="Model Params",
        header=("Type", "Count"),
        rows=rows,
        widths=(25, 32),
    )


def log_training_dynamics(args: TrainingArguments, num_examples: int):
    world_size = max(int(getattr(args, "world_size", 1)), 1)
    per_device = int(args.per_device_train_batch_size)
    grad_accum = int(args.gradient_accumulation_steps)
    total_train_batch_size = per_device * grad_accum * world_size
    num_update_steps_per_epoch = max(len(range(0, num_examples, max(total_train_batch_size, 1))), 1)

    if args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
    else:
        max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs

    rows = [
        ("Num Examples", f"{num_examples:,}"),
        ("Num Epochs", f"{num_train_epochs:.2f}"),
        ("Per Device Batch", f"{per_device}"),
        ("Grad Accum Steps", f"{grad_accum}"),
        ("World Size", f"{world_size}"),
        ("Effective Batch", f"{total_train_batch_size}"),
        ("Total Steps", f"{max_steps:,}"),
        ("Warmup Steps", f"{args.warmup_steps}"),
        ("Learning Rate", f"{args.learning_rate:.2e}"),
        ("FP16 / BF16", f"{args.fp16} / {args.bf16}"),
    ]
    _log_table(
        title="Training Dynamics",
        header=("Metric", "Value"),
        rows=rows,
        widths=(25, 28),
    )

    if total_train_batch_size < 16:
        logger.warning(
            f"Effective batch size is small ({total_train_batch_size}), "
            "training may be unstable."
        )


def parse_args() -> Tuple[CustomizedArguments, DidMusicArguments, TrainingArguments, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, required=True, help="Path to training JSON config.")
    cmd_args = parser.parse_args()

    if not os.path.exists(cmd_args.train_args_file):
        raise FileNotFoundError(f"Configuration file not found: {cmd_args.train_args_file}")

    hf_parser = HfArgumentParser((CustomizedArguments, DidMusicArguments, TrainingArguments))
    args, music_config, training_args = hf_parser.parse_json_file(json_file=cmd_args.train_args_file)

    # if args.gpu_devices is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    set_seed(training_args.seed)
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(training_args.output_dir) / "train_args_backup.json", "w", encoding="utf-8") as f:
        all_args = {**vars(args), **vars(music_config), **vars(training_args)}
        serializable_args = {
            k: v
            for k, v in all_args.items()
            if isinstance(v, (int, float, str, bool, list, dict, type(None)))
        }
        json.dump(serializable_args, f, indent=4, ensure_ascii=False)

    return args, music_config, training_args, cmd_args.train_args_file


def load_model_and_tokenizer(args: CustomizedArguments, music_config: DidMusicArguments, training_args: TrainingArguments):
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
        attention_dropout=0.1,
        use_cache=not training_args.gradient_checkpointing,
        tie_word_embeddings=False,
    )
    logger.info("Time Decoder: attention_drouput = 0.1")
    logger.info("Time Decoder: attention_drouput = 0.1")
    logger.info("Time Decoder: attention_drouput = 0.1")

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
        attention_dropout=0,
        use_cache=not training_args.gradient_checkpointing,
        tie_word_embeddings=False,
    )


    torch_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    model = DidForMusicLM(time_config=time_config, note_config=note_config, music_config=music_config)
    model.to(dtype=torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name_or_path, trust_remote_code=True)
    return model, tokenizer


def verify_checkpoint_bin_exists(checkpoint_bin: str):
    if not os.path.exists(checkpoint_bin):
        raise FileNotFoundError(f"Final checkpoint not found: {checkpoint_bin}")
    logger.info(f"Final checkpoint saved: {checkpoint_bin}")


def train():
    args, music_config, training_args, config_path = parse_args()

    setup_logger(training_args.output_dir)
    log_environment_info()
    logger.info(f"Start training with config: {config_path}")

    if bool(getattr(training_args, "save_safetensors", False)):
        logger.warning(
            "Detected save_safetensors=True in config. "
            "Force save_safetensors=False to avoid shared-storage checkpoint issues."
        )
    training_args.save_safetensors = False

    model, tokenizer = load_model_and_tokenizer(args, music_config, training_args)
    logger.info("Use unified single-stage training protocol.")
    count_parameters(model)

    logger.info(f"Load dataset from: {args.dataset_path}")
    train_aug_type = "random" if getattr(music_config, "use_augmentation", False) else None
    logger.info(f"Train augmentation: {'enabled(random shift)' if train_aug_type is not None else 'disabled'}")

    train_dataset = DidDataset(
        music_config,
        args.dataset_path,
        args.max_seq_length,
        split="train",
        metadata_path=args.metadata_path,
        aug_type=train_aug_type,
    )
    val_dataset = DidDataset(
        music_config,
        args.dataset_path,
        args.max_seq_length,
        split="valid",
        metadata_path=args.metadata_path,
        aug_type=None,
    )
    test_dataset = DidDataset(
        music_config,
        args.dataset_path,
        args.max_seq_length,
        split="test",
        metadata_path=args.metadata_path,
        aug_type=None,
    )

    inspect_dataset_logic(train_dataset, num_samples=2)
    data_collator = DidDataCollator(music_config, args.max_seq_length)
    log_training_dynamics(training_args, len(train_dataset))

    callbacks = [
        PPLCallback(),
        LoggingCallback(),
        GpuMemoryCallback(),
        CheckpointIntegrityCallback(),
        SaveDidConfigCallback(music_config),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("*** Start training (single-stage) ***")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint

        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resume from checkpoint: {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    logger.info(f"Save final model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    final_music_config_path = os.path.join(training_args.output_dir, "music_config.json")
    with open(final_music_config_path, "w", encoding="utf-8") as f:
        json.dump(vars(music_config), f, indent=4, ensure_ascii=False)

    logger.info(
        "Saved final music_config to: "
        f"{final_music_config_path}"
    )
    final_checkpoint_bin = os.path.join(training_args.output_dir, "pytorch_model.bin")
    verify_checkpoint_bin_exists(final_checkpoint_bin)

    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    if "eval_loss" in test_metrics:
        logger.info(f"Test eval_loss: {test_metrics['eval_loss']:.4f}")
    if "eval_global_ppl" in test_metrics:
        logger.info(f"Test eval_global_ppl: {test_metrics['eval_global_ppl']:.4f}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.success("Training completed.")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        logger.exception("Training failed.")
        raise
