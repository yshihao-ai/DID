import json
import os
import time

import torch
from loguru import logger
from transformers import TrainerCallback


class PPLCallback(TrainerCallback):
    """Log per-feature evaluation loss on evaluation."""

    def on_evaluate(self, args, state, control, metrics, model, **kwargs):
        actual_model = model.module if hasattr(model, "module") else model

        logger.info(f"\n{'=' * 65}")
        logger.info(f"[Step {state.global_step}] Validation Metrics")
        logger.info(f"{'-' * 65}")

        for feat in actual_model.note_features:
            if feat == "type":
                continue

            feat_loss_sum = actual_model.eval_feature_loss_sum.get(feat, 0.0)
            feat_count = actual_model.eval_feature_count.get(feat, 0)

            if feat_count > 0:
                feat_loss = feat_loss_sum / feat_count
                metrics[f"eval_{feat}_loss"] = feat_loss
                logger.info(f" {feat.capitalize():<12} Loss: {feat_loss:<7.4f} (valid tokens: {feat_count})")

        logger.info(f"{'-' * 65}")

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            logger.info(f" Global eval loss: {eval_loss:.4f}")

        logger.info(f"{'=' * 65}\n")
        actual_model.reset_eval_trackers()


class LoggingCallback(TrainerCallback):
    """Minimal training/eval logging callback."""

    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        current_time = time.time()
        time_delta = max(current_time - self.last_log_time, 1e-8)
        step_delta = max(state.global_step - self.last_log_step, 0)
        steps_per_sec = step_delta / time_delta

        self.last_log_time = current_time
        self.last_log_step = state.global_step

        max_steps = max(int(state.max_steps), 1)
        progress = min(max(state.global_step / max_steps * 100.0, 0.0), 100.0)
        remaining_steps = max(max_steps - state.global_step, 0)
        eta_seconds = int(remaining_steps / steps_per_sec) if steps_per_sec > 1e-8 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        if "loss" in logs:
            logger.info(
                f"[Train] {state.global_step}/{max_steps} ({progress:.2f}%)"
                f" | epoch {float(logs.get('epoch', 0.0)):.2f}"
                f" | loss {float(logs['loss']):.4f}"
                f" | learning_rate {float(logs.get('learning_rate', 0.0)):.2e}"
                f" | ETA {eta_str}"
            )
            return

        if "eval_loss" in logs:
            msg = (
                f"[Eval] step {state.global_step}/{max_steps} ({progress:.2f}%)"
                f" | eval_loss {float(logs['eval_loss']):.4f}"
            )
            if "eval_global_ppl" in logs:
                msg += f" | eval_ppl {float(logs['eval_global_ppl']):.2f}"
            logger.info(msg)


class GpuMemoryCallback(TrainerCallback):
    """Log GPU memory usage at each logging interval."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
            logger.debug(f"[GPU Mem] Alloc: {allocated:.2f}GB | Rsrv: {reserved:.2f}GB | Max: {max_allocated:.2f}GB")


class CheckpointIntegrityCallback(TrainerCallback):
    """After each save, verify `pytorch_model.bin` is present and readable."""

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        checkpoint_bin = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if not os.path.exists(checkpoint_bin):
            logger.warning(f"Checkpoint integrity check skipped: {checkpoint_bin} not found.")
            return
        _ = torch.load(checkpoint_bin, map_location="cpu")
        logger.info(f"Checkpoint integrity OK: readable pytorch_model.bin in {checkpoint_folder}")


class SaveDidConfigCallback(TrainerCallback):
    """Save `music_config` into each checkpoint directory."""

    def __init__(self, music_config):
        self.music_config = music_config

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_folder):
            return

        config_path = os.path.join(checkpoint_folder, "music_config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(vars(self.music_config), f, indent=4, ensure_ascii=False)
            logger.info(f"Saved music_config to {checkpoint_folder}")
        except Exception as e:
            logger.error(f"Failed to save music_config: {e}")
