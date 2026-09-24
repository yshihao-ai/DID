import random
import numpy as np
from loguru import logger


def _note_type_id(note_obj):
    if isinstance(note_obj, (list, tuple, np.ndarray)):
        return int(note_obj[0])
    return int(note_obj)


def _note_feature(note_obj, idx, default=0):
    if isinstance(note_obj, (list, tuple, np.ndarray)) and len(note_obj) > idx:
        return int(note_obj[idx])
    return default


def find_densest_time_step(n_ids, sto_type_id):
    max_count = -1
    max_idx = 0
    for t, curr_notes in enumerate(n_ids):
        count = 0
        for n in curr_notes:
            if _note_type_id(n) != sto_type_id:
                count += 1
        if count > max_count:
            max_count = count
            max_idx = t
    return max_idx, max_count


def inspect_dataset_logic(dataset, num_samples=2, check_vocab_size=None):
    """
    Lightweight DID bar-level sanity checker.
    """
    config = dataset.music_config

    sto_type_id = int(config.sto_note_id[0])
    sos_type_id = int(config.sos_note_id[0])
    eos_type_id = int(config.eos_note_id[0])
    con_type_id = int(config.con_note_id[0])
    plan_type_id = int(getattr(config, "plan_type_id", -1))

    print(f">>> Start Detailed Dataset Inspection (STO_TYPE={sto_type_id})...")

    total_len = len(dataset)
    if total_len == 0:
        logger.warning("Empty dataset, skip inspection.")
        return

    indices = [0]
    if total_len > 1:
        indices += random.sample(range(1, total_len), min(num_samples - 1, total_len - 1))
    indices = sorted(list(set(indices)))

    for idx in indices:
        print(f"\n{'=' * 40}")
        print(f"[Sample Index]: {idx}")

        try:
            processed_data = dataset[idx]
            if processed_data is None:
                logger.error(f"[Check] dataset[{idx}] returned None, skip.")
                continue

            n_ids = processed_data["note_input_ids"]
            n_masks = processed_data["note_attention_mask"]
            p_ids = processed_data["pos_input_ids"]

            if len(n_ids) != len(n_masks):
                logger.critical(f"[CRITICAL] Length mismatch! Notes: {len(n_ids)}, Masks: {len(n_masks)}")
                continue

            time_steps = len(n_ids)
            densest_idx, max_notes = find_densest_time_step(n_ids, sto_type_id)
            print(f">>> Found densest step at index {densest_idx} (Count: {max_notes} notes)")

            steps_to_show = {0, 1, 2, time_steps - 1, densest_idx}
            for t in sorted(list(steps_to_show)):
                if t < 0 or t >= time_steps:
                    continue

                curr_notes = n_ids[t]
                curr_mask = n_masks[t]
                curr_pos = p_ids[t]

                note_str_list = []
                valid_indices = []

                for i, nid in enumerate(curr_notes):
                    cur_type = _note_type_id(nid)
                    if cur_type == sto_type_id:
                        s = "STO"
                    elif cur_type == sos_type_id:
                        s = "SOS"
                        valid_indices.append(i)
                    elif cur_type == eos_type_id:
                        s = "EOS"
                        valid_indices.append(i)
                    elif plan_type_id >= 0 and cur_type == plan_type_id:
                        s = "PLAN"
                        valid_indices.append(i)
                    elif cur_type == con_type_id:
                        s = "CON"
                        valid_indices.append(i)
                    else:
                        s = f"T{cur_type}"
                        valid_indices.append(i)
                    note_str_list.append(s)

                col_width = 8
                header_str = "".join([f"{i:>{col_width}}" for i in range(len(curr_notes))])
                note_row = "".join([f"{s:>{col_width}}" for s in note_str_list])
                mask_row = "".join([f"{m:>{col_width}}" for m in curr_mask])

                logger.debug(f"--- Time Step {t} ---")
                print(f"POS : {curr_pos}")
                print(f"IDX : {header_str}")
                print(f"NOTE: {note_row}")
                print(f"MASK: {mask_row}")

                last_valid_idx = valid_indices[-1] if valid_indices else -1
                actual_ones = int(sum(curr_mask))

                if last_valid_idx == -1:
                    logger.error("[Check] Empty Bar: FAIL (all tokens are STO)")
                    continue

                last_type = _note_type_id(curr_notes[last_valid_idx])
                expected_ones = len(valid_indices)
                if last_type not in (sos_type_id, eos_type_id):
                    expected_ones += 1
                expected_ones = min(expected_ones, len(curr_notes))

                if actual_ones == expected_ones:
                    logger.success(f"   [Check] Mask Logic: PASS (Valid={len(valid_indices)}, Mask={actual_ones})")
                else:
                    logger.error(
                        f"   [Check] Mask Logic: FAIL! Valid Notes: {len(valid_indices)}. "
                        f"Expected Mask Ones: {expected_ones}, Got: {actual_ones}."
                    )

        except Exception as e:
            logger.exception(f"Error at index {idx}: {e}")

    print(f"{'=' * 40}\nCheck Finished.")
