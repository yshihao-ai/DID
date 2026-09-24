"""Onset-only PLAN built from model-target tracks rather than condition tracks.

This variant inherits the complete onset-only Dataset pipeline.  In
accompaniment mode, configured condition-track blocks remain in the returned bar
and therefore remain available to the condition encoder, but their NOTE tokens
are excluded from PLAN construction.  Chord and continuation modes have no MIDI
condition-track prefix, so their behavior is identical to dataset_onset_plan.
"""

from .dataset_onset_plan import DidDataset as OnsetPlanDataset
from .dataset_onset_plan import QuarterPlanMeterError


class DidDataset(OnsetPlanDataset):
    """Build quarter PLAN from only the tracks the autoregressive model generates."""

    def _insert_harmonic_plan(self, notes_data, note_pos_arr):
        """Insert target-track-only, onset-consistent quarter PLAN.

        In accompaniment mode the bar grammar remains:
          [CHORD][condition tracks][PLAN][target tracks][STOP]

        Only NOTE onsets from ``target tracks`` contribute to QuarterBassPC and
        QuarterOnsetIntervalSet.  Condition tracks are retained byte-for-byte in
        the returned bar, but are invisible to PLAN statistics.
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

            # Track-major DID grammar puts selected condition INST blocks in one
            # continuous prefix.  Keep that prefix in the output, but compute PLAN
            # strictly from the remaining target-track blocks.
            condition_cut = (
                self._condition_prefix_body_cut(body)
                if self.task_mode == "accompaniment"
                else 0
            )
            plan_source_body = body[condition_cut:]

            attacks_by_position = {}
            for tok in plan_source_body:
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
                            "Target-track onset PLAN must contain interval 0."
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
                final_bar = (
                    prefix_rt
                    + body_rt[:condition_cut]
                    + plan_tokens
                    + body_rt[condition_cut:]
                    + stop_rt
                )
            else:
                final_bar = prefix_rt + plan_tokens + body_rt + stop_rt
            output.append(final_bar)

        return output
