# V18-D5.1 -> D6-V PLAN-Motion Expressive Decoding

D6-V is a decode-only continuation experiment.

## Core idea

Adjacent generated Quarter PLANs define structural motion from three normalized
distances:

- QuarterBass circular pitch-class motion
- QuarterSet Jaccard distance
- active chord-template Jaccard distance

The motion is compared with the running history of previous generated PLAN
motions using parameter-free Welford mean/variance statistics. The standardized
surprise is squashed with `tanh` to a drive in `[-1,+1]`.

For NOTE velocity only:

    velocity_logits' =
        velocity_logits
        + normalized_velocity_entropy
        * plan_drive
        * velocity_level

where `velocity_level` maps the existing vocabulary monotonically from
soft `-1` to loud `+1`.

Therefore:
- positive structural surprise softly favors louder velocities;
- negative structural surprise softly favors softer velocities;
- confident velocity distributions are changed less;
- all logits remain finite;
- no velocity token is hard-masked;
- no trainable parameter/loss/checkpoint change;
- pitch/chord/PLAN distributions are untouched by D6-V.

D6-V is enabled ONLY for `task_mode=continuation`.

## Important implementation detail

PLAN events are generated as a prefix before NOTE realization. D6 therefore
stores a beat-indexed piecewise-constant drive lookup for each bar:
later quarter PLANs overwrite later beat regions, and every NOTE reads the drive
at its actual inherited beat.

Prompt bars are replaced by the supplied prompt after free decoding in the
existing generation implementation. To avoid contaminating D6 with discarded
free samples, D6-V v1 does NOT learn its running motion statistics from those
prompt bars. It begins neutral and adapts from generated continuation PLANs.

## Install

From the DID project root:

    bash /path/to/v18_d51_d6v_patch/apply_d6v.sh

This creates:

    .d6v_backup/did_model.py.pre_d6v

The patched model keeps D6-V OFF by default.

## Strict A/B

Baseline, same model code:

    CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
      --infer_args_file infer_args/your_config.json

D6-V:

    DID_D6V=1 CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
      --infer_args_file infer_args/your_config.json

Thus seed/top-p/checkpoint remain identical and the only change is the finite
velocity-logit bias.

## Suggested current configuration

Included JSON:

    infer_args/did_Pop909_tiny_qplan_v18_d51_cont_d6v.json

It uses the current continuation Pareto point:

    plan_bass_top_p = 0.0
    plan_set_top_p  = 0.6
    chord_top_p     = 0.6
    pitch_top_p     = 0.9
    duration        = 0.9
    velocity        = 0.9
    instrument      = 0.9
    tempo/type/beat = 0

Run inference + normal first32 evaluation:

    bash /path/to/v18_d51_d6v_patch/run_d6v.sh

## Sanity

    python /path/to/v18_d51_d6v_patch/diagnostics/check_d6v_patch.py

## Velocity-dynamics diagnostic

    python /path/to/v18_d51_d6v_patch/diagnostics/evaluate_velocity_dynamics.py \
      output/.../song_predict \
      --prompt-bars 4 \
      --max-bars 32

Compare baseline vs D6-V on:
- bar_velocity_std
- bar_velocity_range
- bar_velocity_mean_abs_change

D6-V is successful only if expressive variation rises without materially
damaging simChr/simGrv, harmony, or general distribution metrics.

## Restore

    bash /path/to/v18_d51_d6v_patch/restore_pre_d6v.sh
