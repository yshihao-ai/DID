# Continuation 
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop1k7_tiny/chr_small_search_ckpt7837/c70_p90/song_predict \
  output/did_Pop1k7_tiny/chr_small_search_ckpt7837/c70_p90/song_answer \
  output/did_Pop1k7_tiny/chr_small_search_ckpt7837_c70_p90-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/Pop909_baseline/BEAT/pre \
  baseline/Pop909/Pop909_baseline/BEAT/gt \
  baseline/Pop909/Pop909_baseline/BEAT/beat_pop909-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/Pop909_baseline/pop909_test_best/prediction \
  baseline/Pop909/Pop909_baseline/pop909_test_best/gt_codec \
  baseline/Pop909/Pop909_baseline/pop909_test_best-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/Pop909_baseline/polyffusion_pop909_test/prediction \
  baseline/Pop909/Pop909_baseline/polyffusion_pop909_test/gt_codec \
  baseline/Pop909/Pop909_baseline/polyffusion_pop909_test-32bars.json \
  --max-bars 32


CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/Pop909_baseline/whole_song_gen_pop909_e100_test/prediction \
  baseline/Pop909/Pop909_baseline/whole_song_gen_pop909_e100_test/gt_codec \
  baseline/Pop909/Pop909_baseline/whole_song_gen_pop909_e100_test-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/Pop909_baseline/whole_song_gen_pop909_best_e200_test/prediction \
  baseline/Pop909/Pop909_baseline/whole_song_gen_pop909_best_e200_test/gt_codec \
  baseline/Pop909/Pop909_baseline/whole_song_gen_pop909_best_e200_test-32bars.json \
  --max-bars 32


# Chord

# Accompaniment
## Plan
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96_target_onset/pop909_infer_A0B4_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96_target_onset/pop909_infer_A0B4_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96_target_onset/pop909_infer_A0B4_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96_shared/pop909_infer_A0B4_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96_shared/pop909_infer_A0B4_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96_shared/pop909_infer_A0B4_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_base_acc_plan_chord96_shared/pop909_infer_A0B4_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_base_acc_plan_chord96_shared/pop909_infer_A0B4_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_base_acc_plan_chord96_shared/pop909_infer_A0B4_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96_wo_condition_encoder/pop909_infer_A0B4_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96_wo_condition_encoder/pop909_infer_A0B4_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96_wo_condition_encoder/pop909_infer_A0B4_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_h1/pop909_infer_A0B4_noln2_checkpoint-7000/song_predict_full \
  output/did_Pop909_tiny_acc_plan_h1/pop909_infer_A0B4_noln2_checkpoint-7000/song_answer_full \
  output/did_Pop909_tiny_acc_plan_h1/pop909_infer_A0B4_noln2_checkpoint-7000-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_h1/pop909_infer_A0B4_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_h1/pop909_infer_A0B4_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_h1/pop909_infer_A0B4_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B4_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B4_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B4_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-9472/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-9472/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-9472-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_beat0_noln2_checkpoint-11100/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_beat0_noln2_checkpoint-11100/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_beat0_noln2_checkpoint-11100-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_beat0_inst0_noln2_checkpoint-11100/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_beat0_inst0_noln2_checkpoint-11100/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_beat0_inst0_noln2_checkpoint-11100-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-11100/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-11100/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-11100-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_noln2_checkpoint-11100/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_noln2_checkpoint-11100/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B5_noln2_checkpoint-11100-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B4_noln2_checkpoint-11100/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B4_noln2_checkpoint-11100/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A0B4_noln2_checkpoint-11100-32bars.json \
  --max-bars 32 
  
## No Plan
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_A3B5_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

