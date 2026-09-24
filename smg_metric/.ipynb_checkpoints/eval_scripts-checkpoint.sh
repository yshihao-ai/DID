# Continuation

# Chord

# Accompaniment
## Plan
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-11100/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-11100/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_A3B5_noln2_checkpoint-11100-32bars.json \
  --max-bars 32
  
## No Plan
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_A3B5_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_A3B5_noln2_checkpoint-5550-32bars.json \
  --max-bars 32

