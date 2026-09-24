# Continuation
# Pop909
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny/575_cpt2304/song_predict \
  output/did_Pop909_tiny/575_cpt2304/song_answer \
  output/did_Pop909_tiny/575_cpt2304-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny/575_cpt7200/song_predict \
  output/did_Pop909_tiny/575_cpt7200/song_answer \
  output/did_Pop909_tiny/575_cpt7200-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop1k7_baseline/MuseTok/pre \
  baseline/Pop1k7_baseline/MuseTok/gt \
  baseline/Pop1k7_baseline/MuseTok/pre-16bars.json \
  --max-bars 16

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/nmt_answer_filtered \
  baseline/Pop909/Pop909_baseline/nmt_nb_cond_7features \
  baseline/Pop909/Pop909_baseline/16bars/nmt_nb_cond_7features-16bars.json \
  --max-bars 16

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/nmt_answer_filtered \
  baseline/Pop909/Pop909_baseline/nmt_remi_cond_7features \
  baseline/Pop909/Pop909_baseline/16bars/nmt_remi_cond_7features-16bars.json \
  --max-bars 16

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/nmt_answer_filtered \
  baseline/Pop909/Pop909_baseline/nmt_cp_cond_7features \
  baseline/Pop909/Pop909_baseline/16bars/nmt_cp_cond_7features-16bars.json \
  --max-bars 16

# Pop1k7
# epoch 48
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop1k7_tiny/575_cpt6840/song_predict \
  output/did_Pop1k7_tiny/575_cpt6840/song_answer \
  output/did_Pop1k7_tiny/575_cpt6840-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop1k7_tiny/575_cpt6840_96bars/song_predict \
  output/did_Pop1k7_tiny/575_cpt6840_96bars/song_answer \
  output/did_Pop1k7_tiny/575_cpt6840_96bars-32bars.json \
  --max-bars 32




# ACC
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A0B4_cpt2038/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A0B4_cpt2038/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A0B4_cpt2038-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909/FGG/pre \
  baseline/Pop909/FGG/gt \
  baseline/Pop909/FGG/pre-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A3B5_cpt2038/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A3B5_cpt2038/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A3B5_cpt2038-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/T85A3B5_cpt3624/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/T85A3B5_cpt3624/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/T85A3B5_cpt3624-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A4B6_cpt2038/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A4B6_cpt2038/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A4B6_cpt2038-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A0B4_cpt3624/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A0B4_cpt3624/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A0B4_cpt3624-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A0B65_cpt5550/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A0B65_cpt5550/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A0B65_cpt5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A0B6_cpt5550/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A0B6_cpt5550/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A0B6_cpt5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_acc_plan/A0B5_cpt5550/song_predict_full \
  ./output/did_Pop909_tiny_acc_plan/A0B5_cpt5550/song_answer_full \
  ./output/did_Pop909_tiny_acc_plan/A0B5_cpt5550-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  ./output/did_Pop909_tiny_chord_plan/A0B5_cpt5550/song_predict_full \
  ./output/did_Pop909_tiny_chord_plan/A0B5_cpt5550/song_answer_full \
  ./output/did_Pop909_tiny_chord_plan/A0B5_cpt5550-32bars.json \
  --max-bars 32