# Pop909
# - CP
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop909_baseline/nmt_cp_cond_7features \
  dataset/MIDI_test_dataset/Pop909/ \
  baseline_smg_metrics/Pop909_baseline/32_nmt_cp_cond_7features.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  baseline/Pop909_baseline/nmt_cp_cond_7features \
  dataset/MIDI_test_dataset/Pop909/ \
  baseline_smg_metrics/Pop909_baseline/32_nmt_cp_cond_7features.json \
  --max-bars 32



# - NB
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop909_baseline/nmt_nb_cond_7features \
  dataset/MIDI_test_dataset/Pop909/ \
  baseline_smg_metrics/Pop909_baseline/32_nmt_nb_cond_7features.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop909_baseline/nmt_nb_cond_7features \
  output/did_Pop909_tiny_pb/pop909_infer_checkpoint-8305-0.9-bias0-pitch/song_answer \
  baseline_smg_metrics/Pop909_baseline/32_nmt_nb_cond_7features_ans.json \
  --max-bars 32

# - REMI
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py baseline/Pop909_baseline/nmt_remi_cond_7features \
  dataset/MIDI_test_dataset/Pop909 \
  baseline_smg_metrics/Pop909_baseline/32_nmt_remi_cond_7features.json \
  --max-bars 32

# - MUSETOK
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop909_baseline/musetok_pop909_32bars_cond \
  dataset/MIDI_test_dataset/Pop909 \
  baseline_smg_metrics/Pop909_baseline/32bars/32_musetok_pop909_32bars_cond.json \
  --max-bars 32

# - DID
python smg_metric/eval.py \
  output/did_Pop909_tiny/pop909_infer_checkpoint-9472/song_predict \
  dataset/MIDI_test_dataset/Pop909 \
  output/did_Pop909_tiny/pop909_infer_checkpoint-9472-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test1_checkpoint-2038/song_predict \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test1_checkpoint-2038/song_answer \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test1_checkpoint-2038.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test2_checkpoint-2038/song_predict \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test2_checkpoint-2038/song_answer \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test2_checkpoint-2038.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test3_checkpoint-2038/song_predict \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test3_checkpoint-2038/song_answer \
  output/did_Pop909_tiny_pb_qplan_v18.1-d51/continuation_test3_checkpoint-2038.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_v18_seg/pop909_infer_beat0.6_checkpoint-7200/song_predict \
  output/did_Pop909_tiny_v18_seg/pop909_infer_beat0.6_checkpoint-7200/song_answer \
  output/did_Pop909_tiny_v18_seg/pop909_infer_beat0.6_checkpoint-7200.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_v18_seg_share/pop909_infer_checkpoint-2016/song_predict \
  output/did_Pop909_tiny_v18_seg_share/pop909_infer_checkpoint-2016/song_answer \
  output/did_Pop909_tiny_v18_seg_share/pop909_infer_checkpoint-2016.json \
  --max-bars 32
  
# Pop1k7
# - REMI
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop1k7_baseline/nmt_pre_pop1k7_remi_cond \
  dataset/MIDI_test_dataset/Pop1k7 \
  baseline_smg_metrics/Pop1k7_baseline/32bars/32_nmt_remi_cond_7features.json \
  --max-bars 32
# - CP
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop1k7_baseline/nmt_pre_pop1k7_cp_cond \
  dataset/MIDI_test_dataset/Pop1k7 \
  baseline_smg_metrics/Pop1k7_baseline/32bars/32_nmt_cp_cond_7features.json \
  --max-bars 32
# - NB
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/Pop1k7_baseline/nmt_pre_pop1k7_nmt_cond \
  dataset/MIDI_test_dataset/Pop1k7 \
  baseline_smg_metrics/Pop1k7_baseline/32bars/32_nmt_nb_cond_7features.json \
  --max-bars 32

# - DID
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_95_checkpoint-18447/song_predict \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_95_checkpoint-18447/song_answer \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_95_checkpoint-18447-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_95_checkpoint-9152/song_predict \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_95_checkpoint-9152/song_answer \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_95_checkpoint-9152-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_checkpoint-9152/song_predict \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_checkpoint-9152/song_answer \
  output/did_Pop1k7_tiny_v18_seg_share_large/pop1k7_infer_checkpoint-9152-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg_share/pop1k7_infer_checkpoint-18447/song_predict \
  output/did_Pop1k7_tiny_v18_seg_share/pop1k7_infer_checkpoint-18447/song_answer \
  output/did_Pop1k7_tiny_v18_seg_share/pop1k7_infer_checkpoint-18447-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_577_checkpoint-18447/song_predict \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_577_checkpoint-18447/song_answer \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_577_checkpoint-18447-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-28171/song_predict \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-28171/song_answer \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-28171-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_r85_checkpoint-21450/song_predict \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_r85_checkpoint-21450/song_answer \
  output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_r85_checkpoint-21450-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-27027/song_predict \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-27027/song_answer \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-27027-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-18018/song_predict \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-18018/song_answer \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-18018-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-21450/song_predict \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-21450/song_answer \
  ./output/did_Pop1k7_tiny_v18_seg/pop1k7_infer_beat0.6_checkpoint-21450-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_chord0.9_checkpoint-6028/song_predict \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_chord0.9_checkpoint-6028/song_answer \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_chord0.9_checkpoint-6028-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_checkpoint-6028/song_predict \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_checkpoint-6028/song_answer \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_checkpoint-6028-32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny/Pop1k7_prompt4_infer_chord75_checkpoint-22272/song_predict \
  dataset/MIDI_test_dataset/Pop1k7 \
  output/did_Pop1k7_tiny/Pop1k7_prompt4_infer_chord75_checkpoint-22272-32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d1_checkpoint-10006/song_predict \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d1_checkpoint-10006/song_answer \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d1_checkpoint-10006-32bars.json \
  --max-bars 32 

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d2_checkpoint-10006/song_predict \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d2_checkpoint-10006/song_answer \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d2_checkpoint-10006-32bars.json \
  --max-bars 32 

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d3_checkpoint-10006/song_predict \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d3_checkpoint-10006/song_answer \
  output/did_Pop1k7_tiny_pb_qplan_v18-d51/Pop1k7_infer_d3_checkpoint-10006-32bars.json \
  --max-bars 32 


CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop1k7_tiny/c75_B50_p90/song_predict \
  output/did_Pop1k7_tiny/c75_B50_p90/song_answer \
  output/did_Pop1k7_tiny/c75_B50_p90-32bars.json \
  --max-bars 32 

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_checkpoint-9960/song_predict \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_checkpoint-9960/song_answer \
  output/did_Pop1k7_tiny_v18/Pop1k7_infer_d1_checkpoint-9960-32bars.json \
  --max-bars 32 

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop1k7_tiny_v18_chord/Pop1k7_infer_d1_checkpoint-9960/song_predict \
  output/did_Pop1k7_tiny_v18_chord/Pop1k7_infer_d1_checkpoint-9960/song_answer \
  output/did_Pop1k7_tiny_v18_chord/Pop1k7_infer_d1_checkpoint-9960-32bars.json \
  --max-bars 32 

# acc
# D3PIA
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/D3PIA_baseline/acc/predict \
  baseline/D3PIA_baseline/acc/answer \
  baseline/D3PIA_baseline_acc_wobr_32.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/D3PIA_baseline/acc_br/predict \
  baseline/D3PIA_baseline/acc_br/answer \
  baseline/D3PIA_baseline_acc_br_32.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  baseline/D3PIA_baseline/acc_br/predict \
  baseline/D3PIA_baseline/acc_br/answer \
  baseline/D3PIA_baseline_acc_br_32_wo_bass.json \
  --max-bars 32
  
# DID
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_v2_chord96/pop909_infer_all0.5_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_v2_chord96/pop909_infer_all0.5_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_v2_chord96/pop909_infer_all0.5_noln2_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_greedy_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_greedy_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_greedy_checkpoint-2038_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_onlyGlobal_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_onlyGlobal_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_onlyGlobal_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-2038_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_onlyGlobal_chord96/pop909_infer_all0.5_noln2_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_onlyGlobal_chord96/pop909_infer_all0.5_noln2_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_onlyGlobal_chord96/pop909_infer_all0.5_noln2_checkpoint-2038_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-2038_full_32bars.json \
  --max-bars 32


CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_v5_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_v5_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_v5_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-2038_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_noln2_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_ln2_n128_checkpoint-2038_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_p7b5_ln2_n128_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_p7b5_ln2_n128_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_p7b5_ln2_n128_checkpoint-2038_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_chord96/pop909_infer_all0.5_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_all0.5_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_plan_chord96/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_plan0.5_d0.5_v0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_plan0.5_d0.5_v0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_plan0.5_d0.5_v0.5_checkpoint-2038_answer_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_i0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_i0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_i0.5_checkpoint-2038_answer_full_32bars.json \
  --max-bars 32
  
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038_answer_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038_answer_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18_d51/pop909_infer_d0.5_v0.5_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_greedy_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_greedy_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_greedy_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_d0.5_v0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_d0.5_v0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_d0.5_v0.5_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d5/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d5/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d5/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d4/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d4/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d4/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d3/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d3/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d3/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d2/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d2/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d2/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d1/pop909_infer_i0_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d1/pop909_infer_i0_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d1/pop909_infer_i0_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d1/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d1/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18-d1/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22.1/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22.1/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22.1/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1525/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1525/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1525_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1525/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1525/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1525_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1019/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1019/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v22/pop909_infer_checkpoint-1019_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v21/pop909_infer_checkpoint-2000/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v21/pop909_infer_checkpoint-2000/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v21/pop909_infer_checkpoint-2000_full_32bars.json \
  --max-bars 32
  
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v21/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v21/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v21/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v20/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v20/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v20/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v19/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v19/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v19/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v19/pop909_infer_checkpoint-1019/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v19/pop909_infer_checkpoint-1019/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v19/pop909_infer_checkpoint-1019_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v18/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v17/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v17/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v17/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.2/pop909_infer_checkpoint-1510/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.2/pop909_infer_checkpoint-1510/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.2/pop909_infer_checkpoint-1510_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.2/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.2/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.2/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_i0.5_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_i0.5_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_i0.5_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2718/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2718/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2718_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-3000/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-3000/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-3000_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2529/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2529/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2529_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2038/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2038/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-2038_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-1208/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-1208/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16.1/pop909_infer_checkpoint-1208_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-4492/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-4492/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-4492_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-3850/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-3850/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-3850_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-3057/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-3057/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v16/pop909_infer_checkpoint-3057_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15.1/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15.1/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15.1/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r2/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r2/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r2/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_p0.5_d0.5_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_p0.5_d0.5_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_p0.5_d0.5_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_p0.4_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_p0.4_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_p0.4_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_i0_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15_r1/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_i0_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_i0_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_i0_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.6_checkpoint-4416/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.6_checkpoint-4416/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.6_checkpoint-4416_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.4_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.4_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.4_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.6_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.6_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_p0.6_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v15/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550-dur0.5/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550-dur0.5/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550-dur0.5_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13.1/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13.1/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13.1/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t4/pop909_infer_checkpoint-5032/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t4/pop909_infer_checkpoint-5032/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t4/pop909_infer_checkpoint-5032_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t3/pop909_infer_checkpoint-5032/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t3/pop909_infer_checkpoint-5032/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t3/pop909_infer_checkpoint-5032_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t0/pop909_infer_checkpoint-5032/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t0/pop909_infer_checkpoint-5032/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t0/pop909_infer_checkpoint-5032_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t1/pop909_infer_checkpoint-5032/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t1/pop909_infer_checkpoint-5032/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10.2_t1/pop909_infer_checkpoint-5032_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v13/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v12/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v12/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v12/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v122/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v122/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v122/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v123/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v123/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v123/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=0 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v124/pop909_infer_checkpoint-5550_posthoc_rawk/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v124/pop909_infer_checkpoint-5550_posthoc_rawk/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v124/pop909_infer_checkpoint-5550_posthoc_rawk_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550-bias2/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550-bias2/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550-bias2_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550-greedy/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550-greedy/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550-greedy_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v11/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-3133/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-3133/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-3133_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5032/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5032/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5032_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5032-greedy/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5032-greedy/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5032-greedy_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5550/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5550/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v10/pop909_infer_checkpoint-5550_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-3359/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-3359/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-3359_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5028/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5028/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5028_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5028-greedy/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5028-greedy/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5028-greedy_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5561/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5561/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-5561_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-6682/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-6682/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-6682_full_32bars.json \
  --max-bars 32

CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-8015/song_predict_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-8015/song_answer_full \
  output/did_Pop909_tiny_acc_bridge_pb_asc_qplan_v9/pop909_infer_checkpoint-8015_full_32bars.json \
  --max-bars 32



# chord
# DID
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_chord_wobr_pb_qplan_v16.1/pop909_infer_checkpoint-3552/song_predict_full \
  output/did_Pop909_tiny_chord_wobr_pb_qplan_v16.1/pop909_infer_checkpoint-3552/song_answer_full \
  output/did_Pop909_tiny_chord_wobr_pb_qplan_v16.1/pop909_infer_checkpoint-3552_full_32bars.json \
  --max-bars 32
  
CUDA_VISIBLE_DEVICES=1 python smg_metric/eval.py \
  output/did_Pop909_tiny_chord_wobr_pb/pop909_infer_checkpoint-8305-0.5-bias0-pitch/song_predict_full \
  output/did_Pop909_tiny_chord_wobr_pb/pop909_infer_checkpoint-8305-0.5-bias0-pitch/song_answer_full \
  output/did_Pop909_tiny_chord_wobr_pb/pop909_infer_checkpoint-8305-0.5-bias0-pitch_full_32bars.json \
  --max-bars 32



  