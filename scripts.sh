# Pop909
CUDA_VISIBLE_DEVICES=1 python train.py --train_args_file train_args/did_Pop909_tiny.json
CUDA_VISIBLE_DEVICES=1 python train.py --train_args_file train_args/did_Pop909_tiny_chord.json
CUDA_VISIBLE_DEVICES=1 python train.py --train_args_file train_args/did_Pop909_tiny_acc.json
CUDA_VISIBLE_DEVICES=1 python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny.json
CUDA_VISIBLE_DEVICES=1 python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny_acc.json

# Pop1k7
CUDA_VISIBLE_DEVICES=0 python train.py --train_args_file train_args/did_Pop1k7_tiny.json
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --infer_args_file infer_args/did_Pop1k7_tiny.json

# EMOPIA+
CUDA_VISIBLE_DEVICES=3 python train.py --train_args_file train_args/did_EMOPIA+_tiny.json
CUDA_VISIBLE_DEVICES=3 python inference/infer.py --infer_args_file infer_args/did_EMOPIA+_tiny.json

# PIAST
CUDA_VISIBLE_DEVICES=3 python train.py --train_args_file train_args/did_PIAST_tiny.json
CUDA_VISIBLE_DEVICES=3 python inference/infer.py --infer_args_file infer_args/did_PIAST_tiny.json

# LakhClean
CUDA_VISIBLE_DEVICES=3 python train.py --train_args_file train_args/did_LakhClean_tiny.json
CUDA_VISIBLE_DEVICES=1 python inference/infer.py --infer_args_file infer_args/did_LakhClean_tiny.json


# extract test dataset
python inference/extract_test_set.py \
    --metadata_path ./dataset/metadata/LakhClean_metadata.json \
    --input_dir ./dataset/MIDI_dataset/LakhClean \
    --output_dir ./dataset/MIDI_test_dataset/LakhClean

python inference/extract_test_set.py \
    --metadata_path ./dataset/metadata/Pop909_metadata.json \
    --input_dir ./dataset/MIDI_dataset/Pop909 \
    --output_dir ./dataset/MIDI_test_dataset/Pop909

python inference/extract_test_set.py \
    --metadata_path ./dataset/metadata/Pop1k7_metadata.json \
    --input_dir ./dataset/MIDI_dataset/Pop1k7 \
    --output_dir ./dataset/MIDI_test_dataset/Pop1k7

python inference/extract_test_set.py \
    --metadata_path ./dataset/metadata/EMOPIA+_metadata.json \
    --input_dir ./dataset/MIDI_dataset/EMOPIA+ \
    --output_dir ./dataset/MIDI_test_dataset/EMOPIA+

python inference/extract_test_set.py \
    --metadata_path ./dataset/metadata/PIAST_metadata.json \
    --input_dir ./dataset/MIDI_dataset/PIAST \
    --output_dir ./dataset/MIDI_test_dataset/PIAST

# prompt4 evlaute
conda activate did_eval
python inference/music_evaluation/music_evaluator.py \
  --set1dir dataset/MIDI_test_dataset/Pop909 \
  --set2dir baseline/Pop909_baseline/nmt_cp_cond_7features \
  --outdir baseline_metrics/Pop909_baseline/nmt_cp_cond_7features \
  --num_sample 2048 \
  --num_runs 3 \
  --crop_bars 32



python ./tools/count_did_bars.py \
  --input-dir ./dataset/represented_data/events/events_Pop1k7/did8 \
  --segment-bars 256