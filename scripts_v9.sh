#!/usr/bin/env bash
set -e
# Train
## Continuation
CUDA_VISIBLE_DEVICES=0 python train.py --train_args_file train_args/did_Pop909_tiny.json
CUDA_VISIBLE_DEVICES=0 python train.py --train_args_file train_args/did_Pop1k7_tiny.json
## Chord
CUDA_VISIBLE_DEVICES=0 python train.py --train_args_file train_args/did_Pop909_tiny_chord_plan.json
## Acc
CUDA_VISIBLE_DEVICES=0 python train.py --train_args_file train_args/did_Pop909_tiny_acc.json
CUDA_VISIBLE_DEVICES=3 python train.py --train_args_file train_args/did_Pop909_tiny_acc_plan.json

# Inference
## continuation
## chord
## acc
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny_acc_plan.json
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny_acc.json
