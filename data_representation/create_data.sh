#!/bin/bash

DATASET="Pop1k7"
NUM_FEATURES=8
ENCODING="did"

echo "Running step1_midi2corpus.py..."
python step1_midi2corpus.py --dataset $DATASET --num_features $NUM_FEATURES

echo "Running step2_corpus2event.py..."
python step2_corpus2event.py --dataset $DATASET --num_features $NUM_FEATURES --encoding $ENCODING

echo "Running step3_creating_vocab.py..."
python step3_creating_vocab.py --dataset $DATASET --num_features $NUM_FEATURES --encoding $ENCODING

echo "Running step4_event2tuneidx.py..."
python step4_event2tuneidx.py --dataset $DATASET --num_features $NUM_FEATURES --encoding $ENCODING

echo "All steps completed!"
