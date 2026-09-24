#!/bin/bash

# ====================================================
# Configuration Section
# ====================================================

# Dataset identifier (e.g., Pop1k7)
# DATASET="LakhClean"
# DATASET="EMOPIA+"
# DATASET="PIAST"
# DATASET="Pop909"
# DATASET="SOD"
DATASET="Pop1k7"


# Number of features for processing
NUM_FEATURES=8

# Encoding type (e.g., did)
ENCODING="did"

# Path to the Python executable
PYTHON_BIN="python"

# ====================================================
# Execution Pipeline
# ====================================================

# Terminate script immediately if any command fails
set -e

echo "[INFO] Starting data processing pipeline..."
echo "[INFO] Configuration: Dataset=$DATASET, Encoding=$ENCODING, Features=$NUM_FEATURES"
START_TIME=$(date +%s)

# --- Step 1: MIDI to Corpus ---
cd ./data_representation
echo "----------------------------------------------------"
echo "[STEP 1/4] Executing: MIDI to Corpus conversion..."
$PYTHON_BIN step1_midi2corpus.py \
    --dataset "$DATASET" \
    --num_features "$NUM_FEATURES"

# --- Step 2: Corpus to Event ---
echo "----------------------------------------------------"
echo "[STEP 2/4] Executing: Corpus to Event conversion ($ENCODING)..."
$PYTHON_BIN step2_corpus2event.py \
    --dataset "$DATASET" \
    --num_features "$NUM_FEATURES" \
    --encoding "$ENCODING"

# --- Step 3: Creating Vocabulary ---
echo "----------------------------------------------------"
echo "[STEP 3/4] Executing: Vocabulary generation..."
$PYTHON_BIN step3_creating_vocab.py \
    --dataset "$DATASET" \
    --num_features "$NUM_FEATURES" \
    --encoding "$ENCODING"

# --- Step 4: Event to Tune Index ---
echo "----------------------------------------------------"
echo "[STEP 4/4] Executing: Event indexing..."
$PYTHON_BIN step4_event2tuneidx.py \
    --dataset "$DATASET" \
    --num_features "$NUM_FEATURES" \
    --encoding "$ENCODING"

# --- Finalization ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "----------------------------------------------------"
echo "[SUCCESS] Pipeline execution completed successfully."
echo "[INFO] Total execution time: ${DURATION} seconds."
cd ..