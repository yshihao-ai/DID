# DID: 2D Autoregressive Modeling with a Decoder-in-Decoder Architecture for Symbolic Music Generation

This repository contains the anonymous implementation of **DID**, a two-dimensional autoregressive framework for symbolic music generation. DID models music along both the time dimension and the note dimension, allowing it to generate variable-length musical sequences while retaining detailed note-level structure.

## Overview

Conventional token-level autoregressive models flatten music into a single long sequence. DID instead treats each bar as a time step and generates the events within that bar along a separate note axis. Its Decoder-in-Decoder architecture contains:

- a **time decoder** that maintains the evolving musical history across bars;
- a **note decoder** that autoregressively generates the compound events inside each bar;
- a **bar encoder** that summarizes the completed bar before the next time step; and
- an optional **condition encoder** for chord-conditioned piano generation and accompaniment generation.

For conditional generation, DID also introduces **Harmonic Plan events**. Before generating the notes of a quarter-note region, the model predicts a compact harmonic plan consisting of a bass pitch class and a bass-relative pitch-class set. These intermediate events provide soft harmonic guidance without prescribing a fixed voicing.

## Main ideas

### Two-dimensional autoregressive factorization

DID separates global temporal development from local note realization. At time step $t$, the time decoder summarizes the music completed so far. The note decoder then generates the events in the current bar until a stop event is produced.

The note decoder receives two complementary histories:

- **global history**: compressed summaries of completed bars up to $t-2$; and
- **local history**: the complete note events from bar $t-1$.

This design preserves detailed short-range continuity while keeping long-range generation computationally manageable.

### Harmonic planning

In chord-conditioned tasks, Harmonic Plan events form an intermediate layer between the supplied condition and the generated notes:

```text
condition -> harmonic plan -> note events
```

The plan expresses coarse harmonic intent, while the note decoder remains free to determine register, voicing, duration, velocity, and instrumentation.

### Supported tasks

The paper evaluates one model family on three settings:

1. **Music continuation**: generate 28 bars from a 4-bar prompt.
2. **Chord-conditioned piano generation**: generate a 32-bar piano performance from a chord sequence.
3. **Accompaniment generation**: generate a 32-bar accompaniment from melody and chord conditions.

Continuation experiments use **Pop1K7** and **POP909**. Conditional-generation experiments use **POP909**. Songs are split into training, validation, and test sets with an 80/10/10 ratio.

## Repository structure

```text
component/             data loading, collation, arguments, and callbacks
data_representation/   MIDI-to-event preprocessing pipeline
infer_args/            inference configurations for each task
inference/             generation, decoding, and MIDI utilities
llm_config/did/        tokenizer and base model configuration
musicmodel/            DID model implementation
smg_metric/            objective evaluation metrics
train_args/            training configurations for each task
vocab/                 event vocabularies
train.py               training entry point
preprocess.sh           preprocessing pipeline
```

The repository does not include the original datasets or trained checkpoints. Update dataset, vocabulary, checkpoint, and output paths in the JSON configuration files before running an experiment.

## Environment

Python 3.12 is recommended.

```bash
conda create -n did python=3.12 -y
conda activate did
pip install -r requirements.txt

cd smg_metric
pip install -e ".[dev,torch]"
cd ..
```

Install a PyTorch build compatible with your CUDA environment if the version required by your system is not already available.

## Data preparation

Place the MIDI datasets under `dataset/MIDI_dataset/` and edit the dataset selection in `preprocess.sh`. The preprocessing pipeline performs four stages:

1. MIDI to corpus;
2. corpus to compound events;
3. vocabulary construction; and
4. event-to-index conversion.

Run the complete pipeline with:

```bash
bash preprocess.sh
```

The training and inference configurations expect processed data under `dataset/represented_data/` and metadata under `dataset/metadata/`.

## Training

Each experiment is defined by a JSON file in `train_args/`:

```bash
# Music continuation
python train.py --train_args_file train_args/did_Pop909_tiny.json
python train.py --train_args_file train_args/did_Pop1k7_tiny.json

# Chord-conditioned piano generation with Harmonic Plan
python train.py --train_args_file train_args/did_Pop909_tiny_chord_plan.json

# Accompaniment generation with Harmonic Plan
python train.py --train_args_file train_args/did_Pop909_tiny_acc_plan.json
```

The paper configuration uses 5-layer time and note decoders with hidden size 384 and 6 attention heads. The bar encoder, and the condition encoder used for conditional tasks, each contain 2 layers. Training uses AdamW with a learning rate of $2\times10^{-4}$, cosine decay, a 0.1 warmup ratio, and an effective batch size of 16.

## Inference

Set `model_name_or_path` in the selected file under `infer_args/` to a trained checkpoint, then run:

```bash
# Music continuation
python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny.json
python inference/infer.py --infer_args_file infer_args/did_Pop1k7_tiny.json

# Chord-conditioned piano generation
python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny_chord_plan.json

# Accompaniment generation
python inference/infer.py --infer_args_file infer_args/did_Pop909_tiny_acc_plan.json
```

Generation uses grammar-constrained sampling so that only fields valid for the current event type are active. Sampling parameters can be adjusted independently for event type, beat, chord, pitch, duration, velocity, instrument, tempo, and Harmonic Plan fields in the inference JSON files.

## Evaluation

The paper reports the following objective metrics:

- **continuation**: chord similarity (SimChr), pitch-class-entropy similarity (SimPCE), scale-consistency similarity (SimSC), groove similarity (SimGrv), and pitch-distribution similarity (PD);
- **chord-conditioned piano generation**: chord accuracy (CA) and chord similarity (CS); and
- **accompaniment generation**: CA, CS, and onset similarity (OS).

Evaluate a directory of generated MIDI files against references with:

```bash
python smg_metric/eval.py <prediction_dir> <reference_dir> <output.json> --max-bars 32
```

The output JSON contains per-file measurements and aggregate statistics. Additional listening-test and analysis utilities are available under `inference/music_evaluation/`.

## Experimental scope

The paper compares DID with representative event-based and sequence-based systems on continuation and conditional generation. The experiments study:

- long-horizon continuation quality on Pop1K7 and POP909;
- chord adherence and accompaniment compatibility on POP909;
- the contribution of Harmonic Plan events and global condition encoding; and
- attention flow between chord conditions, Harmonic Plan events, and generated notes.

## Citation

Citation information will be added after the anonymous review period.
