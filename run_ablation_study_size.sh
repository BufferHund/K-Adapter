#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# This script runs a MODIFIED ablation study on adapter_size for LIMITED COMPUTE.

# --- Configuration ---
GPU_DEVICES="0"
BASE_OUTPUT_DIR="ablation_study_output/fac_adapter_size_sub1e2"
TASK_NAME="trex"
# Using the 1% subset of the data
BASE_DATA_DIR="./data/trex-rc-sub1e2"
MODEL_TYPE="roberta"
MODEL_NAME="roberta-large"
EPOCHS=5
# Reduced batch size for single GPU, compensated with gradient accumulation
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
MAX_SEQ_LENGTH=64
LEARNING_RATE=5e-5
WARMUP_STEPS=1200
SAVE_STEPS=1000

# Define the REDUCED set of adapter sizes to test
ADAPTER_SIZES=(768 256 64 16)

# --- Experiment Loop ---
for SIZE in "${ADAPTER_SIZES[@]}"
do
  echo "----------------------------------------------------"
  echo "Running experiment with adapter_size = $SIZE on 1% data"
  echo "----------------------------------------------------"

  # Create a unique output directory and comment for this run
  RUN_COMMENT="fac-adapter-size-${SIZE}-sub1e2"
  RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_size_${SIZE}"

  # Ensure the output directory exists
  mkdir -p $RUN_OUTPUT_DIR

  # Run the training script
  CUDA_VISIBLE_DEVICES=$GPU_DEVICES python fac-adapter.py \
        --model_type $MODEL_TYPE \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK_NAME \
        --data_dir $BASE_DATA_DIR \
        --output_dir $RUN_OUTPUT_DIR \
        --do_train \
        --do_eval \
        --evaluate_during_training 'True' \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --learning_rate $LEARNING_RATE \
        --num_train_epochs $EPOCHS \
        --max_seq_length $MAX_SEQ_LENGTH \
        --warmup_steps $WARMUP_STEPS \
        --save_steps $SAVE_STEPS \
        --comment "$RUN_COMMENT" \
        --adapter_size $SIZE \
        --adapter_list "0,11,22" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model=""
done

echo "----------------------------------------------------"
echo "Ablation study finished."
echo "Results are in ${BASE_OUTPUT_DIR}"
echo "----------------------------------------------------"