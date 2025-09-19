#!/bin/bash
# Ablation study on Adapter Internal Complexity - STAGE 1: PRE-TRAINING

# --- Configuration ---
GPU_DEVICES="0"
ADAPTER_SIZE=64
ADAPTER_LIST="0,11,22"
BASE_OUTPUT_DIR="ablation_study_output/fac_adapter_layers_size${ADAPTER_SIZE}"

# Define the internal layer configurations to test
# The baseline (2 layers) is skipped as it was run in previous experiments.
ADAPTER_LAYERS=(1 4)

# --- Pre-training Loop ---
for LAYERS in "${ADAPTER_LAYERS[@]}"
do
  echo "----------------------------------------------------"
  echo "PRE-TRAINING for adapter_transformer_layers = ${LAYERS}"
  echo "----------------------------------------------------"

  RUN_COMMENT="fac-adapter-layers-${LAYERS}"
  RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_layers_${LAYERS}"

  mkdir -p $RUN_OUTPUT_DIR

  CUDA_VISIBLE_DEVICES=$GPU_DEVICES python fac-adapter.py \
        --model_type roberta \
        --model_name_or_path roberta-large \
        --task_name trex \
        --data_dir ./data/trex-rc-sub1e2 \
        --output_dir $RUN_OUTPUT_DIR \
        --do_train \
        --do_eval \
        --per_gpu_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --num_train_epochs 5 \
        --max_seq_length 64 \
        --warmup_steps 1200 \
        --save_steps 1000 \
        --comment "$RUN_COMMENT" \
        --adapter_size $ADAPTER_SIZE \
        --adapter_list "$ADAPTER_LIST" \
        --adapter_transformer_layers $LAYERS \
        --meta_adapter_model=""
done

echo "----------------------------------------------------"
echo "Internal complexity ablation pre-training finished."
echo "Results are in ${BASE_OUTPUT_DIR}"
echo "----------------------------------------------------"
