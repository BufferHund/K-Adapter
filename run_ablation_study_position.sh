#!/bin/bash
# Ablation study on Adapter insertion position - STAGE 1: PRE-TRAINING

# --- Configuration ---
GPU_DEVICES="0"
ADAPTER_SIZE=64 # Fixed adapter size for this experiment
BASE_OUTPUT_DIR="ablation_study_output/fac_adapter_position_size${ADAPTER_SIZE}"

# Define the insertion positions to test
POSITIONS=(
    # "0,11,22" # Sparse (Baseline) - 已在实验一中运行，跳过
    "0,1,2"   # Early
    "10,11,12" # Middle
    "21,22,23"  # Late
)

# --- Pre-training Loop ---
for POS in "${POSITIONS[@]}"
do
  # Create a clean name for the position, e.g., 0_11_22
  POS_NAME=$(echo "$POS" | tr ',' '_')
  echo "----------------------------------------------------"
  echo "PRE-TRAINING for position: ${POS_NAME}"
  echo "----------------------------------------------------"

  RUN_COMMENT="fac-adapter-pos-${POS_NAME}"
  RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_pos_${POS_NAME}"

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
        --adapter_list "$POS" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model=""
done

echo "----------------------------------------------------"
echo "Position ablation pre-training finished."
echo "Results are in ${BASE_OUTPUT_DIR}"
echo "----------------------------------------------------"
