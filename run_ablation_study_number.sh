#!/bin/bash
# Ablation study on the number of Adapters - STAGE 1: PRE-TRAINING

# --- Configuration ---
GPU_DEVICES="0"
ADAPTER_SIZE=64 # Fixed adapter size for this experiment
BASE_OUTPUT_DIR="ablation_study_output/fac_adapter_number_size${ADAPTER_SIZE}"

# Define the adapter lists to test (number of adapters)
# The 3-adapter baseline ("0,11,22") is skipped as it was run in experiment 1.
ADAPTER_LISTS=(
    "23" 
    "11,23" 
    "0,4,8,12,16,20" 
    "0,2,4,6,8,10,12,14,16,18,20,22"
)

# --- Pre-training Loop ---
for AD_LIST in "${ADAPTER_LISTS[@]}"
do
  NUM_ADAPTERS=$(echo "$AD_LIST" | tr -cd ',' | wc -c)
  NUM_ADAPTERS=$((NUM_ADAPTERS + 1))
  echo "----------------------------------------------------"
  echo "PRE-TRAINING for ${NUM_ADAPTERS} adapter(s): ${AD_LIST}"
  echo "----------------------------------------------------"

  RUN_COMMENT="fac-adapter-num-${NUM_ADAPTERS}"
  RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_num_${NUM_ADAPTERS}"

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
        --adapter_list "$AD_LIST" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model=""
done

echo "----------------------------------------------------"
echo "Number of adapters ablation pre-training finished."
echo "Results are in ${BASE_OUTPUT_DIR}"
echo "----------------------------------------------------"
