#!/bin/bash
# Ablation study on Adapter insertion position - STAGE 2: FINE-TUNING

# --- Configuration ---
GPU_DEVICE="0"
ADAPTER_SIZE=64 # Must be the same as in the pre-training script

# Base directory where the pre-trained adapters were saved
PRETRAIN_BASE_DIR="./ablation_study_output/fac_adapter_position_size${ADAPTER_SIZE}"
# Base directory for saving the fine-tuning results
FINETUNE_BASE_DIR="./outputs_light/openentity_finetune_position_size${ADAPTER_SIZE}"

# Define the same insertion positions to test
POSITIONS=(
    # "0,11,22" # Sparse (Baseline) - 已在实验一中运行，跳过
    "0,1,2"   # Early
    "10,11,12" # Middle
    "21,22,23"  # Late
)

# --- Fine-tuning Loop ---
for POS in "${POSITIONS[@]}"
do
  POS_NAME=$(echo "$POS" | tr ',' '_')
  echo "----------------------------------------------------"
  echo "FINE-TUNING for position: ${POS_NAME}"
  echo "----------------------------------------------------"

  # Construct the path to the specific pre-trained adapter
  PRETRAINED_ADAPTER_PATH="${PRETRAIN_BASE_DIR}/run_pos_${POS_NAME}/trex_maxlen-64_batch-16_lr-5e-05_warmup-1200_epoch-5_fac-adapter-pos-${POS_NAME}/pytorch_model.bin"
  OUTPUT_DIR="${FINETUNE_BASE_DIR}/finetune_pos_${POS_NAME}"

  # Check if the pre-trained adapter exists
  if [ ! -f "$PRETRAINED_ADAPTER_PATH" ]; then
      echo "ERROR: Pre-trained adapter not found at: $PRETRAINED_ADAPTER_PATH"
      echo "Please run the pre-training script first."
      continue
  fi

  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name entity_type \
    --data_dir data/OpenEntity \
    --output_dir $OUTPUT_DIR  \
    --comment "finetune-fac-pos-${POS_NAME}" \
    --max_seq_length 256  \
    --per_gpu_eval_batch_size 4   \
    --per_gpu_train_batch_size 4   \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --max_steps 12000  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps 120 \
    --save_steps 1000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size $ADAPTER_SIZE \
    --adapter_list "$POS" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="$PRETRAINED_ADAPTER_PATH" \
    --meta_lin_adaptermodel=""
done

echo "----------------------------------------------------"
echo "Position ablation fine-tuning finished."
echo "Results are in ${FINETUNE_BASE_DIR}"
echo "----------------------------------------------------"
