#!/bin/bash
# Fine-tuning for MIDDLE position: 10,11,12

# --- Configuration ---
GPU_DEVICE="0"
ADAPTER_SIZE=64
POS="10,11,12"
POS_NAME="10_11_12"

PRETRAIN_BASE_DIR="./ablation_study_output/fac_adapter_position_size${ADAPTER_SIZE}"
FINETUNE_BASE_DIR="./outputs_light/openentity_finetune_position_size${ADAPTER_SIZE}"

# --- Path Construction ---
PRETRAINED_ADAPTER_PATH="${PRETRAIN_BASE_DIR}/run_pos_${POS_NAME}/trex_maxlen-64_batch-16_lr-5e-05_warmup-1200_epoch-5_fac-adapter-pos-${POS_NAME}/pytorch_model.bin"
OUTPUT_DIR="${FINETUNE_BASE_DIR}/finetune_pos_${POS_NAME}"

# --- Run Fine-tuning ---
echo "Running fine-tuning for position: ${POS_NAME}"

if [ ! -f "$PRETRAINED_ADAPTER_PATH" ]; then
    echo "ERROR: Pre-trained adapter not found at: $PRETRAINED_ADAPTER_PATH"
    exit 1
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
