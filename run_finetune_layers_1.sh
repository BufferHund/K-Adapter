#!/bin/bash
# Fine-tuning for Adapter with 1 internal Transformer layer

# --- Configuration ---
GPU_DEVICE="0"
ADAPTER_SIZE=64
ADAPTER_LIST="0,11,22"
ADAPTER_LAYERS=1

PRETRAIN_BASE_DIR="./ablation_study_output/fac_adapter_layers_size${ADAPTER_SIZE}"
FINETUNE_BASE_DIR="./outputs_light/openentity_finetune_layers_size${ADAPTER_SIZE}"

# --- Path Construction ---
PRETRAINED_ADAPTER_PATH="${PRETRAIN_BASE_DIR}/run_layers_${ADAPTER_LAYERS}/trex_maxlen-64_batch-16_lr-5e-05_warmup-1200_epoch-5_fac-adapter-layers-${ADAPTER_LAYERS}/pytorch_model.bin"
OUTPUT_DIR="${FINETUNE_BASE_DIR}/finetune_layers_${ADAPTER_LAYERS}"

# --- Run Fine-tuning ---
echo "Running fine-tuning for adapter_transformer_layers = ${ADAPTER_LAYERS}"

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
    --comment "finetune-fac-layers-${ADAPTER_LAYERS}" \
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
    --adapter_list "$ADAPTER_LIST" \
    --adapter_transformer_layers $ADAPTER_LAYERS \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="$PRETRAINED_ADAPTER_PATH" \
    --meta_lin_adaptermodel=""
