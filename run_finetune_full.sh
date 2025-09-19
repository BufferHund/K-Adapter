#!/bin/bash
# Ablation study: Full Fine-tuning vs. Adapter-Tuning

# --- Configuration ---
GPU_DEVICE="0"

# --- Run Full Fine-tuning ---
echo "Running Full Fine-tuning experiment..."

# Note the key differences:
# 1. --freeze_bert is omitted (or set to ""), meaning we train the entire model.
# 2. --meta_fac_adaptermodel is empty, no adapter is loaded.
# 3. Learning rate is reduced to 1e-5, a standard practice for full fine-tuning.
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name entity_type \
    --data_dir data/OpenEntity \
    --output_dir ./outputs_light/openentity_full_finetune \
    --comment "full-finetune" \
    --max_seq_length 256  \
    --per_gpu_eval_batch_size 4   \
    --per_gpu_train_batch_size 4   \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --max_steps 12000  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps 120 \
    --save_steps 1000 \
    --freeze_bert="" \
    --meta_fac_adaptermodel="" \
    --meta_lin_adaptermodel=""
