#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# --- Fine-tuning script for OpenEntity using a custom pre-trained Adapter (size 16) ---

task=entity_type
GPU_DEVICE="0"

# Adapter 配置 (必须与预训练时完全一致)
ADAPTER_SIZE=16
ADAPTER_LIST="0,11,22"
ADAPTER_SKIP_LAYERS=0

# --- 配置参数 ---
# 预训练好的 Factual Adapter 路径
PRETRAINED_FAC_ADAPTER="./ablation_study_output/fac_adapter_size_sub1e2/run_size_${ADAPTER_SIZE}/trex_maxlen-64_batch-16_lr-5e-05_warmup-1200_epoch-5_fac-adapter-size-${ADAPTER_SIZE}-sub1e2/pytorch_model.bin"

# 微调实验的输出目录
OUTPUT_DIR="./outputs_light/openentity_finetune_size16"



# --- 执行微调 ---
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=data/OpenEntity \
    --output_dir=$OUTPUT_DIR  \
    --comment "finetune-fac-adapter-size-$ADAPTER_SIZE" \
    --max_seq_length=256  \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4   \
    --learning_rate=1e-4 \
    --gradient_accumulation_steps=1 \
    --max_steps=12000  \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=120 \
    --save_steps=1000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size $ADAPTER_SIZE \
    --adapter_list "$ADAPTER_LIST" \
    --adapter_skip_layers $ADAPTER_SKIP_LAYERS \
    --meta_fac_adaptermodel="$PRETRAINED_FAC_ADAPTER" 
