#!/bin/bash
# Script for standard fine-tuning (baseline) on OpenEntity.

task=entity_type
python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --config_name roberta-large \
    --do_train  \
    --do_eval   \
    --task_name=$task     \
    --data_dir=data/OpenEntity \
    --output_dir=./proc_data  \
    --comment 'baseline-finetune' \
    --max_seq_length=256  \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate=2e-5 \
    --gradient_accumulation_steps=1 \
    --max_steps=12000  \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=120 \
    --save_steps=1000 \
    --freeze_bert="" \
    --freeze_adapter="" \
    --fusion_mode 'add'
