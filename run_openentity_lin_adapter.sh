#!/bin/bash
# Script for fine-tuning with lin-adapter on OpenEntity.
# Uses optimal hyperparameters provided by the user.

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
    --comment 'lin-adapter-finetune' \
    --max_seq_length=256  \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4   \
    --learning_rate=5e-6 \
    --gradient_accumulation_steps=1 \
    --max_steps=12000  \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=1000 \
    --save_steps=1000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
    --fusion_mode 'add'
