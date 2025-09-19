#!/bin/bash

# =====================================================================================
# WARNING: This script is for an experimental run to pre-train an adapter
# on the TACRED dataset. TACRED is a downstream evaluation dataset, and using
# it for pre-training can lead to overfitting and methodologically questionable
# results. This is intended for exploration under severe resource constraints.
# =====================================================================================

# Pre-train a new adapter on TACRED data

task=trex
GPU='0' # Assuming a single GPU for this run, modify if you have more
CUDA_VISIBLE_DEVICES=$GPU python fac-adapter.py  \
        --model_type roberta \
        --model_name=roberta-large  \
        --data_dir=./data/tacred_processed_for_pretrain  \
        --output_dir=./pretrained_models/tacred_pretrained_adapter \
        --restore '' \
        --do_train  \
        --do_eval   \
        --evaluate_during_training 'True' \
        --task_name=$task     \
        --comment 'pretrain-on-tacred' \
        --per_gpu_train_batch_size=16   \
        --per_gpu_eval_batch_size=16   \
        --num_train_epochs 5 \
        --max_seq_length 128 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --warmup_steps=200 \
        --save_steps 1000 \
        --logging_steps 200 \
        --adapter_size 768 \
        --adapter_list "0,11,22" \
        --adapter_skip_layers 0 \
        --adapter_transformer_layers 2 \
        --meta_adapter_model=""
