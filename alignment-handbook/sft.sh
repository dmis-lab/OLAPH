#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes 4 scripts/run_sft.py recipes/biomistral_7b/sft/config_full.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes 4 scripts/run_sft.py recipes/mistral_7b/sft/config_full.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes 4 scripts/run_sft.py recipes/llama2_7b/sft/config_full.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes 4 scripts/run_sft.py recipes/meditron_7b/sft/config_full.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml  --num_processes 4 scripts/run_sft.py recipes/selfbiorag_7b/sft/config_full.yaml