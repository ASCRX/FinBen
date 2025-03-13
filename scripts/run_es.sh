#!/bin/bash

source .env
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export CUDA_VISIBLE_DEVICES=2,3

# Array of models
MODELS=(
    "Qwen/QwQ-32B"
    # Add other Spanish-compatible models here if needed
)

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "Evaluating model: $MODEL"

    # 1024 tokens
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=1024" \
        --tasks es \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-es,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/spanish

    # 8192 tokens
    lm_eval --model vllm \
        --model_args "pretrained=$MODEL,tensor_parallel_size=2,gpu_memory_utilization=0.8,max_length=8192" \
        --tasks es_long \
        --batch_size auto \
        --output_path ../results \
        --hf_hub_log_args "hub_results_org=TheFinAI,details_repo_name=lm-eval-results-es,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" \
        --log_samples \
        --apply_chat_template \
        --include_path ../tasks/spanish

    echo "Finished evaluating model: $MODEL"
done
