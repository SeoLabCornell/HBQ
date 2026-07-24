#!/bin/bash
# AWQ baseline (W4, group size 128) for Llama-3.1-8B-Instruct.
# Tasks: gsm8k_cot_llama, humaneval, mmlu
#
# Run from the llm-awq directory:
#     bash scripts/llama3.1_8b_instruct_example.sh
set -euo pipefail

# The model fits on a single GPU in fp16; run everything on one device.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
MODEL=llama3.1-8b-instruct
AWQ_CACHE=awq_cache/$MODEL-w4-g128.pt
RESULTS=results/$MODEL-w4-g128-awq

# AWQ search (skipped automatically if $AWQ_CACHE already exists)
python -m awq.entry --model_path $MODEL_PATH \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq $AWQ_CACHE

# Evaluate with simulated quantization: INT4/g128 weights, fp16 activations
python -m awq.entry --model_path $MODEL_PATH \
    --tasks gsm8k_cot_llama --batch_size 1 \
    --w_bit 4 --q_group_size 128 \
    --load_awq $AWQ_CACHE \
    --q_backend fake \
    --output_path $RESULTS/gsm8k_cot_llama.json

python -m awq.entry --model_path $MODEL_PATH \
    --tasks humaneval --batch_size 1 \
    --w_bit 4 --q_group_size 128 \
    --load_awq $AWQ_CACHE \
    --q_backend fake \
    --output_path $RESULTS/humaneval.json

python -m awq.entry --model_path $MODEL_PATH \
    --tasks mmlu --batch_size 8 \
    --w_bit 4 --q_group_size 128 \
    --load_awq $AWQ_CACHE \
    --q_backend fake \
    --output_path $RESULTS/mmlu.json
