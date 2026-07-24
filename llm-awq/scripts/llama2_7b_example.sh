#!/bin/bash
# AWQ baseline (W4, group size 128) for Llama-2-7B.
# Tasks: wikitext, winogrande, piqa
#
# Run from the llm-awq directory:
#     bash scripts/llama2_7b_example.sh
set -euo pipefail

# The model fits on a single GPU in fp16; run everything on one device.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=meta-llama/Llama-2-7b-hf
MODEL=llama-2-7b
AWQ_CACHE=awq_cache/$MODEL-w4-g128.pt
RESULTS=results/$MODEL-w4-g128-awq

# AWQ search (skipped automatically if $AWQ_CACHE already exists)
python -m awq.entry --model_path $MODEL_PATH \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq $AWQ_CACHE

# Evaluate with simulated quantization: INT4/g128 weights, fp16 activations
python -m awq.entry --model_path $MODEL_PATH \
    --tasks wikitext --batch_size 1 \
    --w_bit 4 --q_group_size 128 \
    --load_awq $AWQ_CACHE \
    --q_backend fake \
    --output_path $RESULTS/wikitext.json

python -m awq.entry --model_path $MODEL_PATH \
    --tasks winogrande --batch_size 8 \
    --w_bit 4 --q_group_size 128 \
    --load_awq $AWQ_CACHE \
    --q_backend fake \
    --output_path $RESULTS/winogrande.json

python -m awq.entry --model_path $MODEL_PATH \
    --tasks piqa --batch_size 8 \
    --w_bit 4 --q_group_size 128 \
    --load_awq $AWQ_CACHE \
    --q_backend fake \
    --output_path $RESULTS/piqa.json
