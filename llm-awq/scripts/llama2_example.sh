MODEL=llama-2-7b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama-2-7b-w4-g128-hbq_a.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama-2-7b-w4-g128-hbq_a.pt \
    --q_backend fake --dump_fake llama2_7B_AWQ_hbq_a/

# generate real quantized weights (w4)
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama-2-7b-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/llama-2-7b-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks winogrande \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/llama-2-7b-w4-g128-awq.pt

python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --tasks piqa \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/llama-2-7b-w4-g128-awq.pt