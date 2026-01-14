MODEL=llama3-8b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/llama3-8b-w4-g128-hbq_a.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128-hbq_a.pt \
    --q_backend fake --dump_fake llama3_8B_AWQ_hbq_a/

# generate real quantized weights (w4)
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/llama3-8b-w4-g128.pt \
    --q_backend real --dump_quant quant_cache/llama3-8b-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks winogrande \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/llama3-8b-w4-g128-awq.pt

python -m awq.entry --model_path meta-llama/Meta-Llama-3-8B \
    --tasks piqa \
    --w_bit 4 --q_group_size 128 \
    --load_quant quant_cache/llama3-8b-w4-g128-awq.pt