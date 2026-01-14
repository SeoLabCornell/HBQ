MODEL=qwen2.5-3b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path Qwen/Qwen2.5-3B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/qwen2.5-3b-w4-g128-hbq_a.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path Qwen/Qwen2.5-3B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/qwen2.5-3b-w4-g128-hbq_a.pt \
    --q_backend fake --dump_fake qwen2.5_3B_AWQ_hbq_a/




# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path Qwen/Qwen2.5-7B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/qwen2.5-7b-w4-g128-hbq_a.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path Qwen/Qwen2.5-7B \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq awq_cache/qwen2.5-7b-w4-g128-hbq_a.pt \
    --q_backend fake --dump_fake qwen2.5_7B_AWQ_hbq_a/






# generate real quantized weights (w4)
# python -m awq.entry --model_path Qwen/Qwen2.5-3B \
#     --w_bit 4 --q_group_size 128 \
#     --load_awq awq_cache/qwen2.5-3b-w4-g128.pt \
#     --q_backend real --dump_quant quant_cache/qwen2.5-3b-w4-g128-awq.pt

# # load and evaluate the real quantized model (smaller gpu memory usage)
# CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path Qwen/Qwen2.5-3B \
#     --tasks wikitext \
#     --w_bit 4 --q_group_size 128 \
#     --load_quant quant_cache/qwen2.5-3b-w4-g128-awq.pt

# CUDA_VISIBLE_DEVICES=2 python -m awq.entry --model_path Qwen/Qwen2.5-7B \
#     --tasks piqa \
#     --w_bit 4 --q_group_size 128 \
#     --load_quant quant_cache/qwen2.5-7b-w4-g128-awq.pt