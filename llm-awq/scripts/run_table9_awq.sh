#!/bin/bash
# Reproduce the Table 9 "AWQ" row for all five base models, then print a
# summary table (Wikitext-2 PPL + Winogrande/PIQA 0-shot average).
#
# Run from the llm-awq directory:
#     bash scripts/run_table9_awq.sh          # sequential on one GPU
#     PARALLEL=1 bash scripts/run_table9_awq.sh   # one model per GPU (needs 5)
set -uo pipefail
cd "$(dirname "$0")/.."

SCRIPTS=(llama2_7b llama3_8b llama3.2_3b qwen2.5_3b qwen2.5_7b)
MODELS=(llama-2-7b llama3-8b llama3.2-3b qwen2.5-3b qwen2.5-7b)

if [ "${PARALLEL:-0}" = "1" ]; then
    for i in "${!SCRIPTS[@]}"; do
        CUDA_VISIBLE_DEVICES=$i bash "scripts/${SCRIPTS[$i]}_example.sh" \
            > "results/${MODELS[$i]}.log" 2>&1 &
    done
    wait
else
    for s in "${SCRIPTS[@]}"; do
        bash "scripts/${s}_example.sh"
    done
fi

python - <<'EOF'
import json, os

print(f"\n{'Model':<14} {'Wikitext-2 PPL':>14} {'Winogrande':>11} {'PIQA':>7} {'0-shot avg':>11}")
for model in ["llama-2-7b", "llama3-8b", "llama3.2-3b", "qwen2.5-3b", "qwen2.5-7b"]:
    d = f"results/{model}-w4-g128-awq"
    try:
        ppl = json.load(open(f"{d}/wikitext.json"))["ppl"]
        wg = json.load(open(f"{d}/winogrande.json"))["results"]["winogrande"]["acc,none"]
        pq = json.load(open(f"{d}/piqa.json"))["results"]["piqa"]["acc,none"]
        print(f"{model:<14} {ppl:>14.2f} {wg*100:>11.1f} {pq*100:>7.1f} {(wg+pq)/2*100:>11.1f}")
    except FileNotFoundError as e:
        print(f"{model:<14}  INCOMPLETE (missing {os.path.basename(e.filename)})")
EOF
