# Replication script for Table 4
# NVFP4 KV cache PPL results across KV block sizes

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp4_kv.yaml"
MODEL_CONFIG = "config/llama3-8b_wiki.yaml"
OUT_DIR = ROOT / "save/KV_block_size"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

KV_BLOCK_SIZES = [8, 16, 32, 64, 128]

OUT_DIR.mkdir(parents=True, exist_ok=True)
base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
model_cfg = yaml.full_load((ROOT / MODEL_CONFIG).read_text())
log_name = model_cfg["save"]["logger"]
results = {}

for kv_block_size in KV_BLOCK_SIZES:
    quant_name = f"nvfp4_kv_b{kv_block_size}"
    run_dir = OUT_DIR / quant_name
    log_file = run_dir / log_name

    if log_file.exists():
        print(f"[{quant_name}] Skipping; log exists: {log_file}")
        matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
        results[kv_block_size] = float(matches[-1]) if matches else None
        continue

    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["quantization"]["xqtype"] = "mxfp"
    cfg["quantization"]["wqtype"] = "mxfp"
    cfg["quantization"]["quant_4b_kv"] = True
    cfg["quantization"]["quant_post_rope_q"] = True
    cfg["quantization"]["quant_k_cache"] = True
    cfg["quantization"]["quant_v_cache"] = True
    cfg["quantization"]["quant_attn_wgt"] = True
    cfg["block_quant"]["kv_block_size"] = kv_block_size

    quant_config = OUT_DIR / f"{quant_name}.yaml"
    quant_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cmd = [
        "python",
        "main.py",
        "--config_dir",
        MODEL_CONFIG,
        "--quant_config",
        str(quant_config.relative_to(ROOT)),
        "--save_dir",
        str(run_dir),
    ]
    print(f"[{quant_name}] " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
    results[kv_block_size] = float(matches[-1]) if matches else None

headers = ["KV block size", "PPL"]
rows = [
    [str(kv_block_size), f"{results[kv_block_size]:.4f}"
     if results[kv_block_size] is not None else "N/A"]
    for kv_block_size in KV_BLOCK_SIZES
]

tsv_file = OUT_DIR / "results.tsv"
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([headers, *rows])

widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
print("\nKV block size PPL results")
print(separator)
print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
print(separator)
for row in rows:
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
print(separator)
print(f"Saved TSV: {tsv_file}")
