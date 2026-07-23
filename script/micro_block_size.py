# HBQ-A PPL results across L2 micro block sizes

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/hbq_a.yaml"
MODEL_CONFIG = "config/llama3-8b_wiki.yaml"
OUT_DIR = ROOT / "save/micro_block_size"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

L2_BLOCK_SIZES = [4, 8, 16, 32, 64]

OUT_DIR.mkdir(parents=True, exist_ok=True)
base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
model_cfg = yaml.full_load((ROOT / MODEL_CONFIG).read_text())
log_name = model_cfg["save"]["logger"]
results = {}

for l2_block_size in L2_BLOCK_SIZES:
    quant_name = f"hbq_a_l2b{l2_block_size}"
    run_dir = OUT_DIR / quant_name
    log_file = run_dir / log_name

    if log_file.exists():
        print(f"[{quant_name}] Skipping; log exists: {log_file}")
        matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
        results[l2_block_size] = float(matches[-1]) if matches else None
        continue

    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["block_quant"]["wgt_l2_block_size"] = l2_block_size
    cfg["block_quant"]["act_l2_block_size"] = l2_block_size

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
    results[l2_block_size] = float(matches[-1]) if matches else None

headers = ["L2 micro block size", "PPL"]
rows = [
    [str(l2_block_size), f"{results[l2_block_size]:.4f}"
     if results[l2_block_size] is not None else "N/A"]
    for l2_block_size in L2_BLOCK_SIZES
]

tsv_file = OUT_DIR / "results.tsv"
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([headers, *rows])

widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
print("\nL2 micro block size PPL results")
print(separator)
print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
print(separator)
for row in rows:
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
print(separator)
print(f"Saved TSV: {tsv_file}")
