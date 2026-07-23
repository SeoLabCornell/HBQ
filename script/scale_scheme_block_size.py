# Replication script for Figure 3
# NVFP4 PPL results with different scaling scheme (PoT/FP8) across different block sizes

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp4.yaml"
MODEL_CONFIG = "config/llama3-8b_wiki.yaml"
OUT_DIR = ROOT / "save/scale_scheme_block_size"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

# iterate different block sizes and scale schemes
BLOCK_SIZES = [4, 8, 16, 32, 64, 128]

# (name, scale exponent bits, scale mantissa bits, use per-tensor scale, use round for PoT q/k scale)
SCALE_SCHEMES = [
    ("fp8_e5m3", 5, 3, False, False),
    ("pot_e8m0_floor", 8, 0, False, False),
    ("pot_e8m0_round", 8, 0, False, True),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
model_cfg = yaml.full_load((ROOT / MODEL_CONFIG).read_text())
log_name = model_cfg["save"]["logger"]
results = {}

for block_size in BLOCK_SIZES:
    for scale_name, sc_ebit, sc_mbit, per_tensor, use_round in SCALE_SCHEMES:
        run_dir = OUT_DIR / f"{scale_name}_b{block_size}"
        log_file = run_dir / log_name

        if log_file.exists():
            print(f"[{scale_name} block{block_size}] Skipping; log exists: {log_file}")
            matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
            results[(block_size, scale_name)] = float(matches[-1]) if matches else None
            continue

        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["block_quant"]["act_block_size"] = block_size
        cfg["block_quant"]["wgt_block_size"] = block_size
        cfg["block_quant"]["act_sc_ebit"] = sc_ebit
        cfg["block_quant"]["act_sc_mbit"] = sc_mbit
        cfg["block_quant"]["wgt_sc_ebit"] = sc_ebit
        cfg["block_quant"]["wgt_sc_mbit"] = sc_mbit
        cfg["block_quant"]["act_per_tensor_scale"] = per_tensor
        cfg["block_quant"]["wgt_per_tensor_scale"] = per_tensor
        cfg["block_quant"]["qk_mx_round"] = use_round

        quant_config = OUT_DIR / f"nvfp4_{scale_name}_b{block_size}.yaml"
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
        print(f"[{scale_name} block{block_size}] " + " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
        matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
        results[(block_size, scale_name)] = float(matches[-1]) if matches else None

headers = ["Block size", *(scheme[0] for scheme in SCALE_SCHEMES)]
rows = []
for block_size in BLOCK_SIZES:
    rows.append([
        str(block_size),
        *(f"{results[(block_size, scheme[0])]:.4f}"
          if results[(block_size, scheme[0])] is not None else "N/A"
          for scheme in SCALE_SCHEMES),
    ])

widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
print("\nScaling scheme/block size PPL results")
print(separator)
tsv_file = OUT_DIR / "results.tsv"
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([headers, *rows])

print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
print(separator)
for row in rows:
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
print(separator)
print(f"Saved TSV: {tsv_file}")
