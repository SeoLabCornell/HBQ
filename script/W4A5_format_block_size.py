# Replication script for Figure 5
# W4A5 PPL results with different target formats across block sizes

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp_w4a8.yaml"
MODEL_CONFIG = "config/llama3-8b_wiki.yaml"
OUT_DIR = ROOT / "save/W4A5_format_block_size"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

BLOCK_SIZES = [4, 8, 16, 32, 64, 128]

# (name, activation quantizer type, activation exponent bits, activation mantissa bits)
ACT_FORMATS = [
    ("e2m2", "mxfp", 2, 2),
    ("e3m1", "mxfp", 3, 1),
    ("int5", "mxint", 0, 5),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
fp_base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
int_base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
model_cfg = yaml.full_load((ROOT / MODEL_CONFIG).read_text())
log_name = model_cfg["save"]["logger"]
results = {}

for block_size in BLOCK_SIZES:
    for act_name, qtype, act_ebit, act_mbit in ACT_FORMATS:
        wgt_name = "e2m1" if qtype == "mxfp" else "int4"
        quant_name = f"w{wgt_name}_a{act_name}_b{block_size}"
        run_dir = OUT_DIR / quant_name
        log_file = run_dir / log_name

        if log_file.exists():
            print(f"[{quant_name}] Skipping; log exists: {log_file}")
            matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
            results[(block_size, act_name)] = float(matches[-1]) if matches else None
            continue

        if qtype == "mxfp":
            cfg = yaml.safe_load(yaml.safe_dump(fp_base))
            cfg["quantization"]["xqtype"] = "mxfp"
            cfg["quantization"]["wqtype"] = "mxfp"
            cfg["block_quant"]["wgt_ebit"] = 2
            cfg["block_quant"]["wgt_mbit"] = 1
            wgt_name = "e2m1"
        else:
            cfg = yaml.safe_load(yaml.safe_dump(int_base))
            cfg["quantization"]["xqtype"] = "mxint"
            cfg["quantization"]["wqtype"] = "mxint"
            cfg["block_quant"]["wgt_ebit"] = 0
            cfg["block_quant"]["wgt_mbit"] = 4
            wgt_name = "int4"

        cfg["block_quant"]["act_ebit"] = act_ebit
        cfg["block_quant"]["act_mbit"] = act_mbit
        cfg["block_quant"]["act_block_size"] = block_size
        cfg["block_quant"]["wgt_block_size"] = block_size
        cfg["block_quant"]["act_scale_use_ceil"] = False
        cfg["block_quant"]["wgt_scale_use_ceil"] = False
        cfg["block_quant"]["qk_mx_round"] = False

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
        results[(block_size, act_name)] = float(matches[-1]) if matches else None

headers = ["Block size", *(act_format[0] for act_format in ACT_FORMATS)]
rows = []
for block_size in BLOCK_SIZES:
    rows.append([
        str(block_size),
        *(f"{results[(block_size, act_format[0])]:.4f}"
          if results[(block_size, act_format[0])] is not None else "N/A"
          for act_format in ACT_FORMATS),
    ])

tsv_file = OUT_DIR / "results.tsv"
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([headers, *rows])

widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
print("\nW4A5 format/block size PPL results")
print(separator)
print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
print(separator)
for row in rows:
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
print(separator)
print(f"Saved TSV: {tsv_file}")
