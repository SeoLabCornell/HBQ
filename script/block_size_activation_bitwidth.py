# W4A4-W4A8 PPL results across activation bitwidths and block-size/scale settings

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp4.yaml"
MODEL_CONFIG = "config/llama3-8b_wiki.yaml"
OUT_DIR = ROOT / "save/block_size_activation_bitwidth"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

# (name, activation exponent bits, activation mantissa bits, activation bitwidth)
ACT_FORMATS = [
    ("W4A4", 2, 1, 4),
    ("W4A5", 2, 2, 5),
    ("W4A6", 2, 3, 6),
    ("W4A7", 2, 4, 7),
    ("W4A8", 2, 5, 8),
]

# (name, scale exponent bits, scale mantissa bits, block size, use qk round)
SCALE_BLOCK_SETTINGS = [
    ("fp8_b16", 5, 3, 16, False),
    ("fp8_b32", 5, 3, 32, False),
    ("fp8_b64", 5, 3, 64, False),
    ("fp8_b128", 5, 3, 128, False),
    ("pot_b32", 8, 0, 32, True),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
model_cfg = yaml.full_load((ROOT / MODEL_CONFIG).read_text())
log_name = model_cfg["save"]["logger"]
results = {}

for act_name, act_ebit, act_mbit, act_bitwidth in ACT_FORMATS:
    for scale_name, sc_ebit, sc_mbit, block_size, qk_round in SCALE_BLOCK_SETTINGS:
        quant_name = f"{act_name}_{scale_name}"
        run_dir = OUT_DIR / quant_name
        log_file = run_dir / log_name

        if log_file.exists():
            print(f"[{quant_name}] Skipping; log exists: {log_file}")
            matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
            results[(act_name, scale_name)] = float(matches[-1]) if matches else None
            continue

        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["quantization"]["xqtype"] = "mxfp"
        cfg["quantization"]["wqtype"] = "mxfp"
        cfg["block_quant"]["wgt_ebit"] = 2
        cfg["block_quant"]["wgt_mbit"] = 1
        cfg["block_quant"]["act_ebit"] = act_ebit
        cfg["block_quant"]["act_mbit"] = act_mbit
        cfg["block_quant"]["act_block_size"] = block_size
        cfg["block_quant"]["wgt_block_size"] = block_size
        cfg["block_quant"]["act_sc_ebit"] = sc_ebit
        cfg["block_quant"]["act_sc_mbit"] = sc_mbit
        cfg["block_quant"]["wgt_sc_ebit"] = sc_ebit
        cfg["block_quant"]["wgt_sc_mbit"] = sc_mbit
        cfg["block_quant"]["act_scale_use_ceil"] = scale_name.startswith("fp8") and act_bitwidth >= 6
        cfg["block_quant"]["wgt_scale_use_ceil"] = False
        cfg["block_quant"]["qk_mx_round"] = qk_round

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
        results[(act_name, scale_name)] = float(matches[-1]) if matches else None

headers = ["Activation format", *(setting[0] for setting in SCALE_BLOCK_SETTINGS)]
rows = []
for act_name, *_ in ACT_FORMATS:
    rows.append([
        act_name,
        *(f"{results[(act_name, setting[0])]:.4f}"
          if results[(act_name, setting[0])] is not None else "N/A"
          for setting in SCALE_BLOCK_SETTINGS),
    ])

tsv_file = OUT_DIR / "results.tsv"
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([headers, *rows])

widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
print("\nActivation bitwidth/block-size PPL results")
print(separator)
print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
print(separator)
for row in rows:
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
print(separator)
print(f"Saved TSV: {tsv_file}")
