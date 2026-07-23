# Replication script for Table 2
# NVFP4 PPL results with different FP8 scaling formats

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp4.yaml"
OUT_DIR = ROOT / "save/nvfp4_scaling_ppl"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

# Fill this bracket with the model & task configs you want to run.
MODEL_CONFIGS = [
    "config/llama2-7b_wiki.yaml",
    "config/llama3-8b_wiki.yaml",
    "config/llama3.1-8b_wiki.yaml",
]

# Fill this bracket with the scale formats you want to run:
# (name, scale exponent bits, scale mantissa bits, use per-tensor scale)
SCALES = [
    ("e4m3", 4, 3, False),
    ("e4m3_per_tensor", 4, 3, True),
    ("e5m2", 5, 2, False),
    ("e5m3", 5, 3, False),
]

def read_ppl(log_file):
    """Return the last Wikitext2 PPL in a log, or None if none was recorded."""
    matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def print_results(results):
    headers = ["Model", *(scale[0] for scale in SCALES)]
    rows = []
    for model_config in MODEL_CONFIGS:
        model_name = Path(model_config).stem.replace("_wiki", "")
        rows.append([
            model_name,
            *(f"{results[(model_name, scale[0])]:.4f}"
              if results[(model_name, scale[0])] is not None else "N/A"
              for scale in SCALES),
        ])

    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    tsv_file = OUT_DIR / "results.tsv"
    with tsv_file.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerows([headers, *rows])

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print("\nNVFP4 scaling PPL results")
    print(separator)
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
    print(separator)
    print(f"Saved TSV: {tsv_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
    results = {}

    for model_config in MODEL_CONFIGS:
        model_name = Path(model_config).stem.replace("_wiki", "")
        model_cfg = yaml.full_load((ROOT / model_config).read_text())
        log_name = model_cfg["save"]["logger"]

        for scale_name, sc_ebit, sc_mbit, per_tensor in SCALES:
            run_dir = OUT_DIR / f"{model_name}_{scale_name}"
            log_file = run_dir / log_name

            if log_file.exists():
                print(f"[{model_name} {scale_name}] Skipping; log exists: {log_file}")
            else:
                cfg = yaml.safe_load(yaml.safe_dump(base))
                cfg["block_quant"]["act_sc_ebit"] = sc_ebit
                cfg["block_quant"]["act_sc_mbit"] = sc_mbit
                cfg["block_quant"]["wgt_sc_ebit"] = sc_ebit
                cfg["block_quant"]["wgt_sc_mbit"] = sc_mbit
                cfg["block_quant"]["act_per_tensor_scale"] = per_tensor
                cfg["block_quant"]["wgt_per_tensor_scale"] = per_tensor

                quant_config = OUT_DIR / f"nvfp4_{scale_name}.yaml"
                quant_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

                cmd = [
                    "python",
                    "main.py",
                    "--config_dir",
                    model_config,
                    "--quant_config",
                    str(quant_config.relative_to(ROOT)),
                    "--save_dir",
                    str(run_dir),
                ]
                print(f"[{model_name} {scale_name}] " + " ".join(cmd))
                subprocess.run(cmd, cwd=ROOT, check=True)

            results[(model_name, scale_name)] = read_ppl(log_file) if log_file.exists() else None

    print_results(results)



if __name__ == "__main__":
    main()
