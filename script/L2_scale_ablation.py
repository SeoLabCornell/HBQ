# W4A5 HBQ L2-scale ablation on WikiText

import csv
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/hbq_e.yaml"
OUT_DIR = ROOT / "save/L2_scale_ablation"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

MODEL_CONFIGS = [
    ("llama3-8b", "config/llama3-8b_wiki.yaml"),
    ("mixtral-8x7b", "save/L2_scale_ablation/model_configs/mixtral-8x7b_wiki.yaml"),
]

# (name, quantizer type, activation L2 scheme, weight L2 scheme)
ACT_SCALE_SCHEMES = [
    ("L1", "mxfp", None, None),
    ("L2-PoT", "hbq", "PoT", None),
    ("L2-INT", "hbq", "INT", None),
    ("L2-SIG", "hbq", "SIG-1", None),
]
WGT_SCALE_SCHEMES = [
    ("L1", "mxfp", None, None),
    ("L2-PoT", "hbq", None, "PoT"),
    ("L2-INT", "hbq", None, "INT"),
    ("L2-SIG", "hbq", None, "Mix"),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "model_configs").mkdir(exist_ok=True)

(OUT_DIR / "model_configs" / "mixtral-8x7b_wiki.yaml").write_text(
    yaml.safe_dump(
        {
            "model": {"model_type": "mistralai/Mixtral-8x7B-v0.1"},
            "eval": {"tasks": ["wikitext"]},
            "save": {"run_dir": str(OUT_DIR / "mixtral-8x7b"), "logger": "wiki.log"},
        },
        sort_keys=False,
    )
)

base = yaml.full_load(BASE_QUANT_CONFIG.read_text())
results = {}

for model_name, model_config in MODEL_CONFIGS:
    model_cfg = yaml.full_load((ROOT / model_config).read_text())
    log_name = model_cfg["save"]["logger"]

    for act_name, xqtype, act_l2_scheme, _ in ACT_SCALE_SCHEMES:
        for wgt_name, wqtype, _, wgt_l2_scheme in WGT_SCALE_SCHEMES:
            quant_name = f"act_{act_name}_wgt_{wgt_name}"
            run_name = f"{model_name}_{quant_name}"
            run_dir = OUT_DIR / run_name
            log_file = run_dir / log_name

            if log_file.exists():
                print(f"[{run_name}] Skipping; log exists: {log_file}")
                matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
                results[(model_name, act_name, wgt_name)] = float(matches[-1]) if matches else None
                continue

            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["quantization"]["xqtype"] = xqtype
            cfg["quantization"]["wqtype"] = wqtype

            cfg["block_quant"]["act_ebit"] = 2
            cfg["block_quant"]["act_mbit"] = 2
            cfg["block_quant"]["act_block_size"] = 128
            cfg["block_quant"]["act_l2_block_size"] = 32

            cfg["block_quant"]["wgt_ebit"] = 2
            cfg["block_quant"]["wgt_mbit"] = 1
            cfg["block_quant"]["wgt_block_size"] = 128
            cfg["block_quant"]["wgt_l2_block_size"] = 32

            if act_l2_scheme is not None:
                cfg["block_quant"]["act_l2_scheme"] = act_l2_scheme
            if wgt_l2_scheme is not None:
                cfg["block_quant"]["wgt_l2_scheme"] = wgt_l2_scheme

            quant_config = OUT_DIR / f"{quant_name}.yaml"
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
            print(f"[{run_name}] TORCH_COMPILE_DISABLE=1 " + " ".join(cmd))
            env = os.environ.copy()
            env["TORCH_COMPILE_DISABLE"] = "1"

            subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                check=True,
            )
            matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
            results[(model_name, act_name, wgt_name)] = float(matches[-1]) if matches else None

tsv_rows = []
for model_name, _ in MODEL_CONFIGS:
    headers = ["Activation \\ Weight", *(scheme[0] for scheme in WGT_SCALE_SCHEMES)]
    rows = []
    for act_name, *_ in ACT_SCALE_SCHEMES:
        rows.append([
            act_name,
            *(f"{results[(model_name, act_name, wgt_scheme[0])]:.4f}"
              if results[(model_name, act_name, wgt_scheme[0])] is not None else "N/A"
              for wgt_scheme in WGT_SCALE_SCHEMES),
        ])
    tsv_rows.extend([[model_name, *row] for row in rows])

    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(f"\n{model_name} L2-scale ablation PPL results")
    print(separator)
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
    print(separator)

tsv_file = OUT_DIR / "results.tsv"
tsv_headers = ["Model", "Activation \\ Weight", *(scheme[0] for scheme in WGT_SCALE_SCHEMES)]
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([tsv_headers, *tsv_rows])
print(f"\nSaved TSV: {tsv_file}")
