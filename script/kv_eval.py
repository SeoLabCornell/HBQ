# Wikitext PPL evaluation with quantized KV caches

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "save/kv_eval"
PPL_PATTERN = re.compile(r"Wikitext2 PPL:\s*([0-9]+(?:\.[0-9]+)?)")

MODEL_CONFIGS = {
    "llama2-7b": ROOT / "config/llama2-7b_wiki.yaml",
    "llama3.1-8b": ROOT / "config/llama3.1-8b_wiki.yaml",
    "llama3.2-3b": ROOT / "config/llama3.2-3b_wiki.yaml",
}

QUANT_CONFIGS = {
    "mxfp_w4a4": ROOT / "config/mxfp4_kv.yaml",
    "mxfp_w4a8": ROOT / "config/mxfp_w4a8_kv.yaml",
    "nvfp_w4a4": ROOT / "config/nvfp4_kv.yaml",
    "nvfp_w4a8": ROOT / "config/nvfp_w4a8_kv.yaml",
    "hbq_e": ROOT / "config/hbq_e_kv.yaml",
    "hbq_a": ROOT / "config/hbq_a_kv.yaml",
}


def parse_ppl(log_file):
    """Return the last Wikitext PPL in a log, or None if none was recorded."""
    matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def print_and_save_results(results):
    headers = ["Model", *QUANT_CONFIGS]
    rows = []
    for model_name in MODEL_CONFIGS:
        row = [model_name]
        for quant_name in QUANT_CONFIGS:
            value = results[(model_name, quant_name)]
            row.append(f"{value:.4f}" if value is not None else "N/A")
        rows.append(row)

    tsv_file = OUT_DIR / "results.tsv"
    with tsv_file.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerows([headers, *rows])

    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nQuantized KV-cache Wikitext PPL results")
    print(separator)
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
    print(separator)
    print(f"Saved TSV: {tsv_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name, model_config in MODEL_CONFIGS.items():
        model_cfg = yaml.full_load(model_config.read_text())
        log_name = model_cfg["save"]["logger"]

        for quant_name, quant_config in QUANT_CONFIGS.items():
            run_dir = OUT_DIR / model_name / quant_name
            log_file = run_dir / log_name

            if log_file.exists():
                print(f"[{model_name} {quant_name}] Skipping; log exists: {log_file}")
            else:
                cmd = [
                    "python",
                    "main.py",
                    "--config_dir",
                    str(model_config.relative_to(ROOT)),
                    "--quant_config",
                    str(quant_config.relative_to(ROOT)),
                    "--save_dir",
                    str(run_dir),
                ]
                print(f"[{model_name} {quant_name}] " + " ".join(cmd))
                subprocess.run(cmd, cwd=ROOT, check=True)

            results[(model_name, quant_name)] = (
                parse_ppl(log_file) if log_file.exists() else None
            )

    print_and_save_results(results)


if __name__ == "__main__":
    main()
