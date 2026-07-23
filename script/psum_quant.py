# Quantization and accumulation-kernel evaluation matrix

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "save/psum_quant"
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PPL_PATTERN = re.compile(rf"Wikitext2 PPL:\s*({NUMBER_PATTERN})")

EVALUATIONS = {
    "wikitext": ("llama3-8b", ROOT / "config/llama3-8b_wiki.yaml"),
    "gsm8k": ("llama3.1-8b-ins", ROOT / "config/llama3.1-8b-ins_gsm8k.yaml"),
}

QUANT_CONFIGS = {
    "baseline": {
        "fp32": ROOT / "config/baseline.yaml",
        "fp16": ROOT / "config/fp16_accum.yaml",
        "psum_quant": ROOT / "config/psum_quant.yaml",
    },
    "nvfp4": {
        "fp32": ROOT / "config/nvfp4.yaml",
        "fp16": ROOT / "config/nvfp4_fp16.yaml",
        "psum_quant": ROOT / "config/nvfp4_psum_quant.yaml",
    },
    "hbq_e": {
        "fp32": ROOT / "config/hbq_e.yaml",
        "fp16": ROOT / "config/hbq_e_fp16.yaml",
        "psum_quant": ROOT / "config/hbq_e_psum_quant.yaml",
    },
}


def parse_result(log_file, evaluation):
    """Return the last PPL or GSM8K strict exact-match value in a log."""
    text = log_file.read_text(errors="replace")
    if evaluation == "wikitext":
        matches = PPL_PATTERN.findall(text)
    else:
        pattern = re.compile(
            rf"gsm8k_cot_llama result:.*?'gsm8k_cot_llama': "
            rf"\{{.*?'exact_match,strict-match':\s*(?:np\.float64\()?({NUMBER_PATTERN})"
        )
        matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def print_and_save_results(results):
    headers = ["Quantization", "fp32", "fp16", "psum_quant"]
    tsv_rows = []

    for evaluation, (model_name, _) in EVALUATIONS.items():
        rows = []
        for quant_name, kernel_configs in QUANT_CONFIGS.items():
            row = [quant_name]
            for kernel_name in kernel_configs:
                value = results[(evaluation, quant_name, kernel_name)]
                row.append(f"{value:.4f}" if value is not None else "N/A")
            rows.append(row)
        tsv_rows.extend([[evaluation, model_name, *row] for row in rows])

        widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        metric = "PPL" if evaluation == "wikitext" else "exact_match,strict-match"
        print(f"\n{model_name} {evaluation} results ({metric})")
        print(separator)
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
        print(separator)
        for row in rows:
            print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
        print(separator)

    tsv_file = OUT_DIR / "results.tsv"
    tsv_headers = ["Evaluation", "Model", *headers]
    with tsv_file.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerows([tsv_headers, *tsv_rows])
    print(f"\nSaved TSV: {tsv_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for evaluation, (model_name, model_config) in EVALUATIONS.items():
        model_cfg = yaml.full_load(model_config.read_text())
        log_name = model_cfg["save"]["logger"]

        for quant_name, kernel_configs in QUANT_CONFIGS.items():
            for kernel_name, quant_config in kernel_configs.items():
                run_dir = OUT_DIR / model_name / evaluation / quant_name / kernel_name
                log_file = run_dir / log_name

                if log_file.exists():
                    print(
                        f"[{model_name} {evaluation} {quant_name} {kernel_name}] "
                        f"Skipping; log exists: {log_file}"
                    )
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
                    print(
                        f"[{model_name} {evaluation} {quant_name} {kernel_name}] "
                        + " ".join(cmd)
                    )
                    subprocess.run(cmd, cwd=ROOT, check=True)

                results[(evaluation, quant_name, kernel_name)] = (
                    parse_result(log_file, evaluation) if log_file.exists() else None
                )

    print_and_save_results(results)


if __name__ == "__main__":
    main()
