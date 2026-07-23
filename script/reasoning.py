# Reasoning benchmark sweep across quantization schemes

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "save/reasoning"
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

MODELS = {
    "llama3.1-8b-ins": "llama3.1-8b-ins",
    "llama3.2-3b-ins": "llama3.2-3b-ins",
}

TASK_METRICS = {
    "gsm8k": ("gsm8k_cot_llama", "exact_match,strict-match"),
    "humaneval": ("humaneval", "pass@1,create_test"),
    "mmlu": ("mmlu", "acc,none"),
}

QUANT_CONFIGS = {
    "hbq_a_no_kv": ROOT / "config/hbq_a.yaml",
    "hbq_e_no_kv": ROOT / "config/hbq_e.yaml",
    "mxfp_w4a4_kv": ROOT / "config/mxfp4_kv.yaml",
    "mxfp_w4a8_kv": ROOT / "config/mxfp_w4a8_kv.yaml",
    "nvfp_w4a4_kv": ROOT / "config/nvfp4_kv.yaml",
    "nvfp_w4a8_kv": ROOT / "config/nvfp_w4a8_kv.yaml",
    "hbq_e_kv": ROOT / "config/hbq_e_kv.yaml",
    "hbq_a_kv": ROOT / "config/hbq_a_kv.yaml",
}


def parse_result(log_file, task):
    """Return the last configured benchmark metric in a log."""
    logged_task, metric = TASK_METRICS[task]
    pattern = re.compile(
        rf"{re.escape(logged_task)} result:.*?'{re.escape(logged_task)}': "
        rf"\{{.*?'{re.escape(metric)}':\s*"
        rf"(?:np\.float64\()?({NUMBER_PATTERN})"
    )
    matches = pattern.findall(log_file.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def print_and_save_results(results):
    headers = ["Model", *QUANT_CONFIGS]
    tsv_rows = []

    for task, (_, metric) in TASK_METRICS.items():
        rows = []
        for model_name in MODELS:
            row = [model_name]
            for quant_name in QUANT_CONFIGS:
                value = results[(model_name, task, quant_name)]
                row.append(f"{value:.4f}" if value is not None else "N/A")
            rows.append(row)
        tsv_rows.extend([[task, *row] for row in rows])

        widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        print(f"\n{task} results ({metric})")
        print(separator)
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
        print(separator)
        for row in rows:
            print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
        print(separator)

    tsv_file = OUT_DIR / "results.tsv"
    tsv_headers = ["Benchmark", *headers]
    with tsv_file.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerows([tsv_headers, *tsv_rows])
    print(f"\nSaved TSV: {tsv_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name, config_stem in MODELS.items():
        for task in TASK_METRICS:
            model_config = ROOT / f"config/{config_stem}_{task}.yaml"
            model_cfg = yaml.full_load(model_config.read_text())
            log_name = model_cfg["save"]["logger"]

            for quant_name, quant_config in QUANT_CONFIGS.items():
                run_dir = OUT_DIR / model_name / task / quant_name
                log_file = run_dir / log_name

                if log_file.exists():
                    print(f"[{model_name} {task} {quant_name}] Skipping; log exists: {log_file}")
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
                    print(f"[{model_name} {task} {quant_name}] " + " ".join(cmd))
                    subprocess.run(cmd, cwd=ROOT, check=True)

                results[(model_name, task, quant_name)] = (
                    parse_result(log_file, task) if log_file.exists() else None
                )

    print_and_save_results(results)


if __name__ == "__main__":
    main()
