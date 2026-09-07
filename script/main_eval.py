# Quantization evaluation across models and benchmarks

import csv
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "save/main_eval"
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PPL_PATTERN = re.compile(rf"Wikitext2 PPL:\s*({NUMBER_PATTERN})")

MODELS = [
    ("llama2-7b", "meta-llama/Llama-2-7b-hf"),
    ("llama3-8b", "meta-llama/Meta-Llama-3-8B"),
    ("llama3.1-70b", "meta-llama/Llama-3.1-70B"),
    ("llama3.2-3b", "meta-llama/Llama-3.2-3B"),
    ("qwen2.5-3b", "Qwen/Qwen2.5-3B"),
    ("qwen2.5-7b", "Qwen/Qwen2.5-7B"),
    ("mixtral-8x7b", "mistralai/Mixtral-8x7B-v0.1"),
]

TASKS = ["wikitext", "piqa", "winogrande"]
QUANT_CONFIGS = {
    "mxfp4": ROOT / "config/mxfp4.yaml",
    "nvfp4": ROOT / "config/nvfp4.yaml",
    "hbq_e": ROOT / "config/hbq_e.yaml",
    "hbq_a": ROOT / "config/hbq_a.yaml",
    "vsq_w4a4": ROOT / "config/vsq_w4a4.yaml",
    "mxex": ROOT / "config/mxex4.yaml",
}


def parse_result(log_file, task):
    """Return the last PPL or requested accuracy value recorded for a task."""
    if task == "wikitext":
        matches = PPL_PATTERN.findall(log_file.read_text(errors="replace"))
    else:
        metric = "acc_norm,none" if task == "piqa" else "acc,none"
        pattern = re.compile(
            rf"{re.escape(task)} result:.*?'{re.escape(task)}': "
            rf"\{{.*?'{re.escape(metric)}':\s*({NUMBER_PATTERN})"
        )
        matches = pattern.findall(log_file.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def print_and_save_results(results):
    headers = ["Quantization"]
    for model_name, _ in MODELS:
        headers.extend([f"{model_name} PPL", f"{model_name} 0-shot"])

    rows = []
    for quant_name in QUANT_CONFIGS:
        row = [quant_name]
        for model_name, _ in MODELS:
            ppl = results[(model_name, "wikitext", quant_name)]
            piqa = results[(model_name, "piqa", quant_name)]
            winogrande = results[(model_name, "winogrande", quant_name)]
            zero_shot = (
                (piqa + winogrande) / 2
                if piqa is not None and winogrande is not None else None
            )
            row.extend([
                f"{ppl:.4f}" if ppl is not None else "N/A",
                f"{zero_shot:.4f}" if zero_shot is not None else "N/A",
            ])
        rows.append(row)

    tsv_file = OUT_DIR / "results.tsv"
    with tsv_file.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerows([headers, *rows])

    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nEvaluation results (PPL; 0-shot = mean of PIQA acc_norm and Winogrande acc)")
    print(separator)
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
    print(separator)
    print(f"Saved TSV: {tsv_file}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_config_dir = OUT_DIR / "model_configs"
    model_config_dir.mkdir(exist_ok=True)
    results = {}

    for model_name, model_id in MODELS:
        for task in TASKS:
            model_config = model_config_dir / f"{model_name}_{task}.yaml"
            model_config.write_text(
                yaml.safe_dump(
                    {
                        "model": {"model_type": model_id},
                        "eval": {"tasks": [task]},
                        "save": {
                            "run_dir": str(OUT_DIR / model_name / task),
                            "logger": f"{task}.log",
                        },
                    },
                    sort_keys=False,
                )
            )

            for quant_name, quant_config in QUANT_CONFIGS.items():
                run_dir = OUT_DIR / model_name / task / quant_name
                log_file = run_dir / f"{task}.log"

                if log_file.exists():
                    print(
                        f"[{model_name} {task} {quant_name}] "
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
                    print(f"[{model_name} {task} {quant_name}] " + " ".join(cmd))
                    subprocess.run(cmd, cwd=ROOT, check=True)

                results[(model_name, task, quant_name)] = (
                    parse_result(log_file, task) if log_file.exists() else None
                )
    print_and_save_results(results)


if __name__ == "__main__":
    main()
