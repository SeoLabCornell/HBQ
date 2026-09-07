# Replication script for Table 3
# W4A4/W4A5/W4A8 benchmark sweep across models and tasks

import csv
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp4.yaml"
OUT_DIR = ROOT / "save/activation_bit_benchmark"
QUANT_NAMES = ["W4A4", "W4A5", "W4A8"]
RESULTS_ONLY = "--results-only" in sys.argv

# Fill these brackets with the model ids you want to evaluate for each benchmark.
MODELS = {
    "piqa": [
        "meta-llama/Meta-Llama-3-8B",
        'meta-llama/Llama-3.1-70B',
        'meta-llama/Llama-3.2-3B',
        'Qwen/Qwen2.5-3B',
        'Qwen/Qwen2.5-7B',
        'mistralai/Mixtral-8x7B-v0.1',
    ],
    "winogrande": [
        "meta-llama/Meta-Llama-3-8B",
        'meta-llama/Llama-3.1-70B',
        'meta-llama/Llama-3.2-3B',
        'Qwen/Qwen2.5-3B',
        'Qwen/Qwen2.5-7B',
        'mistralai/Mixtral-8x7B-v0.1',
    ],
    "mmlu": [
        "meta-llama/Meta-Llama-3-8B",
        'meta-llama/Llama-3.2-3B-Instruct'
    ],
    "gsm8k_cot_llama": [
        "meta-llama/Llama-3.1-8B-Instruct",
        'meta-llama/Llama-3.2-3B-Instruct'
    ],
}

# (name, weight format, activation format, use activation ceil)
QUANT_FORMATS = [
    ("W4A4", (2, 1), (2, 1), False),
    ("W4A5", (2, 1), (2, 2), False),
    ("W4A8", (2, 1), (2, 5), True),
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "model_configs").mkdir(exist_ok=True)
(OUT_DIR / "quant_configs").mkdir(exist_ok=True)
base_quant = yaml.full_load(BASE_QUANT_CONFIG.read_text())

for task, model_ids in MODELS.items():
    for model_id in model_ids:
        if RESULTS_ONLY:
            continue

        model_name = model_id.split("/")[-1].replace(".", "_")
        model_config = OUT_DIR / "model_configs" / f"{model_name}_{task}.yaml"
        model_config.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_type": model_id},
                    "eval": {"tasks": [task]},
                    "save": {"run_dir": str(OUT_DIR / model_name / task), "logger": f"{task}.log"},
                },
                sort_keys=False,
            )
        )

        for quant_name, wgt_format, act_format, act_use_ceil in QUANT_FORMATS:
            save_dir = OUT_DIR / model_name / task / quant_name
            log_file = save_dir / f"{task}.log"

            if log_file.exists():
                print(f"[{model_name} {task} {quant_name}] Skipping; log exists: {log_file}")
                continue

            cfg = yaml.safe_load(yaml.safe_dump(base_quant))
            cfg["quantization"]["xqtype"] = "mxfp"
            cfg["quantization"]["wqtype"] = "mxfp"
            cfg["block_quant"]["wgt_ebit"] = wgt_format[0]
            cfg["block_quant"]["wgt_mbit"] = wgt_format[1]
            cfg["block_quant"]["act_ebit"] = act_format[0]
            cfg["block_quant"]["act_mbit"] = act_format[1]
            cfg["block_quant"]["act_block_size"] = 64
            cfg["block_quant"]["wgt_block_size"] = 64
            cfg["block_quant"]["act_scale_use_ceil"] = act_use_ceil
            cfg["block_quant"]["wgt_scale_use_ceil"] = False

            quant_config = OUT_DIR / "quant_configs" / f"{quant_name}.yaml"
            quant_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

            cmd = [
                "python",
                "main.py",
                "--config_dir",
                str(model_config.relative_to(ROOT)),
                "--quant_config",
                str(quant_config.relative_to(ROOT)),
                "--save_dir",
                str(save_dir),
            ]
            print(f"[{model_name} {task} {quant_name}] " + " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def read_metric(log_file, task):
    """Return the last requested accuracy metric from a benchmark log."""
    metric_keys = {
        "piqa": "acc_norm,none",
        "winogrande": "acc,none",
        "mmlu": "acc,none",
        "gsm8k_cot_llama": "exact_match,strict-match",
    }
    logged_task = "gsm8k_cot_llama" if task == "gsm8k_cot_llama" else task
    metric_key = metric_keys[task]
    pattern = re.compile(
        rf"{re.escape(logged_task)} result:.*?'{re.escape(logged_task)}': "
        rf"\{{.*?'{re.escape(metric_key)}':\s*"
        rf"(?:np\.float64\()?({NUMBER_PATTERN})"
    )
    matches = pattern.findall(log_file.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def collect_task_scores(model_dir, task):
    return {
        quant_name: read_metric(model_dir / task / quant_name / f"{task}.log", task)
        if (model_dir / task / quant_name / f"{task}.log").exists() else None
        for quant_name in QUANT_NAMES
    }


def collect_zero_shot_scores(model_dir):
    values = {}
    for quant_name in QUANT_NAMES:
        piqa_log = model_dir / "piqa" / quant_name / "piqa.log"
        winogrande_log = model_dir / "winogrande" / quant_name / "winogrande.log"
        piqa = read_metric(piqa_log, "piqa") if piqa_log.exists() else None
        winogrande = read_metric(winogrande_log, "winogrande") if winogrande_log.exists() else None
        values[quant_name] = (piqa + winogrande) / 2 if piqa is not None and winogrande is not None else None
    return values


def make_result_row(task, model, values):
    ordered = [values[quant_name] for quant_name in QUANT_NAMES]
    improvement_4_to_5 = (
        (ordered[1] - ordered[0]) * 100
        if ordered[0] is not None and ordered[1] is not None else None
    )
    improvement_5_to_8 = (
        (ordered[2] - ordered[1]) * 100
        if ordered[1] is not None and ordered[2] is not None else None
    )
    return [
        task,
        model,
        *(f"{value * 100:.1f}" if value is not None else "N/A" for value in ordered),
        f"{improvement_4_to_5:.1f}" if improvement_4_to_5 is not None else "N/A",
        f"{improvement_5_to_8:.1f}" if improvement_5_to_8 is not None else "N/A",
    ]


rows = []
excluded_dirs = {"model_configs", "quant_configs"}
model_dirs = sorted(
    path for path in OUT_DIR.iterdir()
    if path.is_dir() and path.name not in excluded_dirs
)
for model_dir in model_dirs:
    zero_shot = collect_zero_shot_scores(model_dir)
    if any(value is not None for value in zero_shot.values()):
        rows.append(make_result_row("Avg zero-shot", model_dir.name, zero_shot))

    for task, label in [("mmlu", "MMLU"), ("gsm8k_cot_llama", "GSM8K")]:
        if not (model_dir / task).is_dir():
            continue
        values = collect_task_scores(model_dir, task)
        if any(value is not None for value in values.values()):
            rows.append(make_result_row(label, model_dir.name, values))

task_order = {"Avg zero-shot": 0, "MMLU": 1, "GSM8K": 2}
model_order = {
    label: {
        model_id.split("/")[-1].replace(".", "_"): index
        for index, model_id in enumerate(MODELS[task])
    }
    for label, task in {
        "Avg zero-shot": "piqa",
        "MMLU": "mmlu",
        "GSM8K": "gsm8k_cot_llama",
    }.items()
}
rows.sort(key=lambda row: (task_order[row[0]], model_order[row[0]].get(row[1], len(model_order[row[0]])), row[1]))

headers = [
    "Task",
    "Model",
    "W4A4-accuracy (%)",
    "W4A5-accuracy (%)",
    "W4A8-accuracy (%)",
    "W4A4 to W4A5 improvement (pp)",
    "W4A5 to W4A8 improvement (pp)",
]

tsv_file = OUT_DIR / "results.tsv"
with tsv_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t", lineterminator="\n")
    writer.writerows([headers, *rows])

widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
print("\nActivation-bit benchmark final results")
print(separator)
print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)) + " |")
print(separator)
for row in rows:
    print("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |")
print(separator)
