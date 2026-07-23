# Replication script for Table 3
# W4A4/W4A5/W4A8 benchmark sweep across models and tasks

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_QUANT_CONFIG = ROOT / "config/nvfp4.yaml"
OUT_DIR = ROOT / "save/activation_bit_benchmark"

# Fill these brackets with the model ids you want to evaluate for each benchmark.
MODELS = {
    "piqa": [
        "meta-llama/Meta-Llama-3-8B",
        'meta-llama/Llama-3.2-3B',
        'Qwen/Qwen2.5-3B',
        'Qwen/Qwen2.5-7B',
        # 'mistralai/Mixtral-8x7B-v0.1',
        # 'meta-llama/Llama-3.1-70B',
    ],
    "winogrande": [
        "meta-llama/Meta-Llama-3-8B",
        # 'meta-llama/Llama-3.2-3B',
        # 'Qwen/Qwen2.5-3B',
        # 'Qwen/Qwen2.5-7B',
        # 'mistralai/Mixtral-8x7B-v0.1',
        # 'meta-llama/Llama-3.1-70B',
    ],
    "mmlu": [
        "meta-llama/Meta-Llama-3-8B",
        # 'meta-llama/Llama-3.2-3B-Instruct'
    ],
    "gsm8k_cot_llama": [
        "meta-llama/Llama-3.1-8B-Instruct",
        # 'meta-llama/Llama-3.2-3B-Instruct'
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
