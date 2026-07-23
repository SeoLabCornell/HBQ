#!/usr/bin/env python3
'''
This script reproduce area per MAC numbers shown in Figure 3.
Run synthesis across different block sizes among PoT and FP8 scaling schemes.
'''

import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_FILE = SCRIPT_DIR / "params.vh"

BLOCK_SIZES = [4, 8, 16, 32, 64, 128]
ACTIVATION_PRECISIONS = [4, 5, 6, 7, 8]
COMPONENT_LABELS = {
    "mac": "mac_vec",
    "deq": "scaling",
    "acc": "fp16_accum",
}
TCL_SCRIPTS = [
    ("syn_FP8.tcl", "FP8"),
    ("syn_PoT.tcl", "PoT"),
]
ACTIVATION_BLOCKSIZE_CONFIGS = [
    ("PoT", "syn_PoT.tcl", [32]),
    ("FP8", "syn_FP8.tcl", [4, 8, 16, 32, 64, 128]),
]

PARAM_PATTERNS = {
    "X": re.compile(r"^(\s*parameter\s+integer\s+X\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
    "Y": re.compile(r"^(\s*parameter\s+integer\s+Y\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
    "B": re.compile(r"^(\s*parameter\s+integer\s+B\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
}


def replace_parameter(text, name, value):
    pattern = PARAM_PATTERNS[name]

    def replacement(match):
        return f"{match.group(1)}{value}{match.group(3)}"

    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace parameter {name} in {PARAMS_FILE}")
    return new_text


def update_params(x, y, b):
    content = PARAMS_FILE.read_text()
    content = replace_parameter(content, "X", x)
    content = replace_parameter(content, "Y", y)
    content = replace_parameter(content, "B", b)
    PARAMS_FILE.write_text(content)
    print(f"Updated params.vh: X={x}, Y={y}, B={b}")


def prepare_run_dirs():
    for folder_name in ["output", "log"]:
        folder_path = SCRIPT_DIR / folder_name
        if folder_path.exists():
            shutil.rmtree(folder_path)


def run_dc_shell(tcl_name):
    tcl_path = SCRIPT_DIR / tcl_name
    if not tcl_path.exists():
        raise FileNotFoundError(f"Could not find TCL script: {tcl_path}")

    print(f"Running: dc_shell -f {tcl_name}")
    subprocess.run(["dc_shell", "-f", tcl_name], cwd=SCRIPT_DIR, check=True)


def output_dir_candidates(x, y, b, suffix):
    return [
        f"W{x}A{y}B{b}_{suffix}",
    ]


def archive_outputs(x, y, b, suffix):
    dir_names = output_dir_candidates(x, y, b, suffix)
    target_dir = None

    for dir_name in dir_names:
        candidate = SCRIPT_DIR / dir_name
        if candidate.exists():
            target_dir = candidate
            break

    if target_dir is None:
        target_dir = SCRIPT_DIR / dir_names[0]
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Skipping {target_dir.name}: output directory already exists")
        return

    for folder_name in ["output", "log"]:
        src = SCRIPT_DIR / folder_name
        if src.exists():
            dst = target_dir / folder_name
            shutil.copytree(src, dst, dirs_exist_ok=True)

    shutil.copy2(PARAMS_FILE, target_dir / "params.vh")
    print(f"Archived outputs to {target_dir}")


def parse_area_report(report_path):
    lines = report_path.read_text().splitlines()

    total_cell_match = None
    for line in lines:
        if "Total cell area:" in line:
            total_cell_match = re.search(r"Total cell area:\s+([0-9.]+)", line)
            break

    if not total_cell_match:
        raise ValueError(f"Could not find 'Total cell area' in {report_path}")

    sys_reg_area = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Hierarchical cell", "--------------------------------", "Global cell area", "Local cell area", "Total")):
            continue

        parts = stripped.split()
        if len(parts) < 6:
            continue

        try:
            float(parts[1])
            float(parts[2])
            float(parts[3])
            float(parts[4])
        except ValueError:
            continue

        sys_reg_area = float(parts[4])
        break

    if sys_reg_area is None:
        raise ValueError(f"Could not find top-level area row in {report_path}")

    total_cell_area = float(total_cell_match.group(1))
    return total_cell_area, sys_reg_area


def parse_component_area_breakdown(report_path):
    component_areas = {}
    for line in report_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Hierarchical cell", "--------------------------------", "Global cell area", "Local cell area", "Total")):
            continue

        parts = stripped.split()
        if len(parts) < 2:
            continue

        cell_name = parts[0]
        if cell_name not in COMPONENT_LABELS.values():
            continue

        try:
            area_value = float(parts[1])
        except ValueError:
            continue

        for component, label in COMPONENT_LABELS.items():
            if label == cell_name:
                component_areas[component] = area_value
                break

    return component_areas


def collect_area_results():
    reg_amortize_ratio = 4.0
    report_candidates = sorted(SCRIPT_DIR.glob("W4A*/log/*syn_area*.rpt"))
    if not report_candidates:
        print("No area report files found")
        return []

    results = []
    for report_path in report_candidates:
        config_dir = report_path.parent.parent
        match = re.match(r"W4A(\d+)B(\d+)_(FP8|PoT)$", config_dir.name)
        if not match:
            continue

        y = int(match.group(1))
        b = int(match.group(2))
        suffix = match.group(3)
        total_cell_area, sys_reg_area = parse_area_report(report_path)
        component_areas = parse_component_area_breakdown(report_path)
        area_per_mac = (total_cell_area - sys_reg_area * (reg_amortize_ratio - 1) / reg_amortize_ratio) / b

        results.append({
            "config": config_dir.name,
            "y": y,
            "b": b,
            "suffix": suffix,
            "total_cell_area": total_cell_area,
            "sys_reg_area": sys_reg_area,
            "component_areas": component_areas,
            "area_per_mac": area_per_mac,
        })

    return results


def generate_scaling_factor_tsv(results, output_file="scaling_factor_area_breakdown_results.tsv"):
    if not results:
        print("No results collected")
        return

    results_by_suffix_component_and_b = {}
    for result in results:
        if result["y"] != 4:
            continue
        for component, area_value in result.get("component_areas", {}).items():
            key = (result["suffix"], component, result["b"])
            results_by_suffix_component_and_b[key] = area_value / result["b"]

    block_sizes = [4, 8, 16, 32, 64, 128]
    suffixes = ["PoT", "FP8"]
    components = ["mac", "deq", "acc"]

    with open(SCRIPT_DIR / output_file, "w") as f:
        header = "component\t" + "\t".join(str(b) for b in block_sizes)
        f.write(header + "\n")

        for suffix in suffixes:
            for component in components:
                row_label = f"{suffix}_{component}"
                row_values = []
                for b in block_sizes:
                    key = (suffix, component, b)
                    if key in results_by_suffix_component_and_b:
                        row_values.append(f"{results_by_suffix_component_and_b[key]:.4f}")
                    else:
                        row_values.append("N/A")
                row = row_label + "\t" + "\t".join(row_values)
                f.write(row + "\n")

    print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def generate_activation_blocksize_tsv(results, output_file="activation_blocksize_area_results.tsv"):
    if not results:
        print("No results collected")
        return

    block_sizes = [4, 8, 16, 32, 64, 128]
    results_by_scale_and_y_and_b = {}
    for result in results:
        if result["y"] not in ACTIVATION_PRECISIONS:
            continue
        if result["suffix"] == "PoT" and result["b"] != 32:
            continue
        if result["suffix"] == "FP8" and result["b"] not in [16, 32, 64, 128]:
            continue
        key = (result["suffix"], result["y"], result["b"])
        results_by_scale_and_y_and_b[key] = result["area_per_mac"]

    with open(SCRIPT_DIR / output_file, "w") as f:
        header = "scaling_scheme\tactivation_precision\t" + "\t".join(str(b) for b in block_sizes)
        f.write(header + "\n")

        for suffix in ["PoT", "FP8"]:
            for y in ACTIVATION_PRECISIONS:
                row_label = f"{suffix}"
                row_values = []
                for b in block_sizes:
                    key = (suffix, y, b)
                    if key in results_by_scale_and_y_and_b:
                        row_values.append(f"{results_by_scale_and_y_and_b[key]:.4f}")
                    else:
                        row_values.append("N/A")
                row = row_label + "\t" + str(y) + "\t" + "\t".join(row_values)
                f.write(row + "\n")

    print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def run_scaling_factor_sweep():
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Could not find params.vh: {PARAMS_FILE}")

    backup_path = PARAMS_FILE.with_suffix(".vh.bak")
    backup_path.write_text(PARAMS_FILE.read_text())
    print(f"Backed up original params.vh to {backup_path.name}")

    try:
        for tcl_name, suffix in TCL_SCRIPTS:
            for b in BLOCK_SIZES:
                y = 4
                target_dir = SCRIPT_DIR / output_dir_candidates(4, y, b, suffix)[0]
                if target_dir.exists():
                    print(f"Skipping {target_dir.name}: output directory already exists")
                    continue

                update_params(x=4, y=y, b=b)
                prepare_run_dirs()
                run_dc_shell(tcl_name)
                archive_outputs(4, y, b, suffix)
    except subprocess.CalledProcessError as exc:
        print(f"dc_shell failed with exit code {exc.returncode}")
        raise
    except Exception:
        print("An error occurred while updating parameters or running dc_shell.")
        raise
    finally:
        PARAMS_FILE.write_text(backup_path.read_text())
        print(f"Restored original params.vh from {backup_path.name}")


def run_activation_blocksize_sweep():
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Could not find params.vh: {PARAMS_FILE}")

    backup_path = PARAMS_FILE.with_suffix(".vh.bak")
    backup_path.write_text(PARAMS_FILE.read_text())
    print(f"Backed up original params.vh to {backup_path.name}")

    try:
        for suffix, tcl_name, b_values in ACTIVATION_BLOCKSIZE_CONFIGS:
            for y in ACTIVATION_PRECISIONS:
                for b in b_values:
                    target_dir = SCRIPT_DIR / output_dir_candidates(4, y, b, suffix)[0]
                    if target_dir.exists():
                        print(f"Skipping {target_dir.name}: output directory already exists")
                        continue

                    update_params(x=4, y=y, b=b)
                    prepare_run_dirs()
                    run_dc_shell(tcl_name)
                    archive_outputs(4, y, b, suffix)
    except subprocess.CalledProcessError as exc:
        print(f"dc_shell failed with exit code {exc.returncode}")
        raise
    except Exception:
        print("An error occurred while updating parameters or running dc_shell.")
        raise
    finally:
        PARAMS_FILE.write_text(backup_path.read_text())
        print(f"Restored original params.vh from {backup_path.name}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        results = collect_area_results()
        generate_scaling_factor_tsv(results)
        generate_activation_blocksize_tsv(results)
    else:
        run_scaling_factor_sweep()
        run_activation_blocksize_sweep()
        results = collect_area_results()
        generate_scaling_factor_tsv(results)
        generate_activation_blocksize_tsv(results)


if __name__ == "__main__":
    main()
