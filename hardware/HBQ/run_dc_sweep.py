#!/usr/bin/env python3
'''
Sweep the HBQ design by varying SUB_B over 4, 8, 16, 32, and 64.
Each run updates params.vh, executes dc_shell with syn.tcl, archives the
output/log directories, and collects area results for summary export.
'''
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_FILE = SCRIPT_DIR / "params.vh"
TCL_SCRIPT = "syn.tcl"
SUB_B_VALUES = [4, 8, 16, 32, 64]

PARAM_PATTERNS = {
    "SUB_B": re.compile(r"^(\s*parameter\s+integer\s+SUB_B\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
}


def replace_parameter(text, name, value):
    pattern = PARAM_PATTERNS[name]

    def replacement(match):
        return f"{match.group(1)}{value}{match.group(3)}"

    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace parameter {name} in {PARAMS_FILE}")
    return new_text


def update_params(sub_b):
    content = PARAMS_FILE.read_text()
    content = replace_parameter(content, "SUB_B", sub_b)
    PARAMS_FILE.write_text(content)
    print(f"Updated params.vh: SUB_B={sub_b}")


def prepare_run_dirs():
    for folder_name in ["output", "log"]:
        folder_path = SCRIPT_DIR / folder_name
        if folder_path.exists():
            shutil.rmtree(folder_path)


def run_dc_shell():
    tcl_path = SCRIPT_DIR / TCL_SCRIPT
    if not tcl_path.exists():
        raise FileNotFoundError(f"Could not find TCL script: {tcl_path}")

    print(f"Running: dc_shell -f {TCL_SCRIPT}")
    subprocess.run(["dc_shell", "-f", TCL_SCRIPT], cwd=SCRIPT_DIR, check=True)


def output_dir_candidates(sub_b):
    return [f"HBQ_uB{sub_b}"]


def archive_outputs(sub_b):
    dir_names = output_dir_candidates(sub_b)
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

        if stripped.startswith("systolic_reg"):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    sys_reg_area = float(parts[1])
                    break
                except ValueError:
                    pass

    if sys_reg_area is None:
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
        raise ValueError(f"Could not find systolic_reg area row in {report_path}")

    total_cell_area = float(total_cell_match.group(1))
    return total_cell_area, sys_reg_area


def collect_area_results():
    reg_amortize_ratio = 4.0
    report_candidates = sorted(SCRIPT_DIR.glob("**/log/*syn_area*.rpt"))
    if not report_candidates:
        print("No area report files found")
        return []

    results = []
    for report_path in report_candidates:
        config_dir = report_path.parent.parent
        match = re.match(r"HBQ_uB(\d+)$", config_dir.name)
        if not match:
            continue

        sub_b = int(match.group(1))
        total_cell_area, sys_reg_area = parse_area_report(report_path)
        area_per_mac = (total_cell_area - sys_reg_area * (reg_amortize_ratio - 1) / reg_amortize_ratio) / 128

        results.append({
            "config": config_dir.name,
            "sub_b": sub_b,
            "total_cell_area": total_cell_area,
            "sys_reg_area": sys_reg_area,
            "area_per_mac": area_per_mac,
        })

    return results


def generate_tsv(results, output_file="HBQ_uB_area_results.tsv"):
    if not results:
        print("No results collected")
        return

    with open(SCRIPT_DIR / output_file, "w") as f:
        f.write("sub_b\tarea_per_mac\n")
        for result in sorted(results, key=lambda item: item["sub_b"]):
            f.write(f"{result['sub_b']}\t{result['area_per_mac']:.4f}\n")

    print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def print_area_table(results):
    if not results:
        print("No results collected")
        return

    print("{:<10} {:>16} {:>18} {:>12}".format("Config", "Total cell area", "Systolic reg area", "Area/MAC"))
    print("-" * 70)
    for result in sorted(results, key=lambda item: item["sub_b"]):
        print("{:<10} {:>16.4f} {:>18.4f} {:>12.4f}".format(
            result["config"], result["total_cell_area"], result["sys_reg_area"], result["area_per_mac"]
        ))


def run_synthesis_loop():
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Could not find params.vh: {PARAMS_FILE}")

    backup_path = PARAMS_FILE.with_suffix(".vh.bak")
    backup_path.write_text(PARAMS_FILE.read_text())
    print(f"Backed up original params.vh to {backup_path.name}")

    try:
        for sub_b in SUB_B_VALUES:
            target_dir = SCRIPT_DIR / output_dir_candidates(sub_b)[0]
            if target_dir.exists():
                print(f"Skipping {target_dir.name}: output directory already exists")
                continue

            update_params(sub_b)
            prepare_run_dirs()
            run_dc_shell()
            archive_outputs(sub_b)

        results = collect_area_results()
        generate_tsv(results)
        print_area_table(results)
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
        generate_tsv(results)
        print_area_table(results)
    else:
        run_synthesis_loop()


if __name__ == "__main__":
    main()
