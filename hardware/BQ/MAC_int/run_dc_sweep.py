#!/usr/bin/env python3
'''
This script reproduce area per MAC numbers for integer format shown in Figure 4(b) and Figure 5.
Run synthesis across different block sizes for FP8 scaling schemes for W4A5 and W4A8 settings.
'''
import re
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_FILE = SCRIPT_DIR / "params.vh"

ACTIVATION_PRECISIONS = [5, 8]
BLOCK_SIZES = [4, 8, 16, 32, 64, 128]
TCL_SCRIPTS = [
    ("syn_FP8.tcl", "FP8"),
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


def collect_area_results():
    reg_amortize_ratio = 4.0
    report_candidates = sorted(SCRIPT_DIR.glob("W4A*/log/*syn_area*.rpt"))
    if not report_candidates:
        print("No area report files found")
        return []

    results = []
    for report_path in report_candidates:
        config_dir = report_path.parent.parent
        match = re.match(r"W4A(\d+)B(\d+)_(FP8)$", config_dir.name)
        if not match:
            continue

        y = int(match.group(1))
        b = int(match.group(2))
        suffix = match.group(3)
        total_cell_area, sys_reg_area = parse_area_report(report_path)
        area_per_mac = (total_cell_area - sys_reg_area * (reg_amortize_ratio - 1) / reg_amortize_ratio) / b

        results.append({
            "config": config_dir.name,
            "y": y,
            "b": b,
            "suffix": suffix,
            "total_cell_area": total_cell_area,
            "sys_reg_area": sys_reg_area,
            "area_per_mac": area_per_mac,
        })

    return results


def generate_tsv(results, output_file="area_results.tsv"):
    if not results:
        print("No results collected")
        return

    results_by_suffix_y_and_b = {}
    for result in results:
        key = (result["suffix"], result["y"], result["b"])
        results_by_suffix_y_and_b[key] = result["area_per_mac"]

    block_sizes = [4, 8, 16, 32, 64, 128]
    activation_precisions = [5, 8]
    suffixes = ["FP8"]

    with open(SCRIPT_DIR / output_file, "w") as f:
        header = "Precision\tB\t" + "\t".join(str(b) for b in block_sizes)
        f.write(header + "\n")

        for suffix in suffixes:
            for y in activation_precisions:
                row_label = f"W4A{y}_INT"
                row_values = []
                for b in block_sizes:
                    key = (suffix, y, b)
                    if key in results_by_suffix_y_and_b:
                        row_values.append(f"{results_by_suffix_y_and_b[key]:.4f}")
                    else:
                        row_values.append("N/A")
                row = row_label + "\t" + "\t".join(row_values)
                f.write(row + "\n")

    print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def print_area_table(results):
    if not results:
        print("No results collected")
        return

    print("{:<18} {:>4} {:>16} {:>18} {:>12}".format("Config", "B", "Total cell area", "Systolic reg area", "Area/MAC"))
    print("-" * 80)
    for result in sorted(results, key=lambda item: (item['suffix'], item['b'])):
        print("{:<18} {:>4} {:>16.4f} {:>18.4f} {:>12.4f}".format(
            result['config'], result['b'], result['total_cell_area'], result['sys_reg_area'], result['area_per_mac']
        ))


def run_synthesis_loop():
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Could not find params.vh: {PARAMS_FILE}")

    backup_path = PARAMS_FILE.with_suffix(".vh.bak")
    backup_path.write_text(PARAMS_FILE.read_text())
    print(f"Backed up original params.vh to {backup_path.name}")

    try:
        for tcl_name, suffix in TCL_SCRIPTS:
            for y in ACTIVATION_PRECISIONS:
                for b in BLOCK_SIZES:
                    candidates = output_dir_candidates(4, y, b, suffix)
                    target_dir = None
                    for dir_name in candidates:
                        candidate = SCRIPT_DIR / dir_name
                        if candidate.exists():
                            target_dir = candidate
                            break

                    if target_dir is not None:
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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        results = collect_area_results()
        generate_tsv(results)
        print_area_table(results)
    else:
        run_synthesis_loop()
        results = collect_area_results()
        generate_tsv(results)
        print_area_table(results)


if __name__ == "__main__":
    main()
