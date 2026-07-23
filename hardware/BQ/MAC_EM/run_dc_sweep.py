#!/usr/bin/env python3
'''
Runs MAC_EM FP8 synthesis for selected configurations and prints the area breakdown.
'''
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_FILE = SCRIPT_DIR / "params.vh"

ACTIVATION_CONFIGS = {
    5: {"tcl": "syn_W4A5.tcl", "e_values": [3]},
    8: {"tcl": "syn_W4A8.tcl", "e_values": [3, 4, 5]},
}
BLOCK_SIZES = [4, 8, 16, 32, 64, 128]

PARAM_PATTERNS = {
    "X": re.compile(r"^(\s*parameter\s+integer\s+X\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
    "Y": re.compile(r"^(\s*parameter\s+integer\s+Y\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
    "B": re.compile(r"^(\s*parameter\s+integer\s+B\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
    "E": re.compile(r"^(\s*parameter\s+integer\s+E\s*=\s*)(\d+)(\s*;.*)$", re.MULTILINE),
}


def replace_parameter(text, name, value):
    pattern = PARAM_PATTERNS[name]

    def replacement(match):
        return f"{match.group(1)}{value}{match.group(3)}"

    new_text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace parameter {name} in {PARAMS_FILE}")
    return new_text


def update_params(x, y, b, e):
    content = PARAMS_FILE.read_text()
    content = replace_parameter(content, "X", x)
    content = replace_parameter(content, "Y", y)
    content = replace_parameter(content, "B", b)
    content = replace_parameter(content, "E", e)
    PARAMS_FILE.write_text(content)
    print(f"Updated params.vh: X={x}, Y={y}, B={b}, E={e}")


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


def output_dir_name(x, y, b, e):
    return f"W{x}A{y}B{b}_E{e}"


def archive_outputs(x, y, b, e):
    target_dir = SCRIPT_DIR / output_dir_name(x, y, b, e)
    if target_dir.exists():
        print(f"Skipping {target_dir.name}: output directory already exists")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
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
    results = []

    for report_path in report_candidates:
        config_dir = report_path.parent.parent
        match = re.match(r"W4A(\d+)B(\d+)_E(\d+)$", config_dir.name)
        if not match:
            continue
        y = int(match.group(1))
        b = int(match.group(2))
        e = int(match.group(3))
        total_cell_area, sys_reg_area = parse_area_report(report_path)
        area_per_mac = (total_cell_area - sys_reg_area * (reg_amortize_ratio - 1) / reg_amortize_ratio) / b
        results.append({
            "config": config_dir.name,
            "y": y,
            "b": b,
            "e": e,
            "total_cell_area": total_cell_area,
            "sys_reg_area": sys_reg_area,
            "area_per_mac": area_per_mac,
        })

    return results


def print_area_table(results):
    if not results:
        print("No results collected")
        return

    print("{:<18} {:>4} {:>4} {:>16} {:>18} {:>12}".format("Config", "B", "E", "Total cell area", "Systolic reg area", "Area/MAC"))
    print("-" * 86)
    for result in sorted(results, key=lambda item: (item["y"], item["e"], item["b"])):
        print("{:<18} {:>4} {:>4} {:>16.4f} {:>18.4f} {:>12.4f}".format(
            result["config"], result["b"], result["e"], result["total_cell_area"], result["sys_reg_area"], result["area_per_mac"]
        ))


def generate_tsv(results):
    if not results:
        print("No results collected")
        return

    block_sizes = [4, 8, 16, 32, 64, 128]
    
    for target_y in [5, 8]:
        filtered_results = [r for r in results if r["y"] == target_y]
        if not filtered_results:
            continue
        
        e_values = sorted(set(r["e"] for r in filtered_results))
        results_by_e_b = {}
        for result in filtered_results:
            key = (result["e"], result["b"])
            results_by_e_b[key] = result["area_per_mac"]
        
        output_file = f"W4A{target_y}_area_results.tsv"
        with open(SCRIPT_DIR / output_file, "w") as f:
            f.write("E\t" + "\t".join(str(b) for b in block_sizes) + "\n")
            for e in e_values:
                row_values = []
                for b in block_sizes:
                    key = (e, b)
                    if key in results_by_e_b:
                        row_values.append(f"{results_by_e_b[key]:.4f}")
                    else:
                        row_values.append("N/A")
                f.write(f"{e}\t" + "\t".join(row_values) + "\n")
        
        print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def run_synthesis_loop():
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Could not find params.vh: {PARAMS_FILE}")

    backup_path = PARAMS_FILE.with_suffix(".vh.bak")
    backup_path.write_text(PARAMS_FILE.read_text())
    print(f"Backed up original params.vh to {backup_path.name}")

    try:
        for y, config in ACTIVATION_CONFIGS.items():
            tcl_name = config["tcl"]
            e_values = config["e_values"]
            for e in e_values:
                for b in BLOCK_SIZES:
                    target_dir = SCRIPT_DIR / output_dir_name(4, y, b, e)
                    if target_dir.exists():
                        print(f"Skipping {target_dir.name}: output directory already exists")
                        continue

                    update_params(x=4, y=y, b=b, e=e)
                    prepare_run_dirs()
                    run_dc_shell(tcl_name)
                    archive_outputs(4, y, b, e)
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
        results = collect_area_results()
        generate_tsv(results)
        print_area_table(results)


if __name__ == "__main__":
    main()
