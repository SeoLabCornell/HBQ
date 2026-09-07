#!/usr/bin/env python3
'''
This script reproduces the energy-per-MAC numbers shown in Figure 3.

For every (scaling scheme, activation precision, block size) point it

  1. reuses (or, if missing, produces via run_dc_sweep.py) the Design Compiler
     netlist / SDC / SDF for that configuration,
  2. runs a gate-level simulation of the PE against the golden vectors in
     ./testcase, dumping a VCD of the streaming region only,
  3. runs PrimeTime PX on that VCD to get averaged power, and
  4. converts power to energy per MAC, amortizing the systolic pass-through
     register cost over REG_AMORTIZE_RATIO PEs.

The register amortization mirrors the one used for area in run_dc_sweep.py: the
weight/activation pass-through registers at the top level are shared by
REG_AMORTIZE_RATIO MACs in the systolic array, so only 1/REG_AMORTIZE_RATIO of
their cost is charged to a single PE:

    P_amortized   = P_total - P_sys_reg * (R - 1) / R
    E_per_MAC     = P_amortized * clk_period / B

with P_sys_reg taken as the top-level local power, i.e. the total minus the
power of the three instantiated blocks (mac_vec, scaling, fp16_accum).

Usage:
    ./run_pt_sweep.py            # full sweep, then write the TSV
    ./run_pt_sweep.py collect    # only re-parse existing reports into the TSV

Tool paths (dc_shell, vcs, pt_shell) are taken from PATH. The TSMC 28nm
standard-cell library location can be overridden with the STD_CELL_VERILOG and
PT_PDK_DIR environment variables.
'''

import os
import re
import subprocess
import sys
from pathlib import Path

import run_dc_sweep as dc

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMS_FILE = SCRIPT_DIR / "params.vh"
TESTCASE_DIR = SCRIPT_DIR / "testcase"
PT_TCL = SCRIPT_DIR / "pt_power.tcl"

# The systolic pass-through registers are shared by this many MACs.
REG_AMORTIZE_RATIO = 4.0

# Scheme -> (top-level module, testbench define, VCD instance name)
SCHEMES = {
    "FP8": ("WXAY_MAC_NV", "TEST_NV", "dut_nv"),
    "PoT": ("WXAY_MAC_MX", "TEST_MX", "dut_mx"),
}

# Same sweep points as the area figure.
ACTIVATION_PRECISIONS = dc.ACTIVATION_PRECISIONS
BLOCK_SIZES = dc.BLOCK_SIZES
CONFIGS = dc.ACTIVATION_BLOCKSIZE_CONFIGS

# Sub-block names reported by PrimeTime's hierarchical power report. Whatever is
# left of the total after subtracting these is the top-level (systolic register)
# power.
CHILD_BLOCKS = ["mac_vec", "scaling", "fp16_accum"]

DEFAULT_STD_CELL_VERILOG = (
    "<PDK_PATH>"
    "/digital/Front_End/verilog/tcbn28hpcplusbwp30p140_110a"
    "/tcbn28hpcplusbwp30p140.v"
)


def config_name(y, b, suffix):
    return f"W4A{y}B{b}_{suffix}"


def clk_period_for(suffix, y, b):
    return dc.clk_period_for(suffix, y, b)


def syn_output_paths(config_dir, top_level, clk_period):
    """Netlist / SDC / SDF written by syn_FP8.tcl and syn_PoT.tcl."""
    out = config_dir / "output"
    return (
        out / f"{top_level}.{clk_period}.28nm.syn.v",
        out / f"{top_level}.{clk_period}.syn.28nm.sdc",
        out / f"{top_level}.{clk_period}.syn.28nm.sdf",
    )


def ensure_synthesis(y, b, suffix, tcl_name):
    """Run Design Compiler for this configuration if it has not been run yet."""
    config_dir = SCRIPT_DIR / config_name(y, b, suffix)
    top_level = SCHEMES[suffix][0]
    clk_period = clk_period_for(suffix, y, b)
    netlist, sdc, _sdf = syn_output_paths(config_dir, top_level, clk_period)

    if netlist.exists() and sdc.exists():
        return config_dir

    print(f"[{config_dir.name}] no netlist yet, running synthesis first")
    dc.update_params(x=4, y=y, b=b)
    dc.prepare_run_dirs()
    dc.run_dc_shell(tcl_name, clk_period)
    dc.archive_outputs(4, y, b, suffix)

    if not netlist.exists():
        raise FileNotFoundError(f"Synthesis did not produce {netlist}")
    return config_dir


def run_gate_sim(config_dir, y, b, suffix):
    """Compile and run the gate-level simulation, producing tb.vcd."""
    top_level, tb_define, _dut_inst = SCHEMES[suffix]
    clk_period = clk_period_for(suffix, y, b)
    netlist, _sdc, sdf = syn_output_paths(config_dir, top_level, clk_period)

    work_dir = config_dir / "power"
    work_dir.mkdir(parents=True, exist_ok=True)

    vcd_path = work_dir / "tb.vcd"
    window_path = work_dir / "power_window.txt"
    if vcd_path.exists() and window_path.exists():
        print(f"[{config_dir.name}] reusing existing {vcd_path.name}")
        return work_dir, vcd_path, window_path

    std_cell_verilog = os.environ.get("STD_CELL_VERILOG", DEFAULT_STD_CELL_VERILOG)
    if not Path(std_cell_verilog).exists():
        raise FileNotFoundError(
            f"Standard-cell simulation library not found: {std_cell_verilog}\n"
            "Set STD_CELL_VERILOG to the tcbn28hpcplusbwp30p140.v of your PDK."
        )

    vcs_cmd = [
        "vcs", "-full64", "-sverilog", "+v2k", "-lca",
        "-debug_access+all",
        f"+incdir+{SCRIPT_DIR}",
        "+notimingcheck", "-negdelay", "+neg_tchk", "+sdfverbose",
        "-override_timescale=1ns/1ps",
        "+define+syn_simv",
        f"+define+{tb_define}",
        f"+define+TB_CLK_PERIOD={clk_period}",
        "+lint=TFIPC-L",
        str(netlist),
        str(SCRIPT_DIR / "tb_WXAY.sv"),
        str(std_cell_verilog),
        "-o", "simv",
    ]
    if sdf.exists():
        # VCS needs the SDF path as a compile-time string constant.
        vcs_cmd.insert(-3, f'+define+SDF_FILE="{sdf}"')
    print(f"[{config_dir.name}] compiling gate-level simulation")
    with open(work_dir / "vcs_compile.log", "w") as log:
        subprocess.run(vcs_cmd, cwd=work_dir, check=True, stdout=log,
                       stderr=subprocess.STDOUT)

    sim_cmd = [
        "./simv",
        f"+testcase_dir={TESTCASE_DIR}",
        f"+vcd={vcd_path.name}",
        f"+power_window={window_path.name}",
    ]

    print(f"[{config_dir.name}] running gate-level simulation")
    with open(work_dir / "sim.log", "w") as log:
        subprocess.run(sim_cmd, cwd=work_dir, check=True, stdout=log,
                       stderr=subprocess.STDOUT)

    sim_log = (work_dir / "sim.log").read_text()
    if "tests FAILED" in sim_log:
        raise RuntimeError(
            f"[{config_dir.name}] gate-level simulation reported mismatches; "
            f"see {work_dir / 'sim.log'}"
        )
    if not window_path.exists():
        raise FileNotFoundError(f"Testbench did not write {window_path}")

    return work_dir, vcd_path, window_path


def read_power_window(window_path):
    values = {}
    for line in window_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = parts[1]
    if "start" not in values or "end" not in values:
        raise ValueError(f"Malformed power window file: {window_path}")
    return values["start"], values["end"]


def run_primetime(config_dir, work_dir, y, b, suffix, vcd_path, window_path):
    top_level, _tb_define, dut_inst = SCHEMES[suffix]
    clk_period = clk_period_for(suffix, y, b)
    netlist, sdc, _sdf = syn_output_paths(config_dir, top_level, clk_period)
    report_prefix = f"{config_name(y, b, suffix)}_{top_level}"

    hier_report = work_dir / f"{report_prefix}.report_power_hierarchy"
    if hier_report.exists():
        print(f"[{config_dir.name}] reusing existing {hier_report.name}")
        return hier_report

    vcd_start, vcd_end = read_power_window(window_path)

    env = os.environ.copy()
    env.update({
        "PT_TOP": top_level,
        "PT_NETLIST": str(netlist),
        "PT_SDC": str(sdc),
        "PT_VCD": str(vcd_path),
        "PT_STRIP_PATH": f"tb/{dut_inst}",
        "PT_VCD_START": vcd_start,
        "PT_VCD_END": vcd_end,
        "PT_REPORT_PREFIX": report_prefix,
    })

    print(f"[{config_dir.name}] running PrimeTime power ({vcd_start} .. {vcd_end} ns)")
    with open(work_dir / "pt_power.log", "w") as log:
        subprocess.run(["pt_shell", "-f", str(PT_TCL)], cwd=work_dir, check=True,
                       env=env, stdout=log, stderr=subprocess.STDOUT)

    if not hier_report.exists():
        raise FileNotFoundError(f"PrimeTime did not produce {hier_report}")
    return hier_report


NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?")


def parse_power_hierarchy(path):
    """Return (total_power, {block_name: power}) from report_power -hierarchy.

    Rows look like

        WXAY_MAC_MX                      1.20e-03 5.77e-04 1.47e-04 1.93e-03 100.0
          fp16_accum (FP16_ACCUM)        1.52e-04 8.26e-05 2.15e-05 2.56e-04  13.3

    and a long instance/module name pushes the numbers onto the next line.
    """
    lines = path.read_text(errors="ignore").splitlines()
    rows = []
    in_table = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--------"):
            in_table = True
            i += 1
            continue
        if not in_table or not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("Hierarchy"):
            i += 1
            continue

        matches = list(NUMBER_RE.finditer(line))
        if len(matches) >= 5:
            name = line[: matches[-5].start()].strip()
            if name:
                rows.append((name, float(matches[-5:][3].group(0))))
            i += 1
            continue

        # wrapped row: name on this line, numbers on the next non-blank line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            matches2 = list(NUMBER_RE.finditer(lines[j]))
            if len(matches2) >= 5 and line.strip():
                rows.append((line.strip(), float(matches2[-5:][3].group(0))))
                i = j + 1
                continue
        i += 1

    if not rows:
        raise ValueError(f"No power rows found in {path}")

    total = rows[0][1]
    blocks = {}
    for raw_name, power in rows:
        blocks.setdefault(raw_name.split(" (", 1)[0], power)
    return total, blocks


def energy_for_report(report_path, b, clk_period_ns):
    """Energy per MAC in fJ, with the pass-through registers amortized."""
    total_power, blocks = parse_power_hierarchy(report_path)

    missing = [name for name in CHILD_BLOCKS if name not in blocks]
    if missing:
        raise ValueError(f"{report_path}: missing power rows for {missing}")

    child_power = sum(blocks[name] for name in CHILD_BLOCKS)
    sys_reg_power = total_power - child_power
    amortized_power = total_power - sys_reg_power * (REG_AMORTIZE_RATIO - 1) / REG_AMORTIZE_RATIO

    # W * ns = nJ; per MAC, expressed in fJ
    energy_per_mac_fj = amortized_power * float(clk_period_ns) / b * 1e6
    return {
        "total_power": total_power,
        "sys_reg_power": sys_reg_power,
        "amortized_power": amortized_power,
        "clk_period_ns": float(clk_period_ns),
        "energy_per_mac_fj": energy_per_mac_fj,
    }


def collect_energy_results():
    results = []
    for suffix, _tcl_name, b_values in CONFIGS:
        top_level = SCHEMES[suffix][0]
        for y in ACTIVATION_PRECISIONS:
            for b in b_values:
                name = config_name(y, b, suffix)
                report = (SCRIPT_DIR / name / "power" /
                          f"{name}_{top_level}.report_power_hierarchy")
                if not report.exists():
                    continue
                entry = energy_for_report(report, b, clk_period_for(suffix, y, b))
                entry.update({"config": name, "suffix": suffix, "y": y, "b": b})
                results.append(entry)
    return results


def generate_energy_tsv(results, output_file="activation_blocksize_energy_results.tsv"):
    if not results:
        print("No power reports found; nothing to write")
        return

    by_key = {(r["suffix"], r["y"], r["b"]): r["energy_per_mac_fj"] for r in results}

    with open(SCRIPT_DIR / output_file, "w") as f:
        f.write("# energy per MAC in fJ, systolic pass-through registers amortized over "
                f"{int(REG_AMORTIZE_RATIO)} PEs\n")
        f.write("scaling_scheme\tactivation_precision\t"
                + "\t".join(str(b) for b in BLOCK_SIZES) + "\n")
        for suffix in ["PoT", "FP8"]:
            for y in ACTIVATION_PRECISIONS:
                values = []
                for b in BLOCK_SIZES:
                    value = by_key.get((suffix, y, b))
                    values.append(f"{value:.4f}" if value is not None else "N/A")
                f.write(f"{suffix}\t{y}\t" + "\t".join(values) + "\n")

    print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def generate_raw_power_tsv(results, output_file="activation_blocksize_power_raw.tsv"):
    """Audit trail: the PrimeTime numbers the energy table is derived from."""
    if not results:
        return
    with open(SCRIPT_DIR / output_file, "w") as f:
        f.write("config\tscaling_scheme\tactivation_precision\tblock_size\t"
                "clk_period_ns\ttotal_power_W\tsys_reg_power_W\tamortized_power_W\t"
                "energy_per_mac_fJ\n")
        for r in sorted(results, key=lambda r: (r["suffix"], r["y"], r["b"])):
            f.write(f"{r['config']}\t{r['suffix']}\t{r['y']}\t{r['b']}\t"
                    f"{r['clk_period_ns']}\t{r['total_power']:.6e}\t"
                    f"{r['sys_reg_power']:.6e}\t{r['amortized_power']:.6e}\t"
                    f"{r['energy_per_mac_fj']:.4f}\n")
    print(f"TSV file generated: {SCRIPT_DIR / output_file}")


def run_sweep():
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Could not find params.vh: {PARAMS_FILE}")
    if not TESTCASE_DIR.exists():
        raise FileNotFoundError(f"Could not find the golden vectors: {TESTCASE_DIR}")

    backup_path = PARAMS_FILE.with_suffix(".vh.bak")
    backup_path.write_text(PARAMS_FILE.read_text())
    print(f"Backed up original params.vh to {backup_path.name}")

    try:
        for suffix, tcl_name, b_values in CONFIGS:
            for y in ACTIVATION_PRECISIONS:
                for b in b_values:
                    print(f"===== {config_name(y, b, suffix)} =====")
                    config_dir = ensure_synthesis(y, b, suffix, tcl_name)
                    # the bench takes X/Y/B from params.vh, same as the RTL
                    dc.update_params(x=4, y=y, b=b)
                    work_dir, vcd_path, window_path = run_gate_sim(config_dir, y, b, suffix)
                    run_primetime(config_dir, work_dir, y, b, suffix, vcd_path, window_path)
    finally:
        PARAMS_FILE.write_text(backup_path.read_text())
        print(f"Restored original params.vh from {backup_path.name}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        results = collect_energy_results()
    else:
        run_sweep()
        results = collect_energy_results()
    generate_energy_tsv(results)
    generate_raw_power_tsv(results)


if __name__ == "__main__":
    main()
