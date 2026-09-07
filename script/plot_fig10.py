import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)


import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.weight": "bold",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    }
)

L2_UB_ORDER = [64, 32, 16, 8, 4]
L2_UB_LABELS = {
    64: "+L2 μB=64",
    32: "+L2 μB=32",
    16: "+L2 μB=16",
    8: "+L2 μB=8",
    4: "+L2 μB=4",
}
L2_BITWIDTH_OVERHEAD = {
    64: 0.03,
    32: 0.06,
    16: 0.13,
    8: 0.25,
    4: 0.5,
}


def _read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_data():
    activation_rows = _read_tsv("../hardware/BQ/MAC_WXAY/activation_blocksize_area_results.tsv")
    l1_row = next(
        row
        for row in activation_rows
        if row["scaling_scheme"] == "FP8" and int(row["activation_precision"]) == 5
    )
    l1_area = float(l1_row["128"])

    ppl_rows = _read_tsv("../save/block_size_activation_bitwidth/results.tsv")
    w4a5_row = next(row for row in ppl_rows if row["Activation format"] == "W4A5")
    l1_ppl = float(w4a5_row["fp8_b128"])

    hbq_rows = _read_tsv("../hardware/HBQ/HBQ_uB_area_results.tsv")
    hbq_by_sub_b = {int(row["sub_b"]): float(row["area_per_mac"]) for row in hbq_rows}

    result_rows = _read_tsv("../save/micro_block_size/results.tsv")
    ppl_by_sub_b = {
        int(row["L2 micro block size"]): float(row["PPL"]) for row in result_rows
    }

    schemes = ["L1 B=128"]
    areas = [l1_area]
    ppls = [l1_ppl]
    bitwidth_overheads = [0.0]

    for sub_b in L2_UB_ORDER:
        schemes.append(L2_UB_LABELS[sub_b])
        areas.append(hbq_by_sub_b[sub_b])
        ppls.append(ppl_by_sub_b[sub_b])
        bitwidth_overheads.append(L2_BITWIDTH_OVERHEAD[sub_b])

    return schemes, np.array(areas), np.array(ppls), np.array(bitwidth_overheads)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schemes, areas, ppls, bitwidth_overheads = load_data()
    baseline_area = areas[0]
    normalized_areas = areas / baseline_area
    display_labels = [schemes[0]] + [f"{schemes[0]}\n{scheme}" for scheme in schemes[1:]]

    x = np.arange(len(schemes))

    fig, ax1 = plt.subplots(figsize=(9, 4))

    bar_color = "#4e79a7"
    line_color = "#e15759"

    bars = ax1.bar(
        x,
        normalized_areas,
        width=0.62,
        color=bar_color,
        edgecolor="black",
        linewidth=1.0,
        zorder=2,
    )
    # ax1.set_xlabel("Scheme")
    ax1.set_ylabel("Normalized Area")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, ha="center")
    for tick_label in ax1.get_xticklabels()[1:]:
        tick_label.set_linespacing(0.95)
    ax1.tick_params(axis="y")
    # ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    # ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    for x_pos, overhead in zip(x[1:], bitwidth_overheads[1:]):
        ax1.text(
            x_pos,
            -0.19,
            f"(+{overhead:g}b EBW)",
            transform=ax1.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
        )

    for bar, normalized_area in zip(bars[1:], normalized_areas[1:]):
        overhead = (normalized_area - 1.0) * 100.0
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            normalized_area + 0.02,
            f"+{overhead:.1f}%",
            ha="center",
            va="bottom",
            # fontsize=11,
            color=bar_color,
            fontweight="bold",
        )

    ax1.set_ylim(0, max(normalized_areas) + 0.25)

    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        ppls,
        color=line_color,
        marker="o",
        markersize=7,
        linewidth=2.2,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=3,
        label="PPL",
    )[0]
    ax2.set_ylabel("Llama3-8B PPL")
    ax2.set_ylim(6.25, 7.15)
    ax2.tick_params(axis="y")
    woq_line = ax2.axhline(6.55, color="black", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    ax1.legend(
        [bars[0], line, woq_line],
        ["Area", "PPL", "WoQ PPL"],
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(script_dir, "fig10.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
