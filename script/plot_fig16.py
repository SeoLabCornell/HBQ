"""Figure 16: system energy breakdown (DRAM / Compute / SRAM) + perplexity.

All energy numbers are computed by perf_model.py; run this script and the
figure is written to fig16.pdf.
"""

import matplotlib.pyplot as plt
from matplotlib import transforms

import perf_model as pm

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# ===== MEASURED DATA (Wikitext-2 PPL with KV cache quantization) =====
PERPLEXITY = {
    "Llama2-7B":   {"Amove": 6.11, "MXFP": 6.19, "NVFP4": 5.99, "HBQ-A": 5.75, "HBQ-E": 5.88},
    "Llama3-8B":   {"Amove": None, "MXFP": 7.45, "NVFP4": 7.41, "HBQ-A": 6.77, "HBQ-E": 7.11},
    "Llama3.2-3B": {"Amove": None, "MXFP": 9.53, "NVFP4": 9.54, "HBQ-A": 8.60, "HBQ-E": 9.07},
    "Qwen2.5-3B":  {"Amove": None, "MXFP": 8.83, "NVFP4": 9.04, "HBQ-A": 8.58, "HBQ-E": 8.79},
    "Qwen2.5-7B":  {"Amove": None, "MXFP": 7.51, "NVFP4": 7.35, "HBQ-A": 7.10, "HBQ-E": 7.23},
}

FORMATS = ["Amove", "MXFP", "NVFP4", "HBQ-A", "HBQ-E"]
COLOR_DRAM = tuple(c / 255 for c in (31, 119, 180))
COLOR_COMP = tuple(c / 255 for c in (255, 127, 14))
COLOR_SRAM = tuple(c / 255 for c in (44, 160, 44))

# ===== NUMBERS (computed) =====
models = list(pm.MODELS)
data = {(mo, ar): pm.energy(ar, mo) for mo in models for ar in FORMATS}

print("Energy totals (J):")
for mo in models:
    row = "  ".join(f"{ar}={data[(mo, ar)]['total']:.1f}" for ar in FORMATS)
    print(f"  {mo:<12} {row}")

# ===== LAYOUT =====
bar_width = 0.8
intra, inter = 0.15, 0.8
group_width = len(FORMATS) * bar_width + (len(FORMATS) - 1) * intra

x_positions, dram_vals, comp_vals, sram_vals, bar_labels = [], [], [], [], []
for i, mo in enumerate(models):
    start = i * (group_width + inter)
    for j, ar in enumerate(FORMATS):
        x_positions.append(start + j * (bar_width + intra))
        e = data[(mo, ar)]
        dram_vals.append(e["dram"])
        comp_vals.append(e["core"])
        sram_vals.append(e["sram"])
        bar_labels.append(ar)
totals = [d + c + s for d, c, s in zip(dram_vals, comp_vals, sram_vals)]

# ===== STACKED BARS =====
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x_positions, dram_vals, bar_width, label="DRAM", color=COLOR_DRAM,
       edgecolor="black", linewidth=1, zorder=3)
ax.bar(x_positions, comp_vals, bar_width, bottom=dram_vals, label="Compute",
       color=COLOR_COMP, edgecolor="black", linewidth=1, zorder=3)
ax.bar(x_positions, sram_vals, bar_width,
       bottom=[d + c for d, c in zip(dram_vals, comp_vals)], label="SRAM",
       color=COLOR_SRAM, edgecolor="black", linewidth=1, zorder=3)

fontsize = 10
ax.set_xticks(x_positions)
ax.set_xticklabels(bar_labels, rotation=90, ha="center", fontsize=fontsize - 1)
ax.set_ylabel("Energy (J)", fontsize=fontsize, labelpad=0)
for x, total in zip(x_positions, totals):
    ax.text(x, total + 0.15, f"{total:.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=4)
ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
ax.set_ylim(0, 12)
ax.margins(x=0.015)
for label in ax.get_yticklabels():
    label.set_fontsize(fontsize)

# ===== PERPLEXITY LINE (right axis) =====
ax2 = ax.twinx()
ax2.set_ylabel("Perplexity", fontsize=fontsize, labelpad=4)
ppl_vals = [PERPLEXITY[mo][ar] for mo in models for ar in FORMATS]
ax2.plot(x_positions, ppl_vals, color="black", marker="o", linestyle="-",
         zorder=0, markersize=4)
ax2.set_ylim(4.6, 10)
ax2.margins(x=0.015)

# ===== MODEL LABELS + SEPARATORS =====
trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
for i, mo in enumerate(models):
    center = i * (group_width + inter) + group_width / 2
    ax.text(center - 0.3, -0.25, mo, ha="center", va="top",
            transform=trans, fontsize=fontsize)
for i in range(1, len(models)):
    sep = i * (group_width + inter) - inter / 2 - intra / 2 - 0.32
    ax.axvline(x=sep, color="gray", linestyle="--", linewidth=0.8, zorder=1)

# ===== LEGEND =====
ax.plot([], [], color="black", marker="o", linestyle="-", label="Perplexity",
        markersize=4)
handles, labels = ax.get_legend_handles_labels()
handles, labels = handles[1:4] + [handles[0]], labels[1:4] + [labels[0]]
ax.legend(handles, labels, fontsize=fontsize - 2, loc="upper center",
          bbox_to_anchor=(0.5, 1.13), ncol=4)

plt.tight_layout()
plt.savefig("fig16.pdf", pad_inches=0.01, bbox_inches="tight")
print("wrote fig16.pdf")
