"""Figure 18: iso-area speedup (Linear + Attention) per model and scheme.

All speedups are computed by perf_model.py from prefill MAC counts and
per-scheme area/derate parameters; run this script and the figure is written
to fig18.pdf.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

import perf_model as pm

COLOR_LINEAR = "#4C78A8"
COLOR_ATTENTION = "#59A14F"
GROUP_WIDTH, GROUP_GAP = 2.0, 1.0

SEQ_LENS = list(pm.SPEEDUP_SEQ_LENS)
SCHEMES = list(pm.SCHEME_AREA)


def geometric_mean(values):
    positive = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def compute_all():
    """(models, data) with data[seq][model][scheme] = (linear, attention)."""
    models = list(pm.MODELS)
    data = {}
    for seq in SEQ_LENS:
        per_model = {
            mo: {s: pm.speedup_components(mo, seq, s) for s in SCHEMES}
            for mo in models
        }
        # GeoMean cluster: geometric mean of totals, arithmetic mean of split.
        geomean = {}
        for s in SCHEMES:
            totals = [sum(per_model[mo][s]) for mo in models]
            fracs = [per_model[mo][s][0] / t for mo, t in zip(models, totals)]
            total_geo = geometric_mean(totals)
            frac = float(np.clip(np.mean(fracs), 0.0, 1.0))
            geomean[s] = (total_geo * frac, total_geo * (1 - frac))
        per_model["GeoMean"] = geomean
        data[seq] = per_model
    return models + ["GeoMean"], data


def bar_layout(n_models, n_schemes):
    bar_width = GROUP_WIDTH / n_schemes
    bar_centers, cluster_centers, boundaries = [], [], []
    for i in range(n_models):
        start = i * (GROUP_WIDTH + GROUP_GAP)
        bar_centers += [start + (j + 0.5) * bar_width for j in range(n_schemes)]
        cluster_centers.append(start + GROUP_WIDTH / 2)
        boundaries.append(start + GROUP_WIDTH + GROUP_GAP / 2)
    x_min = -GROUP_GAP / 2
    x_max = (n_models - 1) * (GROUP_WIDTH + GROUP_GAP) + GROUP_WIDTH + GROUP_GAP / 2
    return bar_width, bar_centers, cluster_centers, boundaries[:-1], (x_min, x_max)


def draw_panel(ax, seq, models, data, show_ylabel):
    bar_width, bar_centers, cluster_centers, boundaries, xlim = bar_layout(
        len(models), len(SCHEMES)
    )
    linear = np.array([data[seq][m][s][0] for m in models for s in SCHEMES])
    attention = np.array([data[seq][m][s][1] for m in models for s in SCHEMES])
    x = np.array(bar_centers)

    ax.bar(x, linear, bar_width, color=COLOR_LINEAR, label="Linear",
           edgecolor="black", linewidth=1.0)
    ax.bar(x, attention, bar_width, bottom=linear, color=COLOR_ATTENTION,
           label="Attention", edgecolor="black", linewidth=1.0)

    ax.set_title(f"SeqLen={seq}", fontweight="bold")
    ax.set_xticks(bar_centers)
    ax.set_xticklabels(SCHEMES * len(models), rotation=90, fontsize=8)
    ax.tick_params(axis="x", pad=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(*xlim)
    if show_ylabel:
        ax.set_ylabel("Normalized Speedup", fontweight="bold")

    totals = linear + attention
    y_max = totals.max()
    ax.set_ylim(0, y_max * 1.18)

    for center, model in zip(cluster_centers, models):
        ax.text(center, -0.22, model.replace("-", "\n"),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold")
    for boundary in boundaries:
        ax.axvline(boundary, color="black", linestyle=(0, (2, 2)),
                   linewidth=1.0, alpha=0.6)

    geo_start = models.index("GeoMean") * len(SCHEMES)
    for i in range(geo_start, geo_start + len(SCHEMES)):
        ax.text(x[i] - 0.3, totals[i] + y_max * 0.015, f"{totals[i]:.2f}x",
                ha="center", va="bottom", fontsize=6)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    })

    models, data = compute_all()

    print("GeoMean speedups:")
    for seq in SEQ_LENS:
        row = "  ".join(f"{s}={sum(data[seq]['GeoMean'][s]):.2f}x" for s in SCHEMES)
        print(f"  SeqLen={seq:>3}: {row}")

    n = len(SEQ_LENS)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.4), squeeze=False)
    for i, seq in enumerate(SEQ_LENS):
        draw_panel(axes[0][i], seq, models, data, show_ylabel=(i == 0))

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    fig.savefig("fig18.pdf", dpi=200, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig18.pdf")


if __name__ == "__main__":
    main()
