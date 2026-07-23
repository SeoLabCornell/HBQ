import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.rcParams['font.family'] = 'Arial'
except:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent


def load_area_breakdown(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    block_size = np.array([int(x) for x in rows[0][1:]])
    components = {
        row[0]: np.array([float(x) for x in row[1:]])
        for row in rows[1:]
        if row
    }
    return block_size, components


def load_ppl(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    block_size = np.array([int(row['Block size']) for row in rows])
    return block_size, {
        'fp8_e5m3': np.array([float(row['fp8_e5m3']) for row in rows]),
        'pot_e8m0_floor': np.array([float(row['pot_e8m0_floor']) for row in rows]),
        'pot_e8m0_round': np.array([float(row['pot_e8m0_round']) for row in rows]),
    }


# =======================
# Data
# =======================
block_size, area = load_area_breakdown(
    '../hardware/BQ/MAC_WXAY/scaling_factor_area_breakdown_results.tsv'
)
_, ppl_data = load_ppl('../save/scale_scheme_block_size/results.tsv')

mac_pot = area['PoT_mac']
deq_pot = area['PoT_deq']
acc_pot = area['PoT_acc']
ppl_pot = ppl_data['pot_e8m0_round']
ppl_pot_floor = ppl_data['pot_e8m0_floor']

mac_fp8 = area['FP8_mac']
deq_fp8 = area['FP8_deq']
acc_fp8 = area['FP8_acc']
ppl_fp8 = ppl_data['fp8_e5m3']

# =======================
# Plot setup
# =======================
x = np.arange(len(block_size))
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 3))

# Unified colors by component
color_mac = '#4e79a7'
color_deq = '#f28e2b'
color_acc = '#59a14f'

# ---- PoT bars (left of pair) ----
ax1.bar(x - bar_width/2 - bar_width/15, mac_pot, width=bar_width, color=color_mac, edgecolor='black', linewidth=1 )
ax1.bar(x - bar_width/2 - bar_width/15, deq_pot, width=bar_width, bottom=mac_pot, color=color_deq, edgecolor='black', linewidth=1)
ax1.bar(x - bar_width/2 - bar_width/15, acc_pot, width=bar_width, bottom=mac_pot + deq_pot, color=color_acc, edgecolor='black', linewidth=1)

# ---- FP8 bars (right of pair) ----
ax1.bar(x + bar_width/2 + bar_width/15, mac_fp8, width=bar_width, color=color_mac, edgecolor='black', linewidth=1)
ax1.bar(x + bar_width/2 + bar_width/15, deq_fp8, width=bar_width, bottom=mac_fp8, color=color_deq, edgecolor='black', linewidth=1)
ax1.bar(x + bar_width/2 + bar_width/15, acc_fp8, width=bar_width, bottom=mac_fp8 + deq_fp8, color=color_acc, edgecolor='black', linewidth=1)

# ---- Axis formatting ----
ax1.set_xticks(x)
ax1.set_xticklabels(block_size)
ax1.set_xlabel('Block size (B)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Area per MAC (µm²)', fontsize=16, fontweight='bold')
# ax1.set_title('Area Breakdown and Accuracy Trade-off')
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.tick_params(axis='both', which='major', labelsize=14)

# =======================
# PPL overlay (secondary axis)
# =======================
ax2 = ax1.twinx()
ax2.plot(x - bar_width/2 - bar_width/15, ppl_pot, 'o-', color="black", label='PoT PPL', markersize=10)
ax2.plot(x - bar_width/2 - bar_width/15, ppl_pot_floor, 'o-', color="black", label='PoT PPL Floor', markersize=10)
# ax2.plot(x + bar_width/2, ppl_fp8, 's-', color="black", label='FP8 PPL')
# Redraw ppl_fp8 with empty circle marker and remove previous plot for ppl_fp8
# ax2.lines.pop()  # remove the previous ppl_fp8 line (assumes it's last)
ax2.plot(x + bar_width/2 + bar_width/15, ppl_fp8, marker='o', markerfacecolor='white', markeredgecolor='black', linestyle='-', color="black", label='FP8 PPL', markersize=10)


ax2.set_ylabel('Wikitext2 PPL', fontsize=16, fontweight='bold')
ax2.set_ylim(6, 9)

# =======================
# Legends
# =======================
bars_legend = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color_mac, edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, facecolor=color_deq, edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, facecolor=color_acc, edgecolor='black'),
]
# Make the legend font size larger and also the patch larger
# Reduce vertical spacing in legend and move legend upward using bbox_to_anchor
legend1 = ax1.legend(
    bars_legend,
    ['Mul+Adder Tree', 'Dequant', 'FP Accum'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.22),  # high enough above the axes
    ncol=3,
    frameon=False,
    fontsize=14,
    handlelength=1.5,
    handleheight=0.8,
    handletextpad=0.3,
    borderpad=0.1,
    labelspacing=0.1,
    columnspacing=0.8
)
# plt.tight_layout()  # leave space for legend at top








for patch in legend1.get_patches():
    patch.set_linewidth(2)  # Set the edge thickness to 2 (adjust as needed)
for text in legend1.get_texts():
    text.set_fontweight('bold')
# Make x- and y-tick label bold
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_yticklabels():
    label.set_fontsize(13)

# pattern_legend = [
# ]
# legend2 = ax1.legend(pattern_legend, ['PoT', 'FP8'], loc='upper right', ncol=2, frameon=False, handleheight=1.5)

# Add both legends to the axes/artists so both show up
ax1.add_artist(legend1)
# ax1.add_artist(legend2)

# ax2.legend(loc='upper right', frameon=False)

# Annotate labels under each pair
# for i, b in enumerate(x):
#     ax1.text(b - bar_width/2, -15, 'PoT', ha='center', va='top', fontsize=8)
#     ax1.text(b + bar_width/2, -15, 'FP8', ha='center', va='top', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('fig3.png', dpi=300)
plt.close()