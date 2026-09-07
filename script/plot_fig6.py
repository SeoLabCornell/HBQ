import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(__file__)
PPL_PATH = "../save/block_size_activation_bitwidth/results.tsv"
HW_PATHS = {
    "area": "../hardware/BQ/MAC_WXAY/activation_blocksize_area_results.tsv",
    "energy": "../hardware/BQ/MAC_WXAY/activation_blocksize_energy_results.tsv",
}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    'font.size': 13,              # base font size
    'axes.titlesize': 20,         # subplot titles
    'axes.labelsize': 20,         # x/y labels
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'legend.title_fontsize': 14,
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

# area_baseline = 173
area_baseline = 140          # WoQ area per MAC (um^2)
energy_baseline = 280        # WoQ energy per MAC (fJ)


def darken_color(color, factor=0.6):
    """
    Darken a color by a given factor.
    
    Args:
        color: RGB tuple, RGBA tuple, or hex color string
        factor: Darkening factor (0-1), lower values = darker (default: 0.6)
    
    Returns:
        Darkened color as RGB tuple
    """
    import matplotlib.colors as mcolors
    
    # Convert color to RGB if it's a string or RGBA
    if isinstance(color, str):
        rgb = mcolors.to_rgb(color)
    elif len(color) == 4:  # RGBA
        rgb = color[:3]
    else:  # RGB
        rgb = color
    
    # Darken by multiplying RGB values by factor
    darkened = tuple(c * factor for c in rgb)
    return darkened


def load_hw_data(scheme, metric):
    """
    Load area or energy data from the combined wide-format TSV files.

    Values are per MAC: area in um^2, energy in fJ.

    The plotting code expects:
        (activation_bit, block_size) -> value

    Args:
        scheme: 'mx' for PoT or 'nv' for FP8
        metric: 'area' or 'energy'
    """
    scheme_name = {"mx": "PoT", "nv": "FP8"}[scheme]
    path = HW_PATHS[metric]
    result = {}

    with open(path, "r", newline="") as f:
        # Skip leading '#' comment lines (e.g. the units note in the energy TSV).
        rows = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows, delimiter="	")
        block_size_columns = [
            column for column in reader.fieldnames
            if column not in ("scaling_scheme", "activation_precision")
        ]

        for row in reader:
            if row["scaling_scheme"].strip() != scheme_name:
                continue

            act_bit = int(row["activation_precision"])
            for block_size_column in block_size_columns:
                raw_value = row[block_size_column].strip()
                if not raw_value or raw_value.upper() == "N/A":
                    continue
                result[(act_bit, int(block_size_column))] = float(raw_value)

    return result


def load_all_ppl(scheme):
    """
    Load PPL data from result.tsv and reshape it into the matrices expected by
    the existing plotting code.

    Returns:
        Dictionary mapping block_size -> PPL matrix
        Matrix row index = activation_bit - 3
        Matrix col index = weight_bit - 3
    """
    column_prefix = {"mxfp": "pot", "nvfp": "fp8"}[scheme]
    result = {}

    with open(PPL_PATH, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="	")

        ppl_columns = {}
        for column in reader.fieldnames:
            match = re.fullmatch(rf"{column_prefix}_b(\d+)", column.strip(), re.IGNORECASE)
            if match:
                block_size = int(match.group(1))
                ppl_columns[column] = block_size
                # Rows/columns represent bit widths 3 through 8.
                result[block_size] = np.full((6, 6), np.nan, dtype=float)

        for row in reader:
            format_match = re.fullmatch(r"W(\d+)A(\d+)", row["Activation format"].strip(), re.IGNORECASE)
            if not format_match:
                continue

            weight_bit = int(format_match.group(1))
            act_bit = int(format_match.group(2))
            row_idx = act_bit - 3
            col_idx = weight_bit - 3

            if not (0 <= row_idx < 6 and 0 <= col_idx < 6):
                continue

            for column, block_size in ppl_columns.items():
                raw_value = row[column].strip()
                if raw_value:
                    result[block_size][row_idx, col_idx] = float(raw_value)

    return result

def plot_ppl_vs_wgt_ebw(
    mxfp_ppl_by_bs,
    nvfp_ppl_by_bs,
    mx_block_sizes,
    nv_block_sizes,
    weight_precision=4,
    mx_block_sizes_with_lines=None,
    nv_block_sizes_with_lines=None,
    ax=None,
    return_handles_labels=False,
):
    """
    Plot PPL vs Wgt-EBW trade-off.
    Wgt-EBW = 4 + 8/B where B is the block size.
    
    Args:
        mxfp_ppl_by_bs: Dict mapping block_size -> PPL matrix for MXFP
        nvfp_ppl_by_bs: Dict mapping block_size -> PPL matrix for NVFP
        mx_block_sizes: List of block sizes to plot for MX (PoT-scale)
        nv_block_sizes: List of block sizes to plot for NV (FP8-scale)
        weight_precision: Weight bit precision (default: 4)
        mx_block_sizes_with_lines: List of MX block sizes that should have lines
        nv_block_sizes_with_lines: List of NV block sizes that should have lines
        ax: Matplotlib axes to plot on (if None, uses current axes)
        return_handles_labels: If True, return handles and labels for legend
    """
    if ax is None:
        ax = plt.gca()
    
    # Colors by block size (deterministic palette)
    all_block_sizes = sorted(set(mx_block_sizes) | set(nv_block_sizes))
    cmap = plt.get_cmap("tab10")
    bs_to_color = {bs: cmap(i % 10) for i, bs in enumerate(all_block_sizes)}
    # Create darker colors for MX (PoT-scale) series
    bs_to_color_mx = {bs: darken_color(bs_to_color[bs], factor=0.75) for bs in all_block_sizes}
    
    scheme_to_marker = {"mx": "o", "nv": "s"}
    
    # Weight precision index in PPL matrix (weight_bit - 3)
    weight_idx = weight_precision - 3
    
    # Activation bits available (4, 5, 6, 7, 8)
    activation_bits = [4, 5, 6, 7, 8]
    
    # Convert line specifications to sets
    mx_lines_set = set(mx_block_sizes_with_lines) if mx_block_sizes_with_lines else set()
    nv_lines_set = set(nv_block_sizes_with_lines) if nv_block_sizes_with_lines else set()
    
    # Plot MXFP (PoT-scale) series
    for bs in mx_block_sizes:
        color = bs_to_color_mx[bs]  # Use darker color for MX
        should_draw_line = bs in mx_lines_set
        wgt_ebw = 4 + 8.0 / bs  # Calculate Wgt-EBW
        
        if bs in mxfp_ppl_by_bs:
            ppl_mat = mxfp_ppl_by_bs[bs]
            ppl_vals = []
            wgt_ebw_vals = []
            
            for act_bit in activation_bits:
                ppl_row_idx = act_bit - 3
                ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                ppl_vals.append(ppl_val)
                wgt_ebw_vals.append(wgt_ebw)
            
            if ppl_vals:
                # Separate points by activation bit (A5 on top)
                points_by_act = {act_bit: [] for act_bit in activation_bits}
                for act_bit in activation_bits:
                    ppl_row_idx = act_bit - 3
                    ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                    points_by_act[act_bit].append((ppl_val, wgt_ebw))
                
                # Sort by PPL for line connection
                sorted_pairs = sorted(zip(ppl_vals, wgt_ebw_vals))
                if sorted_pairs:
                    sorted_ppl, sorted_wgt_ebw = zip(*sorted_pairs)
                else:
                    sorted_ppl, sorted_wgt_ebw = [], []
                
                # Draw line if needed
                if should_draw_line and len(sorted_ppl) > 1:
                    ax.plot(
                        sorted_ppl,
                        sorted_wgt_ebw,
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        alpha=0.8,
                        zorder=0,
                    )
                
                # Draw scatter points: all except A5 first, then A5 on top
                label_added = False
                for act_bit in activation_bits:
                    if act_bit != 5 and points_by_act[act_bit]:
                        act_ppl, act_wgt_ebw = zip(*points_by_act[act_bit])
                        ax.scatter(
                            act_ppl,
                            act_wgt_ebw,
                            c=[color],
                            marker=scheme_to_marker["mx"],
                            s=110,
                            alpha=0.9,
                            edgecolors="black",
                            linewidth=0.8,
                            label=f"PoT-scale B={bs}" if not label_added else "",
                            zorder=1,
                        )
                        if not label_added:
                            label_added = True
                
                # Draw A5 points last
                if points_by_act[5]:
                    act5_ppl, act5_wgt_ebw = zip(*points_by_act[5])
                    ax.scatter(
                        act5_ppl,
                        act5_wgt_ebw,
                        c=[color],
                        marker=scheme_to_marker["mx"],
                        s=110,
                        alpha=0.9,
                        edgecolors="black",
                        linewidth=2,
                        label=f"PoT-scale B={bs}" if not label_added else "",
                        zorder=2,
                    )
    
    # Plot NVFP (FP8-scale) series
    for bs in nv_block_sizes:
        color = bs_to_color[bs]
        should_draw_line = bs in nv_lines_set
        wgt_ebw = 4 + 8.0 / bs  # Calculate Wgt-EBW
        
        if bs in nvfp_ppl_by_bs:
            ppl_mat = nvfp_ppl_by_bs[bs]
            ppl_vals = []
            wgt_ebw_vals = []
            
            for act_bit in activation_bits:
                ppl_row_idx = act_bit - 3
                ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                ppl_vals.append(ppl_val)
                wgt_ebw_vals.append(wgt_ebw)
            
            if ppl_vals:
                # Separate points by activation bit (A5 on top)
                points_by_act = {act_bit: [] for act_bit in activation_bits}
                for act_bit in activation_bits:
                    ppl_row_idx = act_bit - 3
                    ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                    points_by_act[act_bit].append((ppl_val, wgt_ebw))
                
                # Sort by PPL for line connection
                sorted_pairs = sorted(zip(ppl_vals, wgt_ebw_vals))
                if sorted_pairs:
                    sorted_ppl, sorted_wgt_ebw = zip(*sorted_pairs)
                else:
                    sorted_ppl, sorted_wgt_ebw = [], []
                
                # Draw line if needed
                if should_draw_line and len(sorted_ppl) > 1:
                    ax.plot(
                        sorted_ppl,
                        sorted_wgt_ebw,
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        alpha=0.8,
                        zorder=0,
                    )
                
                # Draw scatter points: all except A5 first, then A5 on top
                label_added = False
                for act_bit in activation_bits:
                    if act_bit != 5 and points_by_act[act_bit]:
                        act_ppl, act_wgt_ebw = zip(*points_by_act[act_bit])
                        ax.scatter(
                            act_ppl,
                            act_wgt_ebw,
                            c=[color],
                            marker=scheme_to_marker["nv"],
                            s=110,
                            alpha=0.9,
                            edgecolors="black",
                            linewidth=0.8,
                            label=f"FP8-scale B={bs}" if not label_added else "",
                            zorder=1,
                        )
                        if not label_added:
                            label_added = True
                
                # Draw A5 points last
                if points_by_act[5]:
                    act5_ppl, act5_wgt_ebw = zip(*points_by_act[5])
                    ax.scatter(
                        act5_ppl,
                        act5_wgt_ebw,
                        c=[color],
                        marker=scheme_to_marker["nv"],
                        s=110,
                        alpha=0.9,
                        edgecolors="black",
                        linewidth=2,
                        label=f"FP8-scale B={bs}" if not label_added else "",
                        zorder=2,
                    )
    
    ax.set_xlabel("Llama3-8B Perplexity ↓")
    ax.set_ylabel("Wgt-EBW (bits)")
    ax.set_title("Wgt-EBW vs. Perplexity")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=6.47)
    ax.set_ylim(bottom=3.97, top=4.55)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    
    if return_handles_labels:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)
        return uniq_handles, uniq_labels


def plot_ppl_vs_metric(
    mxfp_ppl_by_bs,
    nvfp_ppl_by_bs,
    mx_hw_data,
    nv_hw_data,
    mx_block_sizes,
    nv_block_sizes,
    metric_name,
    metric_label,
    weight_precision=4,
    baseline=None,
    mx_block_sizes_with_lines=None,
    nv_block_sizes_with_lines=None,
    ax=None,
    return_handles_labels=False,
):
    """
    Plot PPL vs area or energy trade-off.
    
    Args:
        mxfp_ppl_by_bs: Dict mapping block_size -> PPL matrix for MXFP
        nvfp_ppl_by_bs: Dict mapping block_size -> PPL matrix for NVFP
        mx_hw_data: Dict mapping (activation_bit, block_size) -> value for MX
        nv_hw_data: Dict mapping (activation_bit, block_size) -> value for NV
        mx_block_sizes: List of block sizes to plot for MX (PoT-scale)
        nv_block_sizes: List of block sizes to plot for NV (FP8-scale)
        metric_name: 'area' or 'energy' (for title)
        metric_label: Y-axis label (e.g., 'Total Area' or 'Energy per MAC')
        weight_precision: Weight bit precision (default: 4)
        baseline: Optional baseline value to draw as a horizontal line
        mx_block_sizes_with_lines: List of MX block sizes that should have lines connecting markers (None or empty list for none)
        nv_block_sizes_with_lines: List of NV block sizes that should have lines connecting markers (None or empty list for none)
        ax: Matplotlib axes to plot on (if None, uses current axes)
        return_handles_labels: If True, return handles and labels for legend (default: False)
    """
    # Use provided axis or current axis
    if ax is None:
        ax = plt.gca()
    
    # Colors by block size (deterministic palette) - use union of both scales
    all_block_sizes = sorted(set(mx_block_sizes) | set(nv_block_sizes))
    cmap = plt.get_cmap("tab10")
    bs_to_color = {bs: cmap(i % 10) for i, bs in enumerate(all_block_sizes)}
    # Create darker colors for MX (PoT-scale) series
    bs_to_color_mx = {bs: darken_color(bs_to_color[bs], factor=0.75) for bs in all_block_sizes}
    
    scheme_to_marker = {"mx": "o", "nv": "s"}
    
    # Weight precision index in PPL matrix (weight_bit - 3)
    weight_idx = weight_precision - 3  # 3->0, 4->1, etc.
    
    # Activation bits available in HW data (4, 5, 6, 7, 8)
    activation_bits = [4, 5, 6, 7, 8]
    
    # Convert line specifications to sets for easy checking (handle None)
    mx_lines_set = set(mx_block_sizes_with_lines) if mx_block_sizes_with_lines else set()
    nv_lines_set = set(nv_block_sizes_with_lines) if nv_block_sizes_with_lines else set()
    
    # Plot MXFP (PoT-scale) series
    for bs in mx_block_sizes:
        color = bs_to_color_mx[bs]  # Use darker color for MX
        should_draw_line = bs in mx_lines_set
        
        if bs in mxfp_ppl_by_bs:
            ppl_mat = mxfp_ppl_by_bs[bs]
            ppl_vals = []
            hw_vals = []
            
            for act_bit in activation_bits:
                # PPL matrix: row = activation_bit - 3, column = weight_bit - 3
                ppl_row_idx = act_bit - 3
                ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                key = (act_bit, bs)
                
                if key in mx_hw_data:
                    ppl_vals.append(ppl_val)
                    hw_vals.append(mx_hw_data[key])
            
            if ppl_vals and hw_vals:
                # Separate points by activation bit for proper ordering (A5 on top)
                points_by_act = {act_bit: [] for act_bit in activation_bits}
                for act_bit in activation_bits:
                    key = (act_bit, bs)
                    if key in mx_hw_data:
                        ppl_row_idx = act_bit - 3
                        ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                        hw_val = mx_hw_data[key]
                        points_by_act[act_bit].append((ppl_val, hw_val))
                
                # Sort by PPL value for proper line connection (use all points)
                sorted_pairs = sorted(zip(ppl_vals, hw_vals))
                if sorted_pairs:
                    sorted_ppl, sorted_hw = zip(*sorted_pairs)
                else:
                    sorted_ppl, sorted_hw = [], []
                
                # Draw line if this series should have lines
                if should_draw_line and len(sorted_ppl) > 1:
                    ax.plot(
                        sorted_ppl,
                        sorted_hw,
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        alpha=0.8,
                        zorder=0,  # Draw lines behind markers
                    )
                
                # Draw scatter points: first all except A5, then A5 on top
                label_added = False
                for act_bit in activation_bits:
                    if act_bit != 5 and points_by_act[act_bit]:
                        act_ppl, act_hw = zip(*points_by_act[act_bit])
                        ax.scatter(
                            act_ppl,
                            act_hw,
                            c=[color],
                            marker=scheme_to_marker["mx"],
                            s=110,
                            alpha=0.9,
                            edgecolors="black",
                            linewidth=0.8,
                            label=f"PoT-scale B={bs}" if not label_added else "",
                            zorder=1,
                        )
                        if not label_added:
                            label_added = True
                
                # Draw A5 points last with higher zorder to appear on top
                if points_by_act[5]:
                    act5_ppl, act5_hw = zip(*points_by_act[5])
                    ax.scatter(
                        act5_ppl,
                        act5_hw,
                        c=[color],
                        marker=scheme_to_marker["mx"],
                        s=110,
                        alpha=0.9,
                        edgecolors="black",
                        linewidth=2,
                        label=f"PoT-scale B={bs}" if not label_added else "",
                        zorder=2,  # Higher zorder so A5 appears on top
                    )
    
    # Plot NVFP (FP8-scale) series
    for bs in nv_block_sizes:
        color = bs_to_color[bs]
        should_draw_line = bs in nv_lines_set
        
        if bs in nvfp_ppl_by_bs:
            ppl_mat = nvfp_ppl_by_bs[bs]
            ppl_vals = []
            hw_vals = []
            
            for act_bit in activation_bits:
                # PPL matrix: row = activation_bit - 3, column = weight_bit - 3
                ppl_row_idx = act_bit - 3
                ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                key = (act_bit, bs)
                
                if key in nv_hw_data:
                    ppl_vals.append(ppl_val)
                    hw_vals.append(nv_hw_data[key])
            
            if ppl_vals and hw_vals:
                # Separate points by activation bit for proper ordering (A5 on top)
                points_by_act = {act_bit: [] for act_bit in activation_bits}
                for act_bit in activation_bits:
                    key = (act_bit, bs)
                    if key in nv_hw_data:
                        ppl_row_idx = act_bit - 3
                        ppl_val = ppl_mat[ppl_row_idx, weight_idx]
                        hw_val = nv_hw_data[key]
                        points_by_act[act_bit].append((ppl_val, hw_val))
                
                # Sort by PPL value for proper line connection (use all points)
                sorted_pairs = sorted(zip(ppl_vals, hw_vals))
                if sorted_pairs:
                    sorted_ppl, sorted_hw = zip(*sorted_pairs)
                else:
                    sorted_ppl, sorted_hw = [], []
                
                # Draw line if this series should have lines
                if should_draw_line and len(sorted_ppl) > 1:
                    ax.plot(
                        sorted_ppl,
                        sorted_hw,
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        alpha=0.8,
                        zorder=0,  # Draw lines behind markers
                    )
                
                # Draw scatter points: first all except A5, then A5 on top
                label_added = False
                for act_bit in activation_bits:
                    if act_bit != 5 and points_by_act[act_bit]:
                        act_ppl, act_hw = zip(*points_by_act[act_bit])
                        ax.scatter(
                            act_ppl,
                            act_hw,
                            c=[color],
                            marker=scheme_to_marker["nv"],
                            s=110,
                            alpha=0.9,
                            edgecolors="black",
                            linewidth=0.8,
                            label=f"FP8-scale B={bs}" if not label_added else "",
                            zorder=1,
                        )
                        if not label_added:
                            label_added = True
                
                # Draw A5 points last with higher zorder to appear on top
                if points_by_act[5]:
                    act5_ppl, act5_hw = zip(*points_by_act[5])
                    ax.scatter(
                        act5_ppl,
                        act5_hw,
                        c=[color],
                        marker=scheme_to_marker["nv"],
                        s=110,
                        alpha=0.9,
                        edgecolors="black",
                        linewidth=2,
                        label=f"FP8-scale B={bs}" if not label_added else "",
                        zorder=2,  # Higher zorder so A5 appears on top
                    )
    
    # Add baseline line if provided
    if baseline is not None:
        ax.axhline(
            y=baseline,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            # label='Baseline'
        )
        
        # Add baseline text label
        ax.y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        if metric_name == "area":
            baseline_text = 'WoQ Area Baseline'
        elif metric_name == "energy":
            baseline_text = 'WoQ Energy Baseline'
        ax.text(
            0.97,  # Right edge in axes coordinates (98% from left)
            baseline-0.06*ax.y_range,  # Y position at baseline in data coordinates
            baseline_text,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.get_yaxis_transform(),  # Mix x=axes coords, y=data coords
            color='red',
            fontsize=16,
            fontweight='bold',
            alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7)
        )
    
    ax.set_xlabel("Llama3-8B Perplexity ↓")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_name.capitalize()} vs. Perplexity")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=6.47)
    if metric_name == "area":
        ax.set_ylim(bottom=50)
    elif metric_name == "energy":
        ax.set_ylim(bottom=46)
    
    if return_handles_labels:
        # Get handles and labels, de-duplicate
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)
        return uniq_handles, uniq_labels


def main():
    # ============================================================================
    # CONFIGURATION: Easily select which series to render
    # ============================================================================
    # Specify block sizes for each scale separately
    # Use None to auto-detect all available block sizes for that scale
    # Examples:
    #   MX_BLOCK_SIZES = [4, 16]        # Only MX with block sizes 4 and 16
    #   NV_BLOCK_SIZES = [8, 128]       # Only NV with block sizes 8 and 128
    #   MX_BLOCK_SIZES = None           # All available MX block sizes
    #   NV_BLOCK_SIZES = [4]            # Only NV with block size 4
    MX_BLOCK_SIZES = [32]  # e.g., [4, 16, 128] or None for all available
    NV_BLOCK_SIZES = [16, 32, 64, 128]  # e.g., [4, 8, 64] or None for all available
    
    # Specify which series should have lines connecting their markers
    # Use None or empty list [] for no lines
    # Examples:
    #   MX_BLOCK_SIZES_WITH_LINES = [32]       # Only MX with block size 32 gets lines
    #   NV_BLOCK_SIZES_WITH_LINES = [4, 128]   # NV with block sizes 4 and 128 get lines
    #   MX_BLOCK_SIZES_WITH_LINES = None       # No lines for MX series
    MX_BLOCK_SIZES_WITH_LINES = [32]  # e.g., [32] or None/[] for none
    NV_BLOCK_SIZES_WITH_LINES = [16, 32, 64, 128]  # e.g., [4, 128] or None/[] for none
    # ============================================================================
    
    # Load PPL data across block sizes
    mxfp_ppl = load_all_ppl("mxfp")
    nvfp_ppl = load_all_ppl("nvfp")
    
    # Load area (um^2/MAC) and energy (fJ/MAC) data
    mx_area = load_hw_data("mx", "area")
    nv_area = load_hw_data("nv", "area")
    mx_energy = load_hw_data("mx", "energy")
    nv_energy = load_hw_data("nv", "energy")
    
    # Determine available block sizes from all data sources
    mx_area_bs = sorted(set(bs for _, bs in mx_area.keys()))
    nv_area_bs = sorted(set(bs for _, bs in nv_area.keys()))
    mx_energy_bs = sorted(set(bs for _, bs in mx_energy.keys()))
    nv_energy_bs = sorted(set(bs for _, bs in nv_energy.keys()))
    mx_ppl_bs = sorted(set(mxfp_ppl.keys()))
    nv_ppl_bs = sorted(set(nvfp_ppl.keys()))
    
    # Determine available block sizes for each scale (intersection of area, energy, and ppl)
    mx_available_bs = sorted(set(mx_area_bs) & set(mx_energy_bs) & set(mx_ppl_bs))
    nv_available_bs = sorted(set(nv_area_bs) & set(nv_energy_bs) & set(nv_ppl_bs))
    
    # Determine block sizes to plot for each scale
    if MX_BLOCK_SIZES is None:
        mx_block_sizes = mx_available_bs
    else:
        mx_block_sizes = sorted([bs for bs in MX_BLOCK_SIZES if bs in mx_available_bs])
        if not mx_block_sizes:
            print(f"Warning: No selected MX block sizes are available. Available MX block sizes: {mx_available_bs}")
            mx_block_sizes = []
    
    if NV_BLOCK_SIZES is None:
        nv_block_sizes = nv_available_bs
    else:
        nv_block_sizes = sorted([bs for bs in NV_BLOCK_SIZES if bs in nv_available_bs])
        if not nv_block_sizes:
            print(f"Warning: No selected NV block sizes are available. Available NV block sizes: {nv_available_bs}")
            nv_block_sizes = []
    
    # Check if we have anything to plot
    if not mx_block_sizes and not nv_block_sizes:
        print("Error: No series to plot. Please check your configuration.")
        return
    
    # Create figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))
    
    # Plot area trade-off on first subplot
    handles1, labels1 = plot_ppl_vs_metric(
        mxfp_ppl_by_bs=mxfp_ppl,
        nvfp_ppl_by_bs=nvfp_ppl,
        mx_hw_data=mx_area,
        nv_hw_data=nv_area,
        mx_block_sizes=mx_block_sizes,
        nv_block_sizes=nv_block_sizes,
        metric_name="area",
        metric_label="Area per MAC (µm²)",
        weight_precision=4,
        baseline=area_baseline,
        mx_block_sizes_with_lines=MX_BLOCK_SIZES_WITH_LINES,
        nv_block_sizes_with_lines=NV_BLOCK_SIZES_WITH_LINES,
        ax=ax1,
        return_handles_labels=True,
    )
    
    # Plot energy trade-off on second subplot
    handles2, labels2 = plot_ppl_vs_metric(
        mxfp_ppl_by_bs=mxfp_ppl,
        nvfp_ppl_by_bs=nvfp_ppl,
        mx_hw_data=mx_energy,
        nv_hw_data=nv_energy,
        mx_block_sizes=mx_block_sizes,
        nv_block_sizes=nv_block_sizes,
        metric_name="energy",
        metric_label="Energy per MAC (fJ)",
        weight_precision=4,
        baseline=energy_baseline,
        mx_block_sizes_with_lines=MX_BLOCK_SIZES_WITH_LINES,
        nv_block_sizes_with_lines=NV_BLOCK_SIZES_WITH_LINES,
        ax=ax2,
        return_handles_labels=True,
    )
    
    # Plot Wgt-EBW trade-off on third subplot
    handles3, labels3 = plot_ppl_vs_wgt_ebw(
        mxfp_ppl_by_bs=mxfp_ppl,
        nvfp_ppl_by_bs=nvfp_ppl,
        mx_block_sizes=mx_block_sizes,
        nv_block_sizes=nv_block_sizes,
        weight_precision=4,
        mx_block_sizes_with_lines=MX_BLOCK_SIZES_WITH_LINES,
        nv_block_sizes_with_lines=NV_BLOCK_SIZES_WITH_LINES,
        ax=ax3,
        return_handles_labels=True,
    )
    
    # Combine handles and labels from all three plots, de-duplicate
    all_handles = handles1 + handles2 + handles3
    all_labels = labels1 + labels2 + labels3
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)
    
    # Create a single legend at the top in one row
    fig.legend(unique_handles, unique_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 1.03), ncol=len(unique_labels), 
               frameon=True, fontsize=18, handleheight=0.8, markerscale=1.5, labelspacing=0.2, handletextpad=0.1)
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at top for legend
    
    # Save combined figure
    save_path = os.path.join(os.path.dirname(__file__), "fig6.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()