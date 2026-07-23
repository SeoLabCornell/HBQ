import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'

BLOCK_SIZES = [4, 8, 16, 32, 64, 128]
FORMAT_ORDER = ["int8", "e2m5", "e3m4", "e4m3", "e5m2"]
E_TO_FORMAT = {3: "e3m4", 4: "e4m3", 5: "e5m2"}


def _block_series_from_row(row: pd.Series, block_sizes: List[int]) -> pd.Series:
	"""Map consecutive numeric row values (after the label column) to block sizes."""
	values = [
		float(v)
		for v in row.values[1:]
		if pd.notna(v) and str(v) != "N/A"
	]
	if len(values) != len(block_sizes):
		raise ValueError(
			f"Expected {len(block_sizes)} block values in row {row.iloc[0]!r}, got {len(values)}"
		)
	return pd.Series(values, index=block_sizes)


def _block_series_from_columns(row: pd.Series, block_sizes: List[int]) -> pd.Series:
	"""Map named block-size columns to a series, treating N/A as NaN."""
	series = pd.Series(index=block_sizes, dtype=float)
	for block_size in block_sizes:
		value = row[str(block_size)]
		series[block_size] = pd.NA if value == "N/A" else float(value)
	return series


def load_tradeoff_data() -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Aggregate PPL and area-per-MAC tables from data/*.tsv sources."""
	ppl_raw = pd.read_csv("../save/W4A8_format_block_size/results.tsv", sep="\t")
	ppl_raw = ppl_raw.rename(columns={"Block size": "B"}).set_index("B")
	ppl_df = ppl_raw[FORMAT_ORDER]

	hw_data = {}
	area_results = pd.read_csv("../hardware/BQ/MAC_int/area_results.tsv", sep="\t")
	int8_row = area_results.loc[area_results["Precision"] == "W4A8_INT"].iloc[0]
	hw_data["int8"] = _block_series_from_row(int8_row, BLOCK_SIZES)

	w4a8_area = pd.read_csv("../hardware/BQ/MAC_EM/W4A8_area_results.tsv", sep="\t")
	for _, row in w4a8_area.iterrows():
		fmt = E_TO_FORMAT[int(row["E"])]
		hw_data[fmt] = _block_series_from_row(row, BLOCK_SIZES)

	activation_area = pd.read_csv(
		"../hardware/BQ/MAC_WXAY/activation_blocksize_area_results.tsv", sep="\t"
	)
	e2_row = activation_area[
		(activation_area["scaling_scheme"] == "FP8")
		& (activation_area["activation_precision"] == 8)
	].iloc[0]
	hw_data["e2m5"] = _block_series_from_columns(e2_row, BLOCK_SIZES)

	hw_df = pd.DataFrame(hw_data)[FORMAT_ORDER]
	hw_df.index.name = "B"
	return ppl_df, hw_df


def plot_tradeoff(ppl_df: pd.DataFrame, hw_df: pd.DataFrame, out_path: str) -> None:
	"""Plot area-per-MAC (x) vs perplexity (y) for each numeric format.

	Each format becomes a line, parameterized by the shared index (block size).
	"""

	# Align by shared index (block size) and shared formats (columns)
	shared_index = ppl_df.index.intersection(hw_df.index)
	ppl_df = ppl_df.loc[shared_index]
	hw_df = hw_df.loc[shared_index]

	shared_formats: List[str] = [
		col for col in ppl_df.columns 
		if col in hw_df.columns
	]
	if not shared_formats:
		raise ValueError("No shared formats between ppl and hw TSV files.")

	# Set up global font properties: bold and larger
	plt.rcParams.update({
		"font.size": 12,
		"font.weight": "bold",
	})

	# Set up plot
	plt.figure(figsize=(8.0, 3.5), dpi=500)
	markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
	colors = plt.cm.tab10.colors

	ax = plt.gca()

	for i, fmt in enumerate(shared_formats):
		x = hw_df[fmt].values   # area per MAC (um^2/MAC)
		y = ppl_df[fmt].values  # perplexity
		# Annotate points by block size
		block_sizes = shared_index.tolist()
		plt.plot(
			x,
			y,
			label=fmt,
			marker="o",
			color=colors[i % len(colors)],
			linewidth=2.0,
			markersize=6.0,
		)
		# for xi, yi, b in zip(x, y, block_sizes):
		# 	if b == 4 or b == 128:
		# 		plt.annotate(
		# 			str(b),
		# 			(xi, yi),
		# 			textcoords="offset points",
		# 			xytext=(6, 5),
		# 			fontsize=10,
		# 			fontweight="bold",
		# 			color=colors[i % len(colors)],
		# 		)

	# Baselines
	ax.set_xlim(50, 450)
	WoQ_area_baseline = 200
	WoQ_ppl_baseline = 6.55
	ax.axvline(x=WoQ_area_baseline, color="gray", linestyle="--", linewidth=1.6)
	ax.axhline(y=WoQ_ppl_baseline, color="gray", linestyle="--", linewidth=1.6)
	# Add label above the horizontal line (WoQ Baseline in gray)
	ax.text(
		ax.get_xlim()[1]-5, 
		WoQ_ppl_baseline+0.2, 
		"WoQ PPL Baseline", 
		color="gray",
		fontsize=12,
		fontweight="bold",
		ha="right",
		va="bottom"
	)
	# Add label to the right of the vertical line (WoQ Baseline in gray)
	ax.text(
		WoQ_area_baseline+6, 
		ax.get_ylim()[1]-0.015, 
		"WoQ Area Baseline", 
		color="gray",
		fontsize=12,
		fontweight="bold",
		ha="left",
		va="top"
	)
	# Add light gray background to area where x > WoQ_area_baseline
	# Fill region where both area > WoQ_area_baseline and ppl > WoQ_ppl_baseline, filling to the top of the visible figure
	x_min = WoQ_area_baseline
	x_max = ax.get_xlim()[1]
	# Ensure fill goes all the way from WoQ_ppl_baseline to the top
	# Fix xlim so that fill (axvspan) always covers the visible right of the plot
	ax.axvspan(
		x_min, x_max,
		ymin=(WoQ_ppl_baseline - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
		ymax=1.0,
		color="lightcoral", alpha=0.19, zorder=1
	)
	# ax.axhspan(WoQ_area_baseline, ax.get_ylim()[1], color="lightcoral", alpha=0.23, zorder=0)
	# ax.axhspan(0, WoQ_area_baseline, color="lightgreen", alpha=0.23, zorder=0)
	

	# Labels and title (bold and larger)
	ax.set_xlabel("Area per MAC (µm²)", fontsize=16, fontweight="bold")
	ax.set_ylabel("Perplexity ↓", fontsize=16, fontweight="bold")
	ax.set_title("Area vs. Perplexity", fontsize=16, fontweight="bold")

	# Grid
	plt.grid(True, linestyle=":", linewidth=0.9, alpha=0.6)

	# Legend styling (bold, larger)
	# legend = ax.legend(title="Format", frameon=True, prop={"weight": "bold", "size": 11}, title_fontproperties={"weight": "bold", "size": 12})

	# Tick label styling (bold and larger)
	ax.tick_params(axis="both", which="both", labelsize=14, width=1.2)
	for tick in ax.get_xticklabels() + ax.get_yticklabels():
		tick.set_fontweight("bold")

	plt.tight_layout()

	# Ensure output directory exists and save
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.savefig(out_path)
	# Optional display if running interactively
	# plt.show()


if __name__ == "__main__":
	base_dir = os.path.dirname(os.path.abspath(__file__))
	out_path = os.path.join(base_dir, "fig4b.png")
	ppl_df, hw_df = load_tradeoff_data()
	plot_tradeoff(ppl_df, hw_df, out_path)
	print(f"Saved plot to: {out_path}")


