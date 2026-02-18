#!/usr/bin/env python3
"""
Publication-quality plot: MGHD vs MWPM logical error rate comparison.

Uses evaluation data from the d=9-trained model (Profile S, 50 epochs, 10k shots).
Primary data: component_scope=full with MWPM-projection post-processing (d=3,5,7).
Supplemental d=9 data from active-scope MWPM-projection mode.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE / "data/plan_v5_runs/d9_active_pmix_raw_50ep_10k_b4096"
OUTPUT_DIR = BASE / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Primary data (full scope, d=3,5,7,9) – most internally consistent
EVAL_MAIN = EVAL_DIR / "eval_mwpm_d3to9.json"
# Supplement d=9 with more p-values from active-scope MWPM projection
EVAL_D9_SUPP = EVAL_DIR / "eval_d9_mwpm_proj.json"

# Colors – colorblind-safe palette (Wong 2011)
DIST_COLORS = {
    3:  "#0072B2",  # blue
    5:  "#D55E00",  # vermillion
    7:  "#009E73",  # bluish green
    9:  "#E69F00",  # orange
    11: "#CC79A7",  # reddish purple
}
DIST_MARKERS_MGHD = {3: "o", 5: "s", 7: "D", 9: "p"}
DIST_MARKERS_MWPM = {3: "o", 5: "s", 7: "D", 9: "p"}

# ── Load & merge data ─────────────────────────────────────────────────────
with open(EVAL_MAIN) as f:
    raw_main = json.load(f)

with open(EVAL_D9_SUPP) as f:
    raw_d9 = json.load(f)

# Merge: use main data for d=3,5,7; for d=9 use supplemental (more p-values)
merged = [r for r in raw_main if r["distance"] != 9] + raw_d9

# Organise by distance
mghd_data = defaultdict(lambda: {"p": [], "ler": [], "ci_lo": [], "ci_hi": []})
mwpm_data = defaultdict(lambda: {"p": [], "ler": [], "ci_lo": [], "ci_hi": []})

for r in sorted(merged, key=lambda x: (x["distance"], x["p"])):
    d = r["distance"]
    p = r["p"]
    ci = r.get("confidence_intervals_95", {})

    for decoder, store in [("mghd", mghd_data), ("mwpm", mwpm_data)]:
        ler_key = f"ler_{decoder}"
        store[d]["p"].append(p)
        store[d]["ler"].append(r[ler_key])
        if ci.get(decoder):
            store[d]["ci_lo"].append(ci[decoder]["lo"])
            store[d]["ci_hi"].append(ci[decoder]["hi"])
        else:
            store[d]["ci_lo"].append(r[ler_key])
            store[d]["ci_hi"].append(r[ler_key])

# ── Minimum-count filter ──────────────────────────────────────────────────
# Points with <5 observed errors have unreliable LER ratios.  Flag them.
MIN_COUNTS = 5  # at 10k shots, LER < 5e-4 means <5 errors


# ── Plotting helpers ───────────────────────────────────────────────────────
def _setup_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="major", alpha=0.30, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)


# ── FIGURE ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"wspace": 0.32})

# ────────── Panel (a): LER vs p ──────────────────────────────────────────
ax = axes[0]

for d in sorted(mghd_data.keys()):
    color = DIST_COLORS.get(d, "gray")
    mk_mghd = DIST_MARKERS_MGHD.get(d, "o")

    p_arr      = np.array(mghd_data[d]["p"])
    ler_mghd   = np.array(mghd_data[d]["ler"])
    ci_lo_mghd = np.array(mghd_data[d]["ci_lo"])
    ci_hi_mghd = np.array(mghd_data[d]["ci_hi"])
    ler_mwpm   = np.array(mwpm_data[d]["ler"])
    ci_lo_mwpm = np.array(mwpm_data[d]["ci_lo"])
    ci_hi_mwpm = np.array(mwpm_data[d]["ci_hi"])

    mask_mghd = ler_mghd > 0
    mask_mwpm = ler_mwpm > 0

    # MGHD – solid, filled markers
    if mask_mghd.any():
        ax.errorbar(
            p_arr[mask_mghd], ler_mghd[mask_mghd],
            yerr=[
                ler_mghd[mask_mghd] - ci_lo_mghd[mask_mghd],
                ci_hi_mghd[mask_mghd] - ler_mghd[mask_mghd],
            ],
            fmt=f"{mk_mghd}-", color=color, markersize=6, linewidth=2.0,
            capsize=3, markeredgecolor="white", markeredgewidth=0.5,
            zorder=3,
        )

    # MWPM – dashed, open markers
    if mask_mwpm.any():
        ax.errorbar(
            p_arr[mask_mwpm], ler_mwpm[mask_mwpm],
            yerr=[
                ler_mwpm[mask_mwpm] - ci_lo_mwpm[mask_mwpm],
                ci_hi_mwpm[mask_mwpm] - ler_mwpm[mask_mwpm],
            ],
            fmt=f"{mk_mghd}--", color=color, markersize=5, linewidth=1.4,
            capsize=2, alpha=0.55, markerfacecolor="none", markeredgewidth=1.2,
            zorder=2,
        )

ax.set_yscale("log")
ax.set_xscale("log")
_setup_ax(ax,
          "Physical error rate  $p$",
          "Logical error rate  $p_L$",
          "(a)  Logical error rate vs physical error rate")

# Custom legend
legend_elements = []
for d in sorted(mghd_data.keys()):
    c = DIST_COLORS.get(d, "gray")
    mk = DIST_MARKERS_MGHD.get(d, "o")
    legend_elements.append(
        Line2D([0], [0], color=c, marker=mk, linestyle="-",
               markersize=6, linewidth=1.8, markeredgecolor="white",
               markeredgewidth=0.5, label=f"$d = {d}$")
    )
legend_elements.append(Line2D([0], [0], color="none", label=""))  # spacer
legend_elements.append(
    Line2D([0], [0], color="gray", marker="o", linestyle="-",
           markersize=5, linewidth=1.8, label="MGHD  (solid)")
)
legend_elements.append(
    Line2D([0], [0], color="gray", marker="o", linestyle="--",
           markersize=5, linewidth=1.2, markerfacecolor="none",
           markeredgewidth=1.2, alpha=0.55, label="MWPM  (dashed)")
)
ax.legend(handles=legend_elements, fontsize=10, loc="upper left",
          framealpha=0.90, edgecolor="0.8", fancybox=True)

# Format x-axis nicely
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:.0%}" if x >= 0.01 else f"{x:.1%}"))
ax.set_xlim(0.008, 0.12)

# ────────── Panel (b): MGHD / MWPM ratio vs p ───────────────────────────
ax2 = axes[1]

for d in sorted(mghd_data.keys()):
    color = DIST_COLORS.get(d, "gray")
    mk = DIST_MARKERS_MGHD.get(d, "o")
    p_arr    = np.array(mghd_data[d]["p"])
    ler_mghd = np.array(mghd_data[d]["ler"])
    ler_mwpm = np.array(mwpm_data[d]["ler"])
    shots    = 10_000  # all evals use 10k shots

    # Only compute where both > 0 AND have meaningful statistics
    mask = (ler_mghd > 0) & (ler_mwpm > 0)
    # Filter out points with <MIN_COUNTS events (unreliable ratio)
    mask &= (ler_mwpm * shots) >= MIN_COUNTS

    if mask.any():
        ratio = ler_mghd[mask] / ler_mwpm[mask]
        ax2.plot(p_arr[mask], ratio, f"{mk}-", color=color, markersize=7,
                 linewidth=2.0, markeredgecolor="white", markeredgewidth=0.5,
                 label=f"$d = {d}$", zorder=3)

# Reference lines
ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
ax2.axhspan(0.5, 1.0, alpha=0.06, color="green", zorder=0)
ax2.axhspan(1.0, 3.0, alpha=0.04, color="red", zorder=0)
ax2.text(0.098, 0.92, "MGHD better", fontsize=8.5, color="green",
         alpha=0.6, ha="right", transform=ax2.get_yaxis_transform())
ax2.text(0.098, 1.05, "MWPM better", fontsize=8.5, color="red",
         alpha=0.6, ha="right", transform=ax2.get_yaxis_transform())

_setup_ax(ax2,
          "Physical error rate  $p$",
          "LER ratio   (MGHD / MWPM)",
          "(b)  Decoder performance ratio")
ax2.legend(fontsize=10.5, loc="upper right", framealpha=0.90,
           edgecolor="0.8", fancybox=True)

# Sensible y-limits
all_ratios = []
for d in mghd_data:
    mghd_arr = np.array(mghd_data[d]["ler"])
    mwpm_arr = np.array(mwpm_data[d]["ler"])
    m = (mghd_arr > 0) & (mwpm_arr > 0) & ((mwpm_arr * 10_000) >= MIN_COUNTS)
    if m.any():
        all_ratios.extend((mghd_arr[m] / mwpm_arr[m]).tolist())
if all_ratios:
    ax2.set_ylim(0.5, min(max(all_ratios) * 1.15, 4.0))

# ── Suptitle & annotation ────────────────────────────────────────────────
fig.suptitle(
    "MGHD Neural Decoder vs MWPM Baseline\n"
    "Rotated Surface Code  ·  Code-Capacity IID Depolarising Noise  ·  "
    "10 000 shots per point",
    fontsize=12.5, fontweight="bold", y=1.01,
)

fig.text(
    0.5, -0.02,
    "Model: MGHD Profile S  ($d_{\\mathrm{model}}$=192, 8 message-passing iterations)  ·  "
    "Trained 50 epochs on $d$=9 active-scope component crops  ·  "
    "Evaluated with NN-guided MWPM projection",
    ha="center", fontsize=9.5, color="0.35", style="italic",
)

# ── Save ──────────────────────────────────────────────────────────────────
for fmt in ["pdf", "png"]:
    out = OUTPUT_DIR / f"mghd_vs_mwpm_ler.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved  {out}")

plt.close(fig)
print("\nDone.")
