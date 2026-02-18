#!/usr/bin/env python3
"""Publication-quality plot: MGHD vs MWPM logical error rate comparison.

Reads evaluation JSON produced by ``python -m mghd.cli.eval`` and generates
two-panel figures: (a) LER vs p, (b) MGHD/MWPM ratio vs p.

Usage
-----
    # From eval.py output (new format):
    python scripts/plot_mghd_vs_mwpm.py --input eval_results.json

    # Legacy format (plan_v5_runs style):
    python scripts/plot_mghd_vs_mwpm.py --input old_eval.json --legacy

    # Merge multiple inputs:
    python scripts/plot_mghd_vs_mwpm.py --input eval_d3.json eval_d5.json

    # Custom output:
    python scripts/plot_mghd_vs_mwpm.py --input eval.json -o plots/my_plot
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


# ── Configuration ──────────────────────────────────────────────────────────

# Colors – colorblind-safe palette (Wong 2011)
DIST_COLORS = {
    3:  "#0072B2",   # blue
    5:  "#D55E00",   # vermillion
    7:  "#009E73",   # bluish green
    9:  "#E69F00",   # orange
    11: "#CC79A7",   # reddish purple
    13: "#56B4E9",   # sky blue
    15: "#F0E442",   # yellow
}
DIST_MARKERS = {3: "o", 5: "s", 7: "D", 9: "p", 11: "^", 13: "v", 15: "X"}
MIN_COUNTS = 5  # points with <5 observed errors have unreliable LER ratios


# ── Data loading ───────────────────────────────────────────────────────────


def load_eval_data(paths: list[str], legacy: bool = False) -> list[dict]:
    """Load and merge eval results from one or more JSON files.

    Supports two formats:
      - **eval.py format**: ``{"results": [{"distance", "p", "ler_mghd",
        "ler_mwpm", "confidence_intervals_95": {"mghd": {"lo", "hi"}, ...}}]}``
      - **legacy format**: ``[{"distance", "p", "ler_mghd", "ler_mwpm",
        "confidence_intervals_95": {...}}]``
    """
    merged = []
    for path in paths:
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            merged.extend(raw)
        elif isinstance(raw, dict) and "results" in raw:
            merged.extend(raw["results"])
        else:
            print(f"Warning: unrecognised format in {path}, skipping", file=sys.stderr)
    return merged


def organise_by_distance(records: list[dict]):
    """Split records into per-distance MGHD and MWPM data dicts."""
    mghd_data = defaultdict(lambda: {"p": [], "ler": [], "ci_lo": [], "ci_hi": []})
    mwpm_data = defaultdict(lambda: {"p": [], "ler": [], "ci_lo": [], "ci_hi": []})

    for r in sorted(records, key=lambda x: (x["distance"], x["p"])):
        d = r["distance"]
        p = r["p"]
        ci = r.get("confidence_intervals_95", {})

        for decoder, store, ler_key in [
            ("mghd", mghd_data, "ler_mghd"),
            ("mwpm", mwpm_data, "ler_mwpm"),
        ]:
            store[d]["p"].append(p)
            store[d]["ler"].append(r[ler_key])
            if ci.get(decoder):
                store[d]["ci_lo"].append(ci[decoder]["lo"])
                store[d]["ci_hi"].append(ci[decoder]["hi"])
            else:
                store[d]["ci_lo"].append(r[ler_key])
                store[d]["ci_hi"].append(r[ler_key])

    return mghd_data, mwpm_data


# ── Plotting helpers ───────────────────────────────────────────────────────


def _setup_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="major", alpha=0.30, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)


def _color(d):
    return DIST_COLORS.get(d, "gray")


def _marker(d):
    return DIST_MARKERS.get(d, "o")


# ── Main plot ──────────────────────────────────────────────────────────────


def plot_mghd_vs_mwpm(
    mghd_data: dict,
    mwpm_data: dict,
    output_stem: str,
    shots_per_point: int = 10_000,
    suptitle_extra: str = "",
):
    """Generate two-panel LER comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"wspace": 0.32})

    # ────────── Panel (a): LER vs p ──────────────────────────────────
    ax = axes[0]

    for d in sorted(mghd_data.keys()):
        color = _color(d)
        mk = _marker(d)

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
                fmt=f"{mk}-", color=color, markersize=6, linewidth=2.0,
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
                fmt=f"{mk}--", color=color, markersize=5, linewidth=1.4,
                capsize=2, alpha=0.55, markerfacecolor="none", markeredgewidth=1.2,
                zorder=2,
            )

    ax.set_yscale("log")
    ax.set_xscale("log")
    _setup_ax(
        ax,
        "Physical error rate  $p$",
        "Logical error rate  $p_L$",
        "(a)  Logical error rate vs physical error rate",
    )

    # Custom legend
    legend_elements = []
    for d in sorted(mghd_data.keys()):
        c = _color(d)
        mk = _marker(d)
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
    ax.legend(
        handles=legend_elements, fontsize=10, loc="upper left",
        framealpha=0.90, edgecolor="0.8", fancybox=True,
    )

    # Format x-axis
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x:.0%}" if x >= 0.01 else f"{x:.1%}"
        )
    )

    # ────────── Panel (b): MGHD / MWPM ratio vs p ───────────────────
    ax2 = axes[1]

    for d in sorted(mghd_data.keys()):
        color = _color(d)
        mk = _marker(d)
        p_arr    = np.array(mghd_data[d]["p"])
        ler_mghd = np.array(mghd_data[d]["ler"])
        ler_mwpm = np.array(mwpm_data[d]["ler"])

        mask = (ler_mghd > 0) & (ler_mwpm > 0)
        mask &= (ler_mwpm * shots_per_point) >= MIN_COUNTS

        if mask.any():
            ratio = ler_mghd[mask] / ler_mwpm[mask]
            ax2.plot(
                p_arr[mask], ratio, f"{mk}-", color=color, markersize=7,
                linewidth=2.0, markeredgecolor="white", markeredgewidth=0.5,
                label=f"$d = {d}$", zorder=3,
            )

    # Reference lines
    ax2.axhline(y=1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax2.axhspan(0.5, 1.0, alpha=0.06, color="green", zorder=0)
    ax2.axhspan(1.0, 3.0, alpha=0.04, color="red", zorder=0)
    ax2.text(0.98, 0.92, "MGHD better", fontsize=8.5, color="green",
             alpha=0.6, ha="right", transform=ax2.get_yaxis_transform())
    ax2.text(0.98, 1.05, "MWPM better", fontsize=8.5, color="red",
             alpha=0.6, ha="right", transform=ax2.get_yaxis_transform())

    _setup_ax(
        ax2,
        "Physical error rate  $p$",
        "LER ratio   (MGHD / MWPM)",
        "(b)  Decoder performance ratio",
    )
    ax2.legend(
        fontsize=10.5, loc="upper right", framealpha=0.90,
        edgecolor="0.8", fancybox=True,
    )

    # Sensible y-limits
    all_ratios = []
    for d in mghd_data:
        mghd_arr = np.array(mghd_data[d]["ler"])
        mwpm_arr = np.array(mwpm_data[d]["ler"])
        m = (mghd_arr > 0) & (mwpm_arr > 0) & ((mwpm_arr * shots_per_point) >= MIN_COUNTS)
        if m.any():
            all_ratios.extend((mghd_arr[m] / mwpm_arr[m]).tolist())
    if all_ratios:
        ax2.set_ylim(0.5, min(max(all_ratios) * 1.15, 4.0))

    # ── Suptitle & annotation ────────────────────────────────────────
    title = "MGHD Neural Decoder vs MWPM Baseline"
    if suptitle_extra:
        title += f"\n{suptitle_extra}"
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)

    fig.text(
        0.5, -0.02,
        f"Circuit-level DEM noise  ·  {shots_per_point:,} shots per point  ·  "
        f"Wilson 95% CIs",
        ha="center", fontsize=9.5, color="0.35", style="italic",
    )

    # ── Save ──────────────────────────────────────────────────────────
    out_dir = Path(output_stem).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_stem).stem

    for fmt in ["pdf", "png"]:
        out = out_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved  {out}")

    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot MGHD vs MWPM LER from evaluation JSON.",
    )
    parser.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="One or more eval JSON files (from mghd.cli.eval or legacy).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path stem (without extension). Default: plots/mghd_vs_mwpm_ler",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Input files use legacy format (flat list of records).",
    )
    parser.add_argument(
        "--shots", type=int, default=10_000,
        help="Shots per point (for ratio filtering). Default: 10000.",
    )
    parser.add_argument(
        "--title", type=str, default="",
        help="Extra text for suptitle.",
    )
    args = parser.parse_args()

    records = load_eval_data(args.input, legacy=args.legacy)
    if not records:
        print("No records loaded – check input files.", file=sys.stderr)
        sys.exit(1)

    mghd_data, mwpm_data = organise_by_distance(records)

    output_stem = args.output or str(
        Path(__file__).resolve().parent.parent / "plots" / "mghd_vs_mwpm_ler"
    )

    plot_mghd_vs_mwpm(
        mghd_data, mwpm_data, output_stem,
        shots_per_point=args.shots,
        suptitle_extra=args.title,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
