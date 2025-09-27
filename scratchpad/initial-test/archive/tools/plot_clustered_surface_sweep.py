from __future__ import annotations

import argparse
import json
import os
import glob
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_data(input_files: List[str]) -> List[Tuple[Dict, str, str]]:
    """Load multiple sweep files and return list of (data, label, filename) tuples."""
    sweep_data = []
    
    for i, input_file in enumerate(input_files):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Generate label from filename or metadata
        filename = os.path.basename(input_file)
        metadata = data.get("metadata", {})
        tier0_mode = metadata.get("tier0_mode")
        
        if tier0_mode == "aggressive":
            label = "T0-only (aggressive)"
        elif tier0_mode == "mixed":
            label = "Mixed (k≤5,r≤6)"
        elif tier0_mode == "off":
            label = "MGHD-only"
        elif tier0_mode is None:
            # Custom mode - check tier0 settings
            if metadata.get("tier0", True):
                k_max = metadata.get("tier0_k_max", "?")
                r_max = metadata.get("tier0_r_max", "?")
                if k_max == 1 and r_max == 1:
                    label = "Mixed (k≤1,r≤1)"
                else:
                    label = f"T0 (k≤{k_max},r≤{r_max})"
            else:
                label = "MGHD-only"
        else:
            label = f"Sweep {i+1}"
        
        sweep_data.append((data, label, filename))
    
    return sweep_data


def load_json(path_pattern: str) -> Dict:
    """Legacy function for single file loading."""
    paths = sorted(glob.glob(path_pattern))
    if not paths:
        raise FileNotFoundError(f"No files match pattern {path_pattern}")
    if len(paths) > 1:
        print(f"[plot] Multiple matches, using {paths[-1]}")
    with open(paths[-1], "r", encoding="utf-8") as f:
        return json.load(f), paths[-1]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_metrics(data: Dict) -> Tuple[List[int], List[float], Dict[Tuple[int, float], Dict[str, float]]]:
    dists = sorted(map(int, data["results"].keys()))
    ps = sorted({float(p) for d in data["results"].values() for p in d.keys()})
    metrics = {}
    for d in dists:
        for p in ps:
            entry = data["results"][str(d)][f"{p:.3f}"]
            X = entry["X"]
            Z = entry["Z"]
            total_mean = X["latency_total_ms"]["mean"] + Z["latency_total_ms"]["mean"]
            total_p95 = X["latency_total_ms"]["p95"] + Z["latency_total_ms"]["p95"]
            tier0_pct = (X["tier0_stats"]["tier0_pct"] + Z["tier0_stats"]["tier0_pct"]) / 2.0
            mghd_per_shot = (
                X["tier0_stats"]["mghd_clusters"] + Z["tier0_stats"]["mghd_clusters"]
            ) / (X["shots"] + Z["shots"]) * 2.0
            max_cluster_p95 = max(
                X["cluster_stats"]["max_cluster_p95"],
                Z["cluster_stats"]["max_cluster_p95"],
            )
            metrics[(d, p)] = dict(
                total_mean=total_mean,
                total_p95=total_p95,
                tier0_pct=tier0_pct,
                mghd_clusters_per_shot=mghd_per_shot,
                max_cluster_p95=max_cluster_p95,
            )
    return dists, sorted(ps), metrics


def autoscale_latency(values: List[float]) -> Tuple[List[float], str]:
    max_val = max(values) if values else 0.0
    if max_val < 0.001:
        return [v * 1e6 for v in values], "µs"
    if max_val < 1.0:
        return [v * 1e3 for v in values], "µs"
    return values, "ms"


def plot_latency_vs_distance(dists, ps, metrics, outdir, dpi):
    n_cols = min(3, len(ps))
    n_rows = int(np.ceil(len(ps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)
    all_vals = []
    for p in ps:
        for d in dists:
            all_vals.append(metrics[(d, p)]["total_p95"])
    scaled_vals, unit = autoscale_latency(all_vals)
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if idx >= len(ps):
                axes[r, c].axis("off")
                continue
            p = ps[idx]
            y_vals = [metrics[(d, p)]["total_p95"] for d in dists]
            y_scaled, _ = autoscale_latency(y_vals)
            axes[r, c].plot(dists, y_scaled, marker="o", label=f"p95")
            axes[r, c].set_title(f"p={p:.3f}")
            axes[r, c].set_xlabel("distance")
            axes[r, c].set_ylabel(f"Total latency (p95) [{unit}]")
            axes[r, c].grid(True, alpha=0.3)
            idx += 1
    fig.suptitle("Per-round latency (p95) vs distance")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "fig_latency_vs_distance.png"), dpi=dpi)
    fig.savefig(os.path.join(outdir, "fig_latency_vs_distance.svg"))
    plt.close(fig)


def plot_latency_vs_p(dists, ps, metrics, outdir, dpi):
    n_cols = min(3, len(dists))
    n_rows = int(np.ceil(len(dists) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)
    for idx, d in enumerate(dists):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        means = [metrics[(d, p)]["total_mean"] for p in ps]
        p95s = [metrics[(d, p)]["total_p95"] for p in ps]
        tier0 = [metrics[(d, p)]["tier0_pct"] for p in ps]
        means_scaled, unit = autoscale_latency(means)
        p95_scaled, _ = autoscale_latency(p95s)
        ax.plot(ps, means_scaled, marker="o", label="mean")
        ax.plot(ps, p95_scaled, marker="s", label="p95")
        for p_val, t_pct in zip(ps, tier0):
            ax.annotate(f"{t_pct:.0f}%", (p_val, p95_scaled[ps.index(p_val)]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        ax.set_xlabel("error rate p")
        ax.set_ylabel(f"Latency [{unit}]")
        ax.set_title(f"distance {d}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for idx in range(len(dists), n_rows * n_cols):
        axes.flat[idx].axis("off")
    fig.suptitle("Per-round latency vs p (mean & p95)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "fig_latency_vs_p.png"), dpi=dpi)
    fig.savefig(os.path.join(outdir, "fig_latency_vs_p.svg"))
    plt.close(fig)


def plot_routing(dists, ps, metrics, outdir, dpi):
    combos = [(d, p) for d in dists for p in ps]
    tier0 = [metrics[(d, p)]["tier0_pct"] for d, p in combos]
    mghd = [metrics[(d, p)]["mghd_clusters_per_shot"] for d, p in combos]
    x = np.arange(len(combos))
    fig, ax1 = plt.subplots(figsize=(max(6, len(combos) * 0.5), 4))
    ax2 = ax1.twinx()
    ax1.bar(x, tier0, color="#4C72B0", alpha=0.7, label="Tier-0 %")
    ax2.plot(x, mghd, color="#DD8452", marker="o", label="MGHD clusters/shot")
    ax1.set_ylabel("Tier-0 clusters [%]")
    ax2.set_ylabel("MGHD clusters / shot")
    labels = [f"d{d}\np{p:.3f}" for d, p in combos]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylim(0, 110)
    ax1.grid(True, axis="y", alpha=0.3)
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="upper right")
    fig.suptitle("Routing mix across (d,p)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig_routing.png"), dpi=dpi)
    fig.savefig(os.path.join(outdir, "fig_routing.svg"))
    plt.close(fig)


def plot_kappa_heatmap(dists, ps, metrics, outdir, dpi):
    mat = np.zeros((len(ps), len(dists)))
    for i, p in enumerate(ps):
        for j, d in enumerate(dists):
            mat[i, j] = metrics[(d, p)]["max_cluster_p95"]
    fig, ax = plt.subplots(figsize=(1.5 * len(dists), 0.5 * len(ps) + 2))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(dists)))
    ax.set_yticks(np.arange(len(ps)))
    ax.set_xticklabels(dists)
    ax.set_yticklabels([f"{p:.3f}" for p in ps])
    ax.set_xlabel("distance")
    ax.set_ylabel("error rate p")
    ax.set_title("kappa_max p95 (largest cluster size)")
    for i in range(len(ps)):
        for j in range(len(dists)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="white" if mat[i, j] > mat.max()/2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="cluster size")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_kappa_heatmap.png"), dpi=dpi)
    fig.savefig(os.path.join(outdir, "fig_kappa_heatmap.svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clustered surface sweep results")
    parser.add_argument("--json", required=True, help="Path or glob to sweep JSON file")
    parser.add_argument("--outdir", default="results/figs")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    data, path_used = load_json(args.json)
    dists, ps, metrics = collect_metrics(data)
    print(f"Loaded sweep data from {path_used}")

    plot_latency_vs_distance(dists, ps, metrics, args.outdir, args.dpi)
    plot_latency_vs_p(dists, ps, metrics, args.outdir, args.dpi)
    plot_routing(dists, ps, metrics, args.outdir, args.dpi)
    plot_kappa_heatmap(dists, ps, metrics, args.outdir, args.dpi)
    print(f"Figures written to {args.outdir}")


if __name__ == "__main__":
    main()
