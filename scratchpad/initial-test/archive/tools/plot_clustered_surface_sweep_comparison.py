#!/usr/bin/env python3
"""
Enhanced plotter for clustered surface sweep results with multi-input comparison support.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def extract_flat_data(sweep_data: List[Tuple[Dict, str, str]]) -> pd.DataFrame:
    """Extract flat data from all sweeps for analysis and plotting."""
    records = []
    
    for data, label, filename in sweep_data:
        metadata = data.get("metadata", {})
        
        for d_str, d_data in data["results"].items():
            d = int(d_str)
            for p_str, p_data in d_data.items():
                p = float(p_str)
                for side, side_data in p_data.items():
                    # Extract timing and stats
                    latency_us = side_data.get("latency_total_us", {})
                    tier0_stats = side_data.get("tier0_stats", {})
                    
                    record = {
                        "sweep_label": label,
                        "distance": d,
                        "p_error": p,
                        "syndrome_type": side,
                        "shots": side_data.get("shots", 0),
                        "failures": side_data.get("failures", 0),
                        "ler": side_data.get("ler", 0.0),
                        "wilson_ci_upper": side_data.get("wilson_ci_upper", 0.0),
                        "latency_p50_us": latency_us.get("p50", 0.0),
                        "latency_p95_us": latency_us.get("p95", 0.0),
                        "latency_mean_us": latency_us.get("mean", 0.0),
                        "tier0_pct": tier0_stats.get("tier0_pct", 0.0),
                        "mghd_clusters_per_shot": tier0_stats.get("mghd_clusters_per_shot", 0.0),
                        "total_clusters": tier0_stats.get("total_clusters", 0),
                        "cluster_stats": side_data.get("cluster_stats", {}),
                    }
                    records.append(record)
    
    return pd.DataFrame(records)


def plot_latency_vs_distance(df: pd.DataFrame, outdir: str, dpi: int):
    """Plot latency p95 and mean vs distance for each sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by sweep and compute aggregate metrics across syndrome types
    agg_df = df.groupby(["sweep_label", "distance", "p_error"]).agg({
        "latency_p95_us": "mean",
        "latency_mean_us": "mean",
    }).reset_index()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df["sweep_label"].unique())))
    
    for i, (sweep_label, sweep_df) in enumerate(agg_df.groupby("sweep_label")):
        color = colors[i]
        
        # Plot p95 latency
        for p in sorted(sweep_df["p_error"].unique()):
            p_df = sweep_df[sweep_df["p_error"] == p]
            ax1.plot(p_df["distance"], p_df["latency_p95_us"], 
                    marker='o', label=f"{sweep_label} (p={p})", color=color, alpha=0.7 + 0.3*p/0.015)
        
        # Plot mean latency  
        for p in sorted(sweep_df["p_error"].unique()):
            p_df = sweep_df[sweep_df["p_error"] == p]
            ax2.plot(p_df["distance"], p_df["latency_mean_us"], 
                    marker='s', label=f"{sweep_label} (p={p})", color=color, alpha=0.7 + 0.3*p/0.015)
    
    ax1.set_xlabel("Distance")
    ax1.set_ylabel("p95 Latency (μs)")
    ax1.set_title("p95 Latency vs Distance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("Mean Latency (μs)")
    ax2.set_title("Mean Latency vs Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_latency_vs_distance.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{outdir}/fig_latency_vs_distance.svg", bbox_inches="tight")
    plt.close()


def plot_latency_vs_p(df: pd.DataFrame, outdir: str, dpi: int):
    """Plot latency vs p with subplots for each distance."""
    distances = sorted(df["distance"].unique())
    ncols = min(len(distances), 3)
    nrows = (len(distances) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df["sweep_label"].unique())))
    
    for i, d in enumerate(distances):
        ax = axes[i]
        d_df = df[df["distance"] == d]
        
        # Aggregate across syndrome types
        agg_df = d_df.groupby(["sweep_label", "p_error"]).agg({
            "latency_p95_us": "mean",
            "tier0_pct": "mean",
            "mghd_clusters_per_shot": "mean",
        }).reset_index()
        
        for j, (sweep_label, sweep_df) in enumerate(agg_df.groupby("sweep_label")):
            color = colors[j]
            ax.plot(sweep_df["p_error"], sweep_df["latency_p95_us"], 
                   marker='o', label=sweep_label, color=color)
            
            # Add annotations for tier0% and MGHD clusters/shot
            for _, row in sweep_df.iterrows():
                ax.annotate(f"T0:{row['tier0_pct']:.0f}%\nMGHD:{row['mghd_clusters_per_shot']:.3f}/shot",
                           xy=(row["p_error"], row["latency_p95_us"]),
                           xytext=(5, 5), textcoords="offset points",
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel("Error Rate p")
        ax.set_ylabel("p95 Latency (μs)")
        ax.set_title(f"Distance d={d}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    
    # Hide unused subplots
    for i in range(len(distances), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_latency_vs_p.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{outdir}/fig_latency_vs_p.svg", bbox_inches="tight")
    plt.close()


def plot_routing(df: pd.DataFrame, outdir: str, dpi: int):
    """Plot stacked Tier-0/MGHD routing by clusters with MGHD clusters/shot line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Aggregate data across syndrome types
    agg_df = df.groupby(["sweep_label", "distance", "p_error"]).agg({
        "tier0_pct": "mean",
        "mghd_clusters_per_shot": "mean",
    }).reset_index()
    
    # Get all unique combinations of distance and p_error
    all_combinations = []
    for d in sorted(agg_df["distance"].unique()):
        for p in sorted(agg_df["p_error"].unique()):
            if not agg_df[(agg_df["distance"] == d) & (agg_df["p_error"] == p)].empty:
                all_combinations.append((d, p))
    
    x_labels = [f"d={d}\np={p:.3f}" for d, p in all_combinations]
    x = np.arange(len(x_labels))
    
    sweeps = sorted(agg_df["sweep_label"].unique())
    width = 0.35
    colors = plt.cm.tab10(np.linspace(0, 1, len(sweeps)))
    
    for i, sweep_label in enumerate(sweeps):
        tier0_vals = []
        mghd_vals = []
        
        for d, p in all_combinations:
            subset = agg_df[(agg_df["sweep_label"] == sweep_label) & 
                           (agg_df["distance"] == d) & 
                           (agg_df["p_error"] == p)]
            
            if not subset.empty:
                tier0_pct = subset.iloc[0]["tier0_pct"]
                tier0_vals.append(tier0_pct)
                mghd_vals.append(100.0 - tier0_pct)
            else:
                tier0_vals.append(0)
                mghd_vals.append(0)
        
        color = colors[i]
        x_pos = x + i * width
        
        ax.bar(x_pos, tier0_vals, width, label=f"{sweep_label} (Tier-0)", 
               color=color, alpha=0.7)
        ax.bar(x_pos, mghd_vals, width, bottom=tier0_vals, 
               label=f"{sweep_label} (MGHD)", color=color, alpha=0.4)
    
    ax.set_xlabel("Distance × Error Rate")
    ax.set_ylabel("Cluster Routing (%)")
    ax.set_title("Tier-0 vs MGHD Routing by Clusters")
    ax.set_xticks(x + width * (len(sweeps) - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_routing.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{outdir}/fig_routing.svg", bbox_inches="tight")
    plt.close()


def plot_kappa_heatmap(df: pd.DataFrame, outdir: str, dpi: int):
    """Plot κ_max p95 heatmap with tier-0 annotations."""
    # Extract cluster max sizes (kappa values)
    kappa_data = []
    
    for _, row in df.iterrows():
        cluster_stats = row["cluster_stats"]
        if cluster_stats:
            kappa_p95 = cluster_stats.get("max_cluster_p95", 0.0)
            kappa_data.append({
                "sweep_label": row["sweep_label"],
                "distance": row["distance"],
                "p_error": row["p_error"],
                "syndrome_type": row["syndrome_type"],
                "kappa_p95": kappa_p95,
                "tier0_pct": row["tier0_pct"],
            })
    
    kappa_df = pd.DataFrame(kappa_data)
    
    # Aggregate across syndrome types and sweeps
    agg_kappa = kappa_df.groupby(["distance", "p_error"]).agg({
        "kappa_p95": "mean",
        "tier0_pct": "mean",
    }).reset_index()
    
    # Create pivot table for heatmap
    distances = sorted(agg_kappa["distance"].unique())
    ps = sorted(agg_kappa["p_error"].unique())
    
    kappa_matrix = np.zeros((len(distances), len(ps)))
    tier0_matrix = np.zeros((len(distances), len(ps)))
    
    for i, d in enumerate(distances):
        for j, p in enumerate(ps):
            subset = agg_kappa[(agg_kappa["distance"] == d) & (agg_kappa["p_error"] == p)]
            if not subset.empty:
                kappa_matrix[i, j] = subset.iloc[0]["kappa_p95"]
                tier0_matrix[i, j] = subset.iloc[0]["tier0_pct"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(kappa_matrix, cmap="viridis", aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("κ_max p95 (cluster size)")
    
    # Add tier-0 annotations
    for i in range(len(distances)):
        for j in range(len(ps)):
            tier0_pct = tier0_matrix[i, j]
            text_color = "white" if kappa_matrix[i, j] > kappa_matrix.max()/2 else "black"
            ax.text(j, i, f"{tier0_pct:.0f}%", ha="center", va="center", 
                   color=text_color, fontsize=10, weight="bold")
    
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"{p:.3f}" for p in ps])
    ax.set_yticks(range(len(distances)))
    ax.set_yticklabels([f"d={d}" for d in distances])
    ax.set_xlabel("Error Rate p")
    ax.set_ylabel("Distance")
    ax.set_title("κ_max p95 Heatmap with Tier-0 Coverage (%)")
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_kappa_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{outdir}/fig_kappa_heatmap.svg", bbox_inches="tight")
    plt.close()


def generate_summary_table(df: pd.DataFrame, outdir: str) -> str:
    """Generate markdown summary table."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"{outdir}/clustered_surface_sweep_summary_{timestamp}.md"
    
    with open(summary_path, "w") as f:
        f.write("# MGHD Clustered Decoder: Mixed-Mode A/B Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Summary Table\n\n")
        f.write("| Distance | Error Rate | Mode | Tier-0 % | MGHD/shot | p50 [μs] | p95 [μs] | Failures/Shots | Wilson CI Upper |\n")
        f.write("|----------|------------|------|----------|-----------|----------|----------|----------------|------------------|\n")
        
        # Sort by distance, p, then sweep label
        sorted_df = df.sort_values(["distance", "p_error", "sweep_label"])
        
        # Aggregate across syndrome types for cleaner table
        agg_df = sorted_df.groupby(["distance", "p_error", "sweep_label"]).agg({
            "tier0_pct": "mean",
            "mghd_clusters_per_shot": "mean",
            "latency_p50_us": "mean",
            "latency_p95_us": "mean",
            "failures": "sum",
            "shots": "sum",
            "wilson_ci_upper": "mean",
        }).reset_index()
        
        for _, row in agg_df.iterrows():
            f.write(f"| {row['distance']} | {row['p_error']:.3f} | {row['sweep_label']} | "
                   f"{row['tier0_pct']:.1f}% | {row['mghd_clusters_per_shot']:.3f} | "
                   f"{row['latency_p50_us']:.1f} | {row['latency_p95_us']:.1f} | "
                   f"{row['failures']:.0f}/{row['shots']:.0f} | {row['wilson_ci_upper']:.2e} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Calculate speedups and insights
        baseline_sweep = None
        for sweep in agg_df["sweep_label"].unique():
            if "MGHD" in sweep and ("only" in sweep or "off" in sweep):
                baseline_sweep = sweep
                break
        
        if baseline_sweep:
            f.write(f"**Baseline**: {baseline_sweep}\n\n")
            baseline_data = agg_df[agg_df["sweep_label"] == baseline_sweep]
            
            for _, baseline_row in baseline_data.iterrows():
                d, p = baseline_row["distance"], baseline_row["p_error"]
                baseline_latency = baseline_row["latency_p95_us"]
                
                other_sweeps = agg_df[(agg_df["distance"] == d) & 
                                    (agg_df["p_error"] == p) & 
                                    (agg_df["sweep_label"] != baseline_sweep)]
                
                for _, other_row in other_sweeps.iterrows():
                    speedup = baseline_latency / other_row["latency_p95_us"] if other_row["latency_p95_us"] > 0 else float('inf')
                    f.write(f"- **d={d}, p={p:.3f}**: {other_row['sweep_label']} achieves {speedup:.1f}x speedup "
                           f"({baseline_latency:.0f}μs → {other_row['latency_p95_us']:.0f}μs) with "
                           f"{other_row['tier0_pct']:.0f}% Tier-0 coverage\n")
        
        f.write(f"\n## Methodology\n\n")
        f.write("- **Timing**: All measurements in microseconds (μs) with CUDA synchronization\n")
        f.write("- **LER**: Logical Error Rate with Wilson confidence interval upper bounds\n")
        f.write("- **Tier-0**: Fast channel-only solver for small clusters\n")
        f.write("- **MGHD**: Neural network decoder for larger clusters\n")
        f.write("- **Mixed-Mode**: Configurable routing between Tier-0 and MGHD\n")
    
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Plot multi-input clustered surface sweep comparison")
    parser.add_argument("--inputs", nargs="+", required=True, 
                       help="Input JSON files from sweep runs")
    parser.add_argument("--outdir", default="results/figs")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load all sweep data
    sweep_data = load_sweep_data(args.inputs)
    print(f"Loaded {len(sweep_data)} sweep datasets:")
    for data, label, filename in sweep_data:
        print(f"  - {label}: {filename}")
    
    # Extract flat data for analysis
    df = extract_flat_data(sweep_data)
    print(f"Extracted {len(df)} data points across {len(df['sweep_label'].unique())} sweeps")
    
    # Generate plots
    print("Generating plots...")
    plot_latency_vs_distance(df, args.outdir, args.dpi)
    plot_latency_vs_p(df, args.outdir, args.dpi)
    plot_routing(df, args.outdir, args.dpi)
    plot_kappa_heatmap(df, args.outdir, args.dpi)
    
    # Generate summary table
    summary_path = generate_summary_table(df, args.outdir)
    
    print(f"Figures and summary written to {args.outdir}")
    print(f"Summary table: {summary_path}")


if __name__ == "__main__":
    main()