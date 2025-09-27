#!/usr/bin/env python3
"""
Enhanced multi-sweep comparison plotter with truth labels for MGHD clustered decoder.
Handles T0-only, Mixed-mode, and MGHD-only sweeps with proper microsecond units.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

def load_sweep_data(json_path: str) -> dict:
    """Load sweep data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def infer_sweep_label(data: dict, filepath: str) -> str:
    """Infer sweep label from metadata."""
    metadata = data.get("metadata", {})
    tier0_mode = metadata.get("tier0_mode", "unknown")
    tier0 = metadata.get("tier0", True)
    tier0_k_max = metadata.get("tier0_k_max", 15)
    tier0_r_max = metadata.get("tier0_r_max", 20)
    
    if tier0_mode == "aggressive":
        return f"T0-only (k≤{tier0_k_max}, r≤{tier0_r_max})"
    elif tier0_mode == "mixed":
        return f"Mixed (k≤{tier0_k_max}, r≤{tier0_r_max})"
    elif tier0_mode == "mixed_tight":
        return f"Mixed-tight (k≤{tier0_k_max}, r≤{tier0_r_max})"
    elif tier0_mode == "off":
        return "MGHD-only"
    elif not tier0:
        return "MGHD-only"
    else:
        return f"Custom (k≤{tier0_k_max}, r≤{tier0_r_max})"

def extract_flat_data(sweep_files: list) -> pd.DataFrame:
    """Extract flattened data from multiple sweep files."""
    all_data = []
    
    for filepath in sweep_files:
        data = load_sweep_data(filepath)
        sweep_label = infer_sweep_label(data, filepath)
        
        results = data.get("results", {})
        for d_str, d_data in results.items():
            distance = int(d_str)
            for p_str, p_data in d_data.items():
                p_error = float(p_str)
                for syndrome, s_data in p_data.items():
                    # Extract metrics
                    shots = s_data.get("shots", 0)
                    failures = s_data.get("failures", 0)
                    ler = s_data.get("ler", 0.0)
                    wilson_ci = s_data.get("wilson_ci_upper", 0.0)
                    
                    # Timing data (microseconds)
                    total_us = s_data.get("latency_total_us", {})
                    cluster_us = s_data.get("t_cluster_us", {})
                    tier0_us = s_data.get("t_tier0_us", {})
                    mghd_us = s_data.get("t_mghd_us", {})
                    project_us = s_data.get("t_project_us", {})
                    
                    # Tier-0 stats
                    tier0_stats = s_data.get("tier0_stats", {})
                    tier0_pct = tier0_stats.get("tier0_pct", 0.0)
                    mghd_clusters_per_shot = tier0_stats.get("mghd_clusters_per_shot", 0.0)
                    p_channel_used = tier0_stats.get("p_channel_used", p_error)
                    
                    # Cluster stats
                    cluster_stats = s_data.get("cluster_stats", {})
                    max_cluster_p95 = cluster_stats.get("max_cluster_p95", 0.0)
                    
                    all_data.append({
                        "sweep_label": sweep_label,
                        "filepath": filepath,
                        "distance": distance,
                        "p_error": p_error,
                        "syndrome": syndrome,
                        "shots": shots,
                        "failures": failures,
                        "ler": ler,
                        "wilson_ci_upper": wilson_ci,
                        # Timing (all in microseconds)
                        "total_p50_us": total_us.get("p50", 0.0),
                        "total_p95_us": total_us.get("p95", 0.0),
                        "total_mean_us": total_us.get("mean", 0.0),
                        "cluster_p50_us": cluster_us.get("p50", 0.0),
                        "cluster_p95_us": cluster_us.get("p95", 0.0),
                        "tier0_p50_us": tier0_us.get("p50", 0.0),
                        "tier0_p95_us": tier0_us.get("p95", 0.0),
                        "tier0_mean_nonzero_us": tier0_us.get("mean_nonzero", 0.0),
                        "mghd_p50_us": mghd_us.get("p50", 0.0),
                        "mghd_p95_us": mghd_us.get("p95", 0.0),
                        "mghd_mean_nonzero_us": mghd_us.get("mean_nonzero", 0.0),
                        "mghd_count_nonzero": mghd_us.get("count_nonzero", 0),
                        "project_p50_us": project_us.get("p50", 0.0),
                        "project_p95_us": project_us.get("p95", 0.0),
                        # Routing
                        "tier0_pct": tier0_pct,
                        "mghd_clusters_per_shot": mghd_clusters_per_shot,
                        "p_channel_used": p_channel_used,
                        "max_cluster_p95": max_cluster_p95,
                    })
    
    return pd.DataFrame(all_data)

def format_time_units(values, threshold_ms=1000):
    """Format time values with appropriate units (μs vs ms)."""
    values_array = np.array(values)
    if np.max(values_array) > threshold_ms:
        return values_array / 1000, "ms"
    else:
        return values_array, "μs"

def plot_latency_vs_distance(df: pd.DataFrame, outdir: str, dpi: int):
    """Plot latency vs distance for different error rates."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Aggregate across syndrome types
    agg_df = df.groupby(["sweep_label", "distance", "p_error"]).agg({
        "total_p50_us": "mean",
        "total_p95_us": "mean",
        "tier0_pct": "mean",
        "mghd_clusters_per_shot": "mean",
    }).reset_index()
    
    error_rates = sorted(agg_df["p_error"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agg_df["sweep_label"].unique())))
    
    for i, p in enumerate(error_rates[:4]):  # Limit to 4 subplots
        ax = axes[i]
        p_data = agg_df[agg_df["p_error"] == p]
        
        for j, sweep_label in enumerate(sorted(p_data["sweep_label"].unique())):
            sweep_data = p_data[p_data["sweep_label"] == sweep_label]
            
            if not sweep_data.empty:
                distances = sweep_data["distance"]
                p50_vals = sweep_data["total_p50_us"]
                p95_vals = sweep_data["total_p95_us"]
                
                # Format units
                p50_scaled, p50_unit = format_time_units(p50_vals)
                p95_scaled, p95_unit = format_time_units(p95_vals)
                
                color = colors[j]
                ax.plot(distances, p50_scaled, 'o-', color=color, alpha=0.8, 
                       label=f"{sweep_label} (p50)")
                ax.plot(distances, p95_scaled, 's--', color=color, alpha=0.6,
                       label=f"{sweep_label} (p95)")
                
                # Add truth labels
                for d, p50, p95, tier0, mghd in zip(distances, p50_scaled, p95_scaled, 
                                                   sweep_data["tier0_pct"], 
                                                   sweep_data["mghd_clusters_per_shot"]):
                    ax.annotate(f"T0:{tier0:.0f}%\\nMGHD:{mghd:.2f}", 
                               xy=(d, p95), xytext=(5, 5), 
                               textcoords="offset points", fontsize=8, alpha=0.7)
        
        ax.set_xlabel("Distance")
        ax.set_ylabel(f"Latency [{p50_unit}]")
        ax.set_title(f"p = {p:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_latency_vs_distance_enhanced.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{outdir}/fig_latency_vs_distance_enhanced.svg", bbox_inches="tight")
    plt.close()

def plot_latency_vs_p(df: pd.DataFrame, outdir: str, dpi: int):
    """Plot latency vs error rate for different distances."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Aggregate across syndrome types
    agg_df = df.groupby(["sweep_label", "distance", "p_error"]).agg({
        "total_p50_us": "mean",
        "total_p95_us": "mean",
        "tier0_pct": "mean",
        "mghd_clusters_per_shot": "mean",
    }).reset_index()
    
    distances = sorted(agg_df["distance"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agg_df["sweep_label"].unique())))
    
    for i, d in enumerate(distances[:4]):  # Limit to 4 subplots
        ax = axes[i]
        d_data = agg_df[agg_df["distance"] == d]
        
        for j, sweep_label in enumerate(sorted(d_data["sweep_label"].unique())):
            sweep_data = d_data[d_data["sweep_label"] == sweep_label]
            
            if not sweep_data.empty:
                p_errors = sweep_data["p_error"]
                p50_vals = sweep_data["total_p50_us"]
                p95_vals = sweep_data["total_p95_us"]
                
                # Format units
                p50_scaled, p50_unit = format_time_units(p50_vals)
                p95_scaled, p95_unit = format_time_units(p95_vals)
                
                color = colors[j]
                ax.plot(p_errors, p50_scaled, 'o-', color=color, alpha=0.8,
                       label=f"{sweep_label} (p50)")
                ax.plot(p_errors, p95_scaled, 's--', color=color, alpha=0.6,
                       label=f"{sweep_label} (p95)")
                
                # Add truth labels
                for p, p50, p95, tier0, mghd in zip(p_errors, p50_scaled, p95_scaled,
                                                   sweep_data["tier0_pct"],
                                                   sweep_data["mghd_clusters_per_shot"]):
                    ax.annotate(f"T0:{tier0:.0f}%\\nMGHD:{mghd:.2f}",
                               xy=(p, p95), xytext=(5, 5),
                               textcoords="offset points", fontsize=8, alpha=0.7)
        
        ax.set_xlabel("Error Rate")
        ax.set_ylabel(f"Latency [{p50_unit}]")
        ax.set_title(f"d = {d}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_latency_vs_p_enhanced.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"{outdir}/fig_latency_vs_p_enhanced.svg", bbox_inches="tight")
    plt.close()

def create_summary_table(df: pd.DataFrame, outdir: str) -> str:
    """Create comprehensive summary table."""
    # Aggregate across syndrome types
    agg_df = df.groupby(["sweep_label", "distance", "p_error"]).agg({
        "shots": "sum",
        "failures": "sum", 
        "total_p50_us": "mean",
        "total_p95_us": "mean",
        "mghd_p50_us": "mean",
        "mghd_p95_us": "mean",
        "mghd_mean_nonzero_us": "mean",
        "mghd_count_nonzero": "sum",
        "tier0_pct": "mean",
        "mghd_clusters_per_shot": "mean",
        "wilson_ci_upper": "mean",
    }).reset_index()
    
    # Calculate per-invoke MGHD timing
    agg_df["mghd_p50_per_invoke"] = np.where(
        agg_df["mghd_count_nonzero"] > 0,
        agg_df["mghd_mean_nonzero_us"],
        0.0
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{outdir}/enhanced_sweep_summary_{timestamp}.md"
    
    with open(output_path, 'w') as f:
        f.write("# MGHD Enhanced Multi-Sweep Comparison\\n\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("## Performance Summary Table\\n\\n")
        f.write("| Distance | Error Rate | Mode | T0% | MGHD/shot | total p50 [μs] | total p95 [μs] | t_MGHD p50/invoke [μs] | failures/shots | Wilson CI Upper |\\n")
        f.write("|----------|------------|------|-----|-----------|---------------|---------------|----------------------|----------------|------------------|\\n")
        
        for _, row in agg_df.iterrows():
            # Format timing values
            p50_val, p50_unit = format_time_units([row["total_p50_us"]])
            p95_val, p95_unit = format_time_units([row["total_p95_us"]])
            mghd_p50_val, mghd_p50_unit = format_time_units([row["mghd_p50_per_invoke"]])
            
            f.write(f"| {row['distance']} | {row['p_error']:.3f} | {row['sweep_label']} | "
                   f"{row['tier0_pct']:.1f}% | {row['mghd_clusters_per_shot']:.3f} | "
                   f"{p50_val[0]:.1f} | {p95_val[0]:.1f} | {mghd_p50_val[0]:.1f} | "
                   f"{int(row['failures'])}/{int(row['shots'])} | {row['wilson_ci_upper']:.2e} |\\n")
        
        f.write("\\n## Key Findings\\n\\n")
        
        # Analyze findings
        t0_only_data = agg_df[agg_df["sweep_label"].str.contains("T0-only")]
        mixed_data = agg_df[agg_df["sweep_label"].str.contains("Mixed")]
        mghd_only_data = agg_df[agg_df["sweep_label"].str.contains("MGHD-only")]
        
        if not t0_only_data.empty:
            f.write(f"- **T0-only Performance**: Consistent {t0_only_data['tier0_pct'].iloc[0]:.0f}% tier-0 routing\\n")
            
        if not mixed_data.empty:
            f.write(f"- **Mixed-mode Performance**: {mixed_data['tier0_pct'].mean():.1f}% avg tier-0 routing\\n")
            
        if not mghd_only_data.empty:
            avg_mghd_latency = mghd_only_data['mghd_p50_per_invoke'].mean()
            f.write(f"- **MGHD-only Performance**: {avg_mghd_latency:.0f} μs average per-invoke latency\\n")
        
        f.write("\\n## Methodology\\n\\n")
        f.write("- **Timing**: All measurements in microseconds (μs) with CUDA synchronization\\n")
        f.write("- **LER**: Logical Error Rate with Wilson confidence interval upper bounds\\n")
        f.write("- **Tier-0**: Fast channel-only solver for small clusters\\n")
        f.write("- **MGHD**: Neural network decoder for larger clusters\\n")
        f.write("- **Per-invoke timing**: MGHD timing divided by number of invocations\\n")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Enhanced multi-sweep comparison plotter")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON sweep files")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PNG output")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    print(f"Loading {len(args.inputs)} sweep datasets...")
    df = extract_flat_data(args.inputs)
    print(f"Extracted {len(df)} data points across {df['sweep_label'].nunique()} sweeps")
    
    # Generate plots
    print("Generating enhanced comparison plots...")
    plot_latency_vs_distance(df, args.outdir, args.dpi)
    plot_latency_vs_p(df, args.outdir, args.dpi)
    
    # Create summary table
    summary_path = create_summary_table(df, args.outdir)
    
    print(f"Enhanced plots and summary written to {args.outdir}")
    print(f"Summary table: {summary_path}")

if __name__ == "__main__":
    main()