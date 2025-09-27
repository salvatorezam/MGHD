#!/usr/bin/env python3
"""
Generate machine-readable summary table for MGHD clustered decoder benchmarks.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def extract_summary_data(json_path: str) -> list:
    """Extract summary data from a sweep JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    tier0_mode = metadata.get('tier0_mode', 'unknown')
    tier0_k_max = metadata.get('tier0_k_max', 'unknown')
    tier0_r_max = metadata.get('tier0_r_max', 'unknown')
    
    # Create mode label
    if tier0_mode == "aggressive":
        mode_label = f"T0-aggressive (k≤{tier0_k_max}, r≤{tier0_r_max})"
    elif tier0_mode == "mixed":
        mode_label = f"Mixed (k≤{tier0_k_max}, r≤{tier0_r_max})"
    elif tier0_mode == "mixed_tight":
        mode_label = f"Mixed-tight (k≤{tier0_k_max}, r≤{tier0_r_max})"
    elif tier0_mode == "off":
        mode_label = "MGHD-only"
    else:
        mode_label = f"Custom (k≤{tier0_k_max}, r≤{tier0_r_max})"
    
    results = data.get('results', {})
    summary_rows = []
    
    for d_str, d_data in results.items():
        distance = int(d_str)
        for p_str, p_data in d_data.items():
            p_error = float(p_str)
            
            # Combine X and Z statistics (average where appropriate)
            x_data = p_data.get('X', {})
            z_data = p_data.get('Z', {})
            
            if not x_data or not z_data:
                continue
            
            # Tier-0 percentage (average of X and Z)
            x_tier0_pct = x_data.get('tier0_stats', {}).get('tier0_pct', 0.0)
            z_tier0_pct = z_data.get('tier0_stats', {}).get('tier0_pct', 0.0)
            tier0_pct = (x_tier0_pct + z_tier0_pct) / 2.0
            
            # MGHD clusters per shot (average of X and Z)
            x_mghd_per_shot = x_data.get('tier0_stats', {}).get('mghd_clusters_per_shot', 0.0)
            z_mghd_per_shot = z_data.get('tier0_stats', {}).get('mghd_clusters_per_shot', 0.0)
            mghd_clusters_per_shot = (x_mghd_per_shot + z_mghd_per_shot) / 2.0
            
            # Total latency (average of X and Z p50/p95)
            x_latency = x_data.get('latency_total_us', {})
            z_latency = z_data.get('latency_total_us', {})
            total_p50 = (x_latency.get('p50', 0.0) + z_latency.get('p50', 0.0)) / 2.0
            total_p95 = (x_latency.get('p95', 0.0) + z_latency.get('p95', 0.0)) / 2.0
            
            # MGHD non-zero statistics (average of X and Z)
            x_mghd_nz = x_data.get('t_mghd_nonzero_stats', {})
            z_mghd_nz = z_data.get('t_mghd_nonzero_stats', {})
            mghd_nz_p50 = (x_mghd_nz.get('p50_nonzero_us', 0.0) + z_mghd_nz.get('p50_nonzero_us', 0.0)) / 2.0
            mghd_nz_p95 = (x_mghd_nz.get('p95_nonzero_us', 0.0) + z_mghd_nz.get('p95_nonzero_us', 0.0)) / 2.0
            
            # Failures (sum X and Z)
            x_failures = x_data.get('failures', 0)
            z_failures = z_data.get('failures', 0)
            total_failures = x_failures + z_failures
            total_shots = x_data.get('shots', 0) + z_data.get('shots', 0)
            
            # Wilson CI upper (average of X and Z)
            x_wilson = x_data.get('wilson_ci_upper', 0.0)
            z_wilson = z_data.get('wilson_ci_upper', 0.0)
            wilson_hi = (x_wilson + z_wilson) / 2.0
            
            summary_rows.append({
                'd': distance,
                'p': p_error,
                'mode': mode_label,
                'tier0_pct': tier0_pct,
                'mghd_clusters_per_shot': mghd_clusters_per_shot,
                'total_p50_us': total_p50,
                'total_p95_us': total_p95,
                't_mghd_nonzero_p50_us': mghd_nz_p50,
                't_mghd_nonzero_p95_us': mghd_nz_p95,
                'failures_per_shots': f"{total_failures}/{total_shots}",
                'wilson_hi': wilson_hi
            })
    
    return summary_rows


def create_summary_table(input_files: list, output_path: str) -> str:
    """Create summary markdown table from multiple JSON files."""
    all_rows = []
    
    for json_file in input_files:
        rows = extract_summary_data(json_file)
        all_rows.extend(rows)
    
    # Sort by distance, then p, then mode
    all_rows.sort(key=lambda r: (r['d'], r['p'], r['mode']))
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f"clustered_surface_sweep_summary_{timestamp}.md")
    
    with open(output_file, 'w') as f:
        f.write("# MGHD Clustered Decoder Performance Summary\n\n")
        f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
        f.write("## Summary Table\n\n")
        
        # Header
        f.write("| d | p | mode | Tier-0% | MGHD cl/shot | total_p50[μs] | total_p95[μs] | t_MGHD_nonzero_p50[μs] | t_MGHD_nonzero_p95[μs] | failures/shots | wilson_hi |\n")
        f.write("|---|---|------|---------|--------------|---------------|---------------|------------------------|------------------------|----------------|-----------|\\n")
        
        # Data rows
        for row in all_rows:
            f.write(f"| {row['d']} | {row['p']:.3f} | {row['mode']} | "
                   f"{row['tier0_pct']:.1f}% | {row['mghd_clusters_per_shot']:.3f} | "
                   f"{row['total_p50_us']:.1f} | {row['total_p95_us']:.1f} | "
                   f"{row['t_mghd_nonzero_p50_us']:.1f} | {row['t_mghd_nonzero_p95_us']:.1f} | "
                   f"{row['failures_per_shots']} | {row['wilson_hi']:.2e} |\\n")
        
        f.write("\\n## Notes\\n\\n")
        f.write("- **Timing**: All measurements in microseconds (μs) with CUDA synchronization\\n")
        f.write("- **Tier-0%**: Percentage of clusters solved by fast channel-only solver\\n")
        f.write("- **MGHD cl/shot**: Average number of clusters per shot routed to neural decoder\\n")
        f.write("- **total_p50/p95**: Percentiles of end-to-end latency per shot\\n")
        f.write("- **t_MGHD_nonzero**: MGHD timing statistics excluding zero invocations\\n")
        f.write("- **wilson_hi**: Wilson confidence interval upper bound for logical error rate\\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate MGHD performance summary table")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON sweep files")
    parser.add_argument("--outdir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    # Generate summary table
    summary_file = create_summary_table(args.inputs, args.outdir)
    
    print(f"Summary table written to: {summary_file}")
    
    # Also print a compact comparison for d=3, p=0.01 if available
    print("\\n=== Quick d=3, p=0.010 Comparison ===")
    
    for json_file in args.inputs:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            mode = metadata.get('tier0_mode', 'unknown')
            
            results = data.get('results', {})
            d3_data = results.get('3', {})
            p01_data = d3_data.get('0.010', {})
            
            if not p01_data:
                continue
            
            x_data = p01_data.get('X', {})
            z_data = p01_data.get('Z', {})
            
            if not x_data or not z_data:
                continue
            
            # Extract key metrics
            tier0_pct = (x_data.get('tier0_stats', {}).get('tier0_pct', 0.0) + 
                        z_data.get('tier0_stats', {}).get('tier0_pct', 0.0)) / 2.0
            mghd_per_shot = (x_data.get('tier0_stats', {}).get('mghd_clusters_per_shot', 0.0) + 
                           z_data.get('tier0_stats', {}).get('mghd_clusters_per_shot', 0.0)) / 2.0
            
            total_p50 = (x_data.get('latency_total_us', {}).get('p50', 0.0) + 
                        z_data.get('latency_total_us', {}).get('p50', 0.0)) / 2.0
            total_p95 = (x_data.get('latency_total_us', {}).get('p95', 0.0) + 
                        z_data.get('latency_total_us', {}).get('p95', 0.0)) / 2.0
            
            mghd_nz_p50 = (x_data.get('t_mghd_nonzero_stats', {}).get('p50_nonzero_us', 0.0) + 
                          z_data.get('t_mghd_nonzero_stats', {}).get('p50_nonzero_us', 0.0)) / 2.0
            mghd_nz_p95 = (x_data.get('t_mghd_nonzero_stats', {}).get('p95_nonzero_us', 0.0) + 
                          z_data.get('t_mghd_nonzero_stats', {}).get('p95_nonzero_us', 0.0)) / 2.0
            
            print(f"{mode:12} | Tier-0: {tier0_pct:4.1f}% | MGHD: {mghd_per_shot:.3f}/shot | "
                  f"p50/p95: {total_p50:5.1f}/{total_p95:5.1f}μs | "
                  f"MGHD_nz: {mghd_nz_p50:5.1f}/{mghd_nz_p95:5.1f}μs")
                  
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    main()