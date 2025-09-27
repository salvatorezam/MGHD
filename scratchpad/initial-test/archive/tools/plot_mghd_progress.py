#!/usr/bin/env python3
"""
Plotting script for MGHD progress figures.

Creates two figures:
1. d=3 latency comparison across methods (BP, LSD-clustered, LSD-monolithic, MGHD→LSD)
2. MGHD-primary end-to-end latency breakdown (clustering vs MGHD forward vs parity projection)
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_json_data(filepath):
    """Load and return JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def setup_output_dir(outdir):
    """Ensure output directory exists."""
    Path(outdir).mkdir(parents=True, exist_ok=True)


def create_figure1(comparison_data, outdir, dpi):
    """
    Create Figure 1: d=3 latency (p50/p95/p99) across methods.
    """
    # Method names and labels
    methods = ['A_bp', 'B_lsd_cluster', 'B_lsd_mono', 'C_mghd_guided']
    method_labels = ['BP', 'LSD-clustered', 'LSD-monolithic', 'MGHD→LSD']
    percentiles = ['p50', 'p95', 'p99']
    sides = ['X', 'Z']
    
    # Extract data and convert ms to μs
    data = {}
    for side in sides:
        data[side] = {}
        for method in methods:
            data[side][method] = {}
            for pct in percentiles:
                # Convert from ms to μs
                data[side][method][pct] = comparison_data[side][method]['latency_ms'][pct] * 1000
    
    # Create figure with subplots for X and Z sides
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    
    for idx, side in enumerate(sides):
        ax = axes[idx]
        
        # Prepare bar positions
        x = np.arange(len(methods))
        width = 0.25
        
        # Plot bars for each percentile
        for i, pct in enumerate(percentiles):
            values = [data[side][method][pct] for method in methods]
            bars = ax.bar(x + (i - 1) * width, values, width, 
                         label=pct.upper(), alpha=0.8, color=colors[i])
            
            # Add numeric labels above p95 bars only
            if pct == 'p95':
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Customize subplot
        ax.set_xlabel('Method')
        ax.set_ylabel('Latency (μs)')
        ax.set_title(f'Side {side}')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Add overall title and footnote
    fig.suptitle('d=3 Latency Comparison Across Methods (p=0.005)', fontsize=14, y=0.98)
    fig.text(0.5, 0.02, 'MGHD→LSD = LSD kernel only (MGHD inference not included).', 
             ha='center', fontsize=10, style='italic')
    
    # Add caption
    fig.text(0.5, 0.06, 'Rotated d=3 (p=0.005). MGHD→LSD bars reflect LSD kernel time only; MGHD inference not measured here.',
             ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    for ext in ['png', 'svg']:
        filepath = os.path.join(outdir, f'fig1_d3_latency_methods.{ext}')
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.close()
    
    return data


def create_figure2(clustered_data, outdir, dpi):
    """
    Create Figure 2: MGHD-primary end-to-end latency breakdown.
    """
    sides = ['X', 'Z']
    metrics = ['mean', 'p95']
    components = ['clustering', 'MGHD inference', 'projection']
    
    # Extract data (in ms)
    data = {}
    for side in sides:
        data[side] = {}
        for metric in metrics:
            data[side][metric] = {
                'clustering': clustered_data[side]['t_cluster_ms'][metric],
                'MGHD inference': clustered_data[side]['t_mghd_ms'][metric],
                'projection': clustered_data[side]['t_project_ms'][metric]
            }
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Colors for components
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # light red, light blue, light green
    
    for idx, side in enumerate(sides):
        ax = axes[idx]
        
        # Prepare data for stacked bars
        x = np.arange(len(metrics))
        width = 0.6
        
        # Stack bars for each metric
        bottom_mean = np.zeros(len(metrics))
        for i, comp in enumerate(components):
            values = [data[side][metric][comp] for metric in metrics]
            bars = ax.bar(x, values, width, bottom=bottom_mean, 
                         label=comp, color=colors[i], alpha=0.8)
            
            # Add numbers on the MGHD segment (the dominant part)
            if comp == 'MGHD inference':
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    y_pos = bar.get_y() + height/2
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{height:.3f}', ha='center', va='center', 
                           fontweight='bold', fontsize=10)
            
            bottom_mean += values
        
        # Customize subplot
        ax.set_xlabel('Metric')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Side {side}')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Add overall title
    fig.suptitle('MGHD-primary (clustered), d=3, p=0.005 — end-to-end breakdown', 
                 fontsize=14, y=0.98)
    
    # Add caption
    fig.text(0.5, 0.06, 'Rotated d=3 (p=0.005). MGHD-primary clustered decoder. Bars show mean and p95 component times; MGHD inference dominates.',
             ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    for ext in ['png', 'svg']:
        filepath = os.path.join(outdir, f'fig2_d3_mghd_primary_breakdown.{ext}')
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.close()
    
    return data


def print_summary(comparison_data, clustered_data):
    """Print console summary with speedups and performance metrics."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    # p95 speedups (μs) for C_mghd_guided vs others
    print("\np95 Speedups (MGHD→LSD vs other methods):")
    print("-" * 50)
    for side in ['X', 'Z']:
        print(f"\nSide {side}:")
        mghd_p95_us = comparison_data[side]['C_mghd_guided']['latency_ms']['p95'] * 1000
        lsd_mono_p95_us = comparison_data[side]['B_lsd_mono']['latency_ms']['p95'] * 1000
        bp_p95_us = comparison_data[side]['A_bp']['latency_ms']['p95'] * 1000
        
        speedup_vs_lsd = lsd_mono_p95_us / mghd_p95_us
        speedup_vs_bp = bp_p95_us / mghd_p95_us
        
        print(f"  MGHD→LSD p95: {mghd_p95_us:.1f} μs")
        print(f"  vs LSD-mono:  {speedup_vs_lsd:.1f}× faster ({lsd_mono_p95_us:.1f} μs)")
        print(f"  vs BP:        {speedup_vs_bp:.1f}× faster ({bp_p95_us:.1f} μs)")
    
    # End-to-end per-round mean for MGHD-primary
    print(f"\nMGHD-primary End-to-End Performance:")
    print("-" * 40)
    x_mean = clustered_data['X']['latency_total_ms']['mean']
    z_mean = clustered_data['Z']['latency_total_ms']['mean']
    total_mean = x_mean + z_mean
    baseline_ms = 1.7
    speedup_factor = baseline_ms / total_mean
    
    print(f"MGHD-primary (d=3, p=0.005): total mean per round = {total_mean:.3f} ms → ×{speedup_factor:.1f} faster vs 1.7 ms baseline.")
    
    # Component breakdown for context
    print(f"\nComponent breakdown (mean latency):")
    for side in ['X', 'Z']:
        t_cluster = clustered_data[side]['t_cluster_ms']['mean']
        t_mghd = clustered_data[side]['t_mghd_ms']['mean']
        t_project = clustered_data[side]['t_project_ms']['mean']
        total = clustered_data[side]['latency_total_ms']['mean']
        
        print(f"  Side {side}: {total:.3f} ms total")
        print(f"    Clustering:     {t_cluster:.3f} ms ({100*t_cluster/total:.1f}%)")
        print(f"    MGHD inference: {t_mghd:.3f} ms ({100*t_mghd/total:.1f}%)")
        print(f"    Projection:     {t_project:.3f} ms ({100*t_project/total:.1f}%)")
    
    print("="*60)


def main():
    """Main function to parse arguments and generate figures."""
    parser = argparse.ArgumentParser(description='Generate MGHD progress figures')
    parser.add_argument('--d3-json', default='results/compare_bp_lsd_mghd_d3.json',
                       help='Path to d=3 comparison JSON file')
    parser.add_argument('--cluster-json', default='results/mghd_primary_clustered_d3_p0.005.json',
                       help='Path to clustered MGHD JSON file')
    parser.add_argument('--outdir', default='results/figs',
                       help='Output directory for figures')
    parser.add_argument('--dpi', type=int, default=240,
                       help='DPI for output figures')
    
    args = parser.parse_args()
    
    # Setup
    setup_output_dir(args.outdir)
    
    # Load data
    print(f"Loading comparison data from: {args.d3_json}")
    comparison_data = load_json_data(args.d3_json)
    
    print(f"Loading clustered data from: {args.cluster_json}")
    clustered_data = load_json_data(args.cluster_json)
    
    # Generate figures
    print(f"\nGenerating figures in: {args.outdir}")
    
    # Figure 1: Method comparison
    fig1_data = create_figure1(comparison_data, args.outdir, args.dpi)
    
    # Figure 2: MGHD breakdown
    fig2_data = create_figure2(clustered_data, args.outdir, args.dpi)
    
    # Print summary
    print_summary(comparison_data, clustered_data)


if __name__ == '__main__':
    main()