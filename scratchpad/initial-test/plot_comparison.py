#!/usr/bin/env python3
"""
Plot comparison between GNN baseline and MGHD performance across error rates.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(filepath):
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison():
    """Create comparison plot of GNN baseline vs MGHD vs MWPM vs MWPF."""
    
    # Load data
    gnn_data = load_results('results/gnn_baseline_evaluation_p_grid.json')
    mghd_data = load_results('results/foundation_S_core_cq_circuit_v1_20250831_093641/ler_S_grid.json')
    mwpm_data = load_results('results/mwpm_evaluation_p_grid.json')
    mwpf_data = load_results('results/mwpf_evaluation_p_grid.json')
    
    # Extract data for plotting
    gnn_p = [result['p'] for result in gnn_data['results']]
    gnn_ler = [result['LER'] for result in gnn_data['results']]
    gnn_ler_low = [result['ler_low'] for result in gnn_data['results']]
    gnn_ler_high = [result['ler_high'] for result in gnn_data['results']]
    
    mghd_p = [result['p'] for result in mghd_data['results']]
    mghd_ler = [result['LER'] for result in mghd_data['results']]
    mghd_ler_low = [result['ler_low'] for result in mghd_data['results']]
    mghd_ler_high = [result['ler_high'] for result in mghd_data['results']]
    
    mwpm_p = [result['p'] for result in mwpm_data['results']]
    mwpm_ler = [result['LER'] for result in mwpm_data['results']]
    mwpm_ler_low = [result['ler_low'] for result in mwpm_data['results']]
    mwpm_ler_high = [result['ler_high'] for result in mwpm_data['results']]
    
    mwpf_p = [result['p'] for result in mwpf_data['results']]
    mwpf_ler = [result['LER'] for result in mwpf_data['results']]
    mwpf_ler_low = [result['ler_low'] for result in mwpf_data['results']]
    mwpf_ler_high = [result['ler_high'] for result in mwpf_data['results']]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot GNN baseline
    gnn_yerr = [np.array(gnn_ler) - np.array(gnn_ler_low), 
                np.array(gnn_ler_high) - np.array(gnn_ler)]
    plt.errorbar(gnn_p, gnn_ler, yerr=gnn_yerr, 
                 marker='o', linewidth=2, markersize=8, capsize=5,
                 label='GNN Baseline', color='#2E86AB', alpha=0.8)
    
    # Plot MGHD
    mghd_yerr = [np.array(mghd_ler) - np.array(mghd_ler_low), 
                 np.array(mghd_ler_high) - np.array(mghd_ler)]
    plt.errorbar(mghd_p, mghd_ler, yerr=mghd_yerr, 
                 marker='s', linewidth=2, markersize=8, capsize=5,
                 label='MGHD', color='#A23B72', alpha=0.8)
    
    # Plot MWPM
    mwpm_yerr = [np.array(mwpm_ler) - np.array(mwpm_ler_low), 
                 np.array(mwpm_ler_high) - np.array(mwpm_ler)]
    plt.errorbar(mwpm_p, mwpm_ler, yerr=mwpm_yerr, 
                 marker='^', linewidth=2, markersize=8, capsize=5,
                 label='MWPM', color='#F18F01', alpha=0.8)
    
    # Plot MWPF
    mwpf_yerr = [np.array(mwpf_ler) - np.array(mwpf_ler_low), 
                 np.array(mwpf_ler_high) - np.array(mwpf_ler)]
    plt.errorbar(mwpf_p, mwpf_ler, yerr=mwpf_yerr, 
                 marker='d', linewidth=2, markersize=8, capsize=5,
                 label='MWPF', color='#C73E1D', alpha=0.8)
    
    # Formatting
    plt.xlabel('Physical Error Rate (p)', fontsize=14)
    plt.ylabel('Logical Error Rate (LER)', fontsize=14)
    plt.title('Quantum Error Correction Performance Comparison\nRotated d=3 Surface Code', 
              fontsize=16, pad=20)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, which='both')
    
    # Position legend to avoid overlap with data
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    # Add statistics text in upper left corner
    gnn_mean_ler = np.mean(gnn_ler)
    mghd_mean_ler = np.mean(mghd_ler)
    mwpm_mean_ler = np.mean(mwpm_ler)
    mwpf_mean_ler = np.mean(mwpf_ler)
    
    stats_text = f'N = 10,000 samples per p\n'
    stats_text += f'95% confidence intervals\n'
    stats_text += f'Best avg: {min(gnn_mean_ler, mghd_mean_ler, mwpm_mean_ler, mwpf_mean_ler):.4f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top', fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('results/decoder_comparison_full.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/decoder_comparison_full.pdf', bbox_inches='tight')
    
    # Show performance summary
    print("\n" + "="*80)
    print("DECODER PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Error Rate (p)':<12} {'GNN LER':<10} {'MGHD LER':<10} {'MWPM LER':<10} {'MWPF LER':<10}")
    print("-"*80)
    
    for i, p in enumerate(gnn_p):
        gnn_val = gnn_ler[i]
        mghd_val = mghd_ler[i]
        mwpm_val = mwpm_ler[i]
        mwpf_val = mwpf_ler[i]
        print(f"{p:<12.3f} {gnn_val:<10.4f} {mghd_val:<10.4f} {mwpm_val:<10.4f} {mwpf_val:<10.4f}")
    
    print("-"*80)
    print(f"{'AVERAGE':<12} {gnn_mean_ler:<10.4f} {mghd_mean_ler:<10.4f} {mwpm_mean_ler:<10.4f} {mwpf_mean_ler:<10.4f}")
    print("="*80)
    
    # Find best performer for each error rate
    print(f"\nBest decoder per error rate:")
    for i, p in enumerate(gnn_p):
        vals = [('GNN', gnn_ler[i]), ('MGHD', mghd_ler[i]), ('MWPM', mwpm_ler[i]), ('MWPF', mwpf_ler[i])]
        best = min(vals, key=lambda x: x[1])
        print(f"p = {p:.3f}: {best[0]} (LER = {best[1]:.4f})")
    
    # Calculate improvements relative to best classical baseline (MWPF)
    print(f"\nImprovements relative to MWPF:")
    for i, p in enumerate(gnn_p):
        mwpf_val = mwpf_ler[i]
        gnn_imp = ((mwpf_val - gnn_ler[i]) / mwpf_val) * 100
        mghd_imp = ((mwpf_val - mghd_ler[i]) / mwpf_val) * 100
        mwpm_imp = ((mwpf_val - mwpm_ler[i]) / mwpf_val) * 100
        print(f"p = {p:.3f}: GNN {gnn_imp:+6.1f}%, MGHD {mghd_imp:+6.1f}%, MWPM {mwpm_imp:+6.1f}%")
    
    plt.show()

if __name__ == "__main__":
    plot_comparison()
