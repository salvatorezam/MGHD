#!/usr/bin/env python
"""
Plot validation results (LER vs p) for MGHD, MWPM, and LSD.
"""

import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_results(args):
    with open(args.input, "r") as f:
        results = json.load(f)
    
    # Group by distance
    data_by_d = {}
    for r in results:
        d = r["distance"]
        if d not in data_by_d:
            data_by_d[d] = {"p": [], "mghd": [], "mwpm": [], "lsd": []}
        data_by_d[d]["p"].append(r["p"])
        data_by_d[d]["mghd"].append(r["ler_mghd"])
        data_by_d[d]["mwpm"].append(r["ler_mwpm"])
        data_by_d[d]["lsd"].append(r["ler_lsd"])
        
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {3: "blue", 5: "red", 7: "green"}
    markers = {"mghd": "o", "mwpm": "x", "lsd": "--"}
    
    for d, data in data_by_d.items():
        # Sort by p
        p_vals = np.array(data["p"])
        idx = np.argsort(p_vals)
        p_vals = p_vals[idx]
        
        mghd = np.array(data["mghd"])[idx]
        mwpm = np.array(data["mwpm"])[idx]
        lsd = np.array(data["lsd"])[idx]
        
        c = colors.get(d, "black")
        
        ax.loglog(p_vals, mghd, label=f"MGHD d={d}", color=c, marker="o", linestyle="-")
        ax.loglog(p_vals, mwpm, label=f"MWPM d={d}", color=c, marker="x", linestyle="--")
        ax.loglog(p_vals, lsd, label=f"LSD d={d}", color=c, marker="", linestyle=":")
        
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate (LER)")
    ax.set_title("Validation: MGHD vs MWPM vs LSD")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()
    
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluation_results.json")
    parser.add_argument("--output", default="validation_plot.png")
    args = parser.parse_args()
    plot_results(args)
