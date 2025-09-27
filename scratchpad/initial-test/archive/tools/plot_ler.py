#!/usr/bin/env python3
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

def load(path):
    with open(path) as f:
        obj=json.load(f)
    xs=[r['p'] for r in obj['results']]
    ys=[r['LER'] for r in obj['results']]
    los=[r['ler_low'] for r in obj['results']]
    his=[r['ler_high'] for r in obj['results']]
    return xs, ys, los, his

def main():
    if len(sys.argv) < 4:
        print("Usage: plot_ler.py <mghd_json> <mwpf_json> <out_png>")
        sys.exit(1)
    mghd, mwpf, out = map(Path, sys.argv[1:4])
    xm, ym, lom, him = load(mghd)
    xw, yw, low, hiw = load(mwpf)
    # Plot
    plt.figure(figsize=(6,4))
    plt.errorbar(xm, ym, yerr=[ [ym[i]-lom[i] for i in range(len(ym))], [him[i]-ym[i] for i in range(len(ym))] ], fmt='-o', label='MGHD')
    plt.errorbar(xw, yw, yerr=[ [yw[i]-low[i] for i in range(len(yw))], [hiw[i]-yw[i] for i in range(len(yw))] ], fmt='-s', label='MWPF')
    plt.xscale('log')
    plt.xlabel('p (circuit-level)')
    plt.ylabel('LER (coset)')
    plt.title('LER vs p (N=10k per point)')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == '__main__':
    main()

