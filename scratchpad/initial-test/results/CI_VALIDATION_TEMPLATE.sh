#!/bin/bash
# MGHD v2 Production CI Validation Template
# Add this to your CI pipeline to ensure MGHD engagement

set -e

CKPT="results/foundation_v2_production/best.pt/best.pt"
cd /path/to/mghd/workspace

# Production sweep with enforced MGHD usage
echo "Running production validation..."
python tools/bench_clustered_sweep_surface.py \
  --ckpt "$CKPT" --expert v2 --graph-capture \
  --tier0-k-max 2 --tier0-r-max 1 \
  --dists 3 --ps 0.005 0.010 0.015 \
  --shots 1000 --enforce-mghd \
  --out-json ci_validation.json

# Parse results and validate
python - << 'PYCHECK'
import json
import sys

data = json.load(open("ci_validation.json"))
grid = data.get("results", {}).get("grid", [])

total_mghd = sum(entry.get("mghd_clusters_per_shot", 0) for entry in grid)
max_tier0 = max(entry.get("tier0_frac", 1.0) for entry in grid)
max_ler = max(entry.get("ler", 0.0) for entry in grid)

print(f"Total MGHD usage: {total_mghd:.3f}")
print(f"Max Tier0 fraction: {max_tier0:.3f}")
print(f"Max LER: {max_ler:.6f}")

# CI Assertions
assert total_mghd > 0.1, f"MGHD usage too low: {total_mghd}"
assert max_tier0 < 0.98, f"Tier0 monopolization detected: {max_tier0}"
assert max_ler < 0.01, f"Error rate too high: {max_ler}"

print("✅ CI validation passed!")
PYCHECK

echo "✅ MGHD v2 production validation complete"
