# MGHD Copilot Instructions

## Project Overview

MGHD (Mamba-GNN Hybrid Decoder) is a universal quantum error correction decoder combining Mamba sequence models with Graph Neural Networks. The project trains on teacher-supervised labels from classical decoders (MWPF, LSD, MWPM) to decode surface codes, color codes, and other QEC code families.

## Architecture

- **Core model**: `mghd/core/core.py` — MGHDv2 model, `pack_cluster()` packer, `PackedCrop`/`CropMeta` dataclasses
- **Decoders/Teachers**: `mghd/decoders/` — MWPF (primary), LSD (BP+OSD), MWPM (fallback), ensemble mixing
- **Samplers**: `mghd/samplers/cudaq_sampler.py` (circuit-level), `stim_sampler.py` (Pauli-channel)
- **CLI tools**: `mghd/cli/train.py`, `preflight_mghd.py`, `make_cluster_crops.py`
- **TAD (Teacher-Assisted Decoding)**: `mghd/tad/` — context features, adaptive weighting, LinTS RL

## Critical Rules

1. **Never initialize CUDA at import time** — all GPU setup must happen inside `main()` or `if __name__ == "__main__":` blocks
2. **Guard CUDA-Q imports** — wrap in try/except and fail gracefully if unavailable; use Stim fallback for testing
3. **Syndrome ordering is Z→X** — detector bits follow Z-checks-then-X-checks, LSBF in bytes, matching Stim/DEM conventions
4. **Parity/coset validation is mandatory** — never accept labels into training without verifying split parity and coset equivalence via `Hx/Hz` kernels

## Environment Setup

```bash
conda activate mlqec-env
pip install -e ".[dev,qec]"  # base + test + QEC baselines
```

## Common Commands

```bash
# Run tests (90% coverage required for CI)
pytest -q --cov=mghd --cov-report=term-missing

# Preflight validation (checks imports, Stim/CUDA-Q smoke)
mghd-preflight --families "surface,steane" --distances "3,5" --run-pytest

# Online training with Stim (phenomenological noise) — REQUIRES --online flag!
mghd-train --online --sampler stim --family surface \
  --distance-curriculum "3,5,7,9,11" \
  --p-curriculum "0.008,0.006,0.004,0.002,0.001" \
  --teacher-mix "lsd=1.0" \
  --epochs-per-p 25 --shots-per-epoch 32768

# Online training with CUDA-Q (circuit-level noise) — needs --qpu-profile
mghd-train --online --sampler cudaq --family surface --distance 3 \
  --qpu-profile mghd/qpu/profiles/iqm_garnet_example.json \
  --teacher-mix "mwpf=0.7,lsd=0.3"

# Offline training from pre-packed crops (no --online flag)
mghd-train --data-root MGHD-data/crops --epochs 30 --batch 512

# Teacher evaluation
mghd-teacher-eval --families surface --distances 3 --sampler stim
```

**CRITICAL**: The `--online` flag is **required** for on-the-fly sampling. Without it, training expects pre-packed crops in `--data-root`.

## Code Patterns

### Teacher fallback chain
```python
# Always implement fallback: MWPF → MWPM → LSD
try:
    result = mwpf_teacher.decode(syndrome)
except Exception:
    result = mwpm_fallback.decode(syndrome)  # never drop batches
```

### Lazy CUDA-Q import
```python
def sample_round(...):
    import cudaq  # import inside function, not at module level
    ...
```

### Crop packing (single source of truth)
```python
from mghd.core.core import pack_cluster, PackedCrop, CropMeta
packed = pack_cluster(H_sub, syndrome, coords, ...)  # returns PackedCrop
```

## Data Formats

- **PackedCrop tensors**: `x_nodes`, `node_mask`, `node_type`, `edge_index`, `edge_attr`, `edge_mask`, `seq_idx`, `seq_mask`, `g_token`, `y_bits`
- **Crop shards**: `.npz` files with `packed` array containing dict items convertible to `PackedCrop`
- **QPU profiles**: JSON in `mghd/qpu/profiles/` with device topology and noise parameters

## Logging & Artifacts

After every material change, append a dated bullet to `docs/IMPLEMENTATION_SUMMARY.md` with:
- Git SHA
- Files touched
- Commands run
- Best LER/p50/p99 metrics
- One-line conclusion

## Testing

- Tests live in `tests/` with `test_*.py` naming
- Use `conftest.py` for fixtures
- Mark GPU tests with `@pytest.mark.gpu`
- Mark slow tests with `@pytest.mark.slow`
- Coverage excludes CLI/samplers/tools by design (see `pyproject.toml`)

## Key Dependencies

- `torch>=2.2` (CUDA build installed separately)
- `stim>=1.13`, `pymatching>=2.3` (QEC baselines)
- `mamba-ssm>=2.2` (optional, for full Mamba support)
- CUDA-Q (optional, for circuit-level noise)

---

## Training & Evaluation Guardrails

### Goal
Train MGHD to **outperform teachers** (lower LER) while achieving **sub-ms latency** on H100. The Mamba+GNN+ChannelSE architecture captures long-range error correlations that local-matching decoders miss—especially valuable at high distances (d≥11) and for qLDPC codes.

### Before Training: Teacher-Noise Compatibility Check

**Critical**: The teacher must support the noise model being sampled, or it will produce invalid labels.

| Noise Model | Compatible Teachers | Notes |
|-------------|---------------------|-------|
| Phenomenological (Pauli) | MWPF, MWPM, LSD | All work; use `--sampler stim` |
| Circuit-level (CUDA-Q) | MWPF, MWPM | LSD may fail on correlated noise |
| Erasure-aware | MWPF, LSD (with masks) | Pass `--erasure-frac`; teachers consume masks |
| qLDPC / HGP codes | LSD (BP+OSD), MWPF | MWPM may not support hypergraph structure |

**Self-check before launching training:**
```python
# Verify teacher can decode the code family
assert teacher.supports_code(code_obj), f"{teacher_name} incompatible with {code_family}"
# Verify teacher handles the noise model
assert teacher.supports_noise_model(noise_type), f"{teacher_name} cannot decode {noise_type}"
```

### During Training: Logical Consistency Checks

1. **Non-zero loss sanity**: If loss stays at 0 for multiple epochs, teachers are likely returning trivial/invalid labels
2. **Parity validation**: Every teacher label must satisfy `Hx @ correction % 2 == 0` and `Hz @ correction % 2 == 0`
3. **Coset equivalence**: Teacher corrections must be in the same coset as ground truth (kernel of `Hx/Hz`)
4. **Teacher failure rate**: Log and alert if any teacher fails >5% of samples; consider removing from mix

### Evaluation: Fair Comparison Requirements

**Always compare MGHD against appropriate baselines:**

| Code Family | Required Baselines | Why |
|-------------|-------------------|-----|
| Surface codes | MWPM, MWPF, Union-Find | Standard baselines; MWPM is threshold reference |
| Color codes | MWPF, Restriction decoder | MWPM doesn't apply directly |
| qLDPC / HGP | LSD (BP+OSD), MWPF | These handle non-local structure |
| High-distance (d≥11) | All above + MGHD teachers | Show MGHD captures long-range correlations |

**Evaluation checklist:**
```bash
# 1. Same noise model for all decoders
# 2. Same shot count (N≥10k for Wilson CIs)
# 3. Same distance range
# 4. Report Wilson 95% confidence intervals

python scripts/evaluate_model.py \
  --checkpoint best.pt \
  --sampler phenomenological \
  --distances "3,5,7,9,11,13,15" \
  --p-values "0.001,0.003,0.005,0.007,0.01" \
  --shots 10000 \
  --compare-to mwpm,lsd,mwpf
```

### Logical Consistency Self-Checks (Run After Every Task)

Before concluding any training or evaluation task, verify:

1. **LER decreases with distance** at fixed p (for p < threshold) — if not, model may be undertrained or labels corrupted
2. **LER increases with p** at fixed distance — basic sanity
3. **MGHD LER ≤ teacher LER** on training distribution — otherwise training failed
4. **No NaN/Inf in metrics** — check all outputs
5. **Latency reported** — always measure and report B=1 and B=256 inference time

```python
# Quick sanity after evaluation
for d in distances:
    lers = [r['ler_mghd'] for r in results if r['distance'] == d]
    ps = [r['p'] for r in results if r['distance'] == d]
    assert all(l > 0 for l in lers), f"Zero LER at d={d} is suspicious"
    assert lers == sorted(lers), f"LER should increase with p at d={d}"
```

### When MGHD Underperforms Teachers

If MGHD LER > teacher LER, investigate:

1. **Insufficient training**: Increase epochs or shots-per-epoch
2. **Wrong teacher mix**: Ensure dominant teacher handles the noise model
3. **Distance mismatch**: Model trained on d=3-7 may not generalize to d=15
4. **Noise distribution shift**: Evaluation noise differs from training noise
5. **Packing errors**: Verify `pack_cluster()` matches training format

### High-Distance / qLDPC Focus

MGHD's value proposition is **long-range correlation capture**:

- **Mamba**: Processes check sequences in Hilbert order; captures temporal/spatial correlations across the lattice
- **ChannelSE**: Reweights feature channels based on global context; helps identify correlated error patterns
- **GNN message passing**: Propagates information across graph structure; n_iters controls receptive field

For d≥11 surface codes or qLDPC, expect MGHD to show **larger LER improvement** over MWPM/MWPF than at small distances. If not:
- Increase `n_iters` (GNN depth)
- Increase `d_state` (Mamba state dimension)
- Train with curriculum from low-d to high-d
