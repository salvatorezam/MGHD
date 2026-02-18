# MGHD Copilot Instructions

## Project Overview

MGHD (Mamba-GNN Hybrid Decoder) is a neural decoder for quantum error correction that operates on circuit-level DEM (Detector Error Model) graphs sampled via Stim. It combines Mamba sequence models with GNN message passing, supervised by PyMatching (MWPM) teacher labels.

## Architecture

- **Core model**: `mghd/core/core.py` — MGHDv2 model, `pack_cluster()`, `PackedCrop`/`CropMeta` dataclasses
- **Training**: `mghd/cli/train.py` — online DEM pipeline, `OnlineSurfaceDataset`, `pack_dem_cluster`, `_build_dem_info`
- **Teachers**: `mghd/decoders/mwpm_ctx.py` (primary MWPM), `mwpf_teacher.py` (MWPF), `lsd_teacher.py` + `lsd/clustered.py` (LSD)
- **Sampler**: `mghd/samplers/stim_sampler.py` (Stim DEM sampling)
- **Utils**: `mghd/utils/metrics.py` (LER + Wilson CIs), `mghd/utils/graphlike.py` (graph validation)

## Critical Rules

1. **Never initialize CUDA at import time** — all GPU setup must happen inside `main()` or `if __name__ == "__main__":` blocks
2. **Do not create new files without explicit need** — prefer editing existing files. Delete temporary files before finishing.
3. **Syndrome features**: node features are 10-dim `[x,y,t,type,degree,k,r,d_norm,rounds_norm,syndrome]`; `SYND_FEAT_IDX = 8` in core.py
4. **Edge-nodes are type-0, detector-nodes are type-1** in the merged-edge bipartite graph

## Environment

```bash
conda activate mlqec-env
pip install -e ".[dev,qec]"
```

## Common Commands

```bash
# Run tests
pytest -q

# Quick smoke test
python -m mghd.cli.train --online --sampler stim \
  --family surface --distance 3 --p-phys 0.005 \
  --epochs 2 --shots-per-epoch 64

# Circuit-level training run
python -m mghd.cli.train --online --sampler stim \
  --family surface \
  --distance-curriculum "3,5,7,9" \
  --p-curriculum "0.008,0.005,0.003,0.001" \
  --epochs-per-p 25 --shots-per-epoch 32768 \
  --edge-prune-thresh 1e-6 \
  --save checkpoints/circuit_run
```

## Code Patterns

### Crop packing (single source of truth)
```python
from mghd.core.core import pack_cluster, PackedCrop, CropMeta
packed = pack_cluster(H_sub, syndrome, coords, ...)
```

### DEM pipeline (circuit-level)
```python
from mghd.cli.train import _build_dem_info, pack_dem_cluster, _teacher_labels_from_matching
dem_info = _build_dem_info(dem, d, rounds)  # returns edge_index, node_coords, L_obs, etc.
teacher_labels = _teacher_labels_from_matching(dem, detection_events)
crops = pack_dem_cluster(dem_info, syndrome, teacher_labels, ...)
```

### Teacher fallback
```python
try:
    result = mwpm_teacher.decode(syndrome)
except Exception:
    logger.warning("MWPM decode failed, skipping batch")
```

## Data Formats

- **PackedCrop**: `x_nodes [N_max,F_n]`, `node_mask`, `node_type`, `edge_index [2,E_max]`, `edge_attr [E_max,F_e]`, `edge_mask`, `seq_idx`, `seq_mask`, `g_token [F_g]`, `y_bits [N_max]`
- **DEM info**: `edge_index`, `node_coords`, `node_types`, `L_obs`, `dem_edge_probs`, `det_to_node`, `edge_to_node`

## Testing

- Tests in `tests/` with `test_*.py` naming
- Use `conftest.py` for fixtures
- Mark GPU tests with `@pytest.mark.gpu`
- Run `pytest -q` after every material change

## Key Dependencies

- `torch>=2.2`, `stim>=1.13`, `pymatching>=2.3`
- `mamba-ssm>=2.2` (optional), `scipy`
