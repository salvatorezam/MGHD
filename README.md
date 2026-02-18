# MGHD: Mamba-Graph Hybrid Decoder

**A Universal Hybrid Architecture for Quantum Error Correction on Circuit-Level DEM Graphs**

MGHD combines Mamba sequence models with Graph Neural Networks to decode quantum error correction codes. The model operates on merged-edge bipartite graphs derived from Stim Detector Error Models (DEMs), with PyMatching (MWPM) providing teacher supervision.

---

## Overview

MGHD (Mamba-GNN Hybrid Decoder) is a neural decoder that leverages:

- **Mamba Sequence Models**: Efficient processing of detector/edge-node sequences
- **Graph Neural Networks (GNN)**: Spatial reasoning over DEM graph structure
- **Circuit-Level Noise**: Stim-based DEM sampling for realistic error models
- **Teacher Supervision**: MWPM (PyMatching) labels on DEM matching graphs

The decoder currently targets surface code memory experiments with plans to extend to other code families.

---

## How It Works

### Data Flow (Circuit-Level DEM Pipeline)

1. **Stim** generates a `DetectorErrorModel` for surface code memory at distance `d`, `rounds`, physical error rate `p`.
2. **`_build_dem_info()`** converts the DEM into a **merged-edge bipartite graph**:
   - **Type-0 nodes** = DEM error mechanisms (edge-nodes) — the model predicts flip probability for these
   - **Type-1 nodes** = detectors — carry syndrome bits
   - **`L_obs`** matrix maps edge-node flips → observable corrections
3. **`OnlineSurfaceDataset`** samples detection events + observable flips per shot.
4. **`_teacher_labels_from_matching()`** runs PyMatching on the DEM for per-edge-node labels.
5. **`pack_dem_cluster()`** packs each connected component into `PackedCrop` tensors.
6. **MGHDv2** predicts per-node flip probabilities; BCE loss on edge-nodes vs teacher labels.

### Node Features (10-dim)

`[x, y, t, type, degree, k, r, d_norm, rounds_norm, syndrome]`

### Global Token (14-dim)

`[d_norm, rounds_norm, p, N, E, S, mean_degree, max_degree, frac_type0, frac_type1, x_span, y_span, t_span, component_id_norm]`

### Model Architecture

- **Sequence Encoder**: Mamba processes node features with masks
- **ChannelSE**: Channel squeeze-and-excitation reweights features
- **GraphDecoder**: Iterative GNN message passing with GRU updates → 2-logit binary head per node

---

## Installation

### Requirements
- Python 3.10+
- PyTorch
- Stim + PyMatching (QEC baselines)
- mamba-ssm (optional, for full Mamba support)

### Quick Install

```bash
git clone https://github.com/salvatorezam/MGHD.git
cd MGHD
pip install -e ".[dev,qec]"
```

---

## Quick Start

### Circuit-Level Training (Online)

```bash
conda activate mlqec-env

python -m mghd.cli.train --online --sampler stim \
  --family surface \
  --distance-curriculum "3,5,7,9" \
  --p-curriculum "0.008,0.005,0.003,0.001" \
  --epochs-per-p 25 --shots-per-epoch 32768 \
  --edge-prune-thresh 1e-6 \
  --save checkpoints/circuit_run
```

### Quick Smoke Test

```bash
python -m mghd.cli.train --online --sampler stim \
  --family surface --distance 3 --p-phys 0.005 \
  --epochs 2 --shots-per-epoch 64
```

### Run Tests

```bash
pytest -q
```

---

## Project Structure

```
mghd/
├── cli/
│   └── train.py              # Training loop (online DEM pipeline)
├── codes/
│   └── registry.py           # Code family registry
├── config/
│   └── hparams.json          # Model/training hyperparameters
├── core/
│   └── core.py               # MGHDv2 model + PackedCrop + packers
├── decoders/
│   ├── lsd/
│   │   └── clustered.py      # LSD decoder (code-capacity)
│   ├── lsd_teacher.py        # LSD teacher wrapper
│   ├── mwpf_teacher.py       # MWPF teacher wrapper
│   └── mwpm_ctx.py           # PyMatching MWPM teacher
├── samplers/
│   └── stim_sampler.py       # Stim DEM sampler
└── utils/
    ├── graphlike.py           # Graph validation
    └── metrics.py             # LER + Wilson CIs

scripts/                       # Analysis and plotting scripts
tests/                         # pytest suite (~40 tests)
tools/                         # Benchmarking utilities
docs/                          # Architecture and planning docs
data/                          # Experiment outputs (gitignored)
```

---

## CLI Arguments (Key)

| Argument | Description |
|----------|-------------|
| `--online` | Enable on-the-fly Stim sampling (required for circuit-level) |
| `--sampler stim` | Use Stim DEM sampler |
| `--family surface` | Code family |
| `--distance-curriculum "3,5,7"` | Train across distances |
| `--p-curriculum "0.005,0.003"` | Physical error rate schedule |
| `--edge-prune-thresh` | Prune low-probability DEM edges |
| `--shots-per-epoch` | Samples per training epoch |
| `--epochs-per-p` | Epochs per curriculum stage |
| `--save` | Checkpoint directory |
| `--hparams` | JSON hyperparameter file |

---

## Hyperparameters

Provide a JSON file via `--hparams` (example: `mghd/config/hparams.json`):

- `model_architecture`: `n_iters`, `msg_net_size`, `msg_net_dropout_p`, `gru_dropout_p`
- `mamba_parameters`: `d_model`, `d_state`
- `attention_mechanism`: `se_reduction`
- `training_parameters`: `lr`, `weight_decay`, `label_smoothing`, `gradient_clip`

---

## Status & Roadmap

### Completed
- Circuit-level DEM pipeline (Stim sampling → merged-edge bipartite graph)
- Online training with PyMatching teacher labels
- Distance/error-rate curriculum
- Tight active decomposition into connected components
- `pack_dem_cluster` packing for variable-size DEM components

### TODO
- **Evaluation pipeline**: wire `L_obs` to convert edge-node predictions → observable corrections → LER
- **Threshold plots**: LER vs p across distances with Wilson CIs
- **Multi-code support**: extend DEM pipeline beyond surface codes

---

## Key Dependencies

- `torch>=2.2`
- `stim>=1.13`, `pymatching>=2.3`
- `mamba-ssm>=2.2` (optional)
- `scipy` (connected components)

---

## License

Proprietary — See LICENSE file for details.
