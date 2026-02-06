# MGHD: Mamba-Graph Hybrid Decoder

**A Universal Hybrid Architecture for Real-Time Quantum Error Correction**

MGHD combines Mamba sequence models with Graph Neural Networks to create a powerful, universal decoder for quantum error correction across multiple code families.

---

## Overview

MGHD (Mamba-GNN Hybrid Decoder) is a state-of-the-art quantum error correction decoder that leverages:

- **Mamba Sequence Models**: Efficient processing of detector sequences and syndrome patterns
- **Graph Neural Networks (GNN)**: Spatial reasoning over qubit connectivity and error correlations
- **Teacher-Assisted Decoding (TAD)**: Knowledge distillation from classical decoders (MWPF, LSD, MWPM)
- **Multi-Code Support**: Surface codes, color codes, repetition codes, Steane, BB codes, and more

The decoder is designed for real-time performance while maintaining high accuracy across varying code distances and noise models.

---

## Key Features

- **Universal Architecture**: Single model works across multiple quantum error correction code families
- **Hybrid Design**: Combines sequential (Mamba) and spatial (GNN) processing for optimal performance
- **Multiple Samplers**: 
  - CUDA-Q for realistic circuit-level noise simulation with Kraus operators
  - Stim for efficient Pauli-channel sampling
- **Teacher Ensemble**: Learns from multiple classical decoding algorithms:
  - Hyperblossom (MWPF)
  - LSD (Belief Propagation + OSD)
  - MWPM (Classical minimum weight perfect matching)
- **Flexible Training**: Supports curriculum learning across code distances, online RL, and adaptive weighting
- **Production Ready**: Includes preflight validation, benchmarking tools, and comprehensive testing

---

## How MGHD Works

- Data Flow
  - Codes: `mghd.codes.registry.get_code()` yields CSS-style codes (Hx/Hz, coordinates, metadata).
  - Sampling: CUDA‑Q (`mghd.samplers.cudaq_sampler.CudaQSampler`) or Stim (`mghd.samplers.stim_sampler.StimSampler`) generates detection events and metadata per round.
  - Teachers: MWPF, LSD, and MWPM produce supervision labels; a weighted mix selects the target.
  - Packing: `mghd.core.core.pack_cluster` converts each local cluster into tensors for the model.
  - Model: `mghd.core.core.MGHDv2` consumes packed crops and predicts data‑qubit corrections.

- Sampling Backends
  - CUDA‑Q (trajectories): Circuit‑level noise driven by the QPU profile JSON; used for online training via `--online`.
  - Stim (Pauli): Fast Pauli/twirled approximation for A/B checks and smoke tests.
  - Canonical detector packing is Z→X across backends to match DEM/Stim conventions.

- Teachers and Labels
- Hyperblossom (MWPF) (`mghd.decoders.mwpf_teacher`): Hypergraph decoder on detector streams; accepts optional `mwpf_scale` (per‑fault scaling derived from TAD LLRs).
  - LSDTeacher (`mghd.decoders.lsd_teacher`): BP+OSD on CSS parity checks; supports LLR overrides from TAD.
  - MWPMFallback (`mghd.decoders.mwpm_fallback`): Classical matching from H, used as fallback or in teacher mixes.
  - Mix selection (`--teacher-mix`): Weighted random choice per crop among available teachers.

- TAD and RL
  - QPU profile JSON (`mghd/qpu/profiles/*.json`) and optional schedule IR (qiskit/cirq/cudaq) yield per‑qubit weight maps and a compact context vector.
  - LLR overrides bias teacher decisions; also converted to `mwpf_scale` for MWPF.
  - Optional online RL (`--online-rl`) applies LinTS to adapt TAD scaling per epoch from context features.

- Packing and Features
  - `pack_cluster` builds per‑node features: normalized coordinates, node type (data/check), degree, crop stats, and optional erasure flag for data qubits.
  - Global context (`g_token`) appends crop stats and TAD context features to condition the model.
  - Sequence indices order checks by a 2D Hilbert curve to supply a stable temporal structure to the sequence encoder.

- MGHD Model
  - Sequence encoder: Mamba‑family encoder processes node features in sequence space with masks.
  - ChannelSE: Channel squeeze‑and‑excitation reweights feature channels (configurable `se_reduction`).
  - GraphDecoder: Iterative message passing (n_iters), GRU‑based updates, and an MLP message network; outputs a 2‑logit binary head per node; only data nodes are supervised.

- Training Losses
  - BCE/CE on data‑node logits with optional label smoothing and example weighting.
  - Parity auxiliary loss: encourages parity consistency with the local H submatrix.
  - Projection‑aware loss: projects probabilities via exact GF(2) projector (LSD clustered util) and aligns thresholded bits with targets.

- Modes and Loops
  - Offline: Train from pre‑packed crops (`mghd-make-crops` → `.npz` shards) using `mghd-train --data-root`.
  - Online: On‑the‑fly CUDA‑Q sampling (`--online`) with teachers, TAD, and optional RL; erasure injection supported via `--erasure-frac`.

- Erasure Awareness
  - Samplers can provide or inject erasure masks; `pack_cluster` adds a per‑data‑qubit erasure feature; teachers consume masks when available.
  - The node feature dimension increases by 1 when erasures are used; not tied to code distance or family.
  - Stim mirrors CUDA‑Q interface knobs (e.g., optional observables and erasure masks) with zero overhead when unused; you can drive both samplers with the same flags.

- Inference
  - The wrapper collates per‑crop tensors, applies the model, and scatters per‑data‑qubit probabilities back to the code’s local index set for metric computation.


## Installation

### Requirements
- Python 3.10+
- PyTorch
- CUDA-Q (optional, for circuit-level simulation)
- Stim (for efficient Pauli sampling)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/salvatorezam/MGHD.git
cd MGHD

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

---

## Quick Start

### Training

Offline (pre-packed crops):

```bash
mghd-train \
  --data-root MGHD-data/crops \
  --save checkpoints/run1 \
  --epochs 30 --batch 512
```

If the console entrypoints are not installed yet in your current environment, use:
`python -m mghd.cli.make_cluster_crops ...` and `python -m mghd.cli.train ...`.

Online (on-the-fly CUDA‑Q trajectories) with TAD + optional RL scaling and erasure awareness:

```bash
mghd-train \
  --online \
  --online-fast \
  --family surface --distance 3 \
  --sampler cudaq \
  --p-curriculum 0.01,0.006,0.003 --epochs-per-p 10 \
  --qpu-profile mghd/qpu/profiles/iqm_garnet_example.json \
  --context-source qiskit \
  --teacher-mix "mwpf=0.7,mwpm=0.2,lsd=0.1" \
  --online-rl \
  --erasure-frac 0.05 \
  --shots-per-epoch 256 \
  --progress-seconds 15 \
  --save checkpoints/online
```

Note: `mghd-train --online` trains MGHDv2 on parity-check crops with per-qubit supervision; it supports
`--sampler cudaq` and `--sampler synthetic`.

Teacher evaluation (LER A/B + TAD weighting):

```bash
mghd-teacher-eval \
  --families surface \
  --distances 3 \
  --sampler stim \
  --shots-per-batch 8 \
  --batches 2
```

### Validation & Preflight Checks

Run comprehensive system validation:

```bash
mghd-preflight \
    --families "surface,steane,repetition" \
    --distances "3,5" \
    --run-pytest
```

---

## Architecture

### Core Components

```
mghd/
├── cli/                    # Command-line tools
│   ├── train.py               # Training (offline + online)
│   ├── make_cluster_crops.py  # Crop dataset generator
│   └── preflight_mghd.py      # System validation
│
├── core/                   # Neural network implementation
│   └── core.py               # MGHDv2 model + packers + inference wrapper
│
├── codes/                  # Quantum error correction codes
│   ├── registry.py        # Code family registry and builders
│   ├── external_color_488.py  # Color code implementations
│   └── qpu_profile.py     # QPU-specific code profiles
│
├── samplers/               # Error sampling backends
│   ├── cudaq_sampler.py      # CUDA-Q circuit-level sampler
│   ├── stim_sampler.py       # Stim Pauli-channel sampler
│   └── cudaq_backend/        # CUDA-Q kernels + noise model
│
├── decoders/               # Teacher decoders and baselines
│   ├── mwpf_teacher.py    # MWPF hypergraph decoder
│   ├── lsd_teacher.py     # LSD (BP+OSD) decoder
│   ├── mwpm_fallback.py   # Classical MWPM decoder
│   └── ensemble.py           # Teacher ensemble helpers
│   └── lsd/clustered.py      # Projector + clustered decoder
│
├── tad/                    # Teacher-Assisted Decoding
│   ├── weighting.py       # Adaptive teacher weighting
│   ├── context.py         # Context injection (Qiskit, Cirq, CUDA-Q)
│   └── rl/                # Reinforcement learning components
│
└── utils/                  # Utilities
    ├── metrics.py         # Logical error rate computation
    ├── code_loader.py     # Code loading utilities
    ├── curriculum.py      # Training curriculum
    └── graphlike.py       # Graph structure validation
```

### Model + Training Features

- MGHDv2: Mamba (sequence) + ChannelSE + GNN (graph message passing)
- TAD: schedule-aware priors (LLR) + context features from QPU JSON (qiskit/cirq/cudaq adapters)
- Online RL: optional LinTS scaling of TAD priors per epoch (--online-rl)
- Erasure-aware: sampler can inject erasures (--erasure-frac); teachers consume masks; model sees per-qubit erasure flags in node features

---

## Supported Code Families

- **Surface Codes**: Planar and toric surface codes with varying distances
- **Color Codes**: 4.8.8 and 6.6.6 lattice geometries
- **Repetition Codes**: X and Z basis repetition codes
- **Steane Code**: [[7,1,3]] quantum error correction code
- **Bicycle Codes (BB)**: Quantum LDPC codes
- **Hypergraph Product (HGP)**: Generalized product codes
- **Reed-Muller (RM)**: Classical code-based QEC
- **Generalized Bicycle (GB)**: LDPC constructions

---

## CLI Tools

### `mghd-train`
Main training interface with teacher-supervised learning.

**Key Arguments:**
- `--family` / `--families`: Code family selection
- `--distances`: Code distances (e.g., "3,5,7" or "3-31:2")
- `--sampler`: Sampling backend (`cudaq` or `stim`)
- `--shots-per-batch`: Number of syndrome samples per batch
- `--batches`: Number of batches per distance
- `--context-source`: Context injection source (qiskit, cirq, cudaq)
- `--rl-online`: Enable online reinforcement learning

### `mghd-teacher-eval`
Teacher evaluation and LER estimation across families.

### `mghd-make-crops`
Generate offline crop datasets (packed subgraphs + labels) using teacher supervision. Useful for fast offline training or sharing reproducible corpora.

Example:

```bash
mghd-make-crops \
  --families surface \
  --distances 3-9:2 \
  --ps 0.003 0.006 0.01 \
  --teacher-mix "mwpf=0.7,mwpm=0.2,lsd=0.1" \
  --qpu-profile mghd/qpu/profiles/iqm_garnet_example.json \
  --context-source qiskit \
  --shots-per-grid 256 \
  --out MGHD-data/crops
```

Notes:
- Produces `.npz` shards containing `packed` items (tensors + metadata) compatible with offline `mghd-train`.
- TAD context and LLR overrides are derived from the QPU profile and optional schedule IR when `--context-source` is provided.

### `mghd-preflight`
Comprehensive validation suite checking:
- Dependency versions
- PyTest suite execution
- Stim teacher A/B testing
- CUDA-Q smoke tests
- Logical error rate validation

---

## Hyperparameters

- Provide a JSON hyperparameter file via `--hparams` to control model, Mamba, attention, and training settings. Example: `mghd/config/hparams.json`.
- Keys supported by the trainer:
  - `model_architecture`: `n_iters`, `msg_net_size`, `msg_net_dropout_p`, `gru_dropout_p`
  - `mamba_parameters`: `d_model`, `d_state`
  - `attention_mechanism`: `se_reduction`
  - `training_parameters`: `lr`, `weight_decay`, `label_smoothing`, `gradient_clip`, `noise_injection`, `epochs`, `batch_size`

Example:

```bash
mghd-train \
  --online \
  --family surface --distance 3 \
  --qpu-profile mghd/qpu/profiles/iqm_garnet_example.json \
  --hparams mghd/config/hparams.json \
  --save checkpoints/online
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_codes_registry_css.py
pytest tests/test_cudaq_smoke_no_pymatching.py
pytest tests/test_train_core_smoke.py

# Run with coverage
pytest --cov=mghd --cov-report=html
```

### Code Quality

```bash
# Format code
black mghd/

# Lint
ruff check mghd/

# Type checking
mypy mghd/
```

### Project Structure

The repository follows a modular design:

- **`mghd/`**: Main package with all source code
- **`tests/`**: Comprehensive test suite
- **`color_cache/`**: Precomputed color code data
- **`pyproject.toml`**: Package configuration and dependencies
- **`pytest.ini`**: Test configuration

---

## Plotting Results

Use the plotting tool to render loss curves (and the online p schedule) from a saved run directory. If a post‑run teacher evaluation was written (teacher_eval.txt), a teacher usage plot is generated too.

Example:

```bash
mghd-plot-run data/results/20240101-120000_surface_d3_iqm_garnet_example
```

Outputs are saved next to the run (loss.png, teacher_usage.png, plots_manifest.json). If matplotlib is unavailable, a JSON summary is saved instead.

## Workflow

### Training Workflow

1. **Code Selection**: Choose code family and distances
2. **Sampler Setup**: Configure CUDA-Q or Stim for syndrome generation
3. **Teacher Ensemble**: Mix of MWPF, LSD, MWPM provides supervision signals
4. **Model Training**: MGHD learns from teacher outputs via distillation
5. **Validation**: Track logical error rates across distances
6. **Checkpointing**: Save best models based on validation metrics

### Evaluation Workflow

1. **Model Loading**: Load trained checkpoint
2. **Benchmark Execution**: Generate test syndromes
3. **Decoding**: Run MGHD inference
4. **Metrics**: Compute logical error rates, latency, throughput
5. **Analysis**: Compare against teacher baselines

---

## References & Citation

If you use MGHD in your research, please cite:

```
@article{mghd2024,
  title={MGHD: A Universal Mamba-Graph Hybrid Decoder Architecture for Real-Time Quantum Error Correction},
  author={MGHD Team},
  year={2024}
}
```

---

## License

Proprietary - See LICENSE file for details.

---

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue or contact the MGHD Team.

---

## Acknowledgments

- Built on top of CUDA-Q for quantum circuit simulation
- Uses Stim for efficient Pauli error sampling
- Integrates PyMatching for classical MWPM baseline
- Leverages MWPF for hypergraph-based decoding
