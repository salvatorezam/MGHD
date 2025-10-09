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
  - MWPF (Minimum Weight Perfect Matching with hypergraphs)
  - LSD (Belief Propagation + OSD)
  - MWPM (Classical minimum weight perfect matching)
- **Flexible Training**: Supports curriculum learning across code distances, online RL, and adaptive weighting
- **Production Ready**: Includes preflight validation, benchmarking tools, and comprehensive testing

---

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

### Training a Model

Train on surface codes with distances 3-31, using CUDA-Q sampler:

```bash
mghd-train \
    --family surface \
    --distances 3-31:2 \
    --sampler cudaq \
    --shots-per-batch 128 \
    --batches 10 \
    --seed 0
```

Train across multiple code families:

```bash
mghd-train \
    --families "surface,color_666,steane,repetition" \
    --distances 3-11:2 \
    --sampler stim \
    --shots-per-batch 64 \
    --batches 5
```

### Running Benchmarks

Benchmark the trained decoder:

```bash
mghd-bench \
    --family surface \
    --distance 5 \
    --model-path checkpoints/mghd_best.pt
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
│   ├── train_core.py      # Training entrypoint
│   ├── bench_decode.py    # Benchmarking tool
│   └── preflight_mghd.py  # System validation
│
├── core/                   # Neural network implementation
│   ├── model_v2.py        # Hybrid Mamba-GNN architecture
│   ├── blocks.py          # Building blocks (ChannelSE, AstraGNN)
│   ├── features_v2.py     # Feature extraction from syndromes
│   ├── infer.py           # Inference engine
│   └── config.py          # Model configuration
│
├── codes/                  # Quantum error correction codes
│   ├── registry.py        # Code family registry and builders
│   ├── external_color_488.py  # Color code implementations
│   └── qpu_profile.py     # QPU-specific code profiles
│
├── samplers/               # Error sampling backends
│   ├── cudaq_sampler.py   # CUDA-Q circuit-level sampler
│   ├── stim_sampler.py    # Stim Pauli-channel sampler
│   └── cudaq_backend/     # CUDA-Q integration
│
├── decoders/               # Teacher decoders and baselines
│   ├── mwpf_teacher.py    # MWPF hypergraph decoder
│   ├── lsd_teacher.py     # LSD (BP+OSD) decoder
│   ├── mwpm_fallback.py   # Classical MWPM decoder
│   ├── dem_matching.py    # DEM-based matching
│   └── ensemble.py        # Teacher ensemble
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

### Model Architecture

The MGHD decoder processes syndrome measurements through:

1. **Feature Extraction**: Converts raw syndromes to node/edge features
2. **Mamba Encoder**: Captures temporal/sequential patterns in syndrome sequences
3. **GNN Layers**: Propagates information across qubit connectivity graph
4. **Channel SE**: Attention-based feature refinement
5. **Decoder Head**: Predicts logical error corrections

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
- `--dem-enable`: Use detector error model for training
- `--context-source`: Context injection source (qiskit, cirq, cudaq)
- `--rl-online`: Enable online reinforcement learning

### `mghd-bench`
Benchmarking and evaluation tool for trained models.

### `mghd-preflight`
Comprehensive validation suite checking:
- Dependency versions
- PyTest suite execution
- Stim + DEM A/B testing
- CUDA-Q smoke tests
- Logical error rate validation

---

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
- Inspired by Astra GNN architecture for quantum decoding