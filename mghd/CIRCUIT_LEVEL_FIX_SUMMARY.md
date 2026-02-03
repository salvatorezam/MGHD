# Circuit-Level Architecture for MGHD

---
---

## Executive Summary

This document describes the architectural changes made to support **true circuit-level decoding** in MGHD. The original codebase was designed for **code-level (phenomenological) noise**, where syndromes are direct parity checks (`s = H·e`). When applied to **circuit-level noise** from Stim, this caused a fundamental semantic mismatch resulting in 3-6× worse performance than MWPM.

The solution introduces a parallel architecture (`MGHDCircuit`) that operates natively on Stim's Detector Error Model (DEM) structure, with a differentiable soft-matching layer to approximate MWPM's combinatorial power.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Theoretical Foundation](#theoretical-foundation)
4. [Solution Architecture](#solution-architecture)
5. [Code Changes](#code-changes)
6. [Usage](#usage)
7. [File Reference](#file-reference)

---

## The Problem

### Symptoms (from troubleshoot.md)

**Phase 3:** Attempting to use Stim's circuit-level sampler with the existing MGHDv2 pipeline failed because:
- Hz/Hx matrices don't match Stim's detector ordering
- DEM has 7 error mechanisms vs 9 Hz columns for d=3
- Teacher decoders (MWPF/MWPM) expect code-level syndrome structure

**Phase 4:** All attempted workarounds produced poor results:

| Approach | Result |
|----------|--------|
| DEM-MWPM teacher with observable loss | 3× worse than MWPM |
| BP soft labels | 4× worse |
| Direct observable prediction | 6× worse |
| Attention pooling over detectors | Still 3-4× worse |

### The Core Issue

The entire training stack—model architecture, data pipeline, loss functions, teacher decoders—was designed for **code-level semantics** but was being forced to operate on **circuit-level data**.

---

## Root Cause Analysis

### Code-Level vs Circuit-Level: Semantic Mismatch

| Aspect | Code-Level | Circuit-Level |
|--------|------------|---------------|
| **Noise Model** | Phenomenological (per-round i.i.d.) | Full circuit simulation |
| **Syndrome** | `s = H·e` (parity checks) | Detection events `d[t] = m[t] ⊕ m[t-1]` |
| **Graph Structure** | Hz/Hx bipartite (checks ↔ qubits) | DEM (detectors ↔ error mechanisms) |
| **Nodes** | Check nodes + data qubits | Detectors (space-time) |
| **Edges** | Check-to-qubit connections | Error mechanism pairs |
| **Output** | Per-qubit error probability `p(e_i)` | Observable flip `p(obs_k)` |
| **Ground Truth** | Teacher decoder labels | Stim simulation |

### Why Attention Pooling Failed

The key insight from analyzing Phase 4 failures:

> **MWPM's power comes from the matching structure, not from learned representations.**

When we use attention pooling to aggregate detector embeddings into an observable prediction, we lose the combinatorial structure that makes MWPM effective. MWPM finds a minimum-weight perfect matching on the DEM graph—this is fundamentally different from learning a weighted sum of node features.

### The Hz/Hx Ordering Problem

Stim's circuit-level detector ordering follows space-time conventions:
```
detector D0 = m[t=0, pos=0]
detector D1 = m[t=0, pos=1]  
...
detector Dk = m[t=0, pos=0] ⊕ m[t=1, pos=0]  # temporal XOR
```

The existing Hz/Hx matrices assume:
```
syndrome bit s_i = ⊕_{j ∈ support(H_i)} e_j
```

These are incompatible representations. The DEM has different connectivity, different edge semantics, and different output semantics.

---

## Theoretical Foundation

### Detector Error Model (DEM)

A DEM represents circuit-level noise as a hypergraph:
- **Nodes:** Detectors (space-time locations where errors can be detected)
- **Hyperedges:** Error mechanisms (faults that flip subsets of detectors)
- **Edge weights:** Log-likelihood ratios from error probabilities
- **Observables:** Logical operators that may be flipped by errors

For a detection event vector `d ∈ {0,1}^N` and observable flip `o ∈ {0,1}^K`:
```
d = Σ_e (fault_e occurred) · detector_mask_e   (mod 2)
o = Σ_e (fault_e occurred) · observable_mask_e (mod 2)
```

### Why Matching Works

MWPM succeeds because:
1. **Errors come in pairs:** Most faults flip exactly 2 detectors
2. **Minimum weight:** Correct decoding ≈ most likely error configuration
3. **Combinatorial structure:** Matching enforces that each detector is "explained" exactly once

A neural network that ignores this structure must learn it implicitly—which is harder than building it in.

### Soft Matching Layer

To make matching differentiable, we introduce a **soft matching layer** inspired by Sinkhorn iterations:

```
Input:  edge_scores[i,j] = learned score for edge (i,j)
        active_mask[i] = 1 if detector i is triggered

Output: soft_match[i,j] = soft assignment weight

Algorithm:
    1. Initialize: M[i,j] = exp(edge_scores[i,j]) * active_mask[i] * active_mask[j]
    2. For k iterations:
         M[i,j] = M[i,j] / sum_j(M[i,:])   # row normalization
         M[i,j] = M[i,j] / sum_i(M[:,j])   # column normalization
    3. Return soft matching weights
```

This approximates the discrete matching with a continuous relaxation that can be backpropagated through.

---

## Solution Architecture

### New Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        MGHDCircuit                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Detection Events ──► Node Embedding ──► GNN Message Passing    │
│        [B, N]              [B, N, D]         [B, N, D]          │
│                                                   │              │
│                                                   ▼              │
│                                            Edge Scoring          │
│                                              [B, E, 1]           │
│                                                   │              │
│                                                   ▼              │
│                                          SoftMatchingLayer       │
│                                              [B, E, 1]           │
│                                                   │              │
│                                                   ▼              │
│                                          Observable Head         │
│                                              [B, K, 1]           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```python
# 1. Sample from Stim circuit
samples = StimCircuitSampler.sample(shots)
# Returns: {"dets": [B, N], "obs": [B, K]}

# 2. Build DEM graph
graph = build_dem_graph(circuit, det_events, obs_flips)
# Returns: DEMGraph with edge_index, edge_weight, obs_mask, etc.

# 3. Forward pass
obs_logits, edge_probs = model(graph)

# 4. Loss: BCE on observable prediction
loss = F.binary_cross_entropy_with_logits(obs_logits, graph.y_obs)
```

### Key Design Decisions

1. **No teacher decoder needed:** Ground truth comes directly from Stim simulation
2. **DEM structure preserved:** Graph construction respects circuit-level semantics
3. **Matching built-in:** SoftMatchingLayer explicitly models MWPM's mechanism
4. **Observable output:** Predicts logical errors, not physical qubit errors

---

## Code Changes

### 1. `mghd/core/core.py`

#### `DEMGraph` dataclass
```python
@dataclass
class DEMGraph:
    x_det: Tensor       # [N, 1] detector feature (triggered or not)
    det_events: Tensor  # [B, N] batched detection events
    edge_index: Tensor  # [2, E] sparse graph connectivity
    edge_weight: Tensor # [E] log-likelihood ratio from DEM
    obs_mask: Tensor    # [N, K] which detectors connect to observables
    y_obs: Tensor       # [B, K] ground truth observable flips
    num_detectors: int
    num_edges: int
    num_observables: int
```

**Purpose:** Native representation of circuit-level structure. Unlike `PackedCrop` (which has Hz/Hx semantics), this directly encodes DEM topology.

#### `build_dem_graph()` function
```python
def build_dem_graph(
    circuit_or_dem,
    det_events: np.ndarray,
    obs_flips: np.ndarray,
) -> DEMGraph:
    """Parse Stim DEM and construct graph for MGHDCircuit."""
```

**Purpose:** Extracts edges between detector pairs, computes edge weights from error probabilities, builds observable connectivity mask.

#### `SoftMatchingLayer` class
```python
class SoftMatchingLayer(nn.Module):
    def forward(self, edge_scores, edge_index, active_mask, n_iter=5):
        """Differentiable approximation of minimum-weight matching."""
```

**Purpose:** Replaces attention pooling with structure-aware matching approximation.

#### `MGHDCircuit` model
```python
class MGHDCircuit(nn.Module):
    def __init__(self, d_model=128, n_gnn_iters=6, n_match_iters=5, dropout=0.1):
        self.node_embed = nn.Linear(1, d_model)
        self.edge_embed = nn.Linear(1, d_model)
        self.gnn = GraphDecoderCore(...)  # Reuses existing GNN
        self.edge_scorer = nn.Sequential(...)
        self.soft_match = SoftMatchingLayer(n_match_iters)
        self.obs_head = nn.Sequential(...)
```

**Purpose:** End-to-end circuit-level decoder with proper DEM semantics.

### 2. `mghd/samplers/stim_sampler.py`

#### `_get_surface_circuit()` (cached)
```python
@lru_cache(maxsize=32)
def _get_surface_circuit(distance, rounds, dep, basis="z"):
    return stim.Circuit.generated("surface_code:rotated_memory_" + basis, ...)
```

**Purpose:** Avoids rebuilding identical circuits; significant speedup for training.

#### `StimCircuitSampler` class
```python
class StimCircuitSampler:
    def __init__(self, *, distance, rounds, dep):
        self._distance = distance
        self._rounds = rounds
        self._dep = dep
        
    def sample(self, shots) -> dict[str, np.ndarray]:
        """Returns {"dets": [B, N], "obs": [B, K]}"""
        
    def sample_with_dem_graph(self, shots) -> DEMGraph:
        """Returns ready-to-use DEMGraph with batched data."""
```

**Purpose:** Provides both raw samples and constructed `DEMGraph` for training.

### 3. `mghd/cli/train_circuit.py`

New CLI for circuit-level training with infrastructure from `train.py`:

| Feature | Implementation |
|---------|----------------|
| DDP multi-GPU | `torch.distributed` + `DistributedDataParallel` |
| Mixed precision | `torch.amp.autocast` with bf16/fp16 |
| Resume | `--resume checkpoint.pt` |
| Early stopping | `--early-stop-patience N` |
| Curriculum | `--distance-curriculum`, `--rounds-curriculum`, `--dep-curriculum` |

### 4. `pyproject.toml`

Added CLI entry point:
```toml
mghd-train-circuit = "mghd.cli.train_circuit:main"
```

---

## Usage

### Basic Training
```bash
mghd-train-circuit --distance 5 --rounds 7 --dep 0.001 --epochs 100
```

### Multi-GPU with DDP
```bash
torchrun --nproc_per_node=4 -m mghd.cli.train_circuit \
    --distance 5 --rounds 7 --dep 0.001 --amp bf16
```

### Curriculum Over Distances
```bash
mghd-train-circuit \
    --distance-curriculum 3,5,7 \
    --rounds-curriculum 3,5,7 \
    --dep-curriculum 0.001,0.003,0.005 \
    --epochs-per-curriculum 5 \
    --epochs 150
```

### Resume with Early Stopping
```bash
mghd-train-circuit \
    --resume data/results/circuit_xxx/last.pt \
    --early-stop-patience 10 \
    --early-stop-min-delta 0.001
```

---

## File Reference

| File | Purpose | Changes |
|------|---------|---------|
| `mghd/core/core.py` | Neural network components | Added `DEMGraph`, `build_dem_graph()`, `SoftMatchingLayer`, `MGHDCircuit` |
| `mghd/samplers/stim_sampler.py` | Stim sampling | Added `StimCircuitSampler`, `_get_surface_circuit()` cache |
| `mghd/cli/train_circuit.py` | Training CLI | **New file** — circuit-level training with DDP/AMP/curriculum |
| `pyproject.toml` | Package config | Added `mghd-train-circuit` entry point |

---

## Comparison: Before vs After

| Aspect | Before (train.py + MGHDv2) | After (train_circuit.py + MGHDCircuit) |
|--------|----------------------------|----------------------------------------|
| **Data** | `PackedCrop` with Hz/Hx | `DEMGraph` with DEM structure |
| **Model** | Per-qubit prediction | Observable prediction |
| **Supervision** | Teacher decoders | Ground truth from Stim |
| **Loss** | Per-qubit BCE + parity aux | Observable BCE |
| **Matching** | Implicit in embeddings | Explicit `SoftMatchingLayer` |
| **Expected LER** | 3-6× worse than MWPM | Closer to MWPM (structure-aware) |

---

## Appendix: Why Not Merge Into train.py?

The code-level and circuit-level pipelines have fundamentally different:

1. **Data structures:** `PackedCrop` vs `DEMGraph`
2. **Models:** `MGHDv2` vs `MGHDCircuit`
3. **Supervision:** Teacher decoders vs Stim ground truth
4. **Loss functions:** Per-qubit + auxiliary vs observable-only
5. **Evaluation metrics:** Per-qubit accuracy vs logical error rate

Merging would require duplicating the entire training loop with conditional branches everywhere, making maintenance harder. Keeping them separate with shared infrastructure (DDP, AMP, curriculum) is cleaner.

---

## References

- [Stim documentation](https://github.com/quantumlib/Stim) — Circuit-level noise simulation
- [PyMatching](https://github.com/oscarhiggott/PyMatching) — MWPM decoder for DEMs
- [troubleshoot.md](troubleshoot.md) — Original Phase 3/4 failure analysis
