MGHD Project — Engineering Guide for Agents

Overview
- Mission: Build a foundation decoder for IQM Garnet (circuit‑level noise), then fine‑tune a student on exact calibrations. Achieve MWPM‑class or better logical error rate (LER) with sub‑millisecond single‑shot (B=1) GPU latency on rotated surface d=3 (d=5 next). Prepare for FPGA deployment; CPU performance is a non‑goal.
- Constraints (acceptance gates):
  - Accuracy: LER ≤ MWPM/MWPF baselines; strict parity/coset checks.
  - Latency: single‑shot (B=1) GPU decode < 1 ms for d=3; report p50/p99.
  - Reproducibility: fixed seeds, SHA tagging for matrices/artifacts.
  - Reporting: N≥10k per‑p samples, Wilson CIs, params/FLOPs, latency.

Environment & Conventions
- Always run in `conda activate mlqec-env`.
- GPU: H100 preferred, compute capability 9.0.
- No CUDA at import: Do not initialize CUDA contexts or launch kernels in top‑level import paths. Initialize CUDA‑Q, Torch CUDA, graph capture, and extensions inside `main()` or callables. This keeps imports fast, testable, and compatible with non‑GPU contexts.
- Lazy CUDA‑Q: Use CUDA‑Q for syndrome generation; import/setup CUDA‑Q within functions/CLI (not at module import) to avoid side effects and ensure graceful failure when absent.

Data Formats (rotated d=3)
- Syndromes: 8 bits per round, canonical order Z‑checks then X‑checks; LSBF packing within each byte. One byte per shot.
- Corrections: 9 data‑qubit bits (uint8).
- Canonical packs:
  - `student_pack_p003.npz` (Hx, Hz, meta)
  - `fastpath/rotated_d3_lut_256.npz` (LUT16[256], Hx, Hz, meta)

Teachers & Labels
- Primary teacher: MWPF (circuit‑level matching), installed in venv.
- Fallback teacher: MWPM (phenomenological defect matching, PyMatching).
- Ensemble resolution: When both available, compute both and choose the lower‑weight valid correction under strict parity/coset checks. If MWPF is unavailable/invalid for a batch, use MWPM to preserve throughput.
- Rationale: MWPF provides better supervision under circuit‑level noise with realistic measurement errors/correlations; MWPM ensures stability and availability at scale.

Noise Model (Garnet)
- Source file: `cudaq_backend/garnet_noise.py`
- Foundation training distribution: domain randomization: p1/p2 log‑uniform, bad‑edge tails, T1/T2→Tφ, asymmetric readout, durations, crosstalk, spatial heterogeneity, regime mixing (see section below).
- Student training: exact Garnet calibration (no deltas) to specialize for hardware.

Decoders
- MGHD (Mamba + GNN): `poc_my_models.py`
  - Sequence (Mamba) over check nodes; GNN for spatial message passing on Tanner graph; binary head (2 logits/qubit) for rotated layouts.
  - Static indices derived from Hx/Hz; authoritative sizes; rotated layout enforcement.
- Baseline GNN (Astra‑style): `panq_functions.py` (GNNDecoder) and `poc_gnn_train.py` comparison script.
- Fastpath LUT (rotated d=3): `fastpath/` (PyTorch ext and persistent kernel); `fastpath/c_api` for C ABI.

Training Flows
1) Baseline vs MGHD (small run)
   - Goal: Verify MGHD outperforms baseline GNN before scaling.
   - Dataset: Rotated d=3, CUDA‑Q Garnet foundation sampler with MWPF+MWPM labels.
   - Script: adapt `poc_gnn_train.py` to read syndromes/labels from the CUDA‑Q sampler (same data for both models). Plot LER_X, LER_Z, LER_total and fraction solved.

2) Foundation Model (Step‑11 style)
   - Sampler: CUDA‑Q Garnet with ±10% parameter spread; MWPF primary, MWPM fallback; strict parity/coset validation.
   - Loss: BCEWithLogits (bit‑head), minimal label smoothing, grad clip=1.0.
   - Schedule: cosine with warmup; curriculum over p ∈ {0.02, 0.03, 0.05, 0.08}; 20–40 epochs; early stop/selection by coset‑validated val LER (N≥10k per‑p).
   - Profiles: S/M/L (parameterized), selected by LER ± CI under latency budget.

3) Student Model
   - Initialize from foundation checkpoint; fine‑tune on exact Garnet calibrations (no deltas) under same loss/validation gates.

4) Optuna Sweeps
   - Objective: minimize LER (not only latency). Explore hyperparameters around known good regions under latency‑budget constraints. Report best params with confidence intervals.

Evaluation & Reporting
- LER harness: `tools/eval_ler.py` with `--decoder {mghd,mwpm,mwpf,relay,fastpath}`; parity/coset enforced; Wilson CIs (95%); N≥10k per‑p.
- Latency: `tools/collect_latency_benchmarks.py` — single‑shot (B=1) and throughput (B=256) for S/M/L across backends (eager, TorchScript, CUDA Graph; optional ONNX/TensorRT if installed). Report p50/p99/mean.
- Artifacts: Save JSON/NPZ to `results/` and `reports/` with short SHA‑256 hashes of Hx/Hz/labels where applicable. Update `IMPLEMENTATION_SUMMARY.md` with UTC timestamps and key metrics.

Latency Optimization (B=1)
- Preallocate all decode buffers and capture `decode_one` (or equivalent forward path) using CUDA Graph replay to remove Python/kernel dispatch overhead.
- TorchScript: reduce Python overhead; typically faster than eager for B=1.
- TensorRT/ONNX Runtime (optional): INT8/FP16 engines can further reduce latency; gate usage on measured improvements and availability.
- Fastpath persistent: persistent kernel + zero‑copy ring buffer for LUT decoding achieves ~44–50 µs typical, best ~20–26 µs on H100 for d=3.

File Layout (selected)
- `poc_my_models.py` — MGHD model (Mamba+GNN), rotated layout support, ONNX export.
- `unified_mghd_optimizer.py` — Step‑11 training CLI (foundation path), now extended for S/M/L and auto‑evaluation.
- `tools/eval_ler.py` — Coset‑aware LER evaluator with CIs and latency.
- `tools/collect_latency_benchmarks.py` — Latency collector for S/M/L and fastpath.
- `tools/realtime_decode.py` — Persistent fastpath streaming service (demo).
- `fastpath/*` — PyTorch extensions and persistent kernel for LUT decoding; `c_api/` for libfastpath.so.
- `cudaq_backend/*` — (To be) circuit‑level Garnet sampler and wrappers; keep imports lazy.

Reproducibility
- Seeds: set NumPy, PyTorch, Torch CUDA to a fixed seed (e.g., 42) for each run and record in artifacts.
- Determinism: where performance permits, use deterministic ops; otherwise document nondeterministic kernels in reports.
- Hashing: include short SHA‑256 of Hx/Hz and label buffers in JSON/NPZ for traceability.

Coding Guidelines
- Keep imports light; no GPU initialization at import time.
- Keep changes minimal and focused; avoid creating unnecessary files; delete scratch artifacts after use.
- Use clear module guards (`if __name__ == "__main__":`) for CLIs and heavy initializations.
- Maintain strict parity/coset checks in training/evaluation; never report LER without passing parity gates.
- Update `IMPLEMENTATION_SUMMARY.md` with UTC timestamps for material progress (training start/stop, best LER, latency updates).

Playbooks
1) Baseline vs MGHD small comparison
   - Generate Garnet foundation dataset via CUDA‑Q with MWPF/MWPM labels.
   - Train baseline GNN and MGHD on the same data; plot LER curves; proceed only if MGHD ≥ baseline.

2) Foundation training (selected profile)
   - Run Step‑11 for S/M/L; pick best by val LER ± CI; verify B=1 latency meets target after CUDA Graph capture.

3) Student fine‑tune, distill, quantize
   - Fine‑tune on exact calibration; distill to smaller student if accuracy allows; quantize (PTQ/INT8) without compromising LER or latency; validate with `eval_ler.py` and latency collector.

4) Fastpath reference
   - Build `fastpath/c_api` (CMake) and benchmark libfastpath.so for C/firmware integration; maintain parity with Python persistent results.

Escalation & Dependencies
- If MWPF import/solver fails: fall back to MWPM, log event, continue streaming; do not drop batches.
- If CUDA‑Q is unavailable: skip sampling and raise a clear actionable error; do not attempt CUDA at import.
- If parity checks fail for labels/predictions: quarantine batch, log the SHA keys, and continue; never accept invalid labels into training.

Glossary
- B (batch size): number of syndromes processed simultaneously; B=1 is single‑shot latency target for real‑time QEC; B≫1 improves throughput (amortized per‑shot).
- Parity/Coset checks: Valid correction must reproduce Z/X stabilizer parities and be in the same coset as the reference (kernel of Hx/Hz for delta).
- Foundation vs Student: Foundation trains on a distribution around hardware parameters; student fine‑tunes on exact parameters.

Teacher Integration Details (MWPF + MWPM)
- MWPF 0.2.12 (primary): Use MWPF to decode CUDA‑Q generated detection events with full circuit‑level context. Provide a Stim circuit/DEM strictly as decoder metadata via `SinterMWPFDecoder.with_circuit(circuit)` (never for sampling) so MWPF models heralded/meas errors correctly.
- MWPM 2.0.1 (fallback): Use PyMatching’s `Matching` to produce defect‑matching corrections when MWPF is unavailable/invalid for a batch. If both succeed, pick minimum‑weight valid correction under strict parity/coset gates.
- Parity/Coset enforcement: Verify split parity (Z then X) and coset equivalence (Hz/Hx kernel) before accepting labels.

Garnet Foundation Domain Randomization (rotated d=3)
- 1Q infidelity p1: log‑uniform [1e‑4, 5e‑3] per qubit; drift ×U[0.97,1.03] per epoch.
- 2Q infidelity p2: log‑uniform [3e‑3, 3e‑2] per coupler; bad‑edge tail (f_bad~U[0.05,0.20], κ~U[3,8], clip to 0.12); drift ×U[0.95,1.05].
- Coherence times: T1_us ~ log‑uniform [20,150], T2_us ~ log‑uniform [2,80], enforce T2 ≤ 0.7·T1; derive Tφ and apply dephasing by idle window.
- Durations: t_prx_ns~U[20,80], t_cz_ns~U[25,120], optional t_meas_ns~U[300,800].
- Readout asymmetry: e~U[0.01,0.05], r~LogNormal(0,0.35); eps0=clip(e/r, [0.002,0.10]), eps1=clip(e·r, [0.002,0.10]).
- Crosstalk proxy: α_xtalk~U[1.1,1.5] inflating neighbor dephasing during CZ.
- Spatial heterogeneity: ±5–10% low‑frequency field over p1/p2/eps; enforce ≥1 bad edge per patch.
- Regime mixing: Typical/Strong/Pathological with probs [0.6,0.3,0.1] × multipliers [1.0,1.5,2.5], clip within caps.

CUDA‑Q Only Policy
- All sampling is performed by CUDA‑Q circuit‑level simulation; Stim is used only to provide DEM/circuit metadata to the teacher (MWPF) and never to generate syndromes.
- Guard all CUDA and CUDA‑Q work inside callables/CLIs (no GPU work at import) to keep imports fast, testable, and clean.

Baseline vs MGHD Procedure (rotated d=3)
- Use the same CUDA‑Q Garnet foundation dataset for both baseline GNN and MGHD.
- Train/evaluate both under identical splits; produce LER_X/LER_Z/LER_total curves and fraction solved.
- Proceed to large‑scale foundation training only if MGHD meets or exceeds baseline LER on the same data.

Optuna (LER‑first) and Model Selection
- Sweep S/M/L around prior good regions with latency constraints; objective is to minimize LER (with Wilson CIs) at N≥10k per‑p.
- Select the best profile by LER ± CI; verify B=1 latency after CUDA Graph capture meets targets.

Post‑Student Optimization (no compromises)
- Distillation: teacher = best model (likely L), student = smaller (S/M); preserve LER.
- Quantization Aware Training: FP16→INT8 with calibration; accept only if LER parity is maintained.
- Structured pruning: remove low‑contribution channels/heads guided by LER ablation; re‑validate.
- Engine build: TensorRT/ONNX INT8 engines for sub‑µs single‑shot; measure p50/p99; accept only if LER/latency targets are met.
