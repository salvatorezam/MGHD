# MGHD Decoder (S Profile) — Current Architecture and Results

## Topology (Rotated d=3)
- Hybrid: Mamba SSM over check‑node sequence → ChannelSE attention → Projection → GNN message passing (Astra‑style) → Binary head (2 logits/qubit).
- Rotated layout enforced: 8 checks + 9 data nodes (17 total); indices built from authoritative Hx/Hz (Z→X ordering) used by sampler/evaluator.

### Mamba Block
- Layers: 1 (single Mamba)
- d_model=192, d_state=32, d_conv=2, expand=3
- Attention: ChannelSE (reduction=4)
- Optional: post‑Mamba LayerNorm (configurable)

### GNN Block
- Message‑passing iters: n_iters=7
- Node/edge widths: n_node_features=128, n_edge_features=128
- Message MLP size: msg_net_size=96
- Dropout: msg≈0.04, gru≈0.11

### Interfaces
- Input embedding: Linear(9 → d_model), projection: Linear(d_model → 9)
- Binary head via logit difference; BCEWithLogits training

## Size and Latency
- Parameters: ~566,834 (from evaluator param count)
- Latency (H100, B=1): eager ≈3.0 ms p50; TorchScript ≈2.3 ms; CUDA Graph ≈1.7 ms; batch‑256 ≈109 µs/shot (reference; re‑measure per checkpoint as needed)
- Fastpath persistent LUT (rotated d=3): ≈50 µs single‑shot (separate decoder, LUT‑based)

## Training/Eval (Circuit‑Level)
- Data: CUDA‑Q trajectory sims (Garnet), MWPF primary teacher, parity guard; Hx/Hz aligned end‑to‑end.
- Loss: BCEWithLogits (label smoothing ~0.09); optional parity aux (λ~0.03); EMA 0.999.
- p‑grid support: explicit grids and curriculum presets; evaluator mirrors training forward path.

## Recent LER (N=10k per‑p)
- Acceptance grid (CUDA‑Q): p=0.02→0.1485; 0.03→0.2086; 0.05→0.3280; 0.08→0.4540
- Low grid (CUDA‑Q): p=0.02→0.1302; 0.03→0.2071; 0.05→0.3171; 0.08→0.4621
- Baselines: MWPF p=0.05→0.3139; MWPM p=0.05→0.3248

## Curriculum (Recommended)
- Grid: {0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015}
- Sampling: 40% from [0.001,0.004], 45% from [0.004,0.008], 15% from [0.008,0.015]
- Epoch phases: warm (bias 0.008–0.010), mid (0.005–0.007), final (0.003–0.006)

## Next Improvements
- Add nullspace/coset regularizer (tiny weight) to prefer teacher’s representative while preserving parity.
- p‑aware smoothing (lower at high‑p), keep parity λ small; consider post‑Mamba LN.
- Optional micro‑tuning within size budget to improve p=0.05 without hurting latency.

## Files
- Model: `poc_my_models.py`
- Trainer: `unified_mghd_optimizer.py`
- CUDA‑Q: `cudaq_backend/` (noise scaling via phys_p/noise_scale)
- Sampler: `tools/cudaq_sampler.py`
- Eval: `tools/eval_ler.py`
