# MGHD Decoder (S Profile) — Architecture and Rationale

## Summary
- Profile: S (promoted from Optuna Trial 12)
- Objective: Beat MWPF at p=0.05 with minimal size and sub‑ms B=1 latency.
- Best short‑run val LER (p=0.05): ~0.1758 (sweep); promotion run trending ~0.184–0.188 so far (EMA‑validated).

## Model Topology
- Hybrid: Mamba SSM over check‑node sequence → ChannelSE attention → Projection → GNN message passing on Tanner graph → Binary head per data qubit.
- Rotated d=3 enforcement: 8 checks + 9 data nodes; static indices derived from Hx/Hz.

### Mamba Block
- Layers: 1 (single Mamba block)
- Widths: d_model=192, d_state=80, d_conv=2, expand=4
- Attention: ChannelSE (Squeeze‑Excitation), reduction=8
- Optional stabilization: post‑Mamba LayerNorm (off for the sweep; configurable in trainer)

### GNN Block (Astra‑style `GNNDecoder`)
- Message‑passing iters: n_iters=8
- Node/edge widths: n_node_features=160, n_edge_features=256
- Message MLP size: msg_net_size=80
- Dropouts: msg=0.04135, gru=0.09564

### Interfaces
- Input embedding: Linear(n_node_inputs=9 → d_model)
- Projection: Linear(d_model → n_node_inputs=9)
- Binary head: via logit difference across 2‑logit head on qubit nodes; BCEWithLogits for training

## Parameter Count (S Trial‑12)
- Total: 926,707 parameters
- Breakdown:
  - Mamba: 648,960
  - GNN: 264,658
  - ChannelSE: 9,432
  - Projection: 1,737
  - Other: 1,920

## Training/Eval Configuration
- Data: CUDA‑Q Garnet, rotated d=3; MWPF primary labels, MWPM fallback; strict parity/coset checks.
- p‑regime: Fixed p=0.05 for promotion; sweep used p=0.05 focus for selection.
- Loss:
  - Primary: BCEWithLogits (optionally with label smoothing: y' = (1−s)·y + 0.5·s)
  - Auxiliary (optional): Parity‑aware loss via differentiable XOR expectation on Hx/Hz parities (computed FP32)
- Stabilization: EMA (typ. 0.999) used for validation; optional post‑Mamba LN.
- AMP: bf16; compile optionally enabled (promotion run uses compile; A/B disables to match sweep numerics)
- Evaluation: forward‑path evaluator (`--decoder mghd_forward`) mirrors training path with channel attention.

## Why These Features
- Mamba SSM: Efficient long‑range sequence modeling over check nodes with excellent throughput on GPU.
- ChannelSE attention: Empirically best (channel >> cross >> none) for LER; strengthens salient channels post‑SSM.
- GNN message passing: Encodes Tanner graph structure; complements SSM’s sequence modeling.
- Rotated layout enforcement: Eliminates shape/ordering pitfalls; consistent packing and parity logic.
- EMA: Stabilizes validation; typically improves test LER without runtime cost.
- Parity loss: Aligns logits with parity constraints; reduces coset errors when tuned (λ small).
- Label smoothing: Improves calibration; sweep winners favor ~0.09–0.14.

## Relevant Files
- Model: `poc_my_models.py` (MGHD, ChannelSE, FiLM, forward path, rotated layout helpers)
- Trainer: `unified_mghd_optimizer.py` (foundation training; overrides for S; EMA, parity loss, smoothing, schedules)
- Sampler: `tools/cudaq_sampler.py` (CUDA‑Q / LUT fallback, MWPF/MWPM teachers)
- Eval: `tools/eval_ler.py` (coset‑aware LER; `mghd_forward` to mirror training; channel attention enforced)
- Sweeps: `tools/sweep_s_optuna.py` (S‑profile Optuna; p=0.05 short runs; manual label smoothing)
- Auto Eval: `tools/auto_post_eval.py` (post‑run forward eval watcher)

## What We Tried (Chronology)
- L foundation (30 epochs, curriculum): best forward LER(p=0.05) ≈ 0.207–0.211; underperformed MWPF (~0.1952)
- XL (L+) curriculum: early per‑epoch best ~0.163 but didn’t hold at N=10k; auto forward eval queued
- Parity loss + EMA + post‑Mamba LN integrated into trainer
- Forward‑path evaluator added and attention parity enforced in eval
- S Optuna sweep (p=0.05 focus): best Trial 12 ≈ 0.1758 (beats MWPF baseline); promoted to 20‑epoch run
- A/B queue: constant LR + smoothing (A) vs +small parity (B) to reconcile sweep/promotion differences

## Next Steps
- Complete S promotion run; confirm with N=10k per‑p forward eval and latency
- Select best of A/B and (if better) re‑promote with same post‑run suite
- If needed, small capacity tweak (e.g., edges=384; msg=96) within latency budget; re‑validate
- Lock S for deployment; proceed to distillation / QAT / engines as needed
