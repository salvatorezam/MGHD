MGHD-Primary (Clustered) Sub-Microsecond Roadmap

Last updated: 2025-09-19

0) Goal, Scope, Success Criteria

Goal: Deliver sub-microsecond p95 per-round latency (single code cycle, both CSS sides) using MGHD as the primary decoder with clustered inference and exact GF(2) parity projection. No BP/LSD in the loop.

Targets
	•	H100 GPU: p95 10–50 µs (cloud path).
	•	ZCU216 FPGA: p95 < 1 µs (embedded path).
Stretch: ZCU106 p95 2–4 µs.

Scope: Rotated surface codes d\in\{3,5,9,11\}, p\in[0.002,0.015]. (Later: BB [144,12,12], qLDPC.)

DoD:
	•	Verified p50/p95/p99 and LER on the grid; deterministic parity satisfaction per shot.
	•	Reproducible JSON/plots under results/ and commit-frozen code.

⸻

1) Baseline (current)
	•	Decoder: MGHD-primary, clustered, exact ML GF(2) projection, no LSD/BP.
	•	Measured @ d=3,\ p=0.005: per-round mean 0.378 ms (X: 0.214 ms, Z: 0.164 ms).
Component means: MGHD 0.286 ms (76%), clustering 0.038 ms, projection 0.028 ms.
	•	LER: 0/10k (projection enforces parity).

⸻

2) Steady-State Architecture
	1.	Build active-check clusters (connected components of H_\text{act}^\top H_\text{act}); optional 1-hop halo.
	2.	Pack subgraphs (checks→qubits node order; syndrome in channel-0 on check nodes).
	3.	Run MGHD-C (tiny distilled expert) per cluster → per-qubit marginals.
	4.	Exact ML parity projection on the cluster (GF(2) elimination + nullspace enumeration with cap).
	5.	Scatter corrections; done.
Optionals: Tier-0 LUT for |C|\le 3; temporal state for predict-then-verify.

Scaling: below threshold, cluster counts ∝ p d^2 but largest cluster grows ~O(\log d). With GPU micro-batching / FPGA parallel engines, p95 is flat→O(\log d) in d.

⸻

3) Roadmap (phased; you are the sole owner)

Phase A — GPU kernelization & measurement hardening (Week 1)
	•	A1. Micro-batch clusters (pack many subgraphs into one forward; padded tensors + packed edges).
	•	A2. Persistent kernel on H100 (avoid launches; fuse pre/post + projection if possible).
	•	A3. p/d-sweep harness producing JSON + plots.

Acceptance: H100 per-round p95 < 150 µs @ d=3,\ p=0.005.

⸻

Phase B — Distillation + INT8 (Weeks 2–3)
	•	B1. Distill S→Tiny MGHD-C (e.g., d_model 64, d_state 16, 3 MP iters, features 64). Loss: KL/CE + small parity term. Train on cluster crops from larger d.
	•	B2. INT8 QAT (per-channel; keep SSM/GRU in FP16 if needed). Export ONNX/TRT.
	•	B3. Temperature scaling per side.

Acceptance: H100 per-round p95 < 80 µs; mean ≤ 0.12 ms; LER drift ≤ 5% vs S-teacher on the grid.

⸻

Phase C — Tier-0 LUT + κ-gating (Week 3–4)
	•	C1. LUT solutions for |C|\le 3 canonical shapes (1–2 cycles on FPGA; trivial on GPU).
	•	C2. κ-gate on GPU: if only LUT-eligible clusters appear, skip MGHD.
	•	C3. Telemetry: fraction of LUT-eligible shots vs p,d.

Acceptance: GPU p50 < 30 µs, p95 < 60 µs @ d=3,\ p=0.005.

⸻

Phase D — Temporal streaming (Week 4–5)
	•	Maintain Mamba state per cluster footprint.
	•	Predict-then-verify: emit speculative correction from state; if residual≠0, run MGHD-C; else skip.

Acceptance: ≥ 20–40% fewer MGHD calls on stationary runs; no LER regression.

⸻

Phase E — FPGA prototype (ZCU216) (Weeks 2–6, parallel to B–D)
	•	E1. Architecture: 8–16 INT8/INT4 MGHD-C engines; URAM scratch; small per-engine GF(2) solver; MoE dispatcher by cluster size.
	•	E2. HLS/RTL kernels for SSM/linear, projection, and stream glue (DMA→pack→expert→GF(2)→scatter).
	•	E3. Cycle-accurate model + harness parity checks.

Acceptance: cycle model p95 < 1 µs per round for d\in\{3,5,9\},\ p\in[0.002,0.010]; post-route timing ≥ 300–500 MHz; resource fit.

⸻

Phase F — MoE-C & large-d robustness (Weeks 6–8)
	•	3–5 expert variants (by cluster size/shape).
	•	r-cap policy: exact GF(2) for nullity ≤ 6; fallback expert otherwise.
	•	Stress d=11 and mid-p; show flat→log(d) p95.

Acceptance: ZCU216 p95 < 1 µs across grid; ZCU106 2–4 µs; GPU plots show near-flat p95 vs d at low p.

⸻

4) Estimated Gains (simple)
	•	Micro-batching (GPU): ~×2 on MGHD → 0.38 ms → 0.24 ms.
	•	Distillation (Tiny MGHD-C): ×3 on MGHD → 0.38 → 0.19 ms.
	•	INT8 QAT: ×1.6 on MGHD → 0.38 → 0.27 ms.
	•	Combined tuned (GPU): ~0.12–0.13 ms/round (≈×3 overall).
	•	Tier-0 LUTs: many shots pay ~0; p50 tens of µs.
	•	Temporal streaming: −20–40% MGHD invocations.
	•	FPGA ZCU216: 0.06–0.12 ms/round (deterministic).
	•	FPGA ZCU106: 0.14–0.25 ms/round.

⸻

5) Measurement Playbook
	•	Bench scripts: tools/bench_mghd_primary_clustered_d{D}.py with per-shot timers (clustering, MGHD, projection, total).
	•	Outputs: results/*.json; plots under results/figs/.
	•	KPIs: p50/p95/p99 latencies, LER, fraction LUT-eligible, #clusters/shot, nullity histogram.
	•	Statistical power: ≥ 5k shots per point; Wilson CI for LER.

⸻

6) Risks & Mitigations
	•	Subgraph/model mismatch: unit tests for feature builder; distill on cluster crops.
	•	INT8 accuracy loss: QAT with per-channel scales; temp calibration; mixed precision for SSM if needed.
	•	FPGA timing/resource: INT4 variants; share engines; URAM tiling; early floorplanning.
	•	GPU jitter: persistent kernels; CUDA Graphs; NUMA pinning.

⸻

7) Deliverables & Checkpoints
	•	A: GPU micro-batched persistent decoder; p95 < 150 µs; plots vs p,d.
	•	B: Distilled MGHD-C + INT8; p95 < 80 µs; model card + TRT engine.
	•	C: LUT tier + κ-gate; p50 < 30 µs, p95 < 60 µs.
	•	D: Temporal streaming; ≥ 20% fewer MGHD calls.
	•	E: ZCU216 cycle-model p95 < 1 µs; timing closure plan.
	•	F: Final ZCU216 bitstream p95 < 1 µs; ZCU106 2–4 µs; distance/p sweeps.

⸻

8) Quick Formulas
	•	#detections \sim \lambda p d^2.
	•	Largest cluster \kappa_{\max}(p,d) \sim O(\log d) (subcritical).
	•	GPU p95 T \approx T_\text{overhead} + T_\text{fwd}(\kappa_{\max}) \times \left\lceil\frac{\#\text{clusters}}{B}\right\rceil.
	•	Exact projection: minimize \sum_j \log\frac{1-p_j}{p_j} e_j s.t. H_\text{sub} e = s_\text{sub} (mod 2).
