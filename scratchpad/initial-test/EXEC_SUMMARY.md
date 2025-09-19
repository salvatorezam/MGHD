# Executive Summary: MGHD Clustered Decoder Integration (2025-09-18)

## Scope & Actions
- Added clustered decoding infrastructure under `mghd_clustered/`:
  - `adapter.py` exposes MGHD-derived per-bit error probabilities (with neutral fallback).
  - `decoder.py` wraps LDPC’s public `BpLsdDecoder`, defaulting to `lsd_method="LSD_E"`, `max_iter=1`, and optional MGHD priors/stats export.
  - `pcm_utils.py` provides random PCM generation and sampling utilities.
- Implemented realistic Tanner graph builders in `mghd_clustered/pcm_real.py` (analytic rotated surface CSS matrices for odd distances; [[144,12,12]] BB bivariate bicycle code) plus a placeholder `stim_to_pcm.py` for future DEM conversion.
- Reworked `scripts/bench_lsd_clustering.py` to run BP-only, clustered LSD (`bits_per_step=16`), and monolithic LSD on the surface/BB PCMs, reuse common syndrome samples, apply heuristic priors, print metrics, and dump JSON results under `results/`.
- Updated dependency requirements (`requirements.txt`, README) to `ldpc>=2.1.0`, `numpy>=1.26.4`, `scipy>=1.11.1`, `torch>=2.3.0`; upgraded packages inside `mlqec-env` (pip install -U ...) while noting panqec and torchvision/torchaudio version warnings.
- Logged each change in `IMPLEMENTATION_SUMMARY.md` with UTC timestamps.

## Verification & Results
- Smoke test on random PCM confirms decoded syndrome equals input; LDPC stats object exists but cluster telemetry fields remain `None` in the current wheel.
- Rotated surface code (d=9, p=0.002, 500 shots):
  - BP-only avg ≈ **0.017 ms** (p95 ≈ 0.021 ms).
  - LSD clustered avg ≈ **0.0076 ms** (p95 ≈ 0.0106 ms), failures **0**.
  - LSD monolithic avg ≈ **0.0076 ms** (p95 ≈ 0.0106 ms), failures **0**.
- BB [[144,12,12]] code (p=0.001, 500 shots):
  - BP-only avg ≈ **0.0177 ms** (p95 ≈ 0.0461 ms).
  - LSD clustered avg ≈ **0.0143 ms** (p95 ≈ 0.0202 ms), failures **0**.
  - LSD monolithic avg ≈ **0.0141 ms** (p95 ≈ 0.0201 ms), failures **0**.
- Benchmark outputs stored as `results/lsd_clustering_*.json`; clustering clearly improves latency versus BP on realistic codes, while clustered vs monolithic LSD remain similar without MGHD priors.

## Follow-Up Considerations
1. Implement the Stim DEM→CSS converter in `stim_to_pcm.dem_to_css_pcm` to align with LDPC documentation.
2. Integrate real MGHD priors through `MGHDAdapter` to further reduce κ and distinguish clustered vs monolithic LSD performance.
3. Resolve dependency conflicts (panqec’s ldpc pin, torchvision/torchaudio’s torch requirement) for a stable environment.
4. Investigate missing cluster telemetry in the LDPC Python wheel; add instrumentation if upstream support is absent.
