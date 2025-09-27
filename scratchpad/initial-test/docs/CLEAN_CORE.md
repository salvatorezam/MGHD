# Clean-Core v3

The Clean-Core stack keeps only the distance-agnostic MGHD v2 runtime, cluster tooling, and
teacher ensemble paths required for training and benchmarking. Legacy performance features,
hybrid micro-batching, and v1 assets are relocated under `archive/` and replaced with fail-fast
stubs that point back to this document.

## Scope
- `mghd_public/*` exposes MGHD v2 blocks, inference helpers, and feature packing utilities.
- `mghd_clustered/*` provides clustered decoding, Garnet sampling adapters, and surface PCM helpers.
- `teachers/*` keeps the MWPF primary + MWPM fallback ensemble.
- `tools/make_cluster_crops.py` produces NPZ shards using CUDA-Q Garnet sampling.
- `tools/bench_clustered_sweep_surface.py` runs the minimal sweep used in the runbook.
- `tests/*` cover feature packing, projector correctness, MGHD invocation, and LER injection wiring.

## Archived components
Performance experiments, v1-only models, cached micro-batching, and heavy analysis scripts now
live under `archive/`. Importing an archived module from its original path raises
`RuntimeError("This module is archived in archive/. See docs/CLEAN_CORE.md")`.

Key archived areas include:
- `mghd_public/features.py`, `mghd_public/model.py`, `poc_my_models.py` (legacy v1 stack)
- Micro-batching and perf bench tools (`mghd_clustered/microbatcher.py`, `tools/bench_*`)
- Acceptance caches, optuna sweeps, large plotting/CI harnesses
- v1/legacy tests and training pipelines

## Re-admission checklist
A feature graduating from `archive/` back into the clean core must satisfy:
1. **Justification** – document the runtime requirement and expected benefit in this file.
2. **Zero side effects at import** – no CUDA, CUDA-Q, or file-system initialization on module import.
3. **Tests** – add unit coverage (pure CPU / deterministic) proving correctness for MGHD v2.
4. **Docs** – update `docs/CLEAN_CORE.md` with an overview of the reinstated component.
5. **Inventory** – run `python tools/dev/audit_repo.py` and ensure the module is reported in
   `used_modules`.
