from __future__ import annotations

import json
import pathlib
import statistics as stats
import sys
import time
from typing import Dict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mghd_clustered.adapter import MGHDAdapter
from mghd_clustered.decoder import MGHDClusteredDecoder
from mghd_clustered.pcm_real import bb_144_12_12_pcm, rotated_surface_pcm


def toy_priors(H) -> np.ndarray:
    """Simple heuristic priors using column degree information."""
    deg = np.diff(H.tocsc().indptr).astype(float)
    if deg.std() == 0:
        return np.full(H.shape[1], 0.5, dtype=np.float64)
    bias = 0.05 * (deg - deg.mean()) / deg.std()
    return np.clip(0.5 + bias, 1e-4, 1 - 1e-4)


def run_suite(H, name: str, p: float, shots: int = 500, seed: int = 1) -> Dict[str, Dict]:
    n_bits = H.shape[1]
    results: Dict[str, Dict] = {}
    rng = np.random.default_rng(seed)

    samples = []
    for t in range(shots):
        e = (rng.random(n_bits) < p).astype(np.uint8)
        s = (H @ e) % 2
        samples.append(s.astype(np.uint8))

    common_kwargs = dict(
        pcm=H,
        n_bits=n_bits,
        mghd_adapter=MGHDAdapter(n_bits=n_bits, model_loader=None),
        bp_method="minimum_sum",
        schedule="serial",
        error_rate=p,
        lsd_order=0,
    )

    cfgs = {
        "BP_only": dict(lsd_method="LSD_E", bits_per_step=n_bits, max_iter=8),
        "LSD_clustered": dict(lsd_method="LSD_E", bits_per_step=16, max_iter=1),
        "LSD_nocluster": dict(lsd_method="LSD_E", bits_per_step=n_bits, max_iter=1),
    }

    priors = toy_priors(H)

    for tag, params in cfgs.items():
        dec = MGHDClusteredDecoder(**common_kwargs, **params)
        dec.decoder.update_channel_probs(priors)
        lat_ms = []
        failures = 0
        for s in samples:
            t0 = time.perf_counter()
            r = dec.decode(s, features=None, use_mghd_priors=False)
            lat_ms.append((time.perf_counter() - t0) * 1e3)
            if not np.array_equal((H @ r) % 2, s):
                failures += 1
        results[tag] = dict(
            avg_ms=stats.mean(lat_ms),
            p95_ms=float(np.percentile(lat_ms, 95)),
            failures=failures,
            stats=dec.get_stats(),
        )

    print(f"==== {name} @ p={p} shots={shots} ====")
    for key, val in results.items():
        print(key, val)

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"lsd_clustering_{name.replace(' ', '_')}.json").open("w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import sys

    Hx, Hz = rotated_surface_pcm(d=9)
    run_suite(Hx, name="Surface_d9_X", p=0.002)
    run_suite(Hz, name="Surface_d9_Z", p=0.002)

    H_bb_x = bb_144_12_12_pcm(kind="X")
    run_suite(H_bb_x, name="BB_144_12_12_X", p=0.001)

    H_bb_z = bb_144_12_12_pcm(kind="Z")
    run_suite(H_bb_z, name="BB_144_12_12_Z", p=0.001)
