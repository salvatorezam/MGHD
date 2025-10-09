import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from mghd.samplers.stim_sampler import sample_surface_memory
from mghd.decoders.dem_utils import build_surface_memory_dem


def test_stim_sampler_matches_dem():
    d = 3
    rounds = 3
    shots = 64
    dep = 0.001

    dem = build_surface_memory_dem(
        distance=d,
        rounds=rounds,
        profile={"gate_error": {"after_clifford_depolarization": dep}},
        decompose=True,
    )
    dets, obs = sample_surface_memory(d=d, rounds=rounds, shots=shots, dep=dep, seed=42)

    assert dets.shape == (shots, dem.num_detectors)
    assert obs.shape == (shots, dem.num_observables)
    assert dets.dtype == np.uint8
    assert obs.dtype == np.uint8
