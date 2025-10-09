import numpy as np
import pytest


def test_surface_dem_smoke():
    stim = pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    from teachers.dem_utils import build_surface_memory_dem
    from teachers.dem_matching import DEMMatchingTeacher

    dem = build_surface_memory_dem(
        distance=3,
        rounds=3,
        profile={"gate_error": {"after_clifford_depolarization": 0.003}},
        decompose=True,
    )
    teacher = DEMMatchingTeacher(dem, correlated=False)

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.003,
    )
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=64, separate_observables=True)

    pred = teacher.matching.decode_batch(dets)
    assert pred.shape == obs.shape
    assert pred.dtype == np.uint8
