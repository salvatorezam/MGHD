import numpy as np
import pytest


pytest.importorskip("stim")


def _extract_dem_matrices(dem):
    n_det = int(dem.num_detectors)
    n_err = int(dem.num_errors)
    n_obs = int(dem.num_observables)

    H = np.zeros((n_det, n_err), dtype=np.uint8)
    L = np.zeros((n_obs, n_err), dtype=np.uint8)

    err_idx = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                H[int(target.val), err_idx] ^= 1
            elif target.is_logical_observable_id():
                L[int(target.val), err_idx] ^= 1
        err_idx += 1

    assert err_idx == n_err
    return H, L


def test_stim_dem_sampler_parity_holds():
    import stim

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    H, L = _extract_dem_matrices(dem)

    sampler = dem.compile_sampler()
    dets, obs, err = sampler.sample(shots=2048, return_errors=True, bit_packed=False)
    dets = np.asarray(dets, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    err = np.asarray(err, dtype=np.uint8)

    dets_pred = (err @ H.T) % 2
    obs_pred = (err @ L.T) % 2

    assert dets_pred.shape == dets.shape
    assert obs_pred.shape == obs.shape
    assert np.array_equal(dets_pred, dets)
    assert np.array_equal(obs_pred, obs)
