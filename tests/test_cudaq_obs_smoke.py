import importlib

import numpy as np
import pytest


def test_cudaq_sampler_emits_obs_smoke():
    try:
        sampler_mod = importlib.import_module("mghd.samplers.cudaq_sampler")
    except Exception:
        pytest.skip("cudaq sampler module unavailable")

    CudaQSampler = getattr(sampler_mod, "CudaQSampler", None)
    if CudaQSampler is None:
        pytest.skip("CudaQSampler not defined")

    codes_registry = importlib.import_module("mghd.codes.registry")
    code = codes_registry.get_code("surface", distance=3)

    sampler = CudaQSampler()
    batch = sampler.sample(code, n_shots=8, seed=1)
    assert batch.obs is not None
    assert batch.obs.shape[0] == batch.dets.shape[0]
    assert batch.obs.shape[1] >= 1
    assert batch.obs.dtype == np.uint8
