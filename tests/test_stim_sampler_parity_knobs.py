import numpy as np

from mghd.samplers.stim_sampler import StimSampler
from mghd.samplers import SampleBatch


class _FakeSurface:
    name = "surface_d3"
    distance = 3
    # Provide Hx to define number of data qubits for erasure mask injection
    Hx = np.zeros((1, 9), dtype=np.uint8)


def test_stim_sampler_emit_obs_and_erasure_injection_smoke():
    code = _FakeSurface()

    # With emit_obs=False and inject_erasure_frac>0, obs should be empty and erasure fields present
    samp = StimSampler(rounds=1, dep=0.0, emit_obs=False, inject_erasure_frac=0.5)
    batch: SampleBatch = samp.sample(code, n_shots=4, seed=123)

    assert isinstance(batch, SampleBatch)
    assert batch.obs.shape == (4, 0)
    assert batch.dets.shape[0] == 4
    assert batch.erase_data_mask is not None
    assert batch.p_erase_data is not None
    assert batch.erase_data_mask.shape == (4, 9)
    assert batch.p_erase_data.shape == (4, 9)
    # Zero-cost when not used
    samp2 = StimSampler(rounds=1, dep=0.0, emit_obs=True, inject_erasure_frac=0.0)
    batch2 = samp2.sample(code, n_shots=2, seed=7)
    assert batch2.obs.shape[0] == 2 and batch2.obs.shape[1] >= 0
    assert batch2.erase_data_mask is None
    assert batch2.p_erase_data is None
