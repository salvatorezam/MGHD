import numpy as np

from mghd.samplers.stim_sampler import sample_surface_memory


def test_stim_surface_memory_shapes():
    d = 3
    dets, obs = sample_surface_memory(d=d, rounds=1, shots=4, dep=0.001)
    assert dets.shape[0] == 4
    assert obs.shape[0] == 4
    assert dets.dtype == np.uint8 and obs.dtype == np.uint8
