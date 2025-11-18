import numpy as np

from mghd.samplers.stim_sampler import StimSampler


class FakeSampler:
    def sample(self, *, shots=0, separate_observables=True):
        # 5 detectors, 2 observables
        return np.zeros((shots, 5), dtype=np.uint8), np.zeros((shots, 2), dtype=np.uint8)


class FakeCircuit:
    def compile_detector_sampler(self):
        return FakeSampler()


class FakeCode:
    name = "misc_code"
    stim_circuit = FakeCircuit()
    distance = None


def test_stim_sampler_uses_stim_circuit_path():
    s = StimSampler(rounds=1, dep=0.0, emit_obs=True)
    batch = s.sample(FakeCode(), n_shots=3, seed=0)
    assert batch.dets.shape == (3, 5)
    assert batch.obs.shape == (3, 2)
