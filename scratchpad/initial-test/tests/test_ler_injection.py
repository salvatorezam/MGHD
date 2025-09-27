from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

import tools.bench_clustered_sweep_surface as bench
from mghd_clustered.pcm_real import rotated_surface_pcm
from mghd_public.infer import MGHDDecoderPublic
from mghd_public.model_v2 import MGHDv2


@pytest.fixture
def tmp_ckpt(tmp_path):
    import torch

    model = MGHDv2()
    ckpt_path = tmp_path / "mghd_v2_dummy.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture
def decoder(tmp_ckpt):
    return MGHDDecoderPublic(str(tmp_ckpt), device="cpu")


def fake_sample_round(d: int, p: float, seed: int):
    Hx = rotated_surface_pcm(d, "X")
    Hz = rotated_surface_pcm(d, "Z")
    synZ = np.zeros(Hz.shape[0], dtype=np.uint8)
    synX = np.zeros(Hx.shape[0], dtype=np.uint8)
    return {"Hx": Hx, "Hz": Hz, "synZ": synZ, "synX": synX}


@pytest.mark.parametrize("inject_rate, expected_failures", [(0.0, 0), (1.0, 10)])
def test_inject_ler_changes_failure_count(monkeypatch, decoder, inject_rate, expected_failures):
    monkeypatch.setattr(bench, "sample_round", fake_sample_round)

    d = 3
    Hx = rotated_surface_pcm(d, "X")
    Hz = rotated_surface_pcm(d, "Z")
    decoder.bind_code(Hx, Hz)

    args = SimpleNamespace(
        shots=5,
        halo=1,
        thresh=0.5,
        temp=1.0,
        r_cap=10,
        tier0=True,
        tier0_mode="mixed",
        tier0_k_max=None,
        tier0_r_max=None,
        enforce_mghd=False,
        inject_ler_rate=inject_rate,
    )
    rng = np.random.default_rng(0)
    stats = bench.run_distance(d, 0.01, decoder, Hx, Hz, rng, args)
    assert stats["shots"] == 10
    assert stats["failures"] == expected_failures
