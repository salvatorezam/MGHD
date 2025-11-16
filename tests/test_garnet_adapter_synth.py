import os
import numpy as np

from mghd.qpu.adapters.garnet_adapter import sample_round


def test_garnet_adapter_synthetic_shapes(monkeypatch):
    monkeypatch.setenv("MGHD_SYNTHETIC", "1")
    out = sample_round(d=3, p=0.01, seed=0)
    Hx = out["Hx"]
    Hz = out["Hz"]
    synZ = out["synZ"]
    synX = out["synX"]
    coords_q = out["coords_q"]
    coords_c = out["coords_c"]
    n_data = 3 * 3
    n_check = (n_data - 1) // 2
    assert Hx.shape == (n_check, n_data)
    assert Hz.shape == (n_check, n_data)
    assert synZ.shape[0] == n_check
    assert synX.shape[0] == n_check
    assert coords_q.shape[0] == n_data
    # coords_c may include boundary checks (full lattice), so allow >= expected
    assert coords_c.shape[0] >= 2 * n_check
    assert coords_c.dtype == np.float32
