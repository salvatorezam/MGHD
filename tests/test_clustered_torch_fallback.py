import numpy as np

from mghd.decoders.lsd.clustered import ml_parity_project, ml_parity_project_torch


def test_ml_parity_project_torch_fallback_matches_numpy():
    # Small subproblem with 1-dim nullspace; set r_cap=0 to force fallback
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    p = np.full(H.shape[1], 0.5, dtype=np.float64)
    e_np = ml_parity_project(H, s, p)
    e_tf = ml_parity_project_torch(H, s, p, r_cap=0)
    assert np.array_equal(e_np, e_tf)

