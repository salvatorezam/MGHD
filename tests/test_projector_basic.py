import numpy as np
import scipy.sparse as sp

from mghd.decoders.lsd.cluster_core import ml_parity_project, gf2_nullspace


def test_ml_projector_satisfies_parity():
    H_sub = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    s_sub = np.array([1, 0], dtype=np.uint8)
    probs = np.array([0.2, 0.3, 0.1], dtype=np.float32)

    solution = ml_parity_project(H_sub, s_sub, probs, r_cap=4)
    assert solution.shape == (H_sub.shape[1],)

    parity = (H_sub @ (solution % 2)) % 2
    parity = np.asarray(parity).ravel() % 2
    assert np.array_equal(parity.astype(np.uint8), s_sub)

    # Nullspace rank matches expectation
    nullity = gf2_nullspace(H_sub).shape[1]
    assert nullity == 1
