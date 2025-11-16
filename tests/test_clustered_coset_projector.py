import numpy as np
import scipy.sparse as sp

from mghd.decoders.lsd.clustered import (
    gf2_solve_particular,
    gf2_nullspace_basis,
    gf2_project_to_coset,
)


def test_gf2_project_to_coset_matches_hint_when_possible():
    # Small full-rank-ish example with a non-trivial nullspace
    # H is 2x3, rank 2 -> nullity 1
    H = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    # Choose an arbitrary e_true and compute syndrome s
    e_true = np.array([1, 0, 1], dtype=np.uint8)
    s = (H @ e_true) % 2

    # Compute a particular solution e0 and a nullspace basis N
    e0 = gf2_solve_particular(H, s)
    N = gf2_nullspace_basis(H)
    assert N.shape == (3, 1)
    nvec = N[:, 0]

    # Pick a target hint in the same coset: e_hint = e0 ⊕ nvec
    e_hint = (e0 ^ nvec) & 1

    # Project to coset using the hint; should return e_hint (or another valid coset rep)
    e_proj = gf2_project_to_coset(H, s, e_hint=e_hint)

    # Check parity and equality with the hint (unique for this 1D nullspace)
    assert np.array_equal(((H @ e_proj) % 2).astype(np.uint8), s.astype(np.uint8))
    assert np.array_equal(e_proj, e_hint)
