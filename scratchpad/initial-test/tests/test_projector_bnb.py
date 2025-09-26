import numpy as np
import scipy.sparse as sp

from mghd_clustered.cluster_core import (
    gf2_nullspace,
    gf2_solve_particular,
    ml_parity_project,
)


def _bruteforce_ml(H_sub: sp.csr_matrix, s_sub: np.ndarray, p_flip: np.ndarray):
    w = np.log((1 - p_flip) / p_flip)
    e0 = gf2_solve_particular(H_sub, s_sub)
    N = gf2_nullspace(H_sub)
    r = N.shape[1]
    best_e = e0.copy()
    best_cost = float(np.dot(w, e0))
    if r == 0:
        return best_e, best_cost
    for mask in range(1 << r):
        e = e0.copy()
        for j in range(r):
            if (mask >> j) & 1:
                e ^= N[:, j]
        cost = float(np.dot(w, e))
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_e = e.copy()
    return best_e, best_cost


def _random_instance(seed: int):
    rng = np.random.default_rng(seed)
    n = rng.integers(4, 9)
    m = rng.integers(2, n)
    H_dense = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
    e_true = rng.integers(0, 2, size=n, dtype=np.uint8)
    s = (H_dense @ e_true) % 2
    H = sp.csr_matrix(H_dense)
    return H, s.astype(np.uint8)


def test_branch_and_bound_matches_bruteforce():
    for seed in range(12):
        H_sub, s_sub = _random_instance(seed)
        n = H_sub.shape[1]
        p_flip = np.clip(np.random.default_rng(seed + 100).uniform(0.05, 0.3, size=n), 1e-3, 0.49)
        stats = {}
        e_bnb = ml_parity_project(H_sub, s_sub, p_flip, r_cap=12, stats_out=stats)
        e_ref, cost_ref = _bruteforce_ml(H_sub, s_sub, p_flip)

        # Parity check satisfied
        parity = (H_sub @ e_bnb) % 2
        assert np.array_equal(parity.astype(np.uint8) % 2, s_sub % 2)

        cost_bnb = float(np.dot(np.log((1 - p_flip) / p_flip), e_bnb))
        assert abs(cost_bnb - cost_ref) < 1e-9

        r = gf2_nullspace(H_sub).shape[1]
        if r > 0:
            total_states = 1 << r
            assert stats["states_visited"] <= total_states
        if r >= 3:
            assert stats["states_pruned"] > 0
