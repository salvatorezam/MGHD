from __future__ import annotations

import numpy as np

from mghd.core import solve_on_erasure


def _check_solution(H, s, x, mask_cols, mask_rows=None):
    H = H & 1
    s = s & 1
    erased_cols = mask_cols.astype(bool)
    assert np.all(x[~erased_cols] == 0)
    rows_keep = np.ones(H.shape[0], dtype=bool) if mask_rows is None else ~mask_rows.astype(bool)
    lhs = (H[rows_keep] @ (x & 1)) % 2
    rhs = s[rows_keep]
    assert np.array_equal(lhs, rhs)


def test_solve_on_erasure_random_instances():
    rng = np.random.default_rng(123)
    for _ in range(50):
        m, n = 12, 20
        H = rng.integers(0, 2, size=(m, n), dtype=np.uint8)
        mask_cols = rng.random(n) < 0.4
        if not mask_cols.any():
            mask_cols[rng.integers(n)] = True
        x_true = np.zeros(n, dtype=np.uint8)
        erased_idx = np.flatnonzero(mask_cols)
        x_true[erased_idx] = rng.integers(0, 2, size=erased_idx.shape[0])
        s = (H @ x_true) % 2
        mask_rows = (rng.random(m) < 0.2).astype(np.uint8)
        x_hat = solve_on_erasure(H, s, mask_cols.astype(np.uint8), mask_rows=mask_rows)
        _check_solution(H, s, x_hat, mask_cols.astype(np.uint8), mask_rows)


def test_solve_on_erasure_no_rows():
    H = np.zeros((4, 6), dtype=np.uint8)
    s = np.zeros(4, dtype=np.uint8)
    mask_cols = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    mask_rows = np.ones(4, dtype=np.uint8)
    x_hat = solve_on_erasure(H, s, mask_cols, mask_rows=mask_rows)
    assert np.all(x_hat == 0)
