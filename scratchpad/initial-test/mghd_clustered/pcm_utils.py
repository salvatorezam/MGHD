from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def random_regular_pcm(
    m_checks: int,
    n_bits: int,
    col_w: int = 3,
    row_w: int | None = None,
    seed: int = 1,
):
    rng = np.random.default_rng(seed)
    if row_w is None:
        row_w = max(1, int(round(col_w * n_bits / m_checks)))
    rows: list[int] = []
    cols: list[int] = []
    for j in range(n_bits):
        choices = rng.choice(m_checks, size=col_w, replace=False)
        rows.extend(choices.tolist())
        cols.extend([j] * col_w)
    data = np.ones(len(rows), dtype=np.uint8)
    H = sp.coo_matrix((data, (rows, cols)), shape=(m_checks, n_bits)).tocsr()
    H.data[:] = 1
    H.sum_duplicates()
    H.eliminate_zeros()
    return H


def sample_error_and_syndrome(H, p: float, seed: int | None = None):
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    e = (rng.random(n) < p).astype(np.uint8)
    s = (H @ e) % 2
    return e, s.astype(np.uint8)
