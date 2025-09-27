"""Python-side κ/ν proxy using bipartite graph connectivity."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import deque


def kappa_nu_proxy(H: sp.csr_matrix, syndrome: np.ndarray) -> tuple[int, int]:
    """Estimate LSD cluster stats (largest component κ, cluster count ν)."""

    s = np.asarray(syndrome, dtype=bool).ravel()
    active = np.flatnonzero(s)
    if active.size == 0:
        return 0, 0

    H_active = H[active, :]
    A = (H_active.T @ H_active).astype(bool)
    A.setdiag(False)
    A.eliminate_zeros()

    touched = np.flatnonzero((H_active != 0).sum(axis=0).A.ravel() > 0)
    seen = np.zeros(A.shape[0], dtype=bool)
    kappa = 0
    nu = 0

    for q in touched:
        if seen[q]:
            continue
        nu += 1
        size = 0
        dq = deque([q])
        seen[q] = True
        while dq:
            u = dq.popleft()
            size += 1
            lo, hi = A.indptr[u], A.indptr[u + 1]
            for v in A.indices[lo:hi]:
                if not seen[v]:
                    seen[v] = True
                    dq.append(v)
        kappa = max(kappa, size)

    return int(kappa), int(nu)
