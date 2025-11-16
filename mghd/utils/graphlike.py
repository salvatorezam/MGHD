from __future__ import annotations

import numpy as np
from scipy import sparse


def is_graphlike(H) -> bool:
    """True iff every column has <=2 ones."""
    if sparse.issparse(H):
        col_w = np.asarray(H.astype(np.int8).sum(axis=0)).ravel()
    else:
        col_w = (H != 0).astype(np.int8).sum(axis=0)
    return bool((col_w <= 2).all())
