import numpy as np
from scipy import sparse

from mghd.utils.graphlike import is_graphlike


def test_is_graphlike_dense_and_sparse():
    # Dense graphlike: at most two ones per column
    H = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    assert is_graphlike(H)
    # Dense non-graphlike: a column with three ones
    # Non-graphlike: first column has three ones
    H_ng = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    assert not is_graphlike(H_ng)
    # Sparse path should behave the same
    Hs = sparse.csr_matrix(H)
    Hs_ng = sparse.csr_matrix(H_ng)
    assert is_graphlike(Hs)
    assert not is_graphlike(Hs_ng)
