import numpy as np
from mghd.utils.metrics import logical_error_rate


def test_ler_math():
    true = np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    pred = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.uint8)
    res = logical_error_rate(true, pred)
    expected = np.array([1 / 3, 1 / 3, 0.0])
    assert np.allclose(res.ler_per_logical, expected)
    assert abs(res.ler_mean - (2 / 9)) < 1e-12
