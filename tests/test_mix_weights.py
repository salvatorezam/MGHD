import numpy as np

from mghd.decoders.mix import _resolve_mwpm_weights


def test_resolve_mwpm_weights_dict_and_array():
    # Dict with separate x/z arrays
    w = {"x": [1.0, 2.0], "z": [3.0, 4.0]}
    wx, wz = _resolve_mwpm_weights(w)
    assert isinstance(wx, np.ndarray) and isinstance(wz, np.ndarray)
    # Common fallback
    w2 = {"common": [0.5, 0.5, 0.5]}
    wx2, wz2 = _resolve_mwpm_weights(w2)
    assert np.allclose(wx2, wz2)
    # Scalar/array input returns same for both
    arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    wx3, wz3 = _resolve_mwpm_weights(arr)
    assert np.allclose(wx3, wz3)
    # None returns (None, None)
    assert _resolve_mwpm_weights(None) == (None, None)

