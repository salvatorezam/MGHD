import numpy as np
import pytest

from teachers.mwpm_fallback import MWPMFallback, _is_graphlike


def test_is_graphlike_detection():
    H = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    assert not _is_graphlike(H)


def test_mwpm_raises_when_not_graphlike(monkeypatch):
    H = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    fallback = MWPMFallback(H, require_graphlike=True)
    with pytest.raises(ValueError, match="mwpm_not_graphlike"):
        fallback.decode_batch(np.zeros((1, H.shape[0]), dtype=np.uint8))
