import numpy as np
import pytest

from mghd.codes.registry import get_code
from mghd.decoders.erasure_surface_ml import ErasureSurfaceMLTeacher


def test_erasure_surface_ml_teacher_errors_and_basic():
    code = get_code("surface", distance=3)
    t = ErasureSurfaceMLTeacher(code)
    B = 2
    sx = np.zeros((B, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((B, code.Hz.shape[0]), dtype=np.uint8)
    mask = np.zeros((B, code.Hx.shape[1]), dtype=np.uint8)

    # Missing mask -> error
    with pytest.raises(ValueError):
        t.decode_batch(sx, sz, None)

    # Wrong batch size
    with pytest.raises(ValueError):
        t.decode_batch(sx, sz[:1], mask)

    # Wrong mask length
    with pytest.raises(ValueError):
        t.decode_batch(sx, sz, np.zeros((B, code.Hx.shape[1] - 1), dtype=np.uint8))

    # Wrong det mask shape
    with pytest.raises(ValueError):
        t.decode_batch(sx, sz, mask, erase_det_mask=np.zeros((B, 1), dtype=np.uint8))

    # Basic valid case: all-zero erasures returns zeros
    out = t.decode_batch(sx, sz, mask)
    assert out["ex"].shape == (B, code.Hx.shape[1])
    assert out["ez"].shape == (B, code.Hz.shape[1])
