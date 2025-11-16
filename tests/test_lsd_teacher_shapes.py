import numpy as np
import pytest

from mghd.decoders.lsd_teacher import LSDTeacher


def test_lsd_teacher_basic_and_shape_errors():
    Hx = np.array([[1, 0, 1]], dtype=np.uint8)
    Hz = np.array([[0, 1, 1]], dtype=np.uint8)
    t = LSDTeacher(Hx, Hz)
    sx = np.zeros((2, Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((2, Hz.shape[0]), dtype=np.uint8)
    # Basic decode without overrides
    ex, ez = t.decode_batch_xz(sx, sz)
    assert ex.shape == (2, Hx.shape[1]) and ez.shape == (2, Hz.shape[1])

    # llr_overrides wrong length should raise
    with pytest.raises(ValueError):
        t.decode_batch_xz(sx, sz, llr_overrides=np.zeros(Hx.shape[1] - 1))
    # erase_mask wrong length should raise
    with pytest.raises(ValueError):
        t.decode_batch_xz(sx, sz, erase_mask=np.zeros(Hx.shape[1] - 1, dtype=np.uint8))
