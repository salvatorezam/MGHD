import numpy as np

from mghd.decoders.mwpm_fallback import MWPMFallback


class FakeCode:
    def __init__(self, Hx, Hz):
        self.Hx = Hx
        self.Hz = Hz


def test_mwpm_fallback_gf2_decode_shapes():
    Hx = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    Hz = np.array([[1, 1, 0]], dtype=np.uint8)
    code = FakeCode(Hx, Hz)
    mwpm_x = MWPMFallback(code, basis="x", require_graphlike=False)
    synd = np.zeros((2, Hx.shape[0]), dtype=np.uint8)
    cx = mwpm_x.decode_batch(synd)
    assert cx.shape == (2, Hx.shape[1]) and cx.dtype == np.uint8
