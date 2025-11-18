import numpy as np

from mghd.decoders import mwpm_ctx as ctx_mod
from mghd.decoders.mwpm_ctx import MWPMatchingContext


def test_mwpm_ctx_fallback_when_pymatching_missing(monkeypatch):
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    ctx = MWPMatchingContext()
    # Force fallback path regardless of environment
    monkeypatch.setattr(ctx_mod, "pm", None)
    bits, w = ctx.decode(H, s, side="Z")
    assert bits.dtype == np.uint8 and bits.shape[0] == H.shape[1]
    assert w == H.shape[1]
