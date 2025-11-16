"""Regression coverage for MWPM fallback weight coercion and decode shim."""

from __future__ import annotations

import types

import numpy as np


def test_weights_fraction_like_and_decode_shim(monkeypatch):
    from mghd.decoders import mwpm_fallback as module

    class StubMatching:
        def __init__(self, H: np.ndarray, weights):
            self._ncols = H.shape[1]
            self.weights = weights

        @classmethod
        def from_check_matrix(cls, H, weights=None):
            return cls(np.asarray(H, dtype=np.uint8), weights)

        def decode(self, syndrome):
            # Return a simple pattern that depends on the input to sanity-check stacking.
            syndrome = np.asarray(syndrome, dtype=np.uint8)
            parity = int(np.sum(syndrome) % 2)
            return np.full(self._ncols, parity, dtype=np.uint8)

    stub_pm = types.SimpleNamespace(Matching=StubMatching)
    monkeypatch.setattr(module, "pm", stub_pm, raising=False)
    monkeypatch.setattr(module, "_HAVE_PM", True, raising=False)

    class Rat:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def __float__(self):
            return self.a / self.b

    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    weights = [Rat(1, 2), Rat(3, 4), Rat(5, 6)]

    code = types.SimpleNamespace(Hx=H, Hz=H)
    decoder = module.MWPMFallback(code, basis="x", weights=weights)

    assert decoder.weights is not None
    assert decoder.weights.dtype == np.float32
    np.testing.assert_allclose(decoder.weights, [0.5, 0.75, 0.8333333], rtol=1e-6)

    syndromes = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    out = decoder.decode_batch(syndromes)

    assert out.shape == (2, H.shape[1])
    # First shot has odd parity -> ones; second even -> zeros.
    np.testing.assert_array_equal(out[0], np.ones(H.shape[1], dtype=np.uint8))
    np.testing.assert_array_equal(out[1], np.zeros(H.shape[1], dtype=np.uint8))
