from __future__ import annotations

import numpy as np
import pytest

pm = pytest.importorskip("pymatching", reason="PyMatching not installed")

from mghd.decoders.mwpm_fallback import MWPMFallback, _coerce_weights_to_float


class _DummyCode:
    def __init__(self, H: np.ndarray):
        self.Hx = H
        self.Hz = H


def _toy_hx() -> np.ndarray:
    # Simple 2-check, 4-data matrix that is graphlike and offers ambiguity.
    return np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def test_column_weights_bias_matching_choices():
    code = _DummyCode(_toy_hx())
    decoder = MWPMFallback(code, basis="x", require_graphlike=False)
    det = np.array([[1, 1]], dtype=np.uint8)

    weights_a = np.array([0.9, 0.1, 0.5, 0.3], dtype=np.float32)
    weights_b = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)

    corr_a = decoder.decode_batch(det, column_weights=weights_a)[0]
    corr_b = decoder.decode_batch(det, column_weights=weights_b)[0]

    assert corr_a.shape == corr_b.shape == (code.Hx.shape[1],)
    assert not np.array_equal(corr_a, corr_b), "Column weights should influence MWPM output"


def test_coerce_weights_to_float_handles_iterables():
    entries = ["0.25", 0.5, 1, 0.75]
    result = _coerce_weights_to_float(entries, len(entries))
    assert result is not None
    assert result.shape == (len(entries),)
    assert result.dtype == np.float32
    assert np.allclose(result, [0.25, 0.5, 1.0, 0.75])
