from __future__ import annotations

import numpy as np
import pytest

from mghd.codes.registry import get_code
from mghd.decoders.lsd_teacher import LSDTeacher


@pytest.fixture(scope="module")
def surface_teacher():
    code = get_code("surface", distance=3)
    teacher = LSDTeacher(code.Hx, code.Hz)
    return code, teacher


def test_erasure_path_neutralizes_probabilities(surface_teacher, monkeypatch):
    code, teacher = surface_teacher

    captures: list[np.ndarray | None] = []

    def fake_ml_parity(H, syndrome, probs):
        if probs is not None:
            captures.append(np.asarray(probs, dtype=np.float64))
        else:
            captures.append(None)
        return np.zeros(H.shape[1], dtype=np.uint8)

    monkeypatch.setattr("mghd.decoders.lsd.clustered.ml_parity_project", fake_ml_parity)
    monkeypatch.setattr("mghd.decoders.lsd_teacher.ml_parity_project", fake_ml_parity)

    llr = np.linspace(-0.5, 0.5, code.Hx.shape[1], dtype=np.float64)
    mask = np.zeros(code.Hx.shape[1], dtype=bool)
    mask[1] = True
    mask[4] = True

    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)

    teacher.decode_batch_xz(
        syndromes_x=sx,
        syndromes_z=sz,
        llr_overrides=llr,
        erase_mask=mask,
    )

    probs_x, probs_z = captures[0], captures[1]
    assert probs_x is not None and probs_z is not None

    expected = 1.0 / (1.0 + np.exp(llr))
    np.testing.assert_allclose(probs_x[~mask], expected[~mask], atol=1e-8)
    np.testing.assert_allclose(probs_z[~mask], expected[~mask], atol=1e-8)
    np.testing.assert_allclose(probs_x[mask], 0.5, atol=1e-8)
    np.testing.assert_allclose(probs_z[mask], 0.5, atol=1e-8)


def test_erasure_mask_remains_neutral_under_llr_flip(surface_teacher, monkeypatch):
    code, teacher = surface_teacher

    captures: list[np.ndarray | None] = []

    def fake_ml_parity(_H, _syndrome, probs):
        if probs is not None:
            captures.append(np.asarray(probs, dtype=np.float64))
        else:
            captures.append(None)
        return np.zeros(_H.shape[1], dtype=np.uint8)

    monkeypatch.setattr("mghd.decoders.lsd.clustered.ml_parity_project", fake_ml_parity)
    monkeypatch.setattr("mghd.decoders.lsd_teacher.ml_parity_project", fake_ml_parity)

    mask = np.zeros(code.Hx.shape[1], dtype=bool)
    mask[2] = True

    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)

    llr_a = np.linspace(-0.3, 0.7, code.Hx.shape[1], dtype=np.float64)
    llr_b = -3.0 * llr_a

    teacher.decode_batch_xz(sx, sz, llr_overrides=llr_a, erase_mask=mask)
    teacher.decode_batch_xz(sx, sz, llr_overrides=llr_b, erase_mask=mask)

    # captures order: X1, Z1, X2, Z2 (each may be None if probs absent)
    x_first, _, x_second, _ = captures
    assert x_first is not None and x_second is not None
    np.testing.assert_allclose(x_first[mask], 0.5, atol=1e-8)
    np.testing.assert_allclose(x_second[mask], 0.5, atol=1e-8)
