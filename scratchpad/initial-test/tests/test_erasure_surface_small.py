from __future__ import annotations

import numpy as np

from mghd.codes.registry import get_code
from mghd.decoders.erasure_surface_ml import ErasureSurfaceMLTeacher


def test_erasure_surface_ml_matches_syndrome():
    rng = np.random.default_rng(42)
    for d in (3, 5):
        code = get_code("surface", distance=d)
        teacher = ErasureSurfaceMLTeacher(code)
        n = code.Hx.shape[1]
        mx = code.Hx.shape[0]
        mz = code.Hz.shape[0]
        for _ in range(5):
            mask = (rng.random(n) < 0.4).astype(np.uint8)
            if not mask.any():
                mask[rng.integers(n)] = 1
            ex_true = np.zeros(n, dtype=np.uint8)
            ez_true = np.zeros(n, dtype=np.uint8)
            erased_indices = np.flatnonzero(mask)
            ex_true[erased_indices] = rng.integers(0, 2, size=erased_indices.size)
            ez_true[erased_indices] = rng.integers(0, 2, size=erased_indices.size)
            sx = (code.Hx @ ex_true) % 2
            sz = (code.Hz @ ez_true) % 2
            out = teacher.decode_batch(
                syndromes_x=sx[np.newaxis, :],
                syndromes_z=sz[np.newaxis, :],
                erase_data_mask=mask[np.newaxis, :],
                erase_det_mask=np.zeros((1, mx + mz), dtype=np.uint8),
            )
            ex = out["ex"][0]
            ez = out["ez"][0]
            assert np.all(ex[mask == 0] == 0)
            assert np.all(ez[mask == 0] == 0)
            assert np.array_equal((code.Hx @ ex) % 2, sx)
            assert np.array_equal((code.Hz @ ez) % 2, sz)


def test_erasure_surface_zero_when_no_erasures():
    code = get_code("surface", distance=3)
    teacher = ErasureSurfaceMLTeacher(code)
    sx = np.zeros((1, code.Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, code.Hz.shape[0]), dtype=np.uint8)
    mask = np.zeros((1, code.Hx.shape[1]), dtype=np.uint8)
    out = teacher.decode_batch(sx, sz, mask)
    assert not out["ex"].any()
    assert not out["ez"].any()
