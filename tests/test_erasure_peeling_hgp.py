from __future__ import annotations

import importlib
import numpy as np

from mghd.decoders.erasure_peeling import ErasureQLDPCPeelingTeacher


def _load_small_hgp():
    H1 = np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8)
    H2 = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    cr = importlib.import_module("codes_registry")
    return cr.get_code("hgp", H1=H1, H2=H2, name="hgp_small")


def gf2_mul(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (A.astype(np.uint8) @ x.astype(np.uint8)) % 2


def test_erasure_peeling_hgp_random():
    code = _load_small_hgp()
    Hx, Hz = code.Hx, code.Hz
    teacher = ErasureQLDPCPeelingTeacher(Hx, Hz, max_cluster=64)
    rng = np.random.default_rng(321)
    B = 12
    n = Hx.shape[1]
    erase_mask = (rng.random((B, n)) < 0.15).astype(np.uint8)

    ex_true = np.zeros((B, n), dtype=np.uint8)
    ez_true = np.zeros((B, n), dtype=np.uint8)
    for b in range(B):
        support = np.flatnonzero(erase_mask[b])
        if support.size == 0:
            continue
        k = rng.integers(0, support.size + 1)
        if k:
            chosen = support[:k]
            ex_true[b, chosen] ^= 1
            ez_true[b, chosen] ^= 1
    sx = gf2_mul(Hx, ex_true.T).T
    sz = gf2_mul(Hz, ez_true.T).T
    out = teacher.decode_batch(sx, sz, erase_mask, erase_det_mask=None)
    ex = out["ex"]
    ez = out["ez"]
    assert np.all((ex & ~erase_mask) == 0)
    assert np.all((ez & ~erase_mask) == 0)
    np.testing.assert_array_equal(gf2_mul(Hx, ex.T).T, sx)
    np.testing.assert_array_equal(gf2_mul(Hz, ez.T).T, sz)


def test_erasure_peeling_no_erasures_returns_zero():
    code = _load_small_hgp()
    Hx, Hz = code.Hx, code.Hz
    teacher = ErasureQLDPCPeelingTeacher(Hx, Hz)
    sx = np.zeros((1, Hx.shape[0]), dtype=np.uint8)
    sz = np.zeros((1, Hz.shape[0]), dtype=np.uint8)
    erase_mask = np.zeros((1, Hx.shape[1]), dtype=np.uint8)
    out = teacher.decode_batch(sx, sz, erase_mask, erase_det_mask=None)
    assert not out["ex"].any()
    assert not out["ez"].any()
