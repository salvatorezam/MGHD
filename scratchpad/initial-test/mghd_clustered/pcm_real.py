from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from .pcm_utils import random_regular_pcm  # Re-export convenience
from functools import reduce


def _create_circulant_matrix(n: int, shifts: Tuple[int, ...]) -> sp.csr_matrix:
    mat = sp.dok_matrix((n, n), dtype=np.uint8)
    for i in range(n):
        for shift in shifts:
            mat[(i + shift) % n, i] = 1
    return mat.tocsr()


def _matrix_power(mat: sp.csr_matrix, power: int) -> sp.csr_matrix:
    if power == 0:
        return sp.identity(mat.shape[0], dtype=np.uint8, format="csr")
    result = mat.copy()
    for _ in range(1, power):
        result = result @ mat
    return result


def _bb_code_css(l: int, m: int, A_x_pows, A_y_pows, B_x_pows, B_y_pows) -> Tuple[np.ndarray, np.ndarray]:
    S_l = _create_circulant_matrix(l, (-1,))
    S_m = _create_circulant_matrix(m, (-1,))
    x_shift = sp.kron(S_l, sp.identity(m, dtype=np.uint8, format="csr"), format="csr")
    y_shift = sp.kron(sp.identity(l, dtype=np.uint8, format="csr"), S_m, format="csr")

    def build_terms(shifts, base):
        terms = []
        for p in shifts:
            terms.append(_matrix_power(base, p))
        return terms

    A_terms = build_terms(A_x_pows, x_shift) + build_terms(A_y_pows, y_shift)
    B_terms = build_terms(B_y_pows, y_shift) + build_terms(B_x_pows, x_shift)

    A = sum(A_terms[1:], A_terms[0].copy()) if A_terms else sp.csr_matrix((l * m, l * m), dtype=np.uint8)
    B = sum(B_terms[1:], B_terms[0].copy()) if B_terms else sp.csr_matrix((l * m, l * m), dtype=np.uint8)

    hx = sp.hstack([A, B], format="csr").toarray().astype(np.uint8)
    hz = sp.hstack([B.T, A.T], format="csr").toarray().astype(np.uint8)
    return hx, hz


def _rotated_surface_css(d: int) -> Tuple[np.ndarray, np.ndarray]:
    assert d % 2 == 1, "distance must be odd for rotated surface code"
    n2 = d * d
    m = (n2 - 1) // 2
    hx = np.zeros((m, n2), dtype=np.uint8)
    hz = np.zeros((m, n2), dtype=np.uint8)

    def set_pcm_row(arr, idx, i, j):
        i1, j1 = (i + 1) % d, (j + 1) % d
        arr[idx, i * d + j] = 1
        arr[idx, i1 * d + j1] = 1
        arr[idx, i1 * d + j] = 1
        arr[idx, i * d + j1] = 1

    x_idx = 0
    z_idx = 0
    for i in range(d - 1):
        for j in range(d - 1):
            if (i + j) % 2 == 0:
                set_pcm_row(hz, z_idx, i, j)
                z_idx += 1
            else:
                set_pcm_row(hx, x_idx, i, j)
                x_idx += 1

    for j in range(d - 1):
        if j % 2 == 0:
            hx[x_idx, j] = hx[x_idx, j + 1] = 1
        else:
            hx[x_idx, (d - 1) * d + j] = hx[x_idx, (d - 1) * d + (j + 1)] = 1
        x_idx += 1

    for i in range(d - 1):
        if i % 2 == 0:
            hz[z_idx, i * d + (d - 1)] = hz[z_idx, (i + 1) * d + (d - 1)] = 1
        else:
            hz[z_idx, i * d] = hz[z_idx, (i + 1) * d] = 1
        z_idx += 1

    return hx, hz


def rotated_surface_pcm(d: int, noise_p: float = 0.002, seed: int = 1) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Return (H_X, H_Z) for a distance-d rotated surface code.

    First attempts to use a Stim-generated circuit plus a DEM→CSS converter.
    If Stim conversion is unavailable, falls back to panqec's analytical
    construction, which matches the rotated Tanner graph structure.
    """
    try:
        import stim
        from .stim_to_pcm import dem_to_css_pcm

        circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d)
        dem = circ.detector_error_model(decompose_errors=True)
        Hx, Hz = dem_to_css_pcm(dem)
        return Hx.tocsr(), Hz.tocsr()
    except Exception:
        hx, hz = _rotated_surface_css(d)
        return sp.csr_matrix(hx), sp.csr_matrix(hz)


def bb_144_12_12_pcm(kind: str = "X") -> sp.csr_matrix:
    """Return the [[144,12,12]] bivariate bicycle parity-check matrix.

    Parameters
    ----------
    kind : {"X","Z"}
        Select which CSS component to return.
    """
    hx, hz = _bb_code_css(12, 6, [3], [1, 2], [1, 2], [3])
    H = hx if kind.upper() == "X" else hz
    return sp.csr_matrix(H)


def load_custom_pcm(path: str) -> sp.csr_matrix:
    """Load a CSR matrix from an NPZ archive with (data, indices, indptr, shape)."""
    arr = np.load(Path(path), allow_pickle=False)
    return sp.csr_matrix((arr["data"], arr["indices"], arr["indptr"]), shape=tuple(arr["shape"]))
