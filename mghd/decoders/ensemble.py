# NOTE: No CUDA/CUDA-Q initialization at import.
from __future__ import annotations
from typing import NamedTuple, Dict, Any
import numpy as np

class TeacherOut(NamedTuple):
    bits: np.ndarray     # uint8 [n_qubits_local]
    weight: int
    teacher: str         # 'mwpf' or 'mwpm'
    valid: bool
    matched_local_ml: bool

def _check_parity_coset_valid(correction_bits: np.ndarray, synd_bits: np.ndarray, H_sub: np.ndarray) -> bool:
    # minimal parity check: H * correction % 2 == synd (locally)
    lhs = (H_sub @ (correction_bits & 1)) & 1
    rhs = synd_bits & 1
    return np.array_equal(lhs % 2, rhs % 2)

def get_teacher_label(
    *,
    H_sub: np.ndarray,
    synd_bits: np.ndarray,
    side: str,
    mwpf_ctx,
    mwpm_ctx,
    local_ml_bits: np.ndarray | None = None,
    dem_meta=None,
) -> TeacherOut:
    """
    - Run MWPF (primary) using DEM/circuit metadata within mwpf_ctx.
    - Run MWPM (fallback) using PyMatching via mwpm_ctx.
    - Enforce parity/coset validity. If both valid, choose lower-weight.
    - Return bits (uint8), weight, teacher tag, valid flag, and matched_local_ml flag.
    Expected ctx interfaces:
      mwpf_ctx.decode(H_sub, synd_bits, side, dem_meta=dem_meta) -> (bits_uint8, weight_int)
      mwpm_ctx.decode(H_sub, synd_bits, side) -> (bits_uint8, weight_int)
    """
    # MWPF (primary)
    bits_pf, w_pf = mwpf_ctx.decode(H_sub, synd_bits, side, dem_meta=dem_meta)
    valid_pf = _check_parity_coset_valid(bits_pf, synd_bits, H_sub)
    
    # MWPM (fallback)
    bits_pm, w_pm = mwpm_ctx.decode(H_sub, synd_bits, side)
    valid_pm = _check_parity_coset_valid(bits_pm, synd_bits, H_sub)

    # Choose
    if valid_pf and valid_pm:
        pick_pf = (w_pf <= w_pm)
        bits, w, tname = (bits_pf, w_pf, "mwpf") if pick_pf else (bits_pm, w_pm, "mwpm")
        valid = True
    elif valid_pf:
        bits, w, tname, valid = bits_pf, w_pf, "mwpf", True
    elif valid_pm:
        bits, w, tname, valid = bits_pm, w_pm, "mwpm", True
    else:
        # both invalid: fall back to MWPF output but mark invalid
        bits, w, tname, valid = bits_pf, w_pf, "mwpf", False

    matched = False
    if local_ml_bits is not None and local_ml_bits.size == bits.size:
        matched = np.array_equal(local_ml_bits & 1, bits & 1)
    return TeacherOut(bits=bits.astype(np.uint8), weight=int(w), teacher=tname, valid=valid, matched_local_ml=matched)

__all__ = ["TeacherOut","get_teacher_label"]