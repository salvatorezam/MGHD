#!/usr/bin/env python3
"""Deep analysis: why per-component exact ML is much worse than MWPM.

Key question: is it (A) inconsistent systems from external checks, or
(B) ambiguous solutions when components are near the boundary?
"""

import numpy as np
import scipy.sparse as sp
import pymatching
from mghd.codes.registry import get_code
from mghd.decoders.lsd.clustered import (
    active_components, extract_subproblem, gf2_solve_particular, gf2_nullspace
)

d = 9; p = 0.03; n_shots = 5000
rng = np.random.default_rng(42)
code = get_code("surface", d)
Hx = sp.csr_matrix(code.Hx % 2)
Hz = sp.csr_matrix(code.Hz % 2)
Lx = np.asarray(code.Lx % 2, dtype=np.uint8)
Lz = np.asarray(code.Lz % 2, dtype=np.uint8)
n_q = d * d

mwpm_z = pymatching.Matching(Hx.toarray().astype(np.uint8))
mwpm_x = pymatching.Matching(Hz.toarray().astype(np.uint8))


def solve_component_ml(H_sub, s_sub, p_ch):
    """Channel ML for one component. Returns (correction, status)."""
    try:
        e0 = gf2_solve_particular(H_sub, s_sub)
    except ValueError:
        return np.zeros(H_sub.shape[1], dtype=np.uint8), "inconsistent"

    N = gf2_nullspace(H_sub)
    r = N.shape[1]
    if r == 0:
        return e0.astype(np.uint8), "unique"

    w_val = np.log((1 - p_ch) / p_ch)
    w = np.full(H_sub.shape[1], w_val, dtype=np.float64)
    best_e = e0.copy()
    best_cost = float(np.dot(w, best_e))
    for z_int in range(1, 1 << r):
        e = e0.copy()
        bits = z_int; k = 0
        while bits:
            if bits & 1: e ^= N[:, k]
            bits >>= 1; k += 1
        cost = float(np.dot(w, e))
        if cost < best_cost:
            best_cost = cost; best_e = e
    return best_e.astype(np.uint8), f"r={r}"


def solve_internal_only(H, s, checks, qubits, p_ch):
    """Solve using only checks fully internal to the component."""
    qset = set(qubits.tolist())
    Hc = H.tocsr()
    keep = []
    for i, c in enumerate(checks):
        lo, hi = Hc.indptr[c], Hc.indptr[c + 1]
        support = set(Hc.indices[lo:hi].tolist())
        if support.issubset(qset):
            keep.append(i)
    if not keep:
        return np.zeros(len(qubits), dtype=np.uint8), "no_internal_checks"
    keep = np.array(keep, dtype=np.int64)
    H_sub = H[checks[keep], :][:, qubits].tocsr()
    s_sub = np.asarray(s, dtype=np.uint8).ravel()[checks[keep]]
    return solve_component_ml(H_sub, s_sub, p_ch)


# Decode one side
def decode_side(H, s, p_ch, use_fallback=False):
    """Returns (correction, stats_dict)."""
    result = active_components(H, s, halo=0)
    if isinstance(result, tuple) and len(result) == 2:
        check_c, qubit_c = result
    else:
        return np.zeros(H.shape[1], dtype=np.uint8), {"n_comp": 0}

    e = np.zeros(H.shape[1], dtype=np.uint8)
    n_inconsistent = 0
    n_ambiguous = 0
    n_unique = 0

    for checks, qubits in zip(check_c, qubit_c):
        H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(H, s, checks, qubits)
        e_sub, status = solve_component_ml(H_sub, s_sub, p_ch)

        if status == "inconsistent" and use_fallback:
            # Retry with internal checks only
            e_sub, status2 = solve_internal_only(H, s, checks, qubits, p_ch)
            status = f"fallback({status2})"

        if "inconsistent" in status:
            n_inconsistent += 1
        elif status == "unique":
            n_unique += 1
        else:
            n_ambiguous += 1

        e[q_l2g] ^= e_sub

    return e, {"n_comp": len(check_c), "inconsistent": n_inconsistent,
               "ambiguous": n_ambiguous, "unique": n_unique}


# Run 3 strategies:
# 1. Original (all checks, no fallback) — gets inconsistencies
# 2. With fallback (all checks, retry internal on inconsistency)
# 3. MWPM
stats = {"original": {"fail": 0}, "fallback": {"fail": 0}, "mwpm": {"fail": 0}}
orig_inconsistent_total = 0
fallback_inconsistent_total = 0

for shot_idx in range(n_shots):
    ex_true = (rng.random(n_q) < p).astype(np.uint8)
    ez_true = (rng.random(n_q) < p).astype(np.uint8)
    sx = np.asarray((Hx @ ez_true) % 2, dtype=np.uint8).ravel()
    sz = np.asarray((Hz @ ex_true) % 2, dtype=np.uint8).ravel()
    obs_x_true = (ex_true @ Lz.T) % 2
    obs_z_true = (ez_true @ Lx.T) % 2

    # MWPM
    ez_mwpm = mwpm_z.decode(sx)
    ex_mwpm = mwpm_x.decode(sz)
    if not (np.array_equal(obs_x_true, (ex_mwpm @ Lz.T) % 2) and
            np.array_equal(obs_z_true, (ez_mwpm @ Lx.T) % 2)):
        stats["mwpm"]["fail"] += 1

    # Original (no fallback)
    ez_orig, st_z = decode_side(Hx, sx, p, use_fallback=False)
    ex_orig, st_x = decode_side(Hz, sz, p, use_fallback=False)
    orig_inconsistent_total += st_z.get("inconsistent", 0) + st_x.get("inconsistent", 0)
    if not (np.array_equal(obs_x_true, (ex_orig @ Lz.T) % 2) and
            np.array_equal(obs_z_true, (ez_orig @ Lx.T) % 2)):
        stats["original"]["fail"] += 1

    # Fallback strategy
    ez_fb, st_z2 = decode_side(Hx, sx, p, use_fallback=True)
    ex_fb, st_x2 = decode_side(Hz, sz, p, use_fallback=True)
    fallback_inconsistent_total += st_z2.get("inconsistent", 0) + st_x2.get("inconsistent", 0)
    if not (np.array_equal(obs_x_true, (ex_fb @ Lz.T) % 2) and
            np.array_equal(obs_z_true, (ez_fb @ Lx.T) % 2)):
        stats["fallback"]["fail"] += 1

print(f"d={d}  p={p}  {n_shots} shots")
print(f"  Original LER:  {100*stats['original']['fail']/n_shots:.2f}%  (inconsistent components: {orig_inconsistent_total})")
print(f"  Fallback LER:  {100*stats['fallback']['fail']/n_shots:.2f}%  (remaining inconsistent: {fallback_inconsistent_total})")
print(f"  MWPM LER:      {100*stats['mwpm']['fail']/n_shots:.2f}%")
