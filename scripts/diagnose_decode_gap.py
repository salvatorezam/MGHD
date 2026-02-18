#!/usr/bin/env python3
"""Diagnose why per-component ML decode gives much worse LER than global MWPM.

This script:
1. Generates surface code samples at d=9, p=0.03
2. Decomposes into active components
3. Solves each component with exact ML (same as tier-0)
4. Assembles correction
5. Compares against MWPM on the full syndrome
6. For each shot where MGHD fails but MWPM succeeds, prints detailed info
"""

import numpy as np
import scipy.sparse as sp
import pymatching
from mghd.codes.registry import get_code
from mghd.decoders.lsd.clustered import (
    active_components, extract_subproblem, solve_small_cluster_channel_ml, ml_parity_project
)

d = 9
p = 0.03
n_shots = 5000
rng = np.random.default_rng(42)

code = get_code("surface", d)
Hx = sp.csr_matrix(code.Hx % 2)
Hz = sp.csr_matrix(code.Hz % 2)
Lx = np.asarray(code.Lx % 2, dtype=np.uint8)
Lz = np.asarray(code.Lz % 2, dtype=np.uint8)

n_q = d * d
print(f"d={d}  n_q={n_q}  Hx={Hx.shape}  Hz={Hz.shape}")
print(f"Lx={Lx.shape}  Lz={Lz.shape}")

mwpm_z = pymatching.Matching(Hx.toarray().astype(np.uint8))  # decodes Z-errors from X-syndromes
mwpm_x = pymatching.Matching(Hz.toarray().astype(np.uint8))  # decodes X-errors from Z-syndromes


def component_ml_decode(H, s, p_ch):
    """Decode using active component decomposition + exact ML per component."""
    result = active_components(H, s, halo=0)
    if isinstance(result, tuple) and len(result) == 2:
        check_comps, qubit_comps = result
    else:
        return np.zeros(H.shape[1], dtype=np.uint8), []  # no active comps

    e = np.zeros(H.shape[1], dtype=np.uint8)
    comp_info = []
    for checks, qubits in zip(check_comps, qubit_comps):
        H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(H, s, checks, qubits)
        n_sub = len(q_l2g)

        # Try exact ML
        try:
            e_exact = solve_small_cluster_channel_ml(
                H_sub, s_sub, p_channel=p_ch, k_max=50, r_cap=50
            )
        except (ValueError, AssertionError):
            e_exact = None

        if e_exact is None:
            # Fallback: try ml_parity_project with uniform prior
            p_flip = np.full(n_sub, p_ch, dtype=np.float64)
            try:
                e_exact = ml_parity_project(H_sub, s_sub, p_flip=p_flip, r_cap=50)
            except Exception:
                e_exact = np.zeros(n_sub, dtype=np.uint8)  # give up

        e[q_l2g] ^= e_exact
        comp_info.append({
            "q_l2g": q_l2g,
            "c_l2g": c_l2g,
            "n_sub": n_sub,
            "s_sub": s_sub.copy(),
            "e_sub": e_exact.copy(),
            "H_sub_shape": H_sub.shape,
            "inconsistent": (e_exact is None or e_exact.sum() == 0 and s_sub.sum() > 0),
        })
    return e, comp_info


# Run comparison
mghd_fails_mwpm_ok = 0
both_fail = 0
both_ok = 0
mwpm_fails_mghd_ok = 0
mghd_total_fail = 0
mwpm_total_fail = 0

detailed_count = 0

for shot_idx in range(n_shots):
    ex_true = (rng.random(n_q) < p).astype(np.uint8)
    ez_true = (rng.random(n_q) < p).astype(np.uint8)

    sx = np.asarray((Hx @ ez_true) % 2, dtype=np.uint8).ravel()
    sz = np.asarray((Hz @ ex_true) % 2, dtype=np.uint8).ravel()

    # True observables
    obs_x_true = (ex_true @ Lz.T) % 2  # X-observable
    obs_z_true = (ez_true @ Lx.T) % 2  # Z-observable

    # MWPM global decode
    ez_mwpm = mwpm_z.decode(sx)
    ex_mwpm = mwpm_x.decode(sz)
    obs_x_mwpm = (ex_mwpm @ Lz.T) % 2
    obs_z_mwpm = (ez_mwpm @ Lx.T) % 2
    mwpm_ok = np.array_equal(obs_x_true, obs_x_mwpm) and np.array_equal(obs_z_true, obs_z_mwpm)

    # Component ML decode
    ez_comp, comp_info_z = component_ml_decode(Hx, sx, p)
    ex_comp, comp_info_x = component_ml_decode(Hz, sz, p)
    obs_x_comp = (ex_comp @ Lz.T) % 2
    obs_z_comp = (ez_comp @ Lx.T) % 2
    comp_ok = np.array_equal(obs_x_true, obs_x_comp) and np.array_equal(obs_z_true, obs_z_comp)

    # Verify parity: does the component correction satisfy the syndrome?
    resid_z = (Hx @ ez_comp) % 2
    resid_x = (Hz @ ex_comp) % 2
    parity_ok_z = np.array_equal(resid_z.ravel().astype(np.uint8), sx)
    parity_ok_x = np.array_equal(resid_x.ravel().astype(np.uint8), sz)

    if not mwpm_ok:
        mwpm_total_fail += 1
    if not comp_ok:
        mghd_total_fail += 1

    if comp_ok and mwpm_ok:
        both_ok += 1
    elif not comp_ok and not mwpm_ok:
        both_fail += 1
    elif not comp_ok and mwpm_ok:
        mghd_fails_mwpm_ok += 1
        # Print details for first few failures
        if detailed_count < 5:
            detailed_count += 1
            print(f"\n=== Shot {shot_idx}: MGHD fails, MWPM ok ===")
            print(f"  ex_true wt={ex_true.sum()}, ez_true wt={ez_true.sum()}")
            print(f"  sx wt={sx.sum()}, sz wt={sz.sum()}")
            print(f"  parity_ok_z={parity_ok_z}  parity_ok_x={parity_ok_x}")
            print(f"  obs_true: x={obs_x_true} z={obs_z_true}")
            print(f"  obs_mwpm: x={obs_x_mwpm} z={obs_z_mwpm}")
            print(f"  obs_comp: x={obs_x_comp} z={obs_z_comp}")

            diff_z = (ez_true ^ ez_comp)
            diff_x = (ex_true ^ ex_comp)
            print(f"  ez_true⊕ez_comp wt={diff_z.sum()}, (Hx@diff)%2 wt={(np.asarray((Hx @ diff_z) % 2)).sum()}")
            print(f"  ex_true⊕ex_comp wt={diff_x.sum()}, (Hz@diff)%2 wt={(np.asarray((Hz @ diff_x) % 2)).sum()}")
            print(f"  (diff⊕Lx)%2={(diff_z @ Lx.T) % 2}, (diff⊕Lz)%2={(diff_x @ Lz.T) % 2}")

            # Z-side components
            print(f"  Z-side ({len(comp_info_z)} components):")
            for ci, c in enumerate(comp_info_z):
                print(f"    comp {ci}: n_sub={c['n_sub']} H_sub={c['H_sub_shape']} "
                      f"s_sub_wt={c['s_sub'].sum()} e_sub_wt={c['e_sub'].sum()} "
                      f"q_l2g={c['q_l2g'].tolist()}")
                # Show which true errors are in this component
                true_in_comp = ez_true[c['q_l2g']]
                print(f"            true_err={true_in_comp.tolist()} "
                      f"comp_corr={c['e_sub'].tolist()}")

            # MWPM correction for Z side
            diff_mwpm_z = (ez_true ^ ez_mwpm)
            print(f"  MWPM ez_corr wt={ez_mwpm.sum()}, diff_mwpm_z wt={diff_mwpm_z.sum()}")

            # X-side components
            print(f"  X-side ({len(comp_info_x)} components):")
            for ci, c in enumerate(comp_info_x):
                print(f"    comp {ci}: n_sub={c['n_sub']} H_sub={c['H_sub_shape']} "
                      f"s_sub_wt={c['s_sub'].sum()} e_sub_wt={c['e_sub'].sum()} "
                      f"q_l2g={c['q_l2g'].tolist()}")
                true_in_comp = ex_true[c['q_l2g']]
                print(f"            true_err={true_in_comp.tolist()} "
                      f"comp_corr={c['e_sub'].tolist()}")
    else:
        mwpm_fails_mghd_ok += 1

print(f"\n=== Summary ({n_shots} shots, d={d}, p={p}) ===")
print(f"  Both OK:      {both_ok} ({100*both_ok/n_shots:.2f}%)")
print(f"  Both fail:    {both_fail} ({100*both_fail/n_shots:.2f}%)")
print(f"  MGHD fail, MWPM ok: {mghd_fails_mwpm_ok} ({100*mghd_fails_mwpm_ok/n_shots:.2f}%)")
print(f"  MWPM fail, MGHD ok: {mwpm_fails_mghd_ok} ({100*mwpm_fails_mghd_ok/n_shots:.2f}%)")
print(f"  MGHD total LER: {100*mghd_total_fail/n_shots:.2f}%")
print(f"  MWPM total LER: {100*mwpm_total_fail/n_shots:.2f}%")
