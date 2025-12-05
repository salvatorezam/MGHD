#!/usr/bin/env python
"""
Evaluate a trained MGHD model on a range of distances and physical error rates.
Compares MGHD against MWPM and LSD teachers.
"""

import argparse
import json
import time
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

from mghd.codes.registry import get_code
from mghd.core.core import MGHDDecoderPublic
from mghd.decoders.lsd.clustered import MGHDPrimaryClustered
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.decoders.mwpf_teacher import MWPFTeacher
from mghd.samplers.cudaq_sampler import CudaQSampler
from mghd.samplers.stim_sampler import StimSampler
from mghd.utils.graphlike import is_graphlike
from mghd.utils.metrics import logical_error_rate


@dataclass
class PhenomenologicalBatch:
    """Sample batch from phenomenological noise model."""
    dets: np.ndarray  # Combined syndromes [synX | synZ] for compatibility
    obs: np.ndarray   # Logical observables
    synX: np.ndarray  # X-stabilizer syndromes (detect Z errors)
    synZ: np.ndarray  # Z-stabilizer syndromes (detect X errors)
    err_x: np.ndarray # X errors (for debugging)
    err_z: np.ndarray # Z errors (for debugging)
    meta: dict


class PhenomenologicalSampler:
    """Sampler that generates IID phenomenological noise on surface codes.
    
    This produces both X and Z syndromes from independent data qubit errors,
    matching the training setup used with CUDA-Q.
    """
    
    def __init__(self, p: float = 0.01):
        self.p = p
    
    def sample(self, code_obj, n_shots: int, seed: Optional[int] = None) -> PhenomenologicalBatch:
        """Sample phenomenological errors and compute syndromes."""
        rng = np.random.default_rng(seed)
        
        Hx = np.asarray(code_obj.Hx, dtype=np.uint8)
        Hz = np.asarray(code_obj.Hz, dtype=np.uint8)
        n_data = Hx.shape[1]
        mx = Hx.shape[0]
        mz = Hz.shape[0]
        
        # Generate IID errors
        err_x = (rng.random((n_shots, n_data)) < self.p).astype(np.uint8)
        err_z = (rng.random((n_shots, n_data)) < self.p).astype(np.uint8)
        
        # Compute syndromes
        # X checks (Hx) detect Z errors: synX = (Hx @ err_z.T).T % 2
        synX = (err_z @ Hx.T) % 2
        # Z checks (Hz) detect X errors: synZ = (Hz @ err_x.T).T % 2
        synZ = (err_x @ Hz.T) % 2
        
        # Combined detector array for compatibility with _resolve_syndromes
        # Format: [synX | synZ] so that detectors_to_syndromes works correctly
        dets = np.concatenate([synX, synZ], axis=1).astype(np.uint8)
        
        # Compute logical observables
        # For surface code: Lx detects Z logical errors, Lz detects X logical errors
        Lx = getattr(code_obj, 'Lx', None)
        Lz = getattr(code_obj, 'Lz', None)
        
        obs_list = []
        if Lz is not None:
            Lz_arr = np.asarray(Lz, dtype=np.uint8)
            if Lz_arr.ndim == 1:
                Lz_arr = Lz_arr.reshape(1, -1)
            # X errors flip Z logical: obs_z = (err_x @ Lz.T) % 2
            obs_z = (err_x @ Lz_arr.T) % 2
            obs_list.append(obs_z)
        if Lx is not None:
            Lx_arr = np.asarray(Lx, dtype=np.uint8)
            if Lx_arr.ndim == 1:
                Lx_arr = Lx_arr.reshape(1, -1)
            # Z errors flip X logical: obs_x = (err_z @ Lx.T) % 2  
            obs_x = (err_z @ Lx_arr.T) % 2
            obs_list.append(obs_x)
        
        if obs_list:
            obs = np.concatenate(obs_list, axis=1).astype(np.uint8)
        else:
            obs = np.zeros((n_shots, 1), dtype=np.uint8)
        
        return PhenomenologicalBatch(
            dets=dets,
            obs=obs,
            synX=synX.astype(np.uint8),
            synZ=synZ.astype(np.uint8),
            err_x=err_x,
            err_z=err_z,
            meta={'sampler': 'phenomenological', 'p': self.p},
        )

def align_preds(preds, obs_true):
    """Align predictions with ground truth observables."""
    # Squeeze middle dimension if present (B, 1, num_obs) -> (B, num_obs)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
        
    if preds.shape == obs_true.shape:
        return preds
    if obs_true.shape[1] == 1 and preds.shape[1] == 2:
        # Assume Stim surface code memory X -> logical X observable (2nd column of data_to_observables)
        # data_to_observables returns [Z_obs, X_obs]
        return preds[:, 1:]
    return preds

def evaluate(args):
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = "cuda" if args.cuda else "cpu"
    try:
        decoder_public = MGHDDecoderPublic(
            args.checkpoint,
            device=device,
            profile=args.profile,
            node_feat_dim=args.node_feat_dim,
        )
    except RuntimeError as exc:
        # If CUDA init fails, fall back to CPU but keep going.
        if args.cuda and ("cuda" in str(exc).lower() or "cudnn" in str(exc).lower()):
            print(f"Warning: CUDA failed ({exc}); falling back to CPU for evaluation.")
            device = "cpu"
            decoder_public = MGHDDecoderPublic(
                args.checkpoint,
                device=device,
                profile=args.profile,
                node_feat_dim=args.node_feat_dim,
            )
        else:
            raise
    
    results = []
    if Path(args.output).exists():
        try:
            with open(args.output, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {args.output}")
        except json.JSONDecodeError:
            print(f"Could not load existing results from {args.output}, starting fresh.")

    distances = [int(d) for d in args.distances.split(",")]
    p_values = [float(p) for p in args.p_values.split(",")]
    
    for d in distances:
        print(f"\n=== Distance {d} ===")
        code = get_code(args.family, distance=d)
        
        # Ensure we have both sparse and dense versions
        if sp.issparse(code.Hx):
            Hx_sparse = code.Hx
            Hx_dense = code.Hx.toarray()
        else:
            Hx_sparse = sp.csr_matrix(code.Hx)
            Hx_dense = np.asarray(code.Hx)
            
        if sp.issparse(code.Hz):
            Hz_sparse = code.Hz
            Hz_dense = code.Hz.toarray()
        else:
            Hz_sparse = sp.csr_matrix(code.Hz)
            Hz_dense = np.asarray(code.Hz)
        
        # Bind code to decoder (needed for some internal checks, though PrimaryClustered passes H explicitly)
        decoder_public.bind_code(Hx_sparse, Hz_sparse)
        
        # Initialize decoders
        # MGHDPrimaryClustered for Z errors (using Hx)
        mghd_Z = MGHDPrimaryClustered(Hx_sparse, decoder_public, batched=True)
        # MGHDPrimaryClustered for X errors (using Hz)
        mghd_X = MGHDPrimaryClustered(Hz_sparse, decoder_public, batched=True)
        
        # MWPM
        mwpm_ctx = MWPMatchingContext()
        # Disable MWPM for CUDA-Q as the code definition might be non-graph-like (weight-4 columns)
        # causing pymatching to fail or fallback, which is not a useful baseline.
        mwpm_enabled = args.sampler != "cudaq"
        
        # LSD
        lsd = LSDTeacher(Hx_dense, Hz_dense)

        # MWPF (hypergraph) — DISABLED: crashes with Rust panic on larger/complex syndromes
        # that cannot be caught by Python exception handling
        mwpf_teacher = None
        # try:
        #     mwpf_teacher = MWPFTeacher(code)
        # except Exception as exc:
        #     print(f"  MWPFTeacher unavailable for d={d}: {exc}")
        
        for p in p_values:
            # Check if already computed
            if any(r["distance"] == d and r["p"] == p for r in results):
                print(f"  Skipping p={p} (already computed)")
                continue

            print(f"  Testing p={p}...")
            if args.sampler == "stim":
                sampler = StimSampler(rounds=1, dep=p)  # rounds=1 to match training (single-round phenomenological)
            elif args.sampler == "cudaq":
                # CudaQSampler expects phys_p/noise_scale in profile_kwargs
                pk = {"phys_p": p, "rounds": d}
                if getattr(args, "noise_scale", None) is not None:
                    pk["noise_scale"] = float(args.noise_scale)
                sampler = CudaQSampler(device_profile="garnet", profile_kwargs=pk)
            elif args.sampler == "phenomenological":
                # Phenomenological sampler generates IID X/Z errors with both syndrome types
                # This matches the training setup better than Stim's circuit-level simulation
                sampler = PhenomenologicalSampler(p=p)
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            
            # Setup Pymatching for Surface Code (Stim-based); only meaningful for Stim sampler.
            mwpm_stim_matcher = None
            if args.sampler == "stim" and args.family == "surface":
                try:
                    import stim
                    import pymatching
                    # Re-generate circuit to match StimSampler's configuration
                    # Use rounds=1 to match the single-round sampling (phenomenological-like)
                    mwpm_rounds = 1
                    circuit = stim.Circuit.generated(
                        "surface_code:rotated_memory_x",
                        distance=d,
                        rounds=mwpm_rounds,
                        after_clifford_depolarization=p,
                    )
                    dem = circuit.detector_error_model(decompose_errors=True)
                    mwpm_stim_matcher = pymatching.Matching.from_detector_error_model(dem)
                except ImportError:
                    print("  Stim/Pymatching not available for optimized MWPM.")
                except Exception as e:
                    print(f"  Failed to setup Stim MWPM: {e}")

            total_shots = 0
            failures_mghd = 0.0
            failures_mwpm = 0.0 if mwpm_enabled else None
            failures_lsd = 0.0
            failures_mwpf = 0.0 if mwpf_teacher is not None else None
            
            # Batched evaluation (ceil division so we always process >0 when shots>0)
            n_batches = (args.shots + args.batch_size - 1) // args.batch_size
            
            for b in range(n_batches):
                # Adjust batch size for final (possibly smaller) chunk
                this_batch = min(args.batch_size, args.shots - total_shots)
                if this_batch <= 0:
                    break

                batch = sampler.sample(code, n_shots=this_batch, seed=args.seed + b)
                
                # Ground truth observables
                obs_true = batch.obs
                
                # --- Get syndromes ---
                # For phenomenological sampler, use direct synX/synZ
                # For Stim/CUDA-Q, use _resolve_syndromes
                if hasattr(batch, 'synX') and hasattr(batch, 'synZ'):
                    sx = batch.synX
                    sz = batch.synZ
                else:
                    sx, sz = _resolve_syndromes(code, batch.dets)
                
                # --- MGHD Decoding ---
                preds_mghd = []
                for i in range(this_batch):
                    # Z errors (from sx - X checks detect Z errors)
                    res_z = mghd_Z.decode(sx[i])
                    ez = res_z["e"]
                    
                    # X errors (from sz - Z checks detect X errors)
                    res_x = mghd_X.decode(sz[i])
                    ex = res_x["e"]
                    
                    # Convert correction to observables
                    obs_pred = code.data_to_observables(ex, ez)
                    preds_mghd.append(obs_pred)
                
                preds_mghd = np.array(preds_mghd, dtype=np.uint8)
                preds_mghd = align_preds(preds_mghd, obs_true)
                
                ler_res_mghd = logical_error_rate(obs_true, preds_mghd)
                if ler_res_mghd.ler_mean is None:
                    print(f"MGHD LER Error: {ler_res_mghd.notes}")
                    print(f"obs_true shape: {obs_true.shape}, preds_mghd shape: {preds_mghd.shape}")
                    sys.exit(1)
                failures_mghd += float(ler_res_mghd.ler_mean) * this_batch
                
                # --- MWPM Decoding ---
                if args.sampler == "stim" and mwpm_stim_matcher is not None:
                    # Use Stim-based matching (fast and correct for surface code)
                    preds_mwpm = mwpm_stim_matcher.decode_batch(batch.dets)
                    # Align preds if necessary (usually Stim DEM obs match Stim sampler obs)
                    preds_mwpm = align_preds(preds_mwpm, obs_true)
                    
                    ler_res_mwpm = logical_error_rate(obs_true, preds_mwpm)
                    if ler_res_mwpm.ler_mean is None:
                        print(f"MWPM LER Error: {ler_res_mwpm.notes}")
                        sys.exit(1)
                    failures_mwpm += float(ler_res_mwpm.ler_mean) * this_batch
                elif mwpm_enabled and mwpm_ctx is not None:
                    preds_mwpm = []
                    for i in range(this_batch):
                        ez_pm, _ = mwpm_ctx.decode(Hx_dense, sx[i], "Z")
                        ex_pm, _ = mwpm_ctx.decode(Hz_dense, sz[i], "X")
                        obs_pred = code.data_to_observables(ex_pm, ez_pm)
                        preds_mwpm.append(obs_pred)
                    preds_mwpm = np.array(preds_mwpm, dtype=np.uint8)
                    preds_mwpm = align_preds(preds_mwpm, obs_true)
                    ler_res_mwpm = logical_error_rate(obs_true, preds_mwpm)
                    if ler_res_mwpm.ler_mean is None:
                        print(f"MWPM LER Error: {ler_res_mwpm.notes}")
                        print(f"obs_true shape: {obs_true.shape}, preds_mwpm shape: {preds_mwpm.shape}")
                        sys.exit(1)
                    failures_mwpm += float(ler_res_mwpm.ler_mean) * this_batch
                
                # --- LSD Decoding ---
                # NOTE: LSDTeacher.decode_batch_xz returns (ex, ez) where:
                #   ex = Hx^{-1}(syndromes_x) - solves Hx @ ex = sx
                #   ez = Hz^{-1}(syndromes_z) - solves Hz @ ez = sz
                # But in CSS semantics: Hx detects Z errors, Hz detects X errors!
                # So the naming is misleading - we need to swap:
                #   The "ex" output is actually ez (Z errors from sx via Hx)
                #   The "ez" output is actually ex (X errors from sz via Hz)
                ez_lsd, ex_lsd = lsd.decode_batch_xz(sx, sz)  # Swapped!
                preds_lsd = []
                for i in range(this_batch):
                    obs_pred = code.data_to_observables(ex_lsd[i], ez_lsd[i])
                    preds_lsd.append(obs_pred)
                preds_lsd = np.array(preds_lsd, dtype=np.uint8)
                preds_lsd = align_preds(preds_lsd, obs_true)
                
                ler_res_lsd = logical_error_rate(obs_true, preds_lsd)
                if ler_res_lsd.ler_mean is None:
                    print(f"LSD LER Error: {ler_res_lsd.notes}")
                    print(f"obs_true shape: {obs_true.shape}, preds_lsd shape: {preds_lsd.shape}")
                    sys.exit(1)
                failures_lsd += float(ler_res_lsd.ler_mean) * this_batch

                # --- MWPF Decoding (approximate ex/ez from fault_ids) ---
                # NOTE: MWPF often crashes on large distances or complex syndromes.
                # We catch exceptions and disable it for the rest of this (d, p) combo.
                if mwpf_teacher is not None and failures_mwpf is not None:
                    try:
                        out_mwpf = mwpf_teacher.decode_batch(batch.dets)
                        fault_ids = np.asarray(out_mwpf.get("fault_ids"), dtype=np.int32)
                        if fault_ids.ndim == 2:
                            B = fault_ids.shape[0]
                            n = Hx_dense.shape[1]
                            ex_pf = np.zeros((B, n), dtype=np.uint8)
                            ez_pf = np.zeros((B, n), dtype=np.uint8)
                            for bi in range(B):
                                fids = fault_ids[bi]
                                fids = fids[fids >= 0]
                                if fids.size:
                                    # Interpret each fault id as a data-qubit index; flip both X and Z.
                                    ex_pf[bi, fids] ^= 1
                                    ez_pf[bi, fids] ^= 1

                            preds_mwpf = []
                            for i in range(this_batch):
                                obs_pred = code.data_to_observables(ex_pf[i], ez_pf[i])
                                preds_mwpf.append(obs_pred)
                            preds_mwpf = np.array(preds_mwpf, dtype=np.uint8)
                            preds_mwpf = align_preds(preds_mwpf, obs_true)

                            ler_res_mwpf = logical_error_rate(obs_true, preds_mwpf)
                            if ler_res_mwpf.ler_mean is not None:
                                failures_mwpf += float(ler_res_mwpf.ler_mean) * this_batch
                    except Exception as exc:
                        print(f"\n  MWPF decode failed at d={d}, p={p}: {type(exc).__name__}")
                        # Disable MWPF for the rest of this evaluation
                        failures_mwpf = None
                
                total_shots += this_batch
                
                mwpm_ratio = None if failures_mwpm is None or total_shots == 0 else failures_mwpm / total_shots
                mwpm_str = f"{mwpm_ratio:.10f}" if mwpm_ratio is not None else "NA"
                mwpf_ratio = None if failures_mwpf is None or total_shots == 0 else failures_mwpf / total_shots
                mwpf_str = f"{mwpf_ratio:.10f}" if mwpf_ratio is not None else "NA"
                print(
                    f"    Batch {b+1}/{n_batches}: MGHD={failures_mghd/total_shots:.10f} "
                    f"MWPM={mwpm_str} MWPF={mwpf_str} LSD={failures_lsd/total_shots:.10f}",
                    end="\r",
                )
            
            mwpm_final = None if failures_mwpm is None or total_shots == 0 else failures_mwpm / total_shots
            mwpm_final_str = f"{mwpm_final:.10f}" if mwpm_final is not None else "NA"
            mwpf_final = None if failures_mwpf is None or total_shots == 0 else failures_mwpf / total_shots
            mwpf_final_str = f"{mwpf_final:.10f}" if mwpf_final is not None else "NA"
            print(
                f"    Final: MGHD={failures_mghd/total_shots:.10f} "
                f"MWPM={mwpm_final_str} MWPF={mwpf_final_str} LSD={failures_lsd/total_shots:.10f}"
            )
            
            results.append({
                "distance": d,
                "p": p,
                "shots": total_shots,
                "ler_mghd": failures_mghd / total_shots,
                "ler_mwpm": mwpm_final,
                "ler_mwpf": mwpf_final,
                "ler_lsd": failures_lsd / total_shots
            })
            
            # Save results incrementally
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved progress to {args.output}")
            
    # Final save (redundant but safe)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    
    # Run sanity checks
    sanity_check_results(results)
    
    # Plot results
    plot_results(results, args.output)


def sanity_check_results(results):
    """Validate LER trends follow expected QEC behavior.
    
    Expected:
    1. LER should increase with p at fixed d (more noise → more errors)
    2. LER should decrease with d at fixed p (below threshold) — larger codes correct more
    3. MGHD should be comparable to or better than teacher (LSD) on training distribution
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Validating LER trends")
    print("="*60)
    
    warnings_found = []
    
    # Group by distance
    by_d = defaultdict(list)
    for r in results:
        by_d[r["distance"]].append(r)
    
    # Check 1: LER increases with p at fixed d
    print("\n[Check 1] LER should increase with p at fixed d:")
    for d in sorted(by_d.keys()):
        pts = sorted(by_d[d], key=lambda x: x["p"])
        mghd_lers = [x["ler_mghd"] for x in pts]
        lsd_lers = [x["ler_lsd"] for x in pts if x.get("ler_lsd") is not None]
        
        # Check MGHD trend
        mghd_monotonic = all(mghd_lers[i] <= mghd_lers[i+1] + 0.001 for i in range(len(mghd_lers)-1))
        status = "✓" if mghd_monotonic else "✗ WARNING"
        print(f"  d={d}: MGHD {status}")
        if not mghd_monotonic:
            warnings_found.append(f"d={d}: MGHD LER not monotonic with p")
        
        # Check LSD trend (teacher baseline)
        if lsd_lers:
            lsd_monotonic = all(lsd_lers[i] <= lsd_lers[i+1] + 0.001 for i in range(len(lsd_lers)-1))
            status = "✓" if lsd_monotonic else "✗ WARNING"
            print(f"  d={d}: LSD  {status}")
            if not lsd_monotonic:
                warnings_found.append(f"d={d}: LSD LER not monotonic with p (baseline issue!)")
    
    # Check 2: LER decreases with d at fixed p (below threshold ~1%)
    print("\n[Check 2] LER should decrease with d at fixed p (below threshold):")
    by_p = defaultdict(list)
    for r in results:
        by_p[r["p"]].append(r)
    
    for p in sorted(by_p.keys()):
        if p > 0.008:  # Skip above-threshold points
            continue
        pts = sorted(by_p[p], key=lambda x: x["distance"])
        if len(pts) < 2:
            continue
        
        mghd_lers = [x["ler_mghd"] for x in pts]
        distances = [x["distance"] for x in pts]
        
        # Check if generally decreasing (allow some noise)
        decreasing_count = sum(1 for i in range(len(mghd_lers)-1) if mghd_lers[i] >= mghd_lers[i+1] - 0.002)
        is_decreasing = decreasing_count >= len(mghd_lers) - 2  # Allow 1 violation
        
        status = "✓" if is_decreasing else "✗ WARNING"
        trend = " → ".join(f"d{d}:{ler:.4f}" for d, ler in zip(distances, mghd_lers))
        print(f"  p={p:.4f}: {status} ({trend})")
        if not is_decreasing:
            warnings_found.append(f"p={p}: MGHD LER not decreasing with d")
    
    # Check 3: MGHD vs Teacher comparison
    print("\n[Check 3] MGHD should match or beat LSD teacher:")
    mghd_better = 0
    mghd_worse = 0
    mghd_worse_cases = []
    
    for r in results:
        if r.get("ler_lsd") is None:
            continue
        diff = r["ler_mghd"] - r["ler_lsd"]
        if diff <= 0.001:  # MGHD is better or equal (within noise)
            mghd_better += 1
        else:
            mghd_worse += 1
            mghd_worse_cases.append((r["distance"], r["p"], r["ler_mghd"], r["ler_lsd"]))
    
    total = mghd_better + mghd_worse
    if total > 0:
        pct_better = 100 * mghd_better / total
        status = "✓" if pct_better >= 80 else "✗ WARNING"
        print(f"  {status} MGHD ≤ LSD in {mghd_better}/{total} cases ({pct_better:.1f}%)")
        
        if mghd_worse_cases:
            print("\n  Cases where MGHD > LSD:")
            for d, p, ler_mghd, ler_lsd in mghd_worse_cases[:5]:
                ratio = ler_mghd / max(ler_lsd, 1e-10)
                print(f"    d={d}, p={p:.4f}: MGHD={ler_mghd:.5f}, LSD={ler_lsd:.5f} (ratio={ratio:.2f}x)")
            if len(mghd_worse_cases) > 5:
                print(f"    ... and {len(mghd_worse_cases) - 5} more")
    
    # Summary
    print("\n" + "="*60)
    if warnings_found:
        print(f"⚠️  {len(warnings_found)} warnings found:")
        for w in warnings_found:
            print(f"   - {w}")
    else:
        print("✓ All sanity checks passed!")
    print("="*60 + "\n")

def plot_results(results, output_path):
    """Generate and save a clean LER plot from results."""
    # Organize data by distance
    data_by_d = defaultdict(lambda: {"p": [], "mghd": [], "mwpm": [], "mwpf": [], "lsd": []})
    
    for res in results:
        d = res["distance"]
        data_by_d[d]["p"].append(res["p"])
        data_by_d[d]["mghd"].append(res["ler_mghd"])
        if res.get("ler_mwpm") is not None:
            data_by_d[d]["mwpm"].append(res["ler_mwpm"])
        if res.get("ler_mwpf") is not None:
            data_by_d[d]["mwpf"].append(res["ler_mwpf"])
        if res.get("ler_lsd") is not None:
            data_by_d[d]["lsd"].append(res["ler_lsd"])
            
    # Use a professional style
    plt.style.use('seaborn-v0_8-paper')
    # Set global font sizes for publication
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
    
    plt.figure(figsize=(10, 8))
    
    # Use a qualitative colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data_by_d))]
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    for i, (d, data) in enumerate(sorted(data_by_d.items())):
        p_vals = np.array(data["p"])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Sort by p
        idx = np.argsort(p_vals)
        p_vals = p_vals[idx]
        
        # MGHD
        mghd_vals = np.array(data["mghd"])[idx]
        plt.loglog(p_vals, mghd_vals, marker=marker, linestyle='-', color=color, label=f'MGHD d={d}')
        
        # MWPM
        if data["mwpm"]:
            mwpm_vals = np.array(data["mwpm"])[idx]
            plt.loglog(p_vals, mwpm_vals, marker=marker, linestyle='--', color=color, alpha=0.6, label=f'MWPM d={d}')

        # MWPF
        if data["mwpf"]:
            mwpf_vals = np.array(data["mwpf"])[idx]
            plt.loglog(p_vals, mwpf_vals, marker=marker, linestyle='-.', color=color, alpha=0.6, label=f'MWPF d={d}')
            
        # LSD
        if data["lsd"]:
            lsd_vals = np.array(data["lsd"])[idx]
            plt.loglog(p_vals, lsd_vals, marker=marker, linestyle=':', color=color, alpha=0.4, label=f'LSD d={d}')

    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("Logical Error Rate vs Physical Error Rate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plot_file_png = str(Path(output_path).with_suffix('.png'))
    plot_file_pdf = str(Path(output_path).with_suffix('.pdf'))
    
    plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"Plots saved to {plot_file_png} and {plot_file_pdf}")

def _resolve_syndromes(code_obj, dets):
    """Helper to map detectors to syndromes (copied from teacher_eval.py)"""
    mx = getattr(code_obj, "Hx", None)
    mz = getattr(code_obj, "Hz", None)
    B = dets.shape[0]
    if hasattr(code_obj, "detectors_to_syndromes"):
        sx, sz = code_obj.detectors_to_syndromes(dets)
        return sx.astype(np.uint8), sz.astype(np.uint8)
    if mx is not None:
        sx = np.zeros((B, mx.shape[0]), dtype=np.uint8)
    else:
        sx = np.zeros((B, 0), dtype=np.uint8)
    if mz is not None:
        sz = np.zeros((B, mz.shape[0]), dtype=np.uint8)
    else:
        sz = np.zeros((B, 0), dtype=np.uint8)
    return sx, sz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--family", default="surface")
    parser.add_argument("--distances", default="3,5")
    parser.add_argument("--p-values", default="0.001,0.005,0.01")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--profile", default="S")
    parser.add_argument("--node-feat-dim", type=int, default=8)
    parser.add_argument("--sampler", default="phenomenological", choices=["stim", "cudaq", "phenomenological"],
                       help="Sampler to use. 'phenomenological' (default) matches training setup with IID X/Z errors. "
                            "'stim' uses Stim circuit-level simulation (caution: different noise model). "
                            "'cudaq' uses CUDA-Q backend.")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="Optional global noise scale for CUDA-Q Garnet sampler (cudaq only).",
    )
    
    args = parser.parse_args()
    evaluate(args)
