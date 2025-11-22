#!/usr/bin/env python
"""
Evaluate a trained MGHD model on a range of distances and physical error rates.
Compares MGHD against MWPM and LSD teachers.
"""

import argparse
import json
import time
import sys
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from mghd.core.core import MGHDDecoderPublic
from mghd.decoders.lsd.clustered import MGHDPrimaryClustered
from mghd.samplers.stim_sampler import StimSampler
from mghd.samplers.cudaq_sampler import CudaQSampler
from mghd.codes.registry import get_code
from mghd.decoders.mwpm_ctx import MWPMatchingContext
from mghd.decoders.lsd_teacher import LSDTeacher
from mghd.utils.metrics import logical_error_rate
from mghd.utils.graphlike import is_graphlike

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
    decoder_public = MGHDDecoderPublic(args.checkpoint, device=device, profile=args.profile, node_feat_dim=args.node_feat_dim)
    
    results = []
    
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
        mwpm_enabled = True
        
        # LSD
        lsd = LSDTeacher(Hx_dense, Hz_dense)
        
        for p in p_values:
            print(f"  Testing p={p}...")
            if args.sampler == "stim":
                sampler = StimSampler(rounds=d, dep=p) # rounds=d for phenomenological/circuit
            elif args.sampler == "cudaq":
                # CudaQSampler expects phys_p in profile_kwargs
                sampler = CudaQSampler(device_profile="garnet", profile_kwargs={"phys_p": p, "rounds": d})
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            
            # Setup Pymatching for Surface Code (Stim-based)
            mwpm_stim_matcher = None
            if args.family == "surface":
                try:
                    import stim
                    import pymatching
                    # Re-generate circuit to match StimSampler's configuration
                    circuit = stim.Circuit.generated(
                        "surface_code:rotated_memory_x",
                        distance=d,
                        rounds=d,
                        after_clifford_depolarization=p,
                    )
                    dem = circuit.detector_error_model(decompose_errors=True)
                    mwpm_stim_matcher = pymatching.Matching.from_detector_error_model(dem)
                except ImportError:
                    print("  Stim/Pymatching not available for optimized MWPM.")
                except Exception as e:
                    print(f"  Failed to setup Stim MWPM: {e}")

            total_shots = 0
            failures_mghd = 0
            failures_mwpm = 0 if mwpm_enabled else None
            failures_lsd = 0
            
            # Batched evaluation
            n_batches = args.shots // args.batch_size
            
            for b in range(n_batches):
                batch = sampler.sample(code, n_shots=args.batch_size, seed=args.seed + b)
                
                # Ground truth observables
                obs_true = batch.obs
                
                # --- MGHD Decoding ---
                sx, sz = _resolve_syndromes(code, batch.dets)
                
                preds_mghd = []
                for i in range(args.batch_size):
                    # Z errors (from sx)
                    res_z = mghd_Z.decode(sx[i])
                    ez = res_z["e"]
                    
                    # X errors (from sz)
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
                failures_mghd += ler_res_mghd.ler_mean * args.batch_size
                
                # --- MWPM Decoding ---
                if mwpm_stim_matcher is not None:
                    # Use Stim-based matching (fast and correct for surface code)
                    preds_mwpm = mwpm_stim_matcher.decode_batch(batch.dets)
                    # Align preds if necessary (usually Stim DEM obs match Stim sampler obs)
                    preds_mwpm = align_preds(preds_mwpm, obs_true)
                    
                    ler_res_mwpm = logical_error_rate(obs_true, preds_mwpm)
                    if ler_res_mwpm.ler_mean is None:
                        print(f"MWPM LER Error: {ler_res_mwpm.notes}")
                        sys.exit(1)
                    failures_mwpm += ler_res_mwpm.ler_mean * args.batch_size
                elif mwpm_enabled and mwpm_ctx is not None:
                    preds_mwpm = []
                    for i in range(args.batch_size):
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
                    failures_mwpm += ler_res_mwpm.ler_mean * args.batch_size
                
                # --- LSD Decoding ---
                ex_lsd, ez_lsd = lsd.decode_batch_xz(sx, sz)
                preds_lsd = []
                for i in range(args.batch_size):
                    obs_pred = code.data_to_observables(ex_lsd[i], ez_lsd[i])
                    preds_lsd.append(obs_pred)
                preds_lsd = np.array(preds_lsd, dtype=np.uint8)
                preds_lsd = align_preds(preds_lsd, obs_true)
                
                ler_res_lsd = logical_error_rate(obs_true, preds_lsd)
                if ler_res_lsd.ler_mean is None:
                    print(f"LSD LER Error: {ler_res_lsd.notes}")
                    print(f"obs_true shape: {obs_true.shape}, preds_lsd shape: {preds_lsd.shape}")
                    sys.exit(1)
                failures_lsd += ler_res_lsd.ler_mean * args.batch_size
                
                total_shots += args.batch_size
                
                mwpm_ratio = None if failures_mwpm is None else failures_mwpm / total_shots
                mwpm_str = f"{mwpm_ratio:.4f}" if mwpm_ratio is not None else "NA"
                print(
                    f"    Batch {b+1}/{n_batches}: MGHD={failures_mghd/total_shots:.4f} "
                    f"MWPM={mwpm_str} LSD={failures_lsd/total_shots:.4f}",
                    end="\r",
                )
            
            mwpm_final = None if failures_mwpm is None else failures_mwpm / total_shots
            mwpm_final_str = f"{mwpm_final:.4f}" if mwpm_final is not None else "NA"
            print(
                f"    Final: MGHD={failures_mghd/total_shots:.4f} "
                f"MWPM={mwpm_final_str} LSD={failures_lsd/total_shots:.4f}"
            )
            
            results.append({
                "distance": d,
                "p": p,
                "shots": total_shots,
                "ler_mghd": failures_mghd / total_shots,
                "ler_mwpm": mwpm_final,
                "ler_lsd": failures_lsd / total_shots
            })
            
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    
    # Plot results
    plot_results(results, args.output)
    
    # Plot results
    plot_results(results, args.output)

def plot_results(results, output_path):
    """Generate and save a clean LER plot from results."""
    # Organize data by distance
    data_by_d = defaultdict(lambda: {"p": [], "mghd": [], "mwpm": [], "lsd": []})
    
    for res in results:
        d = res["distance"]
        data_by_d[d]["p"].append(res["p"])
        data_by_d[d]["mghd"].append(res["ler_mghd"])
        if res["ler_mwpm"] is not None:
            data_by_d[d]["mwpm"].append(res["ler_mwpm"])
        if res["ler_lsd"] is not None:
            data_by_d[d]["lsd"].append(res["ler_lsd"])
            
    # Use a professional style
    plt.style.use('seaborn-v0_8-paper')
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
        plt.loglog(p_vals, mghd_vals, marker=marker, linestyle='-', color=color, label=f'MGHD d={d}', linewidth=2, markersize=8)
        
        # MWPM
        if data["mwpm"]:
            mwpm_vals = np.array(data["mwpm"])[idx]
            plt.loglog(p_vals, mwpm_vals, marker=marker, linestyle='--', color=color, alpha=0.6, label=f'MWPM d={d}', linewidth=1.5, markersize=6)
            
        # LSD
        if data["lsd"]:
            lsd_vals = np.array(data["lsd"])[idx]
            plt.loglog(p_vals, lsd_vals, marker=marker, linestyle=':', color=color, alpha=0.4, label=f'LSD d={d}', linewidth=1.5, markersize=6)

    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.xlabel("Physical Error Rate (p)", fontsize=12)
    plt.ylabel("Logical Error Rate (LER)", fontsize=12)
    plt.title("Logical Error Rate vs Physical Error Rate", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plot_file = str(Path(output_path).with_suffix('.png'))
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")

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
    parser.add_argument("--sampler", default="stim", choices=["stim", "cudaq"], help="Sampler to use (stim or cudaq)")
    
    args = parser.parse_args()
    evaluate(args)
