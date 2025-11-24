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
from pathlib import Path

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

        # MWPF (hypergraph) — optional; may fall back internally if mwpf not available
        mwpf_teacher = None
        try:
            mwpf_teacher = MWPFTeacher(code)
        except Exception as exc:
            print(f"  MWPFTeacher unavailable for d={d}: {exc}")
        
        for p in p_values:
            # Check if already computed
            if any(r["distance"] == d and r["p"] == p for r in results):
                print(f"  Skipping p={p} (already computed)")
                continue

            print(f"  Testing p={p}...")
            if args.sampler == "stim":
                sampler = StimSampler(rounds=d, dep=p)  # rounds=d for phenomenological/circuit
            elif args.sampler == "cudaq":
                # CudaQSampler expects phys_p/noise_scale in profile_kwargs
                pk = {"phys_p": p, "rounds": d}
                if getattr(args, "noise_scale", None) is not None:
                    pk["noise_scale"] = float(args.noise_scale)
                sampler = CudaQSampler(device_profile="garnet", profile_kwargs=pk)
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            
            # Setup Pymatching for Surface Code (Stim-based); only meaningful for Stim sampler.
            mwpm_stim_matcher = None
            if args.sampler == "stim" and args.family == "surface":
                try:
                    import stim
                    import pymatching
                    # Re-generate circuit to match StimSampler's configuration
                    # For Stim sampler, we use rounds=d (full circuit).
                    # For CudaQ sampler, we use rounds=1 because CudaQSampler currently returns single-round syndromes.
                    mwpm_rounds = d if args.sampler == "stim" else 1
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
                
                # --- MGHD Decoding ---
                sx, sz = _resolve_syndromes(code, batch.dets)
                
                preds_mghd = []
                for i in range(this_batch):
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
                if mwpf_teacher is not None:
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
                            if ler_res_mwpf.ler_mean is None:
                                print(f"MWPF LER Error: {ler_res_mwpf.notes}")
                                sys.exit(1)
                            failures_mwpf += float(ler_res_mwpf.ler_mean) * this_batch
                    except Exception as exc:
                        print(f"  MWPF decode failed at p={p}, d={d}: {exc}")
                        mwpf_teacher = None
                
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
    
    # Plot results
    plot_results(results, args.output)

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
    parser.add_argument("--sampler", default="stim", choices=["stim", "cudaq"], help="Sampler to use (stim or cudaq)")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=None,
        help="Optional global noise scale for CUDA-Q Garnet sampler (cudaq only).",
    )
    
    args = parser.parse_args()
    evaluate(args)
