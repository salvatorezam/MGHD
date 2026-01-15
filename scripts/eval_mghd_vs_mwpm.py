#!/usr/bin/env python3
"""
Evaluate trained MGHD model LER vs MWPM baseline across physical error rates.
Uses the proper PackedCrop pipeline for MGHD inference.
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mghd_model(checkpoint_path, device="cuda"):
    """Load trained MGHD model from checkpoint."""
    from mghd.core.core import MGHDv2
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MGHDv2(profile="S").to(device)
    
    state_dict = checkpoint["model"]
    if "g_proj.weight" in state_dict:
        g_dim = state_dict["g_proj.weight"].shape[1]
        model.ensure_g_proj(g_dim, device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_mghd(model, code, p, distance, shots, device):
    """Evaluate MGHD model using proper PackedCrop pipeline."""
    from mghd.core.core import pack_cluster
    from mghd.qpu.adapters.surface_sampler import sample_round, split_components_for_side
    
    os.environ["MGHD_SYNTHETIC"] = "1"  # Use synthetic sampling
    
    correct = 0
    total = 0
    
    for shot_idx in range(shots):
        seed = shot_idx + int(p * 1e9)
        
        # Sample syndrome
        sample = sample_round(d=distance, p=p, seed=seed)
        
        Hx = sample["Hx"]
        Hz = sample["Hz"]
        synZ = sample["synZ"]
        synX = sample["synX"]
        ex_glob = sample.get("ex_glob")
        ez_glob = sample.get("ez_glob")
        coords_q = sample["coords_q"]
        coords_c = sample["coords_c"]
        
        # Process Z-side components
        z_comps = split_components_for_side(
            side="Z", Hx=Hx, Hz=Hz, synZ=synZ, synX=synX,
            coords_q=coords_q, coords_c=coords_c
        )
        
        # Process X-side components  
        x_comps = split_components_for_side(
            side="X", Hx=Hx, Hz=Hz, synZ=synZ, synX=synX,
            coords_q=coords_q, coords_c=coords_c
        )
        
        # Decode each component with model
        pred_ex = np.zeros(distance * distance, dtype=np.uint8)
        pred_ez = np.zeros(distance * distance, dtype=np.uint8)
        
        # Z-side: predicts X errors (ex)
        for comp in z_comps:
            if comp["k"] == 0:
                continue
            try:
                pack = pack_cluster(
                    H_sub=comp["H_sub"],
                    xy_qubit=comp["xy_qubit"],
                    xy_check=comp["xy_check"],
                    synd_Z_then_X_bits=comp["synd_bits"],
                    k=int(comp["k"]),
                    r=int(comp["r"]),
                    bbox_xywh=tuple(int(v) for v in comp["bbox_xywh"]),
                    kappa_stats=comp.get("kappa_stats", {}),
                    y_bits_local=None,
                    side="Z",
                    d=distance,
                    p=p,
                    seed=seed,
                    N_max=512,
                    E_max=4096,
                    S_max=512,
                )
                
                # Move to device
                for attr in ["x_nodes", "edge_index", "edge_attr", "edge_mask", 
                            "node_mask", "node_type", "seq_idx", "seq_mask", "g_token"]:
                    if hasattr(pack, attr):
                        tensor = getattr(pack, attr)
                        if tensor is not None:
                            setattr(pack, attr, tensor.to(device))
                
                with torch.no_grad():
                    logits, node_mask = model(packed=pack)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(np.uint8)
                
                # Map local predictions back to global indices
                qubit_indices = comp.get("qubit_indices")
                if qubit_indices is not None:
                    for i, qi in enumerate(qubit_indices):
                        if i < len(preds) and qi < len(pred_ex):
                            pred_ex[qi] = preds[i]
            except Exception as e:
                pass  # Skip failed components
        
        # X-side: predicts Z errors (ez)
        for comp in x_comps:
            if comp["k"] == 0:
                continue
            try:
                pack = pack_cluster(
                    H_sub=comp["H_sub"],
                    xy_qubit=comp["xy_qubit"],
                    xy_check=comp["xy_check"],
                    synd_Z_then_X_bits=comp["synd_bits"],
                    k=int(comp["k"]),
                    r=int(comp["r"]),
                    bbox_xywh=tuple(int(v) for v in comp["bbox_xywh"]),
                    kappa_stats=comp.get("kappa_stats", {}),
                    y_bits_local=None,
                    side="X",
                    d=distance,
                    p=p,
                    seed=seed,
                    N_max=512,
                    E_max=4096,
                    S_max=512,
                )
                
                for attr in ["x_nodes", "edge_index", "edge_attr", "edge_mask",
                            "node_mask", "node_type", "seq_idx", "seq_mask", "g_token"]:
                    if hasattr(pack, attr):
                        tensor = getattr(pack, attr)
                        if tensor is not None:
                            setattr(pack, attr, tensor.to(device))
                
                with torch.no_grad():
                    logits, node_mask = model(packed=pack)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(np.uint8)
                
                qubit_indices = comp.get("qubit_indices")
                if qubit_indices is not None:
                    for i, qi in enumerate(qubit_indices):
                        if i < len(preds) and qi < len(pred_ez):
                            pred_ez[qi] = preds[i]
            except Exception as e:
                pass
        
        # Compare predicted correction to true error
        if ex_glob is not None and ez_glob is not None:
            # Check if (pred XOR true) commutes with logical operators
            pred_obs = code.data_to_observables(pred_ex[np.newaxis, :], pred_ez[np.newaxis, :])
            true_obs = code.data_to_observables(ex_glob[np.newaxis, :], ez_glob[np.newaxis, :])
            
            if pred_obs is not None and true_obs is not None:
                # Logical error if observables differ
                if np.all(pred_obs == true_obs):
                    correct += 1
                total += 1
    
    ler = 1.0 - (correct / total) if total > 0 else 1.0
    return {"ler": ler, "samples": total}


def evaluate_mwpm(code, p, distance, shots):
    """Evaluate MWPM decoder using Stim circuit-level noise."""
    from mghd.samplers import get_sampler
    from mghd.decoders.mix import TeacherMix, MixConfig
    
    sampler = get_sampler("stim", rounds=distance, dep=p)
    mix = TeacherMix(code, code.Hx, code.Hz, 
                    mix_cfg=MixConfig(p_mwpf=0.0, p_lsd=0.0, p_mwpm=1.0))
    
    correct = 0
    total = 0
    batch_size = 256
    n_batches = shots // batch_size
    rng = np.random.default_rng(42)
    
    for batch_idx in range(n_batches):
        seed = int(batch_idx * 1000 + p * 1e6)
        batch = sampler.sample(code, n_shots=batch_size, seed=seed)
        
        if batch.obs is None or batch.obs.size == 0:
            continue
        
        sx, sz = code.detectors_to_syndromes(batch.dets)
        out = mix.route_batch(dets=batch.dets, syndromes_x=sx, syndromes_z=sz, rng=rng)
        
        pred_obs = None
        if out["which"].startswith("mwpm"):
            cx, cz = out.get("cx"), out.get("cz")
            pred_obs = code.data_to_observables(cx, cz)
        
        if pred_obs is None:
            continue
        
        true_obs = batch.obs
        if pred_obs.shape[1] > true_obs.shape[1]:
            pred_obs = pred_obs[:, :true_obs.shape[1]]
        
        correct += np.sum(pred_obs[:, 0] == true_obs[:, 0])
        total += len(true_obs)
    
    ler = 1.0 - (correct / total) if total > 0 else 1.0
    err = np.sqrt(ler * (1 - ler) / total) if total > 0 else 0
    return {"ler": ler, "err": err, "samples": total}


def main():
    parser = argparse.ArgumentParser(description="Compare MGHD vs MWPM LER")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--shots", type=int, default=2000, help="Shots per p value")
    parser.add_argument("--output", type=str, default="mghd_vs_mwpm.png")
    args = parser.parse_args()
    
    p_values = [0.001, 0.003, 0.007, 0.01]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_mghd_model(args.model, device)
    
    # Load code
    from mghd.codes.registry import get_code
    code = get_code("surface", distance=args.distance)
    
    # Evaluate MWPM baseline
    print("\n=== Evaluating MWPM baseline ===")
    mwpm_results = {}
    for p in p_values:
        result = evaluate_mwpm(code, p, args.distance, args.shots)
        mwpm_results[p] = result
        print(f"  p={p:.3f}: MWPM LER={result['ler']:.4f} ± {result['err']:.4f}")
    
    # Evaluate MGHD model
    print("\n=== Evaluating MGHD model ===")
    mghd_results = {}
    for p in p_values:
        print(f"  Evaluating p={p:.3f}...", end=" ", flush=True)
        result = evaluate_mghd(model, code, p, args.distance, args.shots, device)
        mghd_results[p] = result
        print(f"MGHD LER={result['ler']:.4f} ({result['samples']} samples)")
    
    # Plot
    print("\nPlotting...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mwpm_lers = [mwpm_results[p]["ler"] for p in p_values]
    mwpm_errs = [mwpm_results[p]["err"] for p in p_values]
    mghd_lers = [mghd_results[p]["ler"] for p in p_values]
    
    ax.errorbar(p_values, mwpm_lers, yerr=mwpm_errs, 
                fmt='o-', label='MWPM', linewidth=2, markersize=8, capsize=4, color='blue')
    ax.plot(p_values, mghd_lers, 's-', label='MGHD (trained)', 
            linewidth=2, markersize=8, color='red')
    
    ax.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax.set_title(f'MGHD vs MWPM - Surface Code d={args.distance}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.012)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")
    
    # Summary
    print("\n" + "=" * 65)
    print(f"{'p':>8} | {'MWPM LER':>12} | {'MGHD LER':>12} | {'Improvement':>12}")
    print("-" * 65)
    for p in p_values:
        mwpm = mwpm_results[p]["ler"]
        mghd = mghd_results[p]["ler"]
        imp = (mwpm - mghd) / mwpm * 100 if mwpm > 0 else 0
        print(f"{p:>8.4f} | {mwpm:>12.4f} | {mghd:>12.4f} | {imp:>11.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
