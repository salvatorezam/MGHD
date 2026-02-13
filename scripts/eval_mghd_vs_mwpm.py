#!/usr/bin/env python3
"""
Evaluate trained MGHD model LER vs MWPM baseline across physical error rates.

Supports two modes:
1. Per-qubit mode (default): Uses PackedCrop pipeline for phenomenological models
2. Circuit-level mode (--circuit-level): Uses obs_head for circuit-level models

For fair comparison, both MGHD and MWPM should use the same noise model/sampler.
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import stim
    _HAVE_STIM = True
except ImportError:
    stim = None
    _HAVE_STIM = False


def load_mghd_model(checkpoint_path, device="cuda", circuit_level=False):
    """Load trained MGHD model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        circuit_level: If True, initialize obs_head from checkpoint metadata
    
    Returns:
        model: Loaded MGHDv2 model in eval mode
        metadata: Dict with checkpoint metadata (mode, num_detectors, etc.)
    """
    from mghd.core.core import MGHDv2
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MGHDv2(profile="S").to(device)
    
    state_dict = checkpoint["model"]
    metadata = {
        "mode": checkpoint.get("mode", "per_qubit"),
        "num_detectors": checkpoint.get("num_detectors"),
        "sampler": checkpoint.get("sampler"),
    }
    
    # Handle g_proj for per-qubit models
    if "g_proj.weight" in state_dict:
        g_dim = state_dict["g_proj.weight"].shape[1]
        model.ensure_g_proj(g_dim, device)
    
    # Handle obs_head for circuit-level models
    if circuit_level or metadata["mode"] == "circuit_level":
        # Try to get num_detectors from checkpoint or state_dict
        num_det = metadata["num_detectors"]
        if num_det is None and "obs_head.0.weight" in state_dict:
            num_det = state_dict["obs_head.0.weight"].shape[1]
        if num_det is not None:
            model.ensure_obs_head(num_det, device)
            metadata["num_detectors"] = num_det
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata


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


# ---------------------------------------------------------------------------
# Circuit-level evaluation functions (fair comparison using same Stim samples)
# ---------------------------------------------------------------------------

def _build_stim_memory_circuit(distance: int, rounds: int, p: float):
    """Build a rotated surface code memory circuit with circuit-level noise."""
    if not _HAVE_STIM:
        raise RuntimeError("stim package required for circuit-level evaluation")
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def evaluate_mghd_circuit_level(model, p, distance, rounds, shots, device):
    """Evaluate circuit-level trained MGHD model using obs_head.
    
    Args:
        model: MGHDv2 model with obs_head initialized
        p: Physical error rate
        distance: Code distance
        rounds: Number of syndrome extraction rounds
        shots: Number of shots to evaluate
        device: Torch device
        
    Returns:
        dict with ler, err, samples
    """
    circuit = _build_stim_memory_circuit(distance, rounds, p)
    sampler = circuit.compile_sampler()
    m2d = circuit.compile_m2d_converter()
    
    correct = 0
    total = 0
    batch_size = min(1024, shots)
    n_batches = max(1, shots // batch_size)
    
    for batch_idx in range(n_batches):
        # Sample measurements and convert to detectors + observables
        meas = sampler.sample(shots=batch_size)
        dets, obs = m2d.convert(meas, separate_observables=True)
        
        dets_t = torch.from_numpy(dets.astype(np.float32)).to(device)
        
        with torch.no_grad():
            obs_logits = model.forward_obs(dets_t)
            pred_obs = (torch.sigmoid(obs_logits) > 0.5).cpu().numpy().astype(np.uint8)
        
        # Compare predictions to ground truth
        true_obs = obs.astype(np.uint8)
        if pred_obs.shape[1] < true_obs.shape[1]:
            true_obs = true_obs[:, :pred_obs.shape[1]]
        elif pred_obs.shape[1] > true_obs.shape[1]:
            pred_obs = pred_obs[:, :true_obs.shape[1]]
        
        correct += np.sum(pred_obs[:, 0] == true_obs[:, 0])
        total += len(true_obs)
    
    ler = 1.0 - (correct / total) if total > 0 else 1.0
    err = np.sqrt(ler * (1 - ler) / total) if total > 0 else 0
    return {"ler": ler, "err": err, "samples": total}


def evaluate_mwpm_circuit_level(p, distance, rounds, shots):
    """Evaluate MWPM (PyMatching) on circuit-level Stim samples.
    
    Uses PyMatching directly on the detector error model for optimal MWPM.
    This is the fairest comparison for circuit-level MGHD.
    """
    try:
        import pymatching
    except ImportError:
        raise RuntimeError("pymatching required for circuit-level MWPM evaluation")
    
    circuit = _build_stim_memory_circuit(distance, rounds, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    
    sampler = circuit.compile_detector_sampler()
    
    correct = 0
    total = 0
    batch_size = min(4096, shots)
    n_batches = max(1, shots // batch_size)
    
    for batch_idx in range(n_batches):
        dets, obs = sampler.sample(shots=batch_size, separate_observables=True)
        
        # PyMatching decoding
        pred_obs = matcher.decode_batch(dets)
        
        # Compare
        true_obs = obs.astype(np.uint8)
        if pred_obs.ndim == 1:
            pred_obs = pred_obs.reshape(-1, 1)
        
        correct += np.sum(pred_obs[:, 0] == true_obs[:, 0])
        total += len(true_obs)
    
    ler = 1.0 - (correct / total) if total > 0 else 1.0
    err = np.sqrt(ler * (1 - ler) / total) if total > 0 else 0
    return {"ler": ler, "err": err, "samples": total}


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
    parser.add_argument("--rounds", type=int, default=None, 
                        help="Syndrome rounds for circuit-level (default: distance)")
    parser.add_argument("--shots", type=int, default=2000, help="Shots per p value")
    parser.add_argument("--output", type=str, default="mghd_vs_mwpm.png")
    parser.add_argument("--circuit-level", action="store_true",
                        help="Use circuit-level evaluation (obs_head prediction)")
    parser.add_argument("--p-values", type=str, default="0.001,0.003,0.007,0.01",
                        help="Comma-separated physical error rates to evaluate")
    args = parser.parse_args()
    
    p_values = [float(p.strip()) for p in args.p_values.split(",")]
    rounds = args.rounds if args.rounds is not None else args.distance
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, metadata = load_mghd_model(args.model, device, circuit_level=args.circuit_level)
    
    # Auto-detect circuit-level mode from checkpoint
    is_circuit_level = args.circuit_level or metadata.get("mode") == "circuit_level"
    
    if is_circuit_level:
        print(f"Mode: CIRCUIT-LEVEL (observable prediction)")
        print(f"  num_detectors: {metadata.get('num_detectors')}")
        print(f"  rounds: {rounds}")
        
        if not _HAVE_STIM:
            raise RuntimeError("stim package required for circuit-level evaluation")
        
        # Evaluate MWPM baseline (PyMatching on DEM)
        print("\n=== Evaluating MWPM baseline (PyMatching on DEM) ===")
        mwpm_results = {}
        for p in p_values:
            result = evaluate_mwpm_circuit_level(p, args.distance, rounds, args.shots)
            mwpm_results[p] = result
            print(f"  p={p:.4f}: MWPM LER={result['ler']:.4f} ± {result['err']:.4f}")
        
        # Evaluate MGHD model
        print("\n=== Evaluating MGHD model (obs_head) ===")
        mghd_results = {}
        for p in p_values:
            print(f"  Evaluating p={p:.4f}...", end=" ", flush=True)
            result = evaluate_mghd_circuit_level(model, p, args.distance, rounds, 
                                                  args.shots, device)
            mghd_results[p] = result
            print(f"MGHD LER={result['ler']:.4f} ± {result['err']:.4f}")
        
        title_suffix = "(Circuit-Level)"
    else:
        print(f"Mode: PER-QUBIT (phenomenological)")
        
        # Load code for per-qubit evaluation
        from mghd.codes.registry import get_code
        code = get_code("surface", distance=args.distance)
        
        # Evaluate MWPM baseline
        print("\n=== Evaluating MWPM baseline ===")
        mwpm_results = {}
        for p in p_values:
            result = evaluate_mwpm(code, p, args.distance, args.shots)
            mwpm_results[p] = result
            print(f"  p={p:.4f}: MWPM LER={result['ler']:.4f} ± {result['err']:.4f}")
        
        # Evaluate MGHD model
        print("\n=== Evaluating MGHD model ===")
        mghd_results = {}
        for p in p_values:
            print(f"  Evaluating p={p:.4f}...", end=" ", flush=True)
            result = evaluate_mghd(model, code, p, args.distance, args.shots, device)
            mghd_results[p] = result
            ler_str = f"MGHD LER={result['ler']:.4f}"
            if "err" in result:
                ler_str += f" ± {result['err']:.4f}"
            print(f"{ler_str} ({result['samples']} samples)")
        
        title_suffix = "(Phenomenological)"
    
    # Plot
    print("\nPlotting...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mwpm_lers = [mwpm_results[p]["ler"] for p in p_values]
    mwpm_errs = [mwpm_results[p].get("err", 0) for p in p_values]
    mghd_lers = [mghd_results[p]["ler"] for p in p_values]
    mghd_errs = [mghd_results[p].get("err", 0) for p in p_values]
    
    ax.errorbar(p_values, mwpm_lers, yerr=mwpm_errs, 
                fmt='o-', label='MWPM', linewidth=2, markersize=8, capsize=4, color='blue')
    ax.errorbar(p_values, mghd_lers, yerr=mghd_errs,
                fmt='s-', label='MGHD (trained)', linewidth=2, markersize=8, 
                capsize=4, color='red')
    
    ax.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax.set_title(f'MGHD vs MWPM - Surface Code d={args.distance} {title_suffix}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(p_values) * 1.2)
    
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
