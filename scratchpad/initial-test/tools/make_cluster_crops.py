# NOTE: Initialize CUDA/CUDA-Q only in main().
from __future__ import annotations
import argparse, os, sys, json, time, hashlib
from pathlib import Path
import numpy as np

from mghd_public.features_v2 import pack_cluster
from teachers.ensemble import get_teacher_label, TeacherOut
from mghd_clustered import cluster_core as cc
from mghd_clustered.garnet_adapter import sample_round, split_components_for_side
from teachers.mwpf_ctx import MWPFContext
from teachers.mwpm_ctx import MWPMatchingContext

def short_sha(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        if isinstance(o, (bytes, bytearray)):
            h.update(o)
        elif isinstance(o, str):
            h.update(o.encode())
        else:
            h.update(repr(o).encode())
    return h.hexdigest()[:8]

def run(args):
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Contexts for teachers (instantiate inside run(), not at import)
    mwpf_ctx = MWPFContext()
    mwpm_ctx = MWPMatchingContext()
    
    shots_per = args.shots_per_grid
    padN, padE, padS = args.N_max, args.E_max, args.S_max
    
    # Track all generated shards
    manifest = []
    
    print(f"Generating crops with MGHD_SYNTHETIC={os.getenv('MGHD_SYNTHETIC', '0')}")

    for d in args.dists:
        for p in args.ps:
            print(f"Processing d={d}, p={p:.5f}")
            shard_items = []
            for shot in range(shots_per):
                seed = int(rng.integers(0, 2**31-1))
                
                # ---- Sample via CUDA-Q Garnet (inside this call) ----
                S = sample_round(d=d, p=p, seed=seed)
                
                for side in ("Z","X"):
                    # Filter out dem_meta from S to avoid unexpected keyword argument
                    split_args = {k: v for k, v in S.items() if k != 'dem_meta'}
                    comp_list = split_components_for_side(side=side, **split_args)
                    
                    for comp in comp_list:
                        H_sub      = comp["H_sub"]
                        xy_qubit   = comp["xy_qubit"]
                        xy_check   = comp["xy_check"]
                        synd_bits  = comp["synd_bits"]
                        bbox       = tuple(int(v) for v in comp["bbox_xywh"])
                        k          = int(comp["k"])
                        r          = int(comp["r"])
                        kappa_stats= comp.get("kappa_stats", {"size": int(H_sub.shape[0]+H_sub.shape[1])})
                        
                        # ---- teacher ensemble ----
                        t_out: TeacherOut = get_teacher_label(
                            H_sub=H_sub, synd_bits=synd_bits, side=side,
                            mwpf_ctx=mwpf_ctx, mwpm_ctx=mwpm_ctx, dem_meta=S.get("dem_meta", None)
                        )
                        y_bits = t_out.bits
                        
                        # ---- pack ----
                        packed = pack_cluster(
                            H_sub=H_sub, xy_qubit=xy_qubit, xy_check=xy_check,
                            synd_Z_then_X_bits=synd_bits,
                            k=k, r=r, bbox_xywh=bbox, kappa_stats=kappa_stats,
                            y_bits_local=y_bits, side=side, d=d, p=p, seed=seed,
                            N_max=padN, E_max=padE, S_max=padS
                        )
                        
                        # save one item dict for NPZ
                        item = {
                            "x_nodes": packed.x_nodes.numpy(),
                            "node_mask": packed.node_mask.numpy(),
                            "node_type": packed.node_type.numpy(),
                            "edge_index": packed.edge_index.numpy(),
                            "edge_attr": packed.edge_attr.numpy(),
                            "edge_mask": packed.edge_mask.numpy(),
                            "seq_idx": packed.seq_idx.numpy(),
                            "seq_mask": packed.seq_mask.numpy(),
                            "g_token": packed.g_token.numpy(),
                            "y_bits": packed.y_bits.numpy(),
                            "meta": {**packed.meta.__dict__, "side": side},
                            "H_sub": H_sub.astype(np.uint8),  # parity/projection
                            "idx_data_local": np.arange(H_sub.shape[1], dtype=np.int32),
                            "idx_check_local": np.arange(H_sub.shape[0], dtype=np.int32),
                            # Teacher metadata for training
                            "teacher": t_out.teacher,
                            "teacher_weight": int(t_out.weight),
                            "teacher_valid": bool(t_out.valid),
                            "teacher_matched_local_ml": bool(t_out.matched_local_ml),
                        }
                        shard_items.append(item)
            
            # deterministic shard id (order-independent)
            shard_sha = short_sha(d, f"{p:.5f}", args.seed, len(shard_items))
            outp = out_root / f"crops_d{d}_p{p:.5f}_seed{args.seed}_{shard_sha}.npz"
            np.savez_compressed(outp, packed=np.array(shard_items, dtype=object))
            manifest.append({"file": str(outp), "d": d, "p": p, "seed": args.seed, "count": len(shard_items), "sha": shard_sha})
            print(json.dumps({"written": str(outp), "count": len(shard_items), "sha": shard_sha}))
    
    # write manifest
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", type=int, nargs="+", required=True)
    ap.add_argument("--ps", type=float, nargs="+", required=True)
    ap.add_argument("--shots-per-grid", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--N_max", type=int, default=512)
    ap.add_argument("--E_max", type=int, default=4096)
    ap.add_argument("--S_max", type=int, default=512)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()