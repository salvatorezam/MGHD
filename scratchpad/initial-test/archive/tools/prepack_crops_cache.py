"""Convert syndrome-level hard crop caches into pre-packed bucket batches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from mghd_clustered.cluster_core import active_components, extract_subproblem, gf2_nullspace
from mghd_clustered.pcm_real import rotated_surface_pcm
from mghd_public.features_v2 import pack_cluster, infer_bucket_id
from tools.bench_clustered_sweep_surface import ensure_dir, _parse_bucket_spec


def _format_prob_tag(d: int, p: float, side: str) -> str:
    return f"d{d}_p{p:.6f}_{side.upper()}"


def _compute_qubit_coords(H: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    n_qubits = H.shape[1]
    coords_q: List[Tuple[float, float]] = []
    if d * d == n_qubits:
        for r in range(d):
            for c in range(d):
                coords_q.append((float(r + c), float(r - c)))
    else:
        coords_q = [(float(i), 0.0) for i in range(n_qubits)]
    coords_q_arr = np.asarray(coords_q, dtype=np.float32)

    coords_c = np.zeros((H.shape[0], 2), dtype=np.float32)
    for idx in range(H.shape[0]):
        qubits = np.nonzero(H[idx])[0]
        if qubits.size:
            coords_c[idx] = coords_q_arr[qubits].mean(axis=0)
    return coords_q_arr, coords_c


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def pack_syndromes(
    syndromes: np.ndarray,
    *,
    H: np.ndarray,
    H_sp,
    coords_q: np.ndarray,
    coords_c: np.ndarray,
    bucket_spec: List[Tuple[int, int, int]],
    side: str,
    d: int,
    p: float,
) -> Dict[int, List[Dict[str, np.ndarray]]]:
    bucket_map: Dict[int, List[Dict[str, np.ndarray]]] = {}
    bucket_spec_tuple = tuple(bucket_spec)

    for s in syndromes:
        checks_list, qubits_list = active_components(H_sp, s, halo=0)
        for ci, qi in zip(checks_list, qubits_list):
            H_sub, s_sub, q_l2g, c_l2g = extract_subproblem(H_sp, s, ci, qi)
            xy_qubit = np.round(coords_q[q_l2g]).astype(np.int32)
            xy_check = np.round(coords_c[c_l2g]).astype(np.int32)
            all_coords = np.vstack([xy_qubit, xy_check]) if xy_check.size else xy_qubit
            if all_coords.size == 0:
                bbox = (0, 0, 1, 1)
            else:
                mins = np.floor(all_coords.min(axis=0)).astype(int)
                maxs = np.ceil(all_coords.max(axis=0)).astype(int)
                bbox = (
                    int(mins[0]),
                    int(mins[1]),
                    int(maxs[0] - mins[0] + 1),
                    int(maxs[1] - mins[1] + 1),
                )

            n_qubits = int(q_l2g.size)
            kappa_stats = {
                "size": int(H_sub.shape[0] + H_sub.shape[1]),
                "density": float(H_sub.nnz) / max(1.0, float(n_qubits * H_sub.shape[0])),
                "syndrome_weight": int(np.asarray(s_sub, dtype=np.uint8).sum()),
            }
            y_bits = np.zeros(n_qubits, dtype=np.uint8)
            crop = pack_cluster(
                H_sub=H_sub.toarray(),
                xy_qubit=xy_qubit,
                xy_check=xy_check,
                synd_Z_then_X_bits=np.asarray(s_sub, dtype=np.uint8),
                k=n_qubits,
                r=int(gf2_nullspace(H_sub).shape[1]),
                bbox_xywh=bbox,
                kappa_stats=kappa_stats,
                y_bits_local=y_bits,
                side=side,
                d=d,
                p=p,
                seed=0,
                N_max=None,
                E_max=None,
                S_max=None,
                bucket_spec=bucket_spec_tuple,
                add_jump_edges=False,
            )
            bucket_id = crop.meta.bucket_id
            bucket_map.setdefault(bucket_id, []).append({
                "x_nodes": _to_np(crop.x_nodes),
                "node_mask": _to_np(crop.node_mask),
                "node_type": _to_np(crop.node_type),
                "edge_index": _to_np(crop.edge_index),
                "edge_attr": _to_np(crop.edge_attr),
                "edge_mask": _to_np(crop.edge_mask),
                "seq_idx": _to_np(crop.seq_idx),
                "seq_mask": _to_np(crop.seq_mask),
                "g_token": _to_np(crop.g_token),
            })
    return bucket_map


def _stack_bucket(entries: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    B = len(entries)
    N_max = entries[0]["x_nodes"].shape[0]
    E_max = entries[0]["edge_index"].shape[1]
    S_max = entries[0]["seq_idx"].shape[0]
    feat_dim = entries[0]["x_nodes"].shape[1]
    edge_feat_dim = entries[0]["edge_attr"].shape[1]
    g_dim = entries[0]["g_token"].shape[0]

    x_nodes = np.zeros((B, N_max, feat_dim), dtype=np.float32)
    node_mask = np.zeros((B, N_max), dtype=bool)
    node_type = np.zeros((B, N_max), dtype=np.int8)
    edge_index = np.zeros((B, 2, E_max), dtype=np.int64)
    edge_attr = np.zeros((B, E_max, edge_feat_dim), dtype=np.float32)
    edge_mask = np.zeros((B, E_max), dtype=bool)
    seq_idx = np.zeros((B, S_max), dtype=np.int64)
    seq_mask = np.zeros((B, S_max), dtype=bool)
    g_token = np.zeros((B, g_dim), dtype=np.float32)

    for i, entry in enumerate(entries):
        x_nodes[i] = entry["x_nodes"].astype(np.float32, copy=False)
        node_mask[i] = entry["node_mask"].astype(bool, copy=False)
        node_type[i] = entry["node_type"].astype(np.int8, copy=False)
        g_token[i] = entry["g_token"].astype(np.float32, copy=False)

        edge_index[i] = entry["edge_index"].astype(np.int64, copy=False)
        edge_attr[i] = entry["edge_attr"].astype(np.float32, copy=False)
        edge_mask[i] = entry["edge_mask"].astype(bool, copy=False)

        seq_idx[i] = entry["seq_idx"].astype(np.int64, copy=False)
        seq_mask[i] = entry["seq_mask"].astype(bool, copy=False)

    return {
        "x_nodes": x_nodes,
        "node_mask": node_mask,
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mask": edge_mask,
        "seq_idx": seq_idx,
        "seq_mask": seq_mask,
        "g_token": g_token,
        "batch_size": B,
        "nodes_pad": N_max,
        "edges_pad": E_max,
        "seq_pad": S_max,
        "feat_dim": feat_dim,
        "edge_feat_dim": edge_feat_dim,
        "g_dim": g_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack cached hard crops into bucket NPZs")
    parser.add_argument("--in", dest="in_root", required=True, help="Input cache directory (syndrome-level)")
    parser.add_argument("--out", dest="out_root", required=True, help="Output directory for packed cache")
    parser.add_argument("--buckets", type=str, required=True, help="Semicolon-separated bucket ceilings")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bucket_spec = _parse_bucket_spec(args.buckets)
    ensure_dir(args.out_root)

    rng = np.random.default_rng(args.seed)

    manifest: Dict[str, List[Dict[str, object]]] = {}

    in_root = Path(args.in_root)
    files = sorted(in_root.glob("d*_p*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ cache files found in {in_root}")

    cache_coords: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for npz_path in files:
        name = npz_path.stem
        parts = name.split("_")
        d = int(parts[0][1:])
        p = float(parts[1][1:])
        side = parts[2]
        tag = _format_prob_tag(d, p, side)

        with np.load(npz_path) as data:
            if "syndromes" not in data:
                print(f"[skip] {npz_path} missing 'syndromes'")
                continue
            syndromes = data["syndromes"]
        if syndromes.size == 0:
            print(f"[skip] {npz_path} empty syndromes")
            continue

        print(f"Packing {npz_path} ({syndromes.shape[0]} syndromes)")

        H = rotated_surface_pcm(d, side)
        if d not in cache_coords:
            coords_q, coords_c = _compute_qubit_coords(H.toarray(), d)
            cache_coords[d] = (coords_q, coords_c)
        else:
            coords_q, coords_c = cache_coords[d]

        bucket_map = pack_syndromes(
            syndromes,
            H=H.toarray(),
            H_sp=H,
            coords_q=coords_q,
            coords_c=coords_c,
            bucket_spec=bucket_spec,
            side=side,
            d=d,
            p=p,
        )

        if not bucket_map:
            print(f"  [warn] no clusters met bucket spec for {tag}")
            continue

        out_dir = Path(args.out_root) / f"d{d}"
        ensure_dir(out_dir)

        entries = []
        for bucket_id, crops in bucket_map.items():
            sample = crops[0]
            N_max = sample["x_nodes"].shape[0]
            E_max = sample["edge_index"].shape[1]
            S_max = sample["seq_idx"].shape[0]
            packed = _stack_bucket(crops)
            out_path = out_dir / f"p{p:.6f}_{side}_bucket_{N_max}_{E_max}_{S_max}.npz"
            np.savez_compressed(out_path, **packed, bucket_id=bucket_id, count=len(crops), d=d, p=p, side=side)
            entries.append({
                "bucket_id": bucket_id,
                "bucket": [N_max, E_max, S_max],
                "path": str(out_path.relative_to(args.out_root)),
                "count": len(crops),
            })
            print(f"  bucket {bucket_id}: {len(crops)} crops -> {out_path}")

        manifest[tag] = entries

    manifest_path = Path(args.out_root) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"entries": manifest}, f, indent=2)
    print(f"WROTE manifest {manifest_path}")


if __name__ == "__main__":
    main()
