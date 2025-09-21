from __future__ import annotations
import os, json, time, numpy as np, scipy.sparse as sp
from argparse import ArgumentParser
from typing import Dict
from mghd_public.config import MGHDConfig
from mghd_public.infer import MGHDDecoderPublic
from mghd_clustered.clustered_primary import MGHDPrimaryClustered


def _bucket_size(size: int) -> str:
    bucket = 1
    while size > bucket:
        bucket *= 2
    return str(bucket)


def _aggregate_mb_stats(reports):
    agg = dict(
        shots=len(reports),
        fast_path_batches=0,
        fixed_d3_batches=0,
        fallback_loops=0,
        graph_used_shots=0,
    )
    hist: Dict[str, int] = {}
    total = 0.0
    count = 0
    device = None
    for rep in reports:
        if not rep:
            continue
        agg["fast_path_batches"] += int(rep.get("fast_path_batches", 0))
        agg["fixed_d3_batches"] += int(rep.get("fixed_d3_batches", 0))
        agg["fallback_loops"] += int(rep.get("fallback_loops", 0))
        if rep.get("graph_used"):
            agg["graph_used_shots"] += 1
        sizes = rep.get("batch_sizes", [])
        total += float(np.sum(sizes)) if len(sizes) else 0.0
        count += len(sizes)
        for size in sizes:
            bucket = _bucket_size(int(size))
            hist[bucket] = hist.get(bucket, 0) + 1
        if device is None and rep.get("device"):
            device = rep["device"]
    agg["avg_batch_size"] = (total / count) if count else 0.0
    agg["batch_histogram"] = {k: int(v) for k, v in sorted(hist.items(), key=lambda kv: int(kv[0]))}
    agg["graph_used"] = bool(agg["graph_used_shots"])
    agg["device"] = device or {}
    return agg

def load_d3_pack(path="student_pack_p003.npz"):
    pack = np.load(path)
    return sp.csr_matrix(pack["Hx"]), sp.csr_matrix(pack["Hz"])

def sample_bsc(H: sp.csr_matrix, p: float, rng):
    n = H.shape[1]
    e = (rng.random(n) < p).astype(np.uint8)
    s = (H @ e) % 2
    return e, s.astype(np.uint8)

def run_side(H, dec: MGHDPrimaryClustered, p: float, shots: int, seed: int):
    rng = np.random.default_rng(seed)
    times=[]; clus=[]; proj=[]; mghd=[]; fails=0
    mb_reports = []
    tier0_clusters = []
    tier0_qubits = []
    mghd_clusters = []
    mghd_invoked_flags = []
    p_channels = []
    total_clusters = []
    for _ in range(shots):
        e_true, s = sample_bsc(H,p,rng)
        out = dec.decode(s)
        e_hat = out["e_hat"]
        # parity success
        fails += int(not np.array_equal((H @ e_hat) % 2, s))
        times.append(out["t_total_ms"]); clus.append(out["t_cluster_ms"])
        proj.append(out["t_project_ms"]); mghd.append(out["t_mghd_ms"])
        mb_reports.append(out.get("mb_stats", {}))
        tier0_clusters.append(int(out.get("tier0_clusters", 0)))
        tier0_qubits.append(int(out.get("tier0_qubits", 0)))
        mghd_clusters.append(int(out.get("mghd_clusters", 0)))
        mghd_invoked_flags.append(bool(out.get("mghd_invoked", False)))
        p_channels.append(out.get("p_channel_used"))
        total_clusters.append(int(out.get("n_clusters", 0)))
    def summarize(a):
        arr = np.asarray(a, dtype=np.float64)
        stats = {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
        mask = arr > 0
        stats["count_nonzero"] = int(mask.sum())
        stats["mean_nonzero"] = float(np.mean(arr[mask])) if mask.any() else 0.0
        return stats
    total_clusters_sum = int(np.sum(total_clusters))
    tier0_total = int(np.sum(tier0_clusters))
    mghd_total = int(np.sum(mghd_clusters))
    tier0_pct = (100.0 * tier0_total / total_clusters_sum) if total_clusters_sum else 0.0
    tier0_stats = dict(
        total_clusters=total_clusters_sum,
        tier0_clusters=tier0_total,
        tier0_qubits=int(np.sum(tier0_qubits)),
        tier0_pct=tier0_pct,
        mghd_clusters=mghd_total,
        mghd_invoked_shots=int(np.sum(mghd_invoked_flags)),
        mghd_invoked=bool(mghd_total),
        p_channel_used=next((x for x in p_channels if x is not None), None),
    )

    return dict(
        shots=shots,
        failures=int(fails),
        latency_total_ms=summarize(times),
        t_cluster_ms=summarize(clus),
        t_project_ms=summarize(proj),
        t_mghd_ms=summarize(mghd),
        mb_stats=_aggregate_mb_stats(mb_reports),
        tier0_stats=tier0_stats,
    )

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--halo", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--r-cap", type=int, default=20)
    ap.add_argument("--unbatched", action="store_true",
                    help="Force legacy single-cluster MGHD inference for the main run")
    ap.add_argument("--tier0", dest="tier0", action="store_true", default=True,
                    help="Enable Tier-0 small cluster solver (default)")
    ap.add_argument("--no-tier0", dest="tier0", action="store_false",
                    help="Disable Tier-0 small cluster solver")
    ap.add_argument("--tier0-k-max", type=int, default=3,
                    help="Maximum qubits for Tier-0 enumeration")
    ap.add_argument("--tier0-r-max", type=int, default=6,
                    help="Maximum nullity for Tier-0 enumeration")
    ap.add_argument("--p-channel", type=float, default=None,
                    help="Channel prior p used by Tier-0 solver (defaults to --p)")
    args = ap.parse_args()

    Hx, Hz = load_d3_pack()

    cfg = MGHDConfig(
        gnn={"dist":3,"n_node_inputs":9,"n_node_outputs":9,"n_iters":7,
             "n_node_features":128,"n_edge_features":128,
             "msg_net_size":96,"msg_net_dropout_p":0.04,"gru_dropout_p":0.11},
        mamba={"d_model":192,"d_state":32,"d_conv":2,"expand":3,
               "attention_mechanism":"channel_attention","se_reduction":4,
               "post_mamba_ln":False},
        n_checks=8, n_qubits=9, n_node_inputs=9, n_node_outputs=2
    )
    dec_pub = MGHDDecoderPublic(args.ckpt, cfg, device="cuda", graph_capture=True)
    dec_pub.bind_code(Hx, Hz)  # Bind the matrices to the public decoder
    p_channel = args.p_channel if args.p_channel is not None else args.p
    decX = MGHDPrimaryClustered(
        Hx,
        dec_pub,
        halo=args.halo,
        thresh=args.thresh,
        temp=args.temp,
        r_cap=args.r_cap,
        batched=not args.unbatched,
        tier0_enable=args.tier0,
        tier0_k_max=args.tier0_k_max,
        tier0_r_max=args.tier0_r_max,
        p_channel=p_channel,
        default_p=args.p,
    )
    decZ = MGHDPrimaryClustered(
        Hz,
        dec_pub,
        halo=args.halo,
        thresh=args.thresh,
        temp=args.temp,
        r_cap=args.r_cap,
        batched=not args.unbatched,
        tier0_enable=args.tier0,
        tier0_k_max=args.tier0_k_max,
        tier0_r_max=args.tier0_r_max,
        p_channel=p_channel,
        default_p=args.p,
    )

    if args.unbatched:
        decX.mb_mode = "unbatched"
        decZ.mb_mode = "unbatched"

    out = dict(
        X = run_side(Hx, decX, p=args.p, shots=args.shots, seed=123),
        Z = run_side(Hz, decZ, p=args.p, shots=args.shots, seed=456),
        params = dict(halo=args.halo, thresh=args.thresh, temp=args.temp, r_cap=args.r_cap),
        shots=args.shots,
        p=args.p,
        mode="unbatched" if args.unbatched else "batched",
    )
    os.makedirs("results", exist_ok=True)
    base = f"results/mghd_primary_clustered_d3_p{args.p:.3f}"
    if args.unbatched:
        path = f"{base}_ab_unbatched.json"
    else:
        path = f"{base}.json"
    with open(path,"w") as f: json.dump(out,f,indent=2)
    print("WROTE", path)
    
    # Print crisp summary
    for side in ['X', 'Z']:
        data = out[side]
        fail_rate = data['failures'] / data['shots']
        t = data['latency_total_ms']
        print(f"{side}: {data['failures']}/{data['shots']} fails ({fail_rate:.3f}), "
              f"Total p50/p95/p99: {t['p50']:.3f}/{t['p95']:.3f}/{t['p99']:.3f} ms, "
              f"Means: cluster={data['t_cluster_ms']['mean']:.3f}, "
              f"mghd={data['t_mghd_ms']['mean']:.3f}, "
              f"project={data['t_project_ms']['mean']:.3f} ms")
        tier = data.get("tier0_stats", {})
        if tier:
            total = tier.get("total_clusters", 0)
            tier0 = tier.get("tier0_clusters", 0)
            pct = tier.get("tier0_pct", 0.0)
            mghd_c = tier.get("mghd_clusters", 0)
            invoked = tier.get("mghd_invoked_shots", 0)
            print(
                f"    Tier-0: {tier0} clusters ({pct:.1f}% of {total}), "
                f"MGHD clusters: {mghd_c}, MGHD invoked shots: {invoked}"
            )

    # Micro-batching comparison on a short run for speedup visibility
    original_modes = (decX.mb_mode, decZ.mb_mode)
    print("\n=== Micro-batching A/B (100 shots) ===")
    ab_results = {}
    for mode in ["unbatched", "batched"]:
        decX.mb_mode = mode
        decZ.mb_mode = mode
        outX = run_side(Hx, decX, p=args.p, shots=100, seed=321)
        outZ = run_side(Hz, decZ, p=args.p, shots=100, seed=654)
        total_mean = outX['latency_total_ms']['mean'] + outZ['latency_total_ms']['mean']
        print(
            f"{mode}: X t_mghd mean={outX['t_mghd_ms']['mean']:.3f} ms, "
            f"Z t_mghd mean={outZ['t_mghd_ms']['mean']:.3f} ms, total mean={total_mean:.3f} ms"
        )
        ab_results[mode] = dict(X=outX, Z=outZ, total_mean=total_mean)
        ab_path = f"{base}_ab_{mode}.json"
        ab_payload = dict(
            X=outX,
            Z=outZ,
            params=dict(halo=args.halo, thresh=args.thresh, temp=args.temp, r_cap=args.r_cap),
            shots=100,
            p=args.p,
            mode=mode,
        )
        with open(ab_path, "w") as f:
            json.dump(ab_payload, f, indent=2)
        print("WROTE", ab_path)

    if {"batched", "unbatched"}.issubset(ab_results.keys()):
        total_batched = ab_results["batched"]["total_mean"]
        total_unbatched = ab_results["unbatched"]["total_mean"]
        if total_batched > 0:
            speedup = total_unbatched / total_batched
            print(f"Speedup (unbatched/batched total mean): ×{speedup:.3f}")

    decX.mb_mode, decZ.mb_mode = original_modes
