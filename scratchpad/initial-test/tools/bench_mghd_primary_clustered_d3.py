from __future__ import annotations
import os, json, time, numpy as np, scipy.sparse as sp
from argparse import ArgumentParser
from mghd_public.config import MGHDConfig
from mghd_public.infer import MGHDDecoderPublic
from mghd_clustered.clustered_primary import MGHDPrimaryClustered

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
    for _ in range(shots):
        e_true, s = sample_bsc(H,p,rng)
        out = dec.decode(s)
        e_hat = out["e_hat"]
        # parity success
        fails += int(not np.array_equal((H @ e_hat) % 2, s))
        times.append(out["t_total_ms"]); clus.append(out["t_cluster_ms"])
        proj.append(out["t_project_ms"]); mghd.append(out["t_mghd_ms"])
    A = lambda a: dict(mean=float(np.mean(a)), p50=float(np.percentile(a,50)),
                       p95=float(np.percentile(a,95)), p99=float(np.percentile(a,99)))
    return dict(
        shots=shots,
        failures=int(fails),
        latency_total_ms=A(times),
        t_cluster_ms=A(clus),
        t_project_ms=A(proj),
        t_mghd_ms=A(mghd),
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
    dec_pub = MGHDDecoderPublic(args.ckpt, cfg, device="cuda")
    dec_pub.bind_code(Hx, Hz)  # Bind the matrices to the public decoder
    decX = MGHDPrimaryClustered(Hx, dec_pub, halo=args.halo, thresh=args.thresh, temp=args.temp, r_cap=args.r_cap)
    decZ = MGHDPrimaryClustered(Hz, dec_pub, halo=args.halo, thresh=args.thresh, temp=args.temp, r_cap=args.r_cap)

    out = dict(
        X = run_side(Hx, decX, p=args.p, shots=args.shots, seed=123),
        Z = run_side(Hz, decZ, p=args.p, shots=args.shots, seed=456),
        params = dict(halo=args.halo, thresh=args.thresh, temp=args.temp, r_cap=args.r_cap)
    )
    os.makedirs("results", exist_ok=True)
    path = f"results/mghd_primary_clustered_d3_p{args.p:.3f}.json"
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