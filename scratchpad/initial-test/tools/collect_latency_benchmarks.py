#!/usr/bin/env python3
"""
Latency benchmark collector for MGHD profiles (S/M/L) and fastpath persistent LUT.

Outputs a single JSON at reports/latency_benchmark.json with per-backend metrics.

Notes:
- Runs in the active conda env (mlqec-env).
- Uses H100 GPU if available; guards CUDA sync for stable measurements.
- Batches: single-shot (B=1) and throughput (B=256) for MGHD eager.
- Fastpath: persistent decode single-shot (repeated) and small batch.
"""

from __future__ import annotations

import json
import time
import statistics
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from poc_my_models import MGHD

# Optional backends
try:
    import onnxruntime as ort  # type: ignore
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

try:
    import tensorrt as trt  # type: ignore
    _HAS_TRT = True
except Exception:
    _HAS_TRT = False


def _bench_decode_one(model: MGHD, repeats=1000, warmup=200, device="cuda") -> Dict[str, float]:
    synd = torch.randint(0, 256, (1,), dtype=torch.uint8, device=device)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.decode_one(synd, device=device)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.decode_one(synd, device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return {
        "p50_us": statistics.median(times),
        "p99_us": float(np.percentile(times, 99)),
        "mean_us": float(np.mean(times)),
    }


def _bench_decode_one_torchscript(model: MGHD, repeats=1000, warmup=200, device="cuda") -> Dict[str, float]:
    # Wrap decode_one for tracing with fixed device
    class Wrapper(torch.nn.Module):
        def __init__(self, m: MGHD):
            super().__init__()
            self.m = m
        def forward(self, s: torch.Tensor):
            return self.m.decode_one(s, device=device)
    w = Wrapper(model)
    dummy = torch.randint(0, 256, (1,), dtype=torch.uint8, device=device)
    try:
        ts = torch.jit.trace(w, (dummy,)).eval()
    except Exception:
        return {"error": "trace_failed"}
    # Warm
    for _ in range(warmup):
        with torch.no_grad():
            _ = ts(dummy)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter();
        with torch.no_grad(): _ = ts(dummy)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return {
        "p50_us": statistics.median(times),
        "p99_us": float(np.percentile(times, 99)),
        "mean_us": float(np.mean(times)),
    }


def _bench_decode_one_cudagraph(model: MGHD, repeats=1000, warmup=50, device="cuda") -> Dict[str, float]:
    """Capture the core forward path with CUDAGraph for B=1.
    Note: We capture model.forward with prebuilt node_inputs and static indices,
    avoiding Python asserts in decode_one which are not capture-safe.
    """
    num_check_nodes = 8
    num_qubit_nodes = 9
    nodes = num_check_nodes + num_qubit_nodes
    node_inputs = torch.zeros(1, nodes, 9, device=device, dtype=torch.float32)
    node_inputs[:, :num_check_nodes, 0] = torch.randint(0, 2, (1, num_check_nodes), device=device).float()
    flat = node_inputs.view(-1, 9)
    model._ensure_static_indices(device)
    src, dst = model._src_ids, model._dst_ids
    # Warm
    for _ in range(warmup):
        with torch.no_grad(): _ = model(flat, src, dst)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = model(flat, src, dst)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return {
        "p50_us": statistics.median(times),
        "p99_us": float(np.percentile(times, 99)),
        "mean_us": float(np.mean(times)),
    }


def _bench_forward_batch(model: MGHD, B=256, repeats=200, warmup=50, device="cuda") -> Dict[str, float]:
    # Build a batch of packed syndromes (we construct node_inputs and call forward directly)
    num_check_nodes = 8
    num_qubit_nodes = 9
    nodes = num_check_nodes + num_qubit_nodes
    # Random syndromes -> place into first feature channel of check-node slots
    node_inputs = torch.zeros(B, nodes, 9, device=device, dtype=torch.float32)
    node_inputs[:, :num_check_nodes, 0] = torch.randint(0, 2, (B, num_check_nodes), device=device, dtype=torch.int32).float()
    flat = node_inputs.view(-1, 9)
    # Ensure indices exist
    model._ensure_static_indices(device)
    src, dst = model._src_ids, model._dst_ids
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(flat, src, dst)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(flat, src, dst)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6 / B)  # us per shot
    return {
        "p50_us_per_shot": statistics.median(times),
        "p99_us_per_shot": float(np.percentile(times, 99)),
        "mean_us_per_shot": float(np.mean(times)),
        "batch_size": B,
    }


def _build_profile(name: str) -> MGHD:
    if name == "S":
        gnn = dict(dist=3, n_node_inputs=9, n_node_outputs=9,
                   n_iters=7, n_node_features=128, n_edge_features=128,
                   msg_net_size=96, msg_net_dropout_p=0.04, gru_dropout_p=0.11)
        m = dict(d_model=192, d_state=32, d_conv=2, expand=3,
                 attention_mechanism='channel_attention', se_reduction=4)
    elif name == "M":
        gnn = dict(dist=3, n_node_inputs=9, n_node_outputs=9,
                   n_iters=8, n_node_features=192, n_edge_features=192,
                   msg_net_size=128, msg_net_dropout_p=0.04, gru_dropout_p=0.11)
        m = dict(d_model=256, d_state=48, d_conv=2, expand=3,
                 attention_mechanism='channel_attention', se_reduction=4)
    elif name == "L":
        gnn = dict(dist=3, n_node_inputs=9, n_node_outputs=9,
                   n_iters=9, n_node_features=256, n_edge_features=256,
                   msg_net_size=160, msg_net_dropout_p=0.04, gru_dropout_p=0.11)
        m = dict(d_model=320, d_state=64, d_conv=2, expand=3,
                 attention_mechanism='channel_attention', se_reduction=4)
    else:
        raise ValueError(name)
    model = MGHD(gnn, m).to("cuda")
    model.set_rotated_layout()
    model.eval()
    return model


def _bench_fastpath_persist(N=1000) -> Dict[str, Any]:
    try:
        import fastpath  # persistent ext loaded lazily
        lut16, *_ = fastpath.load_rotated_d3_lut_npz()
        from fastpath import PersistentLUT
    except Exception as e:
        return {"error": f"fastpath unavailable: {e}"}
    import numpy as np
    with PersistentLUT(lut16=lut16, capacity=max(1024, N)) as svc:
        # warm single-shot
        for _ in range(32):
            _ = svc.decode_bytes(np.array([0], dtype=np.uint8))
        # measure single-shot
        t0 = time.time()
        for _ in range(N):
            _ = svc.decode_bytes(np.array([84], dtype=np.uint8))
        us_single = (time.time() - t0) * 1e6 / N
        # measure small batch
        arr = np.random.randint(0, 256, size=(256,), dtype=np.uint8)
        t1 = time.time(); _ = svc.decode_bytes(arr); us_batch = (time.time() - t1) * 1e6 / len(arr)
    return {
        "single_us": us_single,
        "batch256_us": us_batch,
    }


def main():
    assert torch.cuda.is_available(), "CUDA required for this benchmark"
    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "pytorch": torch.__version__,
    }
    out: Dict[str, Any] = {"env": info, "MGHD": {}, "fastpath_persist": None}

    # MGHD profiles
    # Try to load trained checkpoints for S and L to reflect realistic runtime
    ckpts = {
        "S": Path("results/step11/step11_garnet_S_best.pt"),
        "L": Path("results/step11_L/step11_garnet_L_best.pt"),
    }
    for prof in ["S", "M", "L"]:
        model = _build_profile(prof)
        if prof in ckpts and ckpts[prof].exists():
            try:
                state = torch.load(str(ckpts[prof]), map_location="cuda")
                model.load_state_dict(state, strict=False)
            except Exception:
                pass
        single = _bench_decode_one(model)
        single_ts = _bench_decode_one_torchscript(model)
        single_graph = _bench_decode_one_cudagraph(model)
        batch = _bench_forward_batch(model, B=256)
        out["MGHD"][prof] = {
            "eager_single": single,
            "ts_single": single_ts,
            "graph_single": single_graph,
            "eager_batch256": batch,
        }

    # Fastpath
    out["fastpath_persist"] = _bench_fastpath_persist()

    reports = Path("reports"); reports.mkdir(exist_ok=True)
    dest = reports / "latency_benchmark.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
