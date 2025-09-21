import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mghd_clustered.cluster_core import active_components, extract_subproblem
from mghd_clustered.clustered_primary import MGHDPrimaryClustered


class FakeTimer:
    def __init__(self):
        self.current = 0.0

    def perf_counter(self):
        return self.current

    def advance(self, delta: float):
        self.current += float(delta)


class FakeDecoder:
    def __init__(self, cluster_probs, timer: FakeTimer, *, force_fallback: bool = False):
        self._cluster_probs = [np.asarray(p, dtype=np.float64) for p in cluster_probs]
        self._timer = timer
        self.calls_batched = 0
        self.calls_single = 0
        self.calls_fixed = 0
        self._force_fallback = force_fallback
        self._fallback_triggered = False
        self.device = torch.device("cpu")
        self.cfg = type("Cfg", (), {"n_checks": 8, "n_qubits": 9})()
        self._qubit_maps = [np.arange(len(p), dtype=np.int64) for p in self._cluster_probs]
        self._map_cursor = 0
        self._syndrome_cursor = 0

    def _device_info(self):
        return {"device": "cpu", "name": "fake", "float32_matmul_precision": "high"}

    def _make_report(self, *, fast: bool, batch_size: int, graph_used: bool = False):
        report = {
            "fast_path_batches": 1 if fast else 0,
            "fixed_d3_batches": 0 if fast else 1,
            "fallback_loops": 0,
            "batch_sizes": [batch_size],
            "graph_used": graph_used,
            "device": self._device_info(),
        }
        report["avg_batch_size"] = float(batch_size) if batch_size else 0.0
        bucket = 1
        while batch_size > bucket:
            bucket *= 2
        report["batch_histogram"] = {str(bucket): 1} if batch_size else {}
        return report

    def set_qubit_maps(self, maps):
        self._qubit_maps = [np.asarray(m, dtype=np.int64) for m in maps]
        self._map_cursor = 0
        self._syndrome_cursor = 0

    def priors_from_d3_fullgraph_batched(self, syndromes, *, temp=1.0, bucket=None):
        del temp, bucket
        self.calls_fixed += 1
        self._timer.advance(0.002)
        outputs = []
        for _ in syndromes:
            idx = self._map_cursor % len(self._cluster_probs)
            arr = np.full(9, 0.5, dtype=np.float64)
            q_map = self._qubit_maps[idx] if self._qubit_maps else np.arange(len(self._cluster_probs[idx]), dtype=np.int64)
            arr[q_map] = self._cluster_probs[idx]
            outputs.append(arr)
            self._map_cursor += 1
        return outputs, self._make_report(fast=False, batch_size=len(outputs))

    def priors_from_syndrome(self, s_full, *, side):
        del s_full, side
        self.calls_single += 1
        self._timer.advance(0.004)
        idx = self._syndrome_cursor % len(self._cluster_probs)
        arr = np.full(9, 0.5, dtype=np.float64)
        q_map = self._qubit_maps[idx] if self._qubit_maps else np.arange(len(self._cluster_probs[idx]), dtype=np.int64)
        arr[q_map] = self._cluster_probs[idx]
        self._syndrome_cursor += 1
        return arr

    def priors_from_subgraphs_batched(
        self,
        items,
        *,
        temp=1.0,
        bucket=None,
        use_masked_fullgraph_fallback=True,
    ):
        del temp, bucket, use_masked_fullgraph_fallback
        self.calls_batched += 1
        self._timer.advance(0.001)
        batch_size = len(items)
        if items and isinstance(items[0], tuple) and len(items[0]) >= 3:
            self._qubit_maps = [np.asarray(entry[2], dtype=np.int64) for entry in items]
            self._map_cursor = 0
        if self._force_fallback and not self._fallback_triggered:
            self._fallback_triggered = True
            outputs, report = self.priors_from_d3_fullgraph_batched([None] * batch_size)
            locals_out = []
            for idx, arr in enumerate(outputs):
                q_map = self._qubit_maps[idx % len(self._qubit_maps)]
                locals_out.append(arr[q_map].copy())
            report["batch_sizes"] = [batch_size]
            return locals_out, report
        return (
            [self._cluster_probs[i % len(self._cluster_probs)].copy() for i in range(batch_size)],
            self._make_report(fast=True, batch_size=batch_size),
        )

    def priors_from_subgraph(self, H_sub, s_sub, *, temp=1.0):
        del H_sub, s_sub, temp
        idx = self.calls_single
        if idx >= len(self._cluster_probs):
            raise AssertionError("FakeDecoder received more calls than clusters")
        self.calls_single += 1
        self._timer.advance(0.004)
        return self._cluster_probs[idx].copy()


def _build_block_code():
    H1 = sp.csr_matrix(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    H2 = sp.csr_matrix(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    return sp.block_diag((H1, H2), format="csr")


def test_batched_equals_unbatched(monkeypatch):
    import mghd_clustered.clustered_primary as clustered_primary

    H = _build_block_code()
    e_true = np.array([1, 0, 0, 0, 1, 1], dtype=np.uint8)
    s = (H @ e_true) % 2

    checks, qubits = active_components(H, s, halo=0)
    assert len(checks) == 2

    cluster_probs = []
    qubit_maps = []
    for ci, qi in zip(checks, qubits):
        _, _, q_l2g, _ = extract_subproblem(H, s, ci, qi)
        probs = np.where(e_true[q_l2g] == 1, 0.8, 0.2)
        cluster_probs.append(probs.astype(np.float64))
        qubit_maps.append(q_l2g)

    # Batched decode
    timer_batched = FakeTimer()
    fake_mghd_batched = FakeDecoder(cluster_probs, timer_batched)
    fake_mghd_batched.set_qubit_maps(qubit_maps)
    decoder_batched = MGHDPrimaryClustered(H, fake_mghd_batched, temp=1.0, r_cap=8, batched=True)

    monkeypatch.setattr(clustered_primary.time, "perf_counter", timer_batched.perf_counter)
    out_batched = decoder_batched.decode(s)

    assert fake_mghd_batched.calls_batched == 1
    assert fake_mghd_batched.calls_single == 0

    parity_batched = (H @ out_batched["e_hat"]) % 2
    assert np.array_equal(parity_batched % 2, s % 2)

    mb_batched = out_batched["mb_stats"]
    assert mb_batched["fast_path_batches"] == 1
    assert mb_batched["fixed_d3_batches"] == 0
    assert mb_batched["fallback_loops"] == 0

    # Unbatched decode
    timer_unbatched = FakeTimer()
    fake_mghd_unbatched = FakeDecoder(cluster_probs, timer_unbatched)
    fake_mghd_unbatched.set_qubit_maps(qubit_maps)
    decoder_unbatched = MGHDPrimaryClustered(H, fake_mghd_unbatched, temp=1.0, r_cap=8, batched=False)

    monkeypatch.setattr(clustered_primary.time, "perf_counter", timer_unbatched.perf_counter)
    out_unbatched = decoder_unbatched.decode(s)

    assert fake_mghd_unbatched.calls_batched == 0
    assert fake_mghd_unbatched.calls_single == len(cluster_probs)

    parity_unbatched = (H @ out_unbatched["e_hat"]) % 2
    assert np.array_equal(parity_unbatched % 2, s % 2)

    mb_unbatched = out_unbatched["mb_stats"]
    assert mb_unbatched["fast_path_batches"] == 0
    assert mb_unbatched["fallback_loops"] == len(cluster_probs)
    assert mb_unbatched["batch_histogram"].get("1", 0) == len(cluster_probs)

    # Equivalence checks
    assert np.array_equal(out_batched["e_hat"], out_unbatched["e_hat"])
    assert out_batched["n_clusters"] == out_unbatched["n_clusters"] == len(cluster_probs)

    # Micro-batching should reduce MGHD timing in this synthetic setup
    assert out_batched["t_mghd_ms"] < out_unbatched["t_mghd_ms"]


def test_batched_d3_fallback(monkeypatch):
    import mghd_clustered.clustered_primary as clustered_primary

    H = _build_block_code()
    e_true = np.array([1, 0, 0, 0, 1, 1], dtype=np.uint8)
    s = (H @ e_true) % 2

    checks, qubits = active_components(H, s, halo=0)
    cluster_probs = []
    qubit_maps = []
    for ci, qi in zip(checks, qubits):
        _, _, q_l2g, _ = extract_subproblem(H, s, ci, qi)
        probs = np.where(e_true[q_l2g] == 1, 0.8, 0.2)
        cluster_probs.append(probs.astype(np.float64))
        qubit_maps.append(q_l2g)

    timer = FakeTimer()
    fake_mghd = FakeDecoder(cluster_probs, timer, force_fallback=True)
    fake_mghd.set_qubit_maps(qubit_maps)
    decoder = MGHDPrimaryClustered(H, fake_mghd, temp=1.0, r_cap=8, batched=True)

    monkeypatch.setattr(clustered_primary.time, "perf_counter", timer.perf_counter)
    out = decoder.decode(s)

    mb = out["mb_stats"]
    assert fake_mghd.calls_batched == 1
    assert fake_mghd.calls_fixed == 1
    assert mb["fixed_d3_batches"] >= 1
    assert mb["fallback_loops"] == 0
    assert mb["avg_batch_size"] == len(cluster_probs)

    parity = (H @ out["e_hat"]) % 2
    assert np.array_equal(parity % 2, s % 2)
