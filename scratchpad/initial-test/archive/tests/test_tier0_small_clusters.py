import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mghd_clustered.cluster_core import solve_small_cluster_channel_ml, TIER0_K_MAX, TIER0_R_MAX
from mghd_clustered.clustered_primary import MGHDPrimaryClustered


def brute_channel_ml(H_sub: sp.csr_matrix, s_sub: np.ndarray, p: float) -> np.ndarray:
    H = H_sub.toarray() % 2
    s = s_sub.astype(np.uint8) % 2
    n = H.shape[1]
    w = np.log((1 - p) / p)
    best_cost = float("inf")
    best_e = None
    for e_int in range(1 << n):
        e = np.array([(e_int >> j) & 1 for j in range(n)], dtype=np.uint8)
        if not np.array_equal((H @ e) % 2, s):
            continue
        cost = float(np.sum(w * e))
        if cost < best_cost:
            best_cost = cost
            best_e = e
    assert best_e is not None
    return best_e


def test_tier0_matches_bruteforce():
    cases = [
        (sp.csr_matrix([[1]]), np.array([1], dtype=np.uint8)),
        (sp.csr_matrix([[1, 1]]), np.array([1], dtype=np.uint8)),
        (sp.csr_matrix([[1, 0, 1], [0, 1, 1]]), np.array([1, 0], dtype=np.uint8)),
        (sp.csr_matrix([[1, 1, 0], [0, 1, 1]]), np.array([0, 1], dtype=np.uint8)),
    ]
    p = 0.005
    for H_sub, s_sub in cases:
        tier0 = solve_small_cluster_channel_ml(H_sub, s_sub, p_channel=p)
        brute = brute_channel_ml(H_sub, s_sub, p)
        assert tier0 is not None
        assert np.array_equal(tier0, brute)
        parity = (H_sub @ tier0) % 2
        assert np.array_equal(parity.astype(np.uint8) % 2, s_sub % 2)


def test_tier0_respects_caps():
    # Both size and nullity above thresholds should return None
    H_large = sp.csr_matrix((0, TIER0_K_MAX + TIER0_R_MAX + 1), dtype=np.uint8)
    s_large = np.zeros(0, dtype=np.uint8)
    assert solve_small_cluster_channel_ml(H_large, s_large, p_channel=0.01) is None
    # Nullity cap can be tightened while size criterion still permits solution
    H_null = sp.csr_matrix((0, 3))
    s_null = np.zeros(0, dtype=np.uint8)
    sol = solve_small_cluster_channel_ml(H_null, s_null, p_channel=0.01, r_cap=1)
    assert sol is not None


class _FakeMGHD:
    def __init__(self):
        self.device = torch.device("cpu")

    def _device_info(self):
        return {"device": "cpu", "name": "fake", "float32_matmul_precision": "high"}

    def priors_from_subgraphs_batched(self, items, *, temp=1.0, bucket=None, use_masked_fullgraph_fallback=True):
        raise AssertionError("Tier-0 should handle all clusters in this test")

    def priors_from_syndrome(self, s_full, *, side):
        raise AssertionError("Tier-0 should prevent MGHD path")


def test_decoder_uses_tier0_exclusively():
    H1 = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8))
    H2 = sp.csr_matrix(np.array([[1, 1], [1, 0]], dtype=np.uint8))
    H = sp.block_diag((H1, H2), format="csr")
    e = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    s = (H @ e) % 2

    decoder = MGHDPrimaryClustered(
        H,
        _FakeMGHD(),
        tier0_enable=True,
        tier0_k_max=TIER0_K_MAX,
        tier0_r_max=TIER0_R_MAX,
        p_channel=0.005,
        default_p=0.005,
    )
    out = decoder.decode(s)
    assert out["tier0_clusters"] == out["n_clusters"]
    assert out["mghd_clusters"] == 0
    assert out["mghd_invoked"] is False
    assert out["mb_stats"]["fixed_d3_batches"] == 0
    parity = (H @ out["e_hat"]) % 2
    assert np.array_equal(parity.astype(np.uint8), s)
