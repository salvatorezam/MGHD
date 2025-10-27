import numpy as np
# Use the actual implementation module after the reorg
from mghd.decoders.lsd import clustered as core


def _bruteforce_ml(H, s, w=None):
    H = H.astype(np.uint8)
    s = s.astype(np.uint8)
    n = H.shape[1]
    best_cost = float("inf")
    best_x = None
    for mask in range(1 << n):
        x = np.array([(mask >> j) & 1 for j in range(n)], dtype=np.uint8)
        if np.all((H @ x) % 2 == s):
            cost = float(np.dot(w, x)) if w is not None else x.sum()
            if cost < best_cost:
                best_cost = cost
                best_x = x.copy()
    return best_x


def test_active_components_two_clusters():
    # 4 checks, 5 data; two disjoint pairs of checks sharing disjoint data
    H = np.array([
        [1, 0, 1, 0, 0],  # rows 0-1 share col 0 and 2
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],  # rows 2-3 share col 1 and 3
        [0, 1, 0, 1, 0],
    ], dtype=np.uint8)
    s = np.array([1, 1, 1, 0], dtype=np.uint8)
    clusters = core.active_components(H, s, basis="X")
    sizes = sorted([c.check_indices.size for c in clusters])
    assert sizes == [1, 2]
    # now activate row 3 to merge into second cluster
    s2 = np.array([1, 1, 1, 1], dtype=np.uint8)
    clusters2 = core.active_components(H, s2, basis="X")
    sizes2 = sorted([c.check_indices.size for c in clusters2])
    assert sizes2 == [2, 2]


def test_ml_projection_matches_bruteforce_small():
    # Tiny local problem with known solution
    H = np.array([
        [1, 1, 0],
        [0, 1, 1],
    ], dtype=np.uint8)
    s = np.array([1, 0], dtype=np.uint8)
    # uniform weights
    x_ml = core.ml_parity_project(H, s, probs_local=None, r_cap=8)
    x_bf = _bruteforce_ml(H, s, w=None)
    assert x_bf is not None
    assert (H @ x_ml % 2 == s).all()
    assert x_ml.sum() == x_bf.sum()


def test_batched_inference_shapes_and_consistency():
    # Build a simple CSS toy (same H for X and Z)
    H = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ], dtype=np.uint8)
    B = 3
    sx = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    sz = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    ex, ez = core.infer_clusters_batched(H, H, sx, sz)
    assert ex.shape == (B, H.shape[1])
    assert ez.shape == (B, H.shape[1])
    # Sanity: each solution should satisfy H x = s (mod 2)
    for b in range(B):
        assert np.all((H @ ex[b]) % 2 == sx[b])
        assert np.all((H @ ez[b]) % 2 == sz[b])
