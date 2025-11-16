import numpy as np
import scipy.sparse as sp

from mghd.decoders.lsd import clustered as cc


def test_active_components_returns_groups_and_legacy_clusters():
    # Build a small H with two checks touching overlapping qubits
    H = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    s = np.array([1, 1, 0], dtype=np.uint8)
    Hs = sp.csr_matrix(H)
    checks, qubits = cc.active_components(Hs, s, halo=0)
    assert isinstance(checks, list) and isinstance(qubits, list)
    assert len(checks) >= 1 and len(qubits) >= 1
    # Legacy basis path returns list[Cluster]
    clusters = cc.active_components(Hs, s, halo=0, basis="Z")
    assert isinstance(clusters, list)
    assert hasattr(clusters[0], "check_indices")
