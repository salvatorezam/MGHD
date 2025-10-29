import numpy as np
import scipy.sparse as sp

from types import SimpleNamespace

from mghd.decoders.lsd.clustered import MGHDPrimaryClustered


def test_clustered_decode_empty_syndrome_returns_zero_and_hist():
    H = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8))
    dummy = SimpleNamespace()
    # Provide required attributes but they won't be used for empty subproblems
    mc = MGHDPrimaryClustered(H, dummy, halo=0, batched=True)
    s = np.zeros(H.shape[0], dtype=np.uint8)
    out = mc.decode(s, perf_only=False)
    assert out["e"].shape[0] == H.shape[1]
    assert isinstance(out.get("sizes_hist"), dict)
    assert out.get("mghd_clusters") == 0

