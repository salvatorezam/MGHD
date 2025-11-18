import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import pack_cluster


def _base_kwargs():
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    xy_q = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
    xy_c = np.array([[0, 1], [1, 2]], dtype=np.int32)
    s = np.array([1, 0], dtype=np.uint8)
    nQ = H.shape[1]
    return {
        "H_sub": H,
        "xy_qubit": xy_q,
        "xy_check": xy_c,
        "synd_Z_then_X_bits": s,
        "k": nQ,
        "r": 1,
        "bbox_xywh": (0, 0, 2, 3),
        "kappa_stats": {"size": float(H.shape[0] + H.shape[1])},
        "y_bits_local": np.zeros(nQ, dtype=np.uint8),
        "side": "Z",
        "d": 3,
        "p": 0.01,
        "seed": 0,
    }


def test_pack_cluster_g_extra_and_jump_edges():
    kwargs = _base_kwargs()
    bucket_spec = [(10, 20, 5)]
    crop = pack_cluster(
        **kwargs,
        N_max=None,
        E_max=None,
        S_max=None,
        bucket_spec=bucket_spec,
        g_extra=np.array([0.5, -0.5], dtype=np.float32),
        add_jump_edges=True,
        jump_k=2,
    )
    # bucket_spec path sets bucket_id 0
    assert crop.meta.bucket_id == 0
    # Jump edges add more edges than the base H connections
    assert crop.edge_attr.shape[0] >= int(kwargs["H_sub"].sum())
    # g_token should include the additional features
    assert crop.g_token.shape[-1] >= 6  # base + extras


class BadExtra:
    def __array__(self, dtype=None):
        raise ValueError("boom")


def test_pack_cluster_handles_bad_g_extra_and_erasure_mismatch():
    kwargs = _base_kwargs()
    nQ = kwargs["k"]
    crop = pack_cluster(
        **kwargs,
        N_max=nQ + kwargs["H_sub"].shape[0],
        E_max=int(kwargs["H_sub"].sum()),
        S_max=kwargs["H_sub"].shape[0],
        bucket_spec=None,
        g_extra=BadExtra(),
        erase_local=np.ones(nQ - 1, dtype=np.float32),
        add_jump_edges=False,
    )
    # Bad g_extra should be ignored gracefully
    assert crop.g_token.shape[-1] >= 4
    # Erasure mismatch prevents extra feature column
    assert crop.x_nodes.shape[1] <= 9


def test_pack_cluster_requires_pad_when_no_bucket_spec():
    kwargs = _base_kwargs()
    with pytest.raises(ValueError):
        pack_cluster(
            **kwargs,
            N_max=None,
            E_max=None,
            S_max=None,
            bucket_spec=None,
            add_jump_edges=False,
        )
