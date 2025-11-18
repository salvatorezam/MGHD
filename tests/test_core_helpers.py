import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import (
    _degree_from_Hsub,
    _hilbert_index_2d,
    _normalize_xy,
    _quantize_xy01,
    ChannelSE,
    GraphDecoder,
    GraphDecoderAdapter,
    MGHDConfig,
    MGHDv2,
    SequenceEncoder,
    hilbert_order_within_bbox,
    infer_bucket_id,
    rotated_surface_pcm,
    to_dict,
)


def test_hilbert_order_within_bbox_stability():
    # Two points within a bbox; verify a stable ordering is returned
    xy = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]], dtype=np.float32)
    bbox = (0, 0, 1, 1)
    order = hilbert_order_within_bbox(xy, bbox)
    assert order.shape[0] == xy.shape[0]
    # Recomputing with same inputs gives identical order
    order2 = hilbert_order_within_bbox(xy, bbox)
    assert np.array_equal(order, order2)


def test_infer_bucket_id_picks_first_fit():
    # Sizes: nodes=5, edges=7, seq=3 should fit into first bucket
    spec = [(8, 8, 4), (16, 16, 8)]
    bid = infer_bucket_id(5, 7, 3, spec)
    assert bid == 0
    # For a larger size, pick the second bucket
    bid2 = infer_bucket_id(9, 9, 5, spec)
    assert bid2 == 1


def test_allocate_static_and_scatter_outputs_shapes():
    model = MGHDv2()
    ns = model.allocate_static_batch(
        batch_size=2,
        nodes_pad=4,
        edges_pad=6,
        seq_pad=3,
        feat_dim=8,
        edge_feat_dim=3,
        g_dim=8,
        device=torch.device("cpu"),
    )
    assert ns.x_nodes.shape == (2, 4, 8)
    # Scatter outputs: make logits for 4 nodes and pick indices [0,2]
    logits = torch.zeros((4, 2), dtype=torch.float32)
    cluster_infos = [{"data_idx": torch.tensor([0, 2], dtype=torch.long)}]
    probs = model.scatter_outputs(logits, cluster_infos)
    assert isinstance(probs, list) and probs[0].shape[0] == 2


def test_rotated_surface_pcm_validates_inputs():
    hx = rotated_surface_pcm(3, "x")
    hz = rotated_surface_pcm(3, "Z")
    assert hx.shape == (4, 9)
    assert hz.shape == (4, 9)
    with pytest.raises(ValueError):
        rotated_surface_pcm(2, "X")
    with pytest.raises(ValueError):
        rotated_surface_pcm(3, "Q")


def test_core_coordinate_helpers():
    bbox = (2, 4, 6, 2)
    xy = np.array([[2.0, 4.0], [5.0, 5.0], [8.0, 6.0]])
    norm = _normalize_xy(xy, bbox)
    assert np.allclose(norm[0], 0.0)
    assert np.allclose(norm[-1], [1.0, 1.0])

    quant = _quantize_xy01(norm, levels=8)
    assert quant.min() >= 0 and quant.max() < 8

    hilbert_idx = _hilbert_index_2d(quant, levels=8)
    # Ensure indices are monotonically non-decreasing for sorted inputs
    assert np.all(hilbert_idx[1:] >= hilbert_idx[:-1])


def test_degree_helper_counts_rows_and_columns():
    h_sub = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    deg_data, deg_check = _degree_from_Hsub(h_sub)
    assert np.array_equal(deg_data, [1, 1, 2])
    assert np.array_equal(deg_check, [2, 2])


def test_mghdconfig_to_dict_round_trip():
    cfg = MGHDConfig(gnn={"iters": 4}, mamba={"d_model": 32})
    cfg_dict = to_dict(cfg)
    assert cfg_dict["gnn"] == {"iters": 4}
    assert cfg_dict["mamba"]["d_model"] == 32


def test_channel_se_hsigmoid_branch():
    se = ChannelSE(channels=4, use_hsigmoid=True)
    x = torch.ones((2, 3, 4), dtype=torch.float32)
    out = se(x)
    assert out.shape == x.shape
    assert torch.all(out >= 0)


def test_graph_decoder_normalizes_kwargs():
    decoder = GraphDecoder(n_node_inputs=6)
    core = decoder.core
    assert core.n_node_features == 6
    assert core.n_edge_features >= 6  # msg_net_size default applied


def test_graph_decoder_adapter_iteration_override():
    adapter = GraphDecoderAdapter(hidden_dim=4, edge_feat_dim=2, n_iters=3)
    nodes = torch.zeros((4, 4), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.zeros((2, 2), dtype=torch.float32)
    mask = torch.ones((2,), dtype=torch.bool)
    node_mask = torch.ones((4,), dtype=torch.bool)
    adapter.set_iteration_override(1)
    out = adapter(nodes, edge_index, edge_attr, node_mask, mask)
    assert out.shape == (4, 2)
    adapter.set_iteration_override(None)


def test_sequence_encoder_constructor_fallback(monkeypatch):
    import mghd.core.core as core_mod

    class AlwaysFail:
        def __init__(self, *_, **__):
            raise TypeError("fail")

    class Dummy:
        def __init__(self, *_, **__):
            pass

        def __call__(self, x):
            return x

    monkeypatch.setattr(core_mod, "_mamba_constructors", lambda: [AlwaysFail, Dummy])
    encoder = SequenceEncoder(d_model=4, d_state=2)
    x = torch.randn(4, 4)
    seq_idx = torch.arange(4)
    seq_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    node_type = torch.zeros(4, dtype=torch.long)
    out = encoder(x, seq_idx, seq_mask, node_type=node_type)
    assert out.shape == x.shape
    empty = encoder(
        x,
        torch.tensor([], dtype=torch.long),
        torch.tensor([], dtype=torch.bool),
        node_type=torch.tensor([], dtype=torch.long),
    )
    assert torch.allclose(empty, x)
