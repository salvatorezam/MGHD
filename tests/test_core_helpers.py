import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd.core.core import (
    hilbert_order_within_bbox,
    infer_bucket_id,
    MGHDv2,
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

