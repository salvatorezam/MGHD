import numpy as np
import pytest

pytest.importorskip("torch")

from tools.bench_clustered_sweep_surface import _format_prob_tag, iter_cached_crops
from mghd_clustered.microbatcher import PackedBucketIterator


def test_iter_cached_crops(tmp_path):
    syndromes = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    tag = _format_prob_tag(3, 0.05, "X")
    path = tmp_path / f"{tag}.npz"
    np.savez(path, syndromes=syndromes)

    items = list(iter_cached_crops(str(tmp_path), 3, 0.05, "X"))
    assert len(items) == 2
    assert np.array_equal(items[0], syndromes[0])

    limited = list(iter_cached_crops(str(tmp_path), 3, 0.05, "X", limit=1))
    assert len(limited) == 1
    assert np.array_equal(limited[0], syndromes[0])


def test_packed_bucket_iterator(tmp_path):
    B = 6
    nodes_pad = 5
    edges_pad = 3
    seq_pad = 2
    feat_dim = 7
    edge_feat_dim = 11
    g_dim = 9

    arrays = dict(
        x_nodes=np.arange(B * nodes_pad * feat_dim, dtype=np.float32).reshape(B, nodes_pad, feat_dim),
        node_mask=np.ones((B, nodes_pad), dtype=bool),
        node_type=np.zeros((B, nodes_pad), dtype=np.int8),
        edge_index=np.arange(B * 2 * edges_pad, dtype=np.int64).reshape(B, 2, edges_pad),
        edge_attr=np.arange(B * edges_pad * edge_feat_dim, dtype=np.float32).reshape(B, edges_pad, edge_feat_dim),
        edge_mask=np.ones((B, edges_pad), dtype=bool),
        seq_idx=np.arange(B * seq_pad, dtype=np.int64).reshape(B, seq_pad),
        seq_mask=np.ones((B, seq_pad), dtype=bool),
        g_token=np.arange(B * g_dim, dtype=np.float32).reshape(B, g_dim),
        count=B,
        nodes_pad=nodes_pad,
        edges_pad=edges_pad,
        seq_pad=seq_pad,
    )

    bucket_path = tmp_path / "bucket.npz"
    np.savez(bucket_path, **arrays)

    entries = [
        {
            "bucket_id": 7,
            "bucket": [nodes_pad, edges_pad, seq_pad],
            "path": bucket_path.name,
            "count": B,
        }
    ]

    iterator = PackedBucketIterator(entries, root=str(tmp_path), microbatch=3, pin_memory=False, limit=5)
    bucket_id, bucket_dims, batch = next(iterator)
    assert bucket_id == 7
    assert bucket_dims == (nodes_pad, edges_pad, seq_pad)
    assert batch.batch_size == 3
    assert batch.x_nodes.shape == (3, nodes_pad, feat_dim)
    assert batch.edge_index.shape == (2, 3 * edges_pad)
    assert batch.edge_attr.shape == (3 * edges_pad, edge_feat_dim)
    assert batch.seq_idx.shape == (3 * seq_pad,)

    # Second batch should respect limit=5 → size 2
    _, _, batch2 = next(iterator)
    assert batch2.batch_size == 2
    with pytest.raises(StopIteration):
        next(iterator)
