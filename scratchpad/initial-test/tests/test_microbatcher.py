import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mghd_clustered.microbatcher import CrossShotBatcher, stack_crops_pinned
from mghd_public.features_v2 import pack_cluster


def _make_crop(bucket_spec, seed: int, n_qubits: int = 2) -> object:
    rng = np.random.default_rng(seed)
    n_checks = 2
    H_sub = np.zeros((n_checks, n_qubits), dtype=np.uint8)
    H_sub[0, 0] = 1
    H_sub[1, -1] = 1
    if n_qubits > 1:
        H_sub[0, 1] = 1

    xy_qubit = np.stack([np.arange(n_qubits), np.zeros(n_qubits)], axis=1).astype(np.int32)
    xy_check = np.stack([np.arange(n_checks), np.ones(n_checks)], axis=1).astype(np.int32)
    synd = np.array([1, 0], dtype=np.uint8)
    y_bits = np.zeros(n_qubits, dtype=np.uint8)

    crop = pack_cluster(
        H_sub=H_sub,
        xy_qubit=xy_qubit,
        xy_check=xy_check,
        synd_Z_then_X_bits=synd,
        k=n_qubits,
        r=1,
        bbox_xywh=(0, 0, int(n_qubits), 2),
        kappa_stats={"size": int(n_qubits + n_checks)},
        y_bits_local=y_bits,
        side="Z",
        d=3,
        p=0.01,
        seed=seed,
        N_max=None,
        E_max=None,
        S_max=None,
        bucket_spec=bucket_spec,
        add_jump_edges=False,
    )
    setattr(crop, "_record_index", seed)
    return crop


def test_cross_shot_batcher_bins_and_stacks():
    bucket_spec = [(16, 32, 16), (32, 64, 32)]
    batcher = CrossShotBatcher(bucket_spec=bucket_spec, microbatch=2, flush_ms=0.0)

    first = [_make_crop(bucket_spec, seed) for seed in range(2)]
    for crop in first:
        batcher.add(crop)

    ready = batcher.pop_ready_batches()
    assert len(ready) == 1
    bucket_id, batch = next(iter(ready.items()))
    assert len(batch) == 2
    assert {c.meta.bucket_id for c in batch} == {bucket_id}

    host_batch, infos, meta = stack_crops_pinned(batch, pin=True)
    assert meta["batch_size"] == 2
    assert host_batch.x_nodes.shape[0] == 2
    assert host_batch.x_nodes.is_pinned()
    assert [info["record_index"] for info in infos] == [0, 1]

    # Add three more crops to exercise deterministic ordering and finalize()
    for seed in range(2, 5):
        batcher.add(_make_crop(bucket_spec, seed))

    # Microbatch is 2, so two should be ready and one left for finalize
    ready2 = batcher.pop_ready_batches()
    ids = [c._record_index for c in ready2[bucket_id]]
    assert ids == sorted(ids)

    leftover = list(batcher.finalize())
    assert len(leftover) == 1
    leftover_bucket, crops = leftover[0]
    assert leftover_bucket == bucket_id
    assert len(crops) == 1
    assert crops[0]._record_index == 4
    # Finalize resets flush flag
    assert batcher.pop_ready_batches() == {}
