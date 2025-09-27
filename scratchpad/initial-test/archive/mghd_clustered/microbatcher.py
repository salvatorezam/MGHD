from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from types import SimpleNamespace

from mghd_public.features_v2 import PackedCrop, infer_bucket_id


BucketSpec = Sequence[Tuple[int, int, int]]


@dataclass(order=True)
class _QueuedCrop:
    order: int
    arrived_ms: float
    crop: PackedCrop = field(compare=False)


class CrossShotBatcher:
    """Micro-batches `PackedCrop`s across shots keeping deterministic order."""

    def __init__(
        self,
        bucket_spec: BucketSpec | None = None,
        *,
        microbatch: int = 64,
        flush_ms: float = 1.0,
        pin_host: bool = True,
    ) -> None:
        self.bucket_spec = tuple(bucket_spec or ())
        self.microbatch = max(1, int(microbatch))
        self.flush_ms = max(0.0, float(flush_ms))
        self.pin_host = bool(pin_host)
        self._queues: Dict[int, List[_QueuedCrop]] = {}
        self._order = 0
        self._force_flush = False

    def add(self, crop: PackedCrop) -> None:
        bucket_id = self._bucket_id_for_crop(crop)
        queue = self._queues.setdefault(bucket_id, [])
        queue.append(
            _QueuedCrop(
                order=self._order,
                arrived_ms=time.monotonic() * 1e3,
                crop=crop,
            )
        )
        self._order += 1

    def flush_due(self, now_ms: float) -> bool:
        if self.flush_ms <= 0:
            return False
        for queue in self._queues.values():
            if not queue:
                continue
            if now_ms - queue[0].arrived_ms >= self.flush_ms:
                self._force_flush = True
                return True
        return False

    def pop_ready_batches(self) -> Dict[int, List[PackedCrop]]:
        ready: Dict[int, List[PackedCrop]] = {}
        for bucket_id, queue in list(self._queues.items()):
            if not queue:
                continue
            if len(queue) >= self.microbatch or self._force_flush:
                count = min(len(queue), self.microbatch) if not self._force_flush else len(queue)
                # Maintain deterministic order
                queue.sort()  # sort by order preserving arrival
                batch = [queue.pop(0).crop for _ in range(count)]
                ready[bucket_id] = batch
        self._force_flush = False
        return ready

    def finalize(self) -> Iterator[Tuple[int, List[PackedCrop]]]:
        for bucket_id, queue in list(self._queues.items()):
            if not queue:
                continue
            queue.sort()
            crops = [item.crop for item in queue]
            queue.clear()
            yield bucket_id, crops
        self._force_flush = False

    def _bucket_id_for_crop(self, crop: PackedCrop) -> int:
        bucket_id = getattr(crop.meta, "bucket_id", None)
        if bucket_id is not None and bucket_id >= 0:
            return bucket_id
        n_pad = int(crop.node_mask.numel())
        e_pad = int(crop.edge_mask.numel())
        s_pad = int(crop.seq_mask.numel())
        if not self.bucket_spec:
            raise ValueError("Bucket specification required to bin crops")
        bucket = infer_bucket_id(n_pad, e_pad, s_pad, self.bucket_spec)
        crop.meta.bucket_id = bucket
        crop.meta.pad_nodes = n_pad
        crop.meta.pad_edges = e_pad
        crop.meta.pad_seq = s_pad
        return bucket


def stack_crops_pinned(
    crops: Sequence[PackedCrop],
    *,
    pin: bool = True,
) -> Tuple[SimpleNamespace, List[Dict[str, torch.Tensor]], Dict[str, int]]:
    if not crops:
        raise ValueError("Cannot stack empty crop list")

    first = crops[0]
    pad_nodes = int(getattr(first.meta, "pad_nodes", first.x_nodes.shape[0]))
    pad_edges = int(getattr(first.meta, "pad_edges", first.edge_index.shape[1]))
    pad_seq = int(getattr(first.meta, "pad_seq", first.seq_idx.shape[0]))

    B = len(crops)
    feat_dim = int(first.x_nodes.shape[1])
    edge_feat_dim = int(first.edge_attr.shape[1])
    g_dim = int(first.g_token.numel())

    def alloc(shape, dtype):
        return torch.zeros(shape, dtype=dtype, pin_memory=pin)

    x_nodes = alloc((B, pad_nodes, feat_dim), torch.float32)
    node_mask = alloc((B, pad_nodes), torch.bool)
    node_type = alloc((B, pad_nodes), torch.int8)
    edge_index = alloc((2, B * pad_edges), torch.long)
    edge_attr = alloc((B * pad_edges, edge_feat_dim), torch.float32)
    edge_mask = alloc((B * pad_edges,), torch.bool)
    seq_idx = alloc((B * pad_seq,), torch.long)
    seq_mask = alloc((B * pad_seq,), torch.bool)
    g_token = alloc((B, g_dim), torch.float32)

    cluster_infos: List[Dict[str, torch.Tensor]] = []
    node_offset = 0

    for i, crop in enumerate(crops):
        if crop.x_nodes.shape[0] != pad_nodes:
            raise ValueError("Crop node pad mismatch within bucket")
        x_nodes[i].copy_(crop.x_nodes)
        node_mask[i].copy_(crop.node_mask)
        node_type[i].copy_(crop.node_type)
        g_token[i].copy_(crop.g_token)

        edge_base = i * pad_edges
        seq_base = i * pad_seq
        edge_index[:, edge_base : edge_base + pad_edges].copy_(crop.edge_index + node_offset)
        edge_attr[edge_base : edge_base + pad_edges].copy_(crop.edge_attr)
        edge_mask[edge_base : edge_base + pad_edges].copy_(crop.edge_mask)

        seq_idx[seq_base : seq_base + pad_seq].copy_(crop.seq_idx + node_offset)
        seq_mask[seq_base : seq_base + pad_seq].copy_(crop.seq_mask)

        data_positions = torch.nonzero((crop.node_type == 0) & crop.node_mask, as_tuple=False).squeeze(-1)
        cluster_infos.append(
            {
                "record_index": getattr(crop, "_record_index", None),
                "data_idx": data_positions + node_offset,
            }
        )
        node_offset += pad_nodes

    batch_ns = SimpleNamespace(
        x_nodes=x_nodes,
        node_mask=node_mask,
        node_type=node_type,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        seq_idx=seq_idx,
        seq_mask=seq_mask,
        g_token=g_token,
        batch_size=B,
        nodes_pad=pad_nodes,
    )

    meta = {
        "batch_size": B,
        "nodes_pad": pad_nodes,
        "edges_pad": pad_edges,
        "seq_pad": pad_seq,
        "feat_dim": feat_dim,
        "edge_feat_dim": edge_feat_dim,
        "g_dim": g_dim,
    }

    return batch_ns, cluster_infos, meta


class PackedBucketIterator:
    """Iterate over pre-packed crop tensors stored on disk."""

    _FIELDS = (
        "x_nodes",
        "node_mask",
        "node_type",
        "edge_index",
        "edge_attr",
        "edge_mask",
        "seq_idx",
        "seq_mask",
        "g_token",
    )

    def __init__(
        self,
        entries: Sequence[Dict[str, object]],
        *,
        root: str,
        microbatch: int,
        pin_memory: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        self._microbatch = max(1, int(microbatch))
        self._pin = bool(pin_memory)
        self._limit = None if limit is None else max(0, int(limit))
        self._consumed = 0
        self._items: List[Dict[str, object]] = []
        root_path = Path(root)
        for entry in entries:
            rel_path = Path(entry["path"])
            npz_path = (root_path / rel_path).resolve()
            with np.load(npz_path) as data:
                arrays = {field: data[field] for field in self._FIELDS}
                nodes_pad = int(data["nodes_pad"]) if "nodes_pad" in data else arrays["x_nodes"].shape[1]
                edges_pad = int(data["edges_pad"]) if "edges_pad" in data else arrays["edge_attr"].shape[1]
                seq_pad = int(data["seq_pad"]) if "seq_pad" in data else arrays["seq_idx"].shape[1]
                count = int(data["count"]) if "count" in data else arrays[self._FIELDS[0]].shape[0]
                bucket = tuple(entry["bucket"])
            self._items.append(
                {
                    "bucket_id": int(entry["bucket_id"]),
                    "bucket": bucket,
                    "arrays": arrays,
                    "count": count,
                    "offset": 0,
                    "nodes_pad": nodes_pad,
                    "edges_pad": edges_pad,
                    "seq_pad": seq_pad,
                }
            )
        self._idx = 0

    def __iter__(self) -> "PackedBucketIterator":
        return self

    def __next__(self) -> Tuple[int, Tuple[int, int, int], SimpleNamespace]:
        if self._limit is not None and self._consumed >= self._limit:
            raise StopIteration
        while self._idx < len(self._items):
            item = self._items[self._idx]
            count = item["count"]
            offset = item["offset"]
            if offset >= count:
                self._idx += 1
                continue
            remaining = count - offset
            if self._limit is not None:
                remaining = min(remaining, self._limit - self._consumed)
                if remaining <= 0:
                    raise StopIteration
            batch_end = offset + min(remaining, self._microbatch)
            batch_ns = self._make_batch(
                arrays=item["arrays"],
                start=offset,
                end=batch_end,
                nodes_pad=item["nodes_pad"],
                edges_pad=item["edges_pad"],
                seq_pad=item["seq_pad"],
            )
            item["offset"] = batch_end
            self._consumed += batch_ns.batch_size
            return item["bucket_id"], tuple(item["bucket"]), batch_ns
        raise StopIteration

    def _make_batch(
        self,
        arrays: Dict[str, np.ndarray],
        start: int,
        end: int,
        *,
        nodes_pad: int,
        edges_pad: int,
        seq_pad: int,
    ) -> SimpleNamespace:
        slice_obj = slice(start, end)
        host = {}
        batch_size = end - start
        for field in self._FIELDS:
            raw = arrays[field]
            if field == "edge_index":
                chunk = raw[start:end]  # (B, 2, E)
                tensor = torch.from_numpy(chunk.reshape(batch_size, 2, edges_pad))
                tensor = tensor.permute(1, 0, 2).reshape(2, batch_size * edges_pad)
            elif field == "edge_attr":
                chunk = raw[start:end]  # (B, E, F)
                tensor = torch.from_numpy(chunk.reshape(batch_size * edges_pad, *chunk.shape[2:]))
            elif field == "edge_mask":
                chunk = raw[start:end]  # (B, E)
                tensor = torch.from_numpy(chunk.reshape(batch_size * edges_pad))
            elif field in {"seq_idx", "seq_mask"}:
                chunk = raw[start:end]  # (B, S)
                tensor = torch.from_numpy(chunk.reshape(batch_size * seq_pad))
            else:
                tensor = torch.from_numpy(raw[slice_obj])
            if self._pin:
                host[field] = tensor.contiguous().pin_memory()
            else:
                host[field] = tensor.contiguous()
        host["batch_size"] = batch_size
        host["nodes_pad"] = nodes_pad
        host["edges_pad"] = edges_pad
        host["seq_pad"] = seq_pad
        return SimpleNamespace(**host)
