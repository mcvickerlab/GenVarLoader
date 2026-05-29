"""ChunkPlanner unit tests. Pure logic, no Dataset dependency."""
import numpy as np
import pytest
from genvarloader._chunked import ChunkPlanner


def test_plan_respects_slot_bytes():
    # 100 instances, each 10 bytes → 1000 total. slot_bytes=200 → ~5 chunks.
    bytes_per_instance = np.full((10, 10), 10, dtype=np.int64)
    # BatchSampler yields batches; simulate batch_size=5.
    batches = [np.arange(i, i + 5) for i in range(0, 100, 5)]
    flat_idx = np.concatenate(batches)
    r = flat_idx // 10
    s = flat_idx % 10
    planner = ChunkPlanner(
        r_idx=r, s_idx=s, batch_size=5,
        bytes_per_instance=bytes_per_instance, slot_bytes=200,
    )
    chunks = list(planner)
    # Each chunk's total bytes ≤ 200; each chunk is a multiple of batch_size.
    for cr, cs, nb in chunks:
        assert len(cr) % 5 == 0
        b = bytes_per_instance[cr, cs].sum()
        assert b <= 200
        assert nb == len(cr) // 5
    # Total instances preserved.
    assert sum(len(cr) for cr, _, _ in chunks) == 100


def test_plan_single_batch_chunks_when_tight():
    bytes_per_instance = np.full((4, 1), 100, dtype=np.int64)
    flat = np.arange(4)
    planner = ChunkPlanner(
        r_idx=flat, s_idx=np.zeros_like(flat), batch_size=2,
        bytes_per_instance=bytes_per_instance, slot_bytes=200,
    )
    chunks = list(planner)
    assert len(chunks) == 2  # 200 bytes per batch fits exactly one chunk
    for cr, cs, nb in chunks:
        assert nb == 1


def test_plan_raises_when_batch_exceeds_slot():
    bytes_per_instance = np.full((2, 1), 300, dtype=np.int64)
    flat = np.arange(2)
    with pytest.raises(ValueError, match="exceeds slot"):
        list(ChunkPlanner(
            r_idx=flat, s_idx=np.zeros_like(flat), batch_size=2,
            bytes_per_instance=bytes_per_instance, slot_bytes=200,
        ))


def test_peak_chunk_bytes_reported():
    bytes_per_instance = np.array([[10, 20], [30, 40]], dtype=np.int64)
    flat = np.array([0, 1, 2, 3])
    r = flat // 2
    s = flat % 2
    planner = ChunkPlanner(
        r_idx=r, s_idx=s, batch_size=2,
        bytes_per_instance=bytes_per_instance, slot_bytes=1000,
    )
    chunks = list(planner)
    # Single chunk of 4 instances, total bytes = 10+20+30+40 = 100.
    assert len(chunks) == 1
    assert planner.peak_chunk_bytes == 100
