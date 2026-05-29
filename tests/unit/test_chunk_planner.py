"""ChunkPlanner unit tests. Pure logic, no Dataset dependency."""
import numpy as np
import pytest
import genvarloader as gvl
from genvarloader._chunked import ChunkPlanner, slice_chunk


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


# ---------------------------------------------------------------------------
# Task 7: slice_chunk parity tests against real dataset outputs
# ---------------------------------------------------------------------------


def _compare(a, b):
    """Recursive equality for ndarray, Ragged, RaggedAnnotatedHaps, AnnotatedHaps,
    ak.Array (including RaggedVariants), and tuples thereof."""
    from seqpro.rag import Ragged
    from genvarloader._types import AnnotatedHaps
    from genvarloader._ragged import RaggedAnnotatedHaps
    import awkward as ak

    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _compare(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, RaggedAnnotatedHaps):
        # RaggedAnnotatedHaps is a dataclass of three Ragged arrays.
        assert isinstance(b, RaggedAnnotatedHaps)
        _compare(a.haps, b.haps)
        _compare(a.var_idxs, b.var_idxs)
        _compare(a.ref_coords, b.ref_coords)
    elif isinstance(a, Ragged):
        # Ragged slicing returns a view where .data may be the full backing
        # array; compare only the actual content via offsets.
        np.testing.assert_array_equal(a.data[a.offsets[0]:a.offsets[-1]], b.data[b.offsets[0]:b.offsets[-1]])
        np.testing.assert_array_equal(a.offsets - a.offsets[0], b.offsets - b.offsets[0])
    elif isinstance(a, AnnotatedHaps):
        _compare(a.haps, b.haps)
        _compare(a.var_idxs, b.var_idxs)
        _compare(a.ref_coords, b.ref_coords)
    elif isinstance(a, ak.Array):
        # Covers RaggedVariants (ak.Array subclass).
        assert ak.to_list(a) == ak.to_list(b)
    else:
        raise AssertionError(f"unsupported {type(a)}")


@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes", "annotated", "variants"])
def test_slice_chunk_matches_direct(seq_kind):
    ds = gvl.get_dummy_dataset().with_seqs(seq_kind)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)
    ds = ds.with_tracks(False)
    n_r = ds.full_shape[0]
    n_s = min(2, ds.full_shape[1])
    r = np.repeat(np.arange(n_r), n_s)
    s = np.tile(np.arange(n_s), n_r)
    chunk = ds[r, s]
    sliced = list(slice_chunk(chunk, batch_size=n_s))
    assert len(sliced) == n_r
    for i, mini in enumerate(sliced):
        direct = ds[r[i * n_s:(i + 1) * n_s], s[i * n_s:(i + 1) * n_s]]
        _compare(mini, direct)
