"""Round-trip tests for the shm slot layout."""

import multiprocessing as mp
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from genvarloader._shm_layout import write_chunk, read_chunk, slot_capacity_for


def test_dense_roundtrip_single_array():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    capacity = slot_capacity_for([arr]) * 2
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [arr], n_instances=10)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 10
        assert len(views) == 1
        np.testing.assert_array_equal(views[0], arr)
    finally:
        shm.close()
        shm.unlink()


def test_dense_roundtrip_multiple_arrays():
    a = np.arange(20, dtype=np.float32).reshape(4, 5)
    b = np.arange(8, dtype=np.int64).reshape(4, 2)
    capacity = slot_capacity_for([a, b]) * 2
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [a, b], n_instances=4)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 4
        np.testing.assert_array_equal(views[0], a)
        np.testing.assert_array_equal(views[1], b)
    finally:
        shm.close()
        shm.unlink()


def _child_read(shm_name, q):
    s = SharedMemory(name=shm_name)
    try:
        n_inst, views = read_chunk(s.buf)
        q.put((n_inst, [np.asarray(v).copy() for v in views]))
    finally:
        s.close()


def test_dense_cross_process():
    arr = np.arange(50, dtype=np.int32).reshape(5, 10)
    capacity = slot_capacity_for([arr]) * 2
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [arr], n_instances=5)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_child_read, args=(shm.name, q))
        p.start()
        n_inst, views = q.get(timeout=10)
        p.join(timeout=10)
        assert n_inst == 5
        np.testing.assert_array_equal(views[0], arr)
    finally:
        shm.close()
        shm.unlink()


def test_ragged_roundtrip():
    from seqpro.rag import Ragged

    data = np.arange(20, dtype=np.int32)
    # Three rows of lengths 5, 8, 7 → offsets [0, 5, 13, 20].
    offsets = np.array([0, 5, 13, 20], dtype=np.int64)
    rag = Ragged.from_offsets(data, (3, None), offsets)
    capacity = 4096 + data.nbytes + offsets.nbytes + 64
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [rag], n_instances=3)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 3
        from seqpro.rag import Ragged as R

        assert isinstance(views[0], R)
        np.testing.assert_array_equal(views[0].data, data)
        np.testing.assert_array_equal(views[0].offsets, offsets)
    finally:
        shm.close()
        shm.unlink()


def test_annotated_haps_roundtrip():
    """Three Ragged arrays in sequence (haps S1, ref_coords int32, var_idxs int32)."""
    from seqpro.rag import Ragged

    haps_data = np.frombuffer(b"ACGTAAAA", dtype="S1")
    haps_offsets = np.array([0, 4, 8], dtype=np.int64)
    haps = Ragged.from_offsets(haps_data, (2, None), haps_offsets)
    coords = Ragged.from_offsets(np.arange(8, dtype=np.int32), (2, None), haps_offsets)
    v_idxs = Ragged.from_offsets(
        np.array([10, 20, 30], dtype=np.int32),
        (2, None),
        np.array([0, 1, 3], dtype=np.int64),
    )
    capacity = 8192
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [haps, coords, v_idxs], n_instances=2)
        n_inst, views = read_chunk(shm.buf)
        assert n_inst == 2 and len(views) == 3
        np.testing.assert_array_equal(views[0].data, haps_data)
        np.testing.assert_array_equal(
            views[1].data.view(np.int32), np.arange(8, dtype=np.int32)
        )
    finally:
        shm.close()
        shm.unlink()


def test_rag_variants_roundtrip():
    import genvarloader as gvl
    from genvarloader._shm_layout import HEADER_RESERVED

    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    r = np.arange(ds.full_shape[0])
    s = np.zeros(len(r), dtype=np.int64)
    rv = ds[r, s]
    nbytes = ds._output_bytes_per_instance(r, s).sum()
    # header + payload + generous slack for ragged offsets
    capacity = HEADER_RESERVED + int(nbytes) + 4096
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [rv], n_instances=len(r))
        n_inst, views = read_chunk(shm.buf)
        from genvarloader._dataset._rag_variants import RaggedVariants

        assert isinstance(views[0], RaggedVariants)
        import awkward as ak

        assert ak.to_list(views[0]) == ak.to_list(rv)
    finally:
        shm.close()
        shm.unlink()
