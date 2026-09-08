"""Round-trip tests for the shm slot layout."""

import multiprocessing as mp
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory
from genvarloader._shm_layout import (
    write_chunk,
    read_chunk,
    slot_capacity_for,
    SlotOverflowError,
    HEADER_RESERVED,
)


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
        rt = views[0]
        # Compare fields without awkward
        np.testing.assert_array_equal(rt.alt.data, rv.alt.data)
        np.testing.assert_array_equal(
            np.asarray(rt.alt._rl.str_offsets), np.asarray(rv.alt._rl.str_offsets)
        )
        np.testing.assert_array_equal(rt.start.data, rv.start.data)
        np.testing.assert_array_equal(
            np.asarray(rt.start.offsets), np.asarray(rv.start.offsets)
        )
    finally:
        shm.close()
        shm.unlink()


def test_rag_variants_roundtrip_asymmetric():
    """Round-trip a RaggedVariants with asymmetric allele lengths (no dummy dataset).

    Core unit test for _write_rag_variants / _read_rag_variants: build a synthetic
    RaggedVariants with KNOWN byte buffers, write it to shm, read it back, and
    assert exact byte/offset equality — no awkward required.
    """
    from seqpro.rag import Ragged
    from genvarloader._dataset._rag_variants import RaggedVariants
    from genvarloader._shm_layout import HEADER_RESERVED

    # Two batches, ploidy=1: batch 0 has 2 variants, batch 1 has 1 variant.
    # alt alleles: "AC" (2 bytes), "G" (1 byte), "TTT" (3 bytes)
    var_off = np.array([0, 2, 3], np.int64)  # b*p=2 groups
    char_off = np.array([0, 2, 3, 6], np.int64)  # 3 variants
    alt_chars = np.frombuffer(b"ACGTTT", dtype="S1").copy()

    b, p = 2, 1
    alt = (
        Ragged.from_offsets(alt_chars, (b * p, None, None), [var_off, char_off])
        .to_strings()
        .reshape(b, p, None)
    )
    start = Ragged.from_offsets(
        np.array([10, 20, 30], np.int32), (b * p, None), var_off
    ).reshape(b, p, None)
    ilen = Ragged.from_offsets(
        np.array([1, -1, 2], np.int32), (b * p, None), var_off
    ).reshape(b, p, None)

    rv = RaggedVariants(alt=alt, start=start, ilen=ilen)

    capacity = HEADER_RESERVED + 4096
    shm = SharedMemory(create=True, size=capacity)
    try:
        write_chunk(shm.buf, [rv], n_instances=b)
        n_inst, views = read_chunk(shm.buf)

        assert n_inst == b
        assert len(views) == 1
        rt = views[0]
        assert isinstance(rt, RaggedVariants)

        # alt: char data and both offset levels
        np.testing.assert_array_equal(rt.alt.data, rv.alt.data)
        np.testing.assert_array_equal(
            np.asarray(rt.alt.offsets), np.asarray(rv.alt.offsets)
        )
        np.testing.assert_array_equal(
            np.asarray(rt.alt._rl.str_offsets),
            np.asarray(rv.alt._rl.str_offsets),
        )
        # start: data and offsets
        np.testing.assert_array_equal(rt.start.data, rv.start.data)
        np.testing.assert_array_equal(
            np.asarray(rt.start.offsets), np.asarray(rv.start.offsets)
        )
        # ilen: data
        np.testing.assert_array_equal(rt.ilen.data, rv.ilen.data)
        # shape is preserved
        assert rt.shape == rv.shape
    finally:
        shm.close()
        shm.unlink()


def test_flat_ragged_roundtrip():
    from genvarloader._flat import _Flat
    from genvarloader._shm_layout import write_chunk, read_chunk, HEADER_RESERVED

    data = np.arange(20, dtype=np.int32)
    offsets = np.array([0, 5, 13, 20], dtype=np.int64)
    flat = _Flat(data, offsets, (3, None))
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(
        create=True, size=HEADER_RESERVED + data.nbytes + offsets.nbytes + 64
    )
    try:
        write_chunk(shm.buf, [flat], n_instances=3)
        n_inst, views = read_chunk(shm.buf, flat=True)
        assert n_inst == 3
        assert isinstance(views[0], _Flat)
        np.testing.assert_array_equal(views[0].data, data)
        np.testing.assert_array_equal(views[0].offsets, offsets)
    finally:
        shm.close()
        shm.unlink()


def test_flat_variants_roundtrip_matches_ragged():
    """Write a _FlatVariants, read back flat; its .to_ragged() equals the
    RaggedVariants read of the SAME source written from the ragged path."""
    from multiprocessing.shared_memory import SharedMemory

    import genvarloader as gvl
    from genvarloader._shm_layout import write_chunk, read_chunk, HEADER_RESERVED

    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    r = np.array([0, 0, 1], dtype=np.int64)
    s = np.array([0, 1, 0], dtype=np.int64)

    flat_fv = ds.with_output_format("flat")[r, s]  # _FlatVariants
    ragged_rv = ds[r, s]  # RaggedVariants

    cap = HEADER_RESERVED + 1024 * 1024
    shm = SharedMemory(create=True, size=cap)
    try:
        write_chunk(shm.buf, [flat_fv], n_instances=len(r))
        n_inst, views = read_chunk(shm.buf, flat=True)
        from genvarloader._dataset._flat_variants import _FlatVariants

        assert isinstance(views[0], _FlatVariants)
        rt = views[0].to_ragged()
        # Compare fields without awkward
        np.testing.assert_array_equal(rt.alt.data, ragged_rv.alt.data)
        np.testing.assert_array_equal(
            np.asarray(rt.alt._rl.str_offsets),
            np.asarray(ragged_rv.alt._rl.str_offsets),
        )
        np.testing.assert_array_equal(rt.start.data, ragged_rv.start.data)
        np.testing.assert_array_equal(
            np.asarray(rt.start.offsets), np.asarray(ragged_rv.start.offsets)
        )
    finally:
        shm.close()
        shm.unlink()


def test_flat_annotated_roundtrip_matches_ragged():
    """Write a _FlatAnnotatedHaps, read it back flat; its .to_ragged() equals
    the RaggedAnnotatedHaps read of the SAME (r, s) via the awkward path."""
    from multiprocessing.shared_memory import SharedMemory

    import genvarloader as gvl
    from genvarloader._shm_layout import write_chunk, read_chunk, HEADER_RESERVED
    from genvarloader._flat import _FlatAnnotatedHaps

    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("annotated")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    r = np.array([0, 0, 1], dtype=np.int64)
    s = np.array([0, 1, 0], dtype=np.int64)

    flat_ah = ds.with_output_format("flat")[r, s]  # _FlatAnnotatedHaps
    ragged_ah = ds[r, s]  # RaggedAnnotatedHaps

    cap = HEADER_RESERVED + 1024 * 1024
    shm = SharedMemory(create=True, size=cap)
    try:
        write_chunk(shm.buf, [flat_ah], n_instances=len(r))
        _n_inst, views = read_chunk(shm.buf, flat=True)
        assert isinstance(views[0], _FlatAnnotatedHaps)
        got = views[0].to_ragged()
        for comp in ("haps", "var_idxs", "ref_coords"):
            np.testing.assert_array_equal(
                np.asarray(getattr(got, comp).data),
                np.asarray(getattr(ragged_ah, comp).data),
            )
            np.testing.assert_array_equal(
                np.asarray(getattr(got, comp).offsets),
                np.asarray(getattr(ragged_ah, comp).offsets),
            )
    finally:
        shm.close()
        shm.unlink()


def test_flat_read_avoids_awkward_variant_funcs(monkeypatch):
    """The flat read/write path must not touch the awkward kind-2 helpers."""
    from multiprocessing.shared_memory import SharedMemory

    import genvarloader as gvl
    import genvarloader._shm_layout as L

    def _boom(*a, **k):
        raise AssertionError("awkward variant helper called in flat mode")

    monkeypatch.setattr(L, "_write_rag_variants", _boom)
    monkeypatch.setattr(L, "_read_rag_variants", _boom)

    ds = gvl.get_dummy_dataset().with_seqs("variants").with_tracks(False)
    r = np.array([0, 1], dtype=np.int64)
    s = np.array([0, 0], dtype=np.int64)
    flat_fv = ds.with_output_format("flat")[r, s]

    shm = SharedMemory(create=True, size=L.HEADER_RESERVED + 1024 * 1024)
    try:
        L.write_chunk(
            shm.buf, [flat_fv], n_instances=2
        )  # must NOT call _write_rag_variants
        L.read_chunk(shm.buf, flat=True)  # must NOT call _read_rag_variants
    finally:
        shm.close()
        shm.unlink()


def test_write_chunk_raises_actionable_on_overflow():
    # A dense array whose payload exceeds a deliberately tiny slot.
    arr = np.arange(4096, dtype=np.int64)  # 32 KiB payload
    tiny = memoryview(bytearray(HEADER_RESERVED + 1024))  # far too small
    with pytest.raises(SlotOverflowError) as ei:
        write_chunk(tiny, [arr], n_instances=1)
    msg = str(ei.value)
    assert "slot" in msg.lower()
    assert "buffer_bytes" in msg or "batch_size" in msg  # actionable remedy
