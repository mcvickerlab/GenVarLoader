"""Round-trip tests for the shm slot layout."""
import multiprocessing as mp
import numpy as np
import pytest
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
