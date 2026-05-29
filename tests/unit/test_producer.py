"""Tests for the producer subprocess in isolation."""
import multiprocessing as mp
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory
import genvarloader as gvl
from genvarloader._producer import producer_main
from genvarloader._shm_layout import read_chunk


def test_producer_writes_chunk_and_signals():
    ctx = mp.get_context("spawn")
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    if not hasattr(ds, "path") or ds.path is None or not ds.path.is_dir():
        pytest.skip("dummy dataset is not file-backed")
    capacity = 64 * 1024
    shm = SharedMemory(create=True, size=capacity)
    free = ctx.Event()
    free.set()
    ready = ctx.Event()
    index_queue = ctx.Queue()
    exc_q = ctx.Queue()
    r = np.array([0], dtype=np.int64)
    s = np.array([0], dtype=np.int64)
    index_queue.put((0, r, s, 1))
    index_queue.put(None)
    p = ctx.Process(
        target=producer_main,
        args=(
            str(ds.path),
            {"with_seqs": "reference", "with_tracks": False},
            [shm.name],
            [(free, ready)],
            index_queue,
            exc_q,
        ),
    )
    p.start()
    assert ready.wait(timeout=30), "producer did not signal ready in time"
    n_inst, views = read_chunk(shm.buf)
    assert n_inst == 1
    p.join(timeout=10)
    assert not p.is_alive()
    # Drain exc queue to ensure no error.
    assert exc_q.empty()
    shm.close()
    shm.unlink()


def test_producer_exception_pushed_to_queue(tmp_path):
    """If the dataset path is bad, the producer pushes an error and exits cleanly."""
    ctx = mp.get_context("spawn")
    shm = SharedMemory(create=True, size=4096)
    free = ctx.Event()
    free.set()
    ready = ctx.Event()
    index_queue = ctx.Queue()
    exc_q = ctx.Queue()
    p = ctx.Process(
        target=producer_main,
        args=(
            str(tmp_path / "does-not-exist.gvl"),
            {"with_seqs": "reference", "with_tracks": False},
            [shm.name],
            [(free, ready)],
            index_queue,
            exc_q,
        ),
    )
    p.start()
    p.join(timeout=15)
    assert not p.is_alive()
    assert not exc_q.empty(), "producer exited without pushing an exception"
    tname, msg, tb = exc_q.get_nowait()
    assert tname  # some exception type
    shm.close()
    shm.unlink()
