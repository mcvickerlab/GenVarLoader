"""mode='double_buffered' dataloader: subprocess producer + 2-slot shm ping-pong."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import uuid
import weakref
from typing import TYPE_CHECKING

import numpy as np
from multiprocessing.shared_memory import SharedMemory

from ._chunked import ChunkPlanner, slice_chunk
from ._shm_layout import read_chunk, HEADER_RESERVED

if TYPE_CHECKING:
    import torch.utils.data as td

    from ._dataset._impl import Dataset


def _cleanup(shms: list, producer_ref) -> None:
    """Finalizer: terminate producer and release/unlink shm slots."""
    proc = producer_ref() if callable(producer_ref) else None
    if proc is not None:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        except Exception:
            pass
    for shm in shms:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def _reshape_ragged_for_chunk(views: list, n_instances: int) -> list:
    """Reshape any Ragged in ``views`` that has lost intermediate fixed dims.

    ``_read_ragged`` always reconstructs Ragged with shape ``(n_groups, None)``.
    For multi-dimensional ragged output such as haplotypes ``(n_inst, ploidy, None)``,
    ``n_groups = n_inst * ploidy``, so we need to re-introduce the ploidy axis.

    If ``n_groups % n_instances != 0`` (already correct, e.g. 1-D ragged), we leave
    the array unchanged.
    """
    from seqpro.rag import Ragged

    result: list = []
    for arr in views:
        if isinstance(arr, Ragged):
            n_groups = arr.shape[0]
            if n_groups != n_instances and n_instances > 0 and n_groups % n_instances == 0:
                ploidy = n_groups // n_instances
                arr = Ragged.from_offsets(arr.data, (n_instances, ploidy, None), arr.offsets)
        result.append(arr)
    return result


def _deep_copy_batch(batch):
    """Recursively copy a batch so it doesn't share memory with the shm slot."""
    from seqpro.rag import Ragged
    from ._types import AnnotatedHaps
    from ._ragged import RaggedAnnotatedHaps
    import awkward as ak

    if isinstance(batch, tuple):
        return tuple(_deep_copy_batch(x) for x in batch)
    if isinstance(batch, np.ndarray):
        return batch.copy()
    if isinstance(batch, RaggedAnnotatedHaps):
        return RaggedAnnotatedHaps(
            haps=batch.haps[...].copy() if hasattr(batch.haps, "data") else batch.haps.copy(),
            var_idxs=batch.var_idxs[...].copy() if hasattr(batch.var_idxs, "data") else batch.var_idxs.copy(),
            ref_coords=batch.ref_coords[...].copy() if hasattr(batch.ref_coords, "data") else batch.ref_coords.copy(),
        )
    if isinstance(batch, Ragged):
        return Ragged.from_offsets(batch.data.copy(), batch.shape, batch.offsets.copy())
    if isinstance(batch, AnnotatedHaps):
        return AnnotatedHaps(
            haps=_deep_copy_batch(batch.haps),
            var_idxs=_deep_copy_batch(batch.var_idxs),
            ref_coords=_deep_copy_batch(batch.ref_coords),
        )
    if isinstance(batch, ak.Array):
        return ak.copy(batch)
    raise TypeError(f"_deep_copy_batch: unsupported {type(batch)}")


class _DoubleBufferedIterable:
    """Iterable that reads chunks from a pair of shared-memory slots.

    The producer subprocess writes chunks into alternating slots; the consumer
    reads and yields mini-batches. Events coordinate access: ``free[i]`` means
    the consumer has finished reading slot ``i`` (producer may write); ``ready[i]``
    means the producer has filled slot ``i`` (consumer may read).
    """

    def __init__(
        self,
        dataset: "Dataset",
        batch_size: int,
        slot_bytes: int,
        bytes_per_instance: np.ndarray,
        flat_r: np.ndarray,
        flat_s: np.ndarray,
        copy: bool,
        heartbeat_seconds: float,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._copy = copy
        self._heartbeat = heartbeat_seconds

        # Build the chunk plan once and store it for reuse across epochs.
        planner = ChunkPlanner(
            r_idx=flat_r,
            s_idx=flat_s,
            batch_size=batch_size,
            bytes_per_instance=bytes_per_instance,
            slot_bytes=slot_bytes,
        )
        self._chunks: list[tuple[np.ndarray, np.ndarray, int]] = list(planner)
        # peak_chunk_bytes is computed eagerly in ChunkPlanner.__init__.
        peak = planner.peak_chunk_bytes

        # Each slot needs header space + payload.
        capacity = HEADER_RESERVED + peak + 4096
        # Unique prefix avoids collisions between concurrent processes.
        suffix = uuid.uuid4().hex[:8]
        self._shm_names = [f"gvl-{os.getpid()}-{suffix}-{i}" for i in range(2)]
        self._shms = [SharedMemory(create=True, name=n, size=capacity) for n in self._shm_names]

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        # events[i] = (free, ready); free starts set, ready starts clear.
        self._events: list[tuple] = [(ctx.Event(), ctx.Event()) for _ in range(2)]
        for free, ready in self._events:
            free.set()
            ready.clear()

        self._index_queue = ctx.Queue()
        self._exc_q: mp.Queue = ctx.Queue()
        self._producer: mp.Process | None = None

        # Register cleanup on GC and process exit.
        shm_snapshot = list(self._shms)
        producer_holder: list[mp.Process | None] = [None]

        def _producer_getter():
            return producer_holder[0]

        self._producer_holder = producer_holder
        weakref.finalize(self, _cleanup, shm_snapshot, _producer_getter)
        atexit.register(self.close)

    def _spawn_producer(self) -> None:
        from ._producer import producer_main
        from ._dataset._reconstruct import Haps
        from ._dataset._insertion_fill import Repeat5p

        ds = self._dataset

        # Reject settings that cannot be serialized into the schema dict.
        if ds.is_spliced:
            raise ValueError(
                "mode='double_buffered' is not supported when splice_info is set; "
                "use mode='buffered' instead."
            )
        tracks = ds._tracks
        if tracks is not None and tracks.insertion_fill:
            non_default = {
                name: fill
                for name, fill in tracks.insertion_fill.items()
                if not isinstance(fill, Repeat5p)
            }
            if non_default:
                raise ValueError(
                    "mode='double_buffered' is not supported when non-default insertion_fill "
                    f"strategies are set ({list(non_default)}); use mode='buffered' instead."
                )

        schema: dict = {
            "with_seqs": ds.sequence_type,
            "with_tracks": ds.active_tracks if ds.active_tracks else False,
            "deterministic": ds.deterministic,
            "rc_neg": ds.rc_neg,
            "jitter": ds.jitter,
        }

        seqs = ds._seqs
        if isinstance(seqs, Haps):
            if seqs.min_af is not None:
                schema["min_af"] = seqs.min_af
            if seqs.max_af is not None:
                schema["max_af"] = seqs.max_af
            if seqs.filter is not None:
                schema["var_filter"] = seqs.filter
            if hasattr(seqs, "var_fields"):
                schema["var_fields"] = list(seqs.var_fields)

        # Pass reference path so the producer can reopen with a reference genome.
        ref = getattr(ds, "reference", None)
        if ref is not None:
            ref_path = getattr(ref, "path", None)
            if ref_path is not None:
                schema["reference_path"] = str(ref_path)
                # Detect whether the reference was opened as a memmap (in_memory=False).
                # If so, tell the producer subprocess to do the same, ensuring
                # identical data is returned for the same (r, s) indices.
                ref_data = getattr(ref, "reference", None)
                schema["reference_in_memory"] = not isinstance(ref_data, np.memmap)

        ds_path = ds.path
        if not ds_path.is_dir():
            raise RuntimeError(
                f"mode='double_buffered' requires a file-backed dataset (got path={ds_path!r}). "
                "The dummy in-memory dataset cannot be used; open a dataset via Dataset.open(path)."
            )

        proc = self._ctx.Process(
            target=producer_main,
            args=(
                str(ds_path),
                schema,
                list(self._shm_names),
                self._events,
                self._index_queue,
                self._exc_q,
            ),
            daemon=True,
        )
        proc.start()
        self._producer = proc
        self._producer_holder[0] = proc

    def __iter__(self):
        # Spawn producer on first iteration; reuse on subsequent epochs.
        if self._producer is None:
            self._spawn_producer()

        # Re-initialise slot events for this epoch.
        for free, ready in self._events:
            free.set()
            ready.clear()

        # Enqueue all chunk work items upfront.
        for i, (cr, cs, nb) in enumerate(self._chunks):
            self._index_queue.put((i % 2, cr, cs, nb))

        # Consume chunks in order, yielding mini-batches.
        for i, (_cr, _cs, _nb) in enumerate(self._chunks):
            slot_idx = i % 2
            _free, ready = self._events[slot_idx]

            # Wait for producer to fill the slot; check liveness on timeout.
            while not ready.wait(timeout=self._heartbeat):
                if not self._producer.is_alive():
                    raise self._reraise_or_die()

            # Also verify no exception slipped through before we got here.
            if not self._exc_q.empty():
                raise self._reraise_or_die()

            _n_inst, views = read_chunk(self._shms[slot_idx].buf, copy=self._copy)
            # Restore intermediate fixed dims (e.g. ploidy) that were flattened
            # in the shm round-trip for Ragged arrays.
            views = _reshape_ragged_for_chunk(views, int(_n_inst))
            chunk_output: object = tuple(views) if len(views) > 1 else views[0]

            for mini in slice_chunk(chunk_output, self._batch_size):
                if self._copy:
                    # read_chunk with copy=True already owns data; no extra copy needed.
                    yield mini
                else:
                    # Zero-copy: caller must consume before next iteration.
                    yield mini

            # Signal the producer that this slot is free to overwrite.
            ready.clear()
            _free.set()

    def _reraise_or_die(self) -> RuntimeError:
        """Build a RuntimeError from exc_q contents or a generic ProducerDied."""
        try:
            if not self._exc_q.empty():
                tname, msg, tb = self._exc_q.get_nowait()
                return RuntimeError(f"ProducerError ({tname}): {msg}\n{tb}")
        except Exception:
            pass
        return RuntimeError(
            "ProducerDied: producer subprocess exited without an exception in the queue"
        )

    def __len__(self) -> int:
        return sum(nb for _r, _s, nb in self._chunks)

    def close(self) -> None:
        """Shut down the producer and release shared-memory resources."""
        producer = self._producer
        if producer is not None and producer.is_alive():
            try:
                self._index_queue.put(None)
                producer.join(timeout=5)
                if producer.is_alive():
                    producer.terminate()
                    producer.join(timeout=2)
            except Exception:
                pass

        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        # Clear list so finalizer is idempotent.
        self._shms.clear()


def make_double_buffered_dataset(
    dataset: "Dataset",
    batch_size: int,
    slot_bytes: int,
    bytes_per_instance: np.ndarray,
    flat_r: np.ndarray,
    flat_s: np.ndarray,
    copy: bool = True,
    heartbeat_seconds: float = 60.0,
) -> "td.IterableDataset":
    """Create a PyTorch IterableDataset backed by the double-buffered producer.

    Parameters
    ----------
    dataset:
        The gvl Dataset to read from (must be file-backed).
    batch_size:
        Mini-batch size.
    slot_bytes:
        Byte budget per shared-memory slot (= buffer_bytes / 2).
    bytes_per_instance:
        2-D array of shape (n_regions, n_samples) with per-instance byte costs.
    flat_r, flat_s:
        Flattened region/sample indices for the epoch.
    copy:
        If True (default), batches own their data. If False, zero-copy views
        are returned — valid only until the next batch is yielded.
    heartbeat_seconds:
        Seconds to wait per slot before checking producer liveness.
    """
    import torch.utils.data as td

    class _DBTorchDataset(td.IterableDataset):
        def __init__(self_inner) -> None:
            self_inner._impl = _DoubleBufferedIterable(
                dataset,
                batch_size,
                slot_bytes,
                bytes_per_instance,
                flat_r,
                flat_s,
                copy,
                heartbeat_seconds,
            )

        def __iter__(self_inner):
            return iter(self_inner._impl)

        def __len__(self_inner) -> int:
            return len(self_inner._impl)

    return _DBTorchDataset()
