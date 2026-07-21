"""mode='double_buffered' dataloader: subprocess producer + 2-slot shm ping-pong."""

from __future__ import annotations

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


def _token_alphabet_from_lut(lut: "np.ndarray", unknown_token: int) -> bytes:
    """Recover the ordered alphabet bytes a ``token_lut`` was built from.

    ``build_token_lut`` (``_flat_flanks.py``) assigns each alphabet byte's
    position in the input alphabet as its token id (``0..len(alphabet)-1``) and
    fills every other byte with ``unknown_token``; ``Haps`` only retains the
    resulting LUT, not the original alphabet. This inverts that mapping by
    sorting the non-``unknown_token`` byte values by their token id, which
    recovers the alphabet verbatim as long as ``unknown_token`` doesn't collide
    with a real alphabet token id (the standard usage, e.g. ``unknown_token=len(alphabet)``).

    Args:
        lut: 256-entry byte->token lookup table.
        unknown_token: Token id assigned to bytes outside the alphabet.

    Returns:
        The reconstructed alphabet, in original order.
    """
    byte_vals = np.nonzero(lut != unknown_token)[0]
    order = np.argsort(lut[byte_vals])
    return bytes(byte_vals[order].astype(np.uint8).tolist())


def _reshape_ragged_for_chunk(views: list, n_instances: int) -> list:
    """Re-introduce the ploidy axis on any Ragged / _Flat that the shm reader flattened to (n_groups, None).

    _FlatVariants / RaggedVariants already carry ploidy (regular_size) and are left unchanged.
    """
    from seqpro.rag import Ragged

    from ._ragged import RaggedAnnotatedHaps
    from ._flat import _Flat, _FlatAnnotatedHaps
    from ._dataset._rag_variants import RaggedVariants

    def _reshape_one(arr):
        # RaggedVariants is a record Ragged that already carries the ploidy axis;
        # it is now a Ragged subclass, so it would otherwise enter the generic
        # `isinstance(arr, Ragged)` branch below, which assumes a single-field
        # Ragged with .data/.offsets (invalid on a record). Leave it unchanged.
        if isinstance(arr, RaggedVariants):
            return arr
        if isinstance(arr, Ragged):
            n_groups = arr.shape[0]
            if (
                n_groups != n_instances
                and n_instances > 0
                and n_groups % n_instances == 0
            ):
                ploidy = n_groups // n_instances
                arr = Ragged.from_offsets(
                    arr.data, (n_instances, ploidy, None), arr.offsets
                )
            return arr
        if isinstance(arr, _Flat):
            n_groups = arr.shape[0]
            if (
                n_groups != n_instances
                and n_instances > 0
                and n_groups % n_instances == 0
            ):
                ploidy = n_groups // n_instances
                arr = arr.reshape((n_instances, ploidy))
            return arr
        return arr

    result: list = []
    for arr in views:
        if isinstance(arr, RaggedAnnotatedHaps):
            arr = RaggedAnnotatedHaps(
                haps=_reshape_one(arr.haps),
                var_idxs=_reshape_one(arr.var_idxs),
                ref_coords=_reshape_one(arr.ref_coords),
            )
        elif isinstance(arr, _FlatAnnotatedHaps):
            arr = _FlatAnnotatedHaps(
                haps=_reshape_one(arr.haps),
                var_idxs=_reshape_one(arr.var_idxs),
                ref_coords=_reshape_one(arr.ref_coords),
            )
        else:
            arr = _reshape_one(arr)
        result.append(arr)
    return result


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

        planner = ChunkPlanner(
            r_idx=flat_r,
            s_idx=flat_s,
            batch_size=batch_size,
            bytes_per_instance=bytes_per_instance,
            slot_bytes=slot_bytes,
        )
        self._chunks: list[tuple[np.ndarray, np.ndarray, int]] = list(planner)
        peak = planner.peak_chunk_bytes

        capacity = HEADER_RESERVED + peak + 4096
        suffix = uuid.uuid4().hex[:8]
        self._shm_names = [f"gvl-{os.getpid()}-{suffix}-{i}" for i in range(2)]
        self._shms = [
            SharedMemory(create=True, name=n, size=capacity) for n in self._shm_names
        ]

        ctx = mp.get_context("spawn")
        self._ctx = ctx
        # events[i] = (free_event, ready_event); free starts set, ready starts clear.
        self._events: list[tuple] = [(ctx.Event(), ctx.Event()) for _ in range(2)]
        for free, ready in self._events:
            free.set()
            ready.clear()

        self._index_queue = ctx.Queue()
        self._exc_q: mp.Queue = ctx.Queue()
        self._producer: mp.Process | None = None

        shm_snapshot = list(self._shms)
        producer_holder: list[mp.Process | None] = [None]

        def _producer_getter():
            return producer_holder[0]

        self._producer_holder = producer_holder
        # weakref.finalize handles cleanup both on garbage collection AND at
        # interpreter exit, without retaining a strong reference. Do NOT also
        # atexit.register(self.close): a bound method keeps the loader alive for
        # the whole process, so per-loader producers + shm would accumulate
        # (e.g. one loader per bench cell) until exit and exhaust RAM.
        weakref.finalize(self, _cleanup, shm_snapshot, _producer_getter)

    def _spawn_producer(self) -> None:
        from ._producer import producer_main
        from ._dataset._reconstruct import Haps
        from ._dataset._insertion_fill import Repeat5p

        ds = self._dataset

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
            "output_format": ds.output_format,
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

            window_opt = getattr(seqs, "window_opt", None)
            if window_opt is not None:
                schema["window_opt"] = {
                    "flank_length": window_opt.flank_length,
                    "token_alphabet": window_opt.token_alphabet,
                    "unknown_token": window_opt.unknown_token,
                    "ref": window_opt.ref,
                    "alt": window_opt.alt,
                }
            elif getattr(seqs, "flank_length", None) and seqs.token_lut is not None:
                # Plain-variants ride-along flank tokens (Config B). ``Haps``
                # only retains the derived LUT, not the original
                # ``token_alphabet`` it was built from, so recover it.
                schema["flank_length"] = seqs.flank_length
                schema["token_alphabet"] = _token_alphabet_from_lut(
                    seqs.token_lut, seqs.unknown_token
                )
                schema["unknown_token"] = seqs.unknown_token

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
        if self._producer is None:
            self._spawn_producer()

        for free, ready in self._events:
            free.set()
            ready.clear()

        # Enqueue all chunk work items upfront.
        for i, (cr, cs, nb) in enumerate(self._chunks):
            # slot_idx = i % 2 works because the producer drains the queue in
            # order, so slot ownership rotates deterministically between the two
            # sides and is never double-assigned.
            self._index_queue.put((i % 2, cr, cs, nb))

        flat = self._dataset.output_format == "flat"

        for i, (_cr, _cs, _nb) in enumerate(self._chunks):
            slot_idx = i % 2
            _free, ready = self._events[slot_idx]

            while not ready.wait(timeout=self._heartbeat):
                if not self._producer.is_alive():
                    raise self._reraise_or_die()

            if not self._exc_q.empty():
                raise self._reraise_or_die()

            _n_inst, views = read_chunk(
                self._shms[slot_idx].buf, copy=self._copy, flat=flat
            )
            views = _reshape_ragged_for_chunk(views, int(_n_inst))
            chunk_output: object = tuple(views) if len(views) > 1 else views[0]

            yield from slice_chunk(chunk_output, self._batch_size)

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
        self._shms.clear()  # idempotent: prevents double-unlink in finalizer


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

    Args:
        dataset: The gvl Dataset to read from (must be file-backed).
        batch_size: Mini-batch size.
        slot_bytes: Byte budget per shared-memory slot (= buffer_bytes / 2).
        bytes_per_instance: 2-D array of shape (n_regions, n_samples) with per-instance byte
            costs.
        flat_r: Flattened region indices for the epoch.
        flat_s: Flattened sample indices for the epoch.
        copy: If True (default), batches own their data. If False, zero-copy views
            are returned — valid only until the next batch is yielded.
        heartbeat_seconds: Seconds to wait per slot before checking producer liveness.
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
