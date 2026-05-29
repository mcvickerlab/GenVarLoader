"""Producer subprocess entrypoint for mode='double_buffered'."""
from __future__ import annotations

import os
import traceback
from multiprocessing.shared_memory import SharedMemory

import genvarloader as gvl
from ._shm_layout import write_chunk


def _apply_schema(ds, schema: dict):
    """Reapply a serializable schema dict to a freshly opened Dataset."""
    if schema.get("with_seqs", "UNSET") != "UNSET":
        ds = ds.with_seqs(schema["with_seqs"])
    if "with_tracks" in schema:
        ds = ds.with_tracks(schema["with_tracks"])
    settings_kwargs: dict = {}
    if schema.get("deterministic") is not None:
        settings_kwargs["deterministic"] = schema["deterministic"]
    if schema.get("rc_neg") is not None:
        settings_kwargs["rc_neg"] = schema["rc_neg"]
    if schema.get("jitter") is not None:
        settings_kwargs["jitter"] = schema["jitter"]
    if schema.get("min_af") is not None:
        settings_kwargs["min_af"] = schema["min_af"]
    if schema.get("max_af") is not None:
        settings_kwargs["max_af"] = schema["max_af"]
    if schema.get("var_filter") is not None:
        settings_kwargs["var_filter"] = schema["var_filter"]
    if schema.get("var_fields") is not None:
        settings_kwargs["var_fields"] = schema["var_fields"]
    if settings_kwargs:
        ds = ds.with_settings(**settings_kwargs)
    return ds


def producer_main(
    dataset_path: str,
    schema: dict,
    shm_names: list[str],
    events: list[tuple],
    index_queue,
    exc_q,
) -> None:
    """Main function for the producer subprocess.

    Opens the dataset, reapplies the schema, and loops on index_queue items.
    Each item is a tuple of (slot_idx, r_idx, s_idx, n_batches).
    Sentinel None item causes clean exit.
    Any exception is pushed to exc_q as (type_name, message, traceback).
    """
    try:
        # Test-only hook (also used in Task 13 crash tests).
        if os.environ.get("GVL_TEST_PRODUCER_RAISE") == "1":
            raise RuntimeError("test-injected producer failure")

        reference_path = schema.get("reference_path")
        reference_in_memory = schema.get("reference_in_memory", True)
        if reference_path is not None:
            from ._dataset._reference import Reference
            ref = Reference.from_path(reference_path, in_memory=reference_in_memory)
            ds = gvl.Dataset.open(dataset_path, reference=ref)
        else:
            ds = gvl.Dataset.open(dataset_path)
        ds = _apply_schema(ds, schema)

        shms = [SharedMemory(name=n) for n in shm_names]
        try:
            while True:
                item = index_queue.get()
                if item is None:
                    return
                slot_idx, r_idx, s_idx, _n_batches = item
                free, ready = events[slot_idx]
                free.wait()
                chunk = ds[r_idx, s_idx]
                arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]
                write_chunk(shms[slot_idx].buf, arrays, n_instances=len(r_idx))
                free.clear()
                ready.set()
        finally:
            for s in shms:
                try:
                    s.close()
                except Exception:
                    pass
    except Exception as e:
        try:
            exc_q.put((type(e).__name__, str(e), traceback.format_exc()))
        except Exception:
            pass
