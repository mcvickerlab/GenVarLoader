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
        if schema.get("window_opt") is not None:
            from ._dataset._flat_variants import VarWindowOpt

            ds = ds.with_seqs(schema["with_seqs"], VarWindowOpt(**schema["window_opt"]))
        else:
            ds = ds.with_seqs(schema["with_seqs"])
    if "with_tracks" in schema:
        ds = ds.with_tracks(schema["with_tracks"])
    if schema.get("output_format") == "flat":
        ds = ds.with_output_format("flat")
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
    if schema.get("unphased_union") is not None:
        settings_kwargs["unphased_union"] = schema["unphased_union"]
    if schema.get("flank_length") is not None:
        settings_kwargs["flank_length"] = schema["flank_length"]
        settings_kwargs["token_alphabet"] = schema["token_alphabet"]
        settings_kwargs["unknown_token"] = schema["unknown_token"]
    if schema.get("dummy_variant") is not None:
        from ._dataset._flat_variants import DummyVariant

        settings_kwargs["dummy_variant"] = DummyVariant(**schema["dummy_variant"])
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
    """Producer subprocess: opens dataset, replays schema, writes chunks to shm slots."""
    try:
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
