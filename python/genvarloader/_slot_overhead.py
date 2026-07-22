"""Per-chunk serialization overhead not captured by _output_bytes_per_instance.

The byte estimate charges exact per-instance payload + per-instance offset entries,
but the serializer (_shm_layout.write_chunk) additionally writes, per serialized
offset array, one +1 terminator entry (8 bytes), and _align-pads (<=8 bytes) before
every serialized array. Those are per-chunk constants (independent of instance count)
that must be covered by the slot's fixed slack. This module derives a true upper
bound on them from the schema, replacing the historical magic 4096.
"""

from __future__ import annotations

_OFF = 8  # int64 offset entry / terminator
_ALIGN = 8  # max _align(8) padding per serialized array


def _array_counts(dataset) -> tuple[int, int]:
    """(n_offset_arrays, n_serialized_arrays) the serializer emits for one chunk.

    Derived from the dataset's active output schema. Upper bound; over-counting
    is safe.
    """
    seq = dataset.sequence_type
    n_off = 0
    n_arr = 0
    seqs = getattr(dataset, "_seqs", None)
    var_fields = list(getattr(seqs, "var_fields", []) or [])
    # scalar .fields: always-emitted "start" plus non-allele var_fields.
    scalars = {f for f in var_fields if f not in ("alt", "ref")}
    scalars.add("start")
    if seq == "variant-windows":
        n_scalar = len(scalars)
        n_window_slots = 2  # exactly one ref-derived + one alt-derived slot
        n_off += (
            n_scalar * 1 + n_window_slots * 2
        )  # scalars: outer; windows: outer+inner
        n_arr += n_scalar * 2 + n_window_slots * 3  # +1 data array each
    elif seq == "variants":
        n_scalar = len(scalars)
        n_allele = sum(1 for f in var_fields if f in ("alt", "ref"))
        n_off += n_scalar * 1 + n_allele * 2
        n_arr += n_scalar * 2 + n_allele * 3
        # optional flank_tokens payload: _write_flat_variants emits one extra
        # data + offset array when present. Charge it unconditionally (an
        # upper bound; harmless over-count when flanks are absent).
        n_off += 1
        n_arr += 2
    else:
        # reference / haplotypes / annotated / none: few arrays; the 4096 floor
        # dominates. Charge a generous constant so the floor is never exceeded.
        n_off += 8
        n_arr += 8
    # active tracks: each is a single-level ragged (1 offset + 1 data array).
    n_tracks = len(getattr(dataset, "active_tracks", None) or {})
    n_off += n_tracks
    n_arr += n_tracks * 2
    return n_off, n_arr


def slot_overhead_bytes(dataset) -> int:
    """Upper bound on per-chunk overhead beyond the per-instance byte estimate.

    Args:
        dataset: The gvl Dataset whose active output schema fixes the field/array
            structure serialized per chunk.

    Returns:
        Bytes to add to peak_chunk_bytes when sizing a double_buffered shm slot,
        floored at 4096 (covers the header-adjacent slack the format has always
        assumed).
    """
    n_off, n_arr = _array_counts(dataset)
    return max(4096, _OFF * n_off + _ALIGN * n_arr)
