"""Shared-memory slot layout: hand-rolled header + payload.

Header layout (little-endian throughout):
  u64 n_instances
  u64 payload_bytes
  u8  n_arrays
  ArrayDescriptor[n_arrays]:
    u8   kind            (0=dense, 1=ragged_seqpro, 2=ragged_variants)
    u8   ndim
    char dtype_str[4]   (np.dtype.str, e.g. '<f4', '|S1', zero-padded to 4 bytes)
    u64  shape[ndim]
    u64  data_offset
    u64  data_nbytes
    u64  offsets_offset   (byte offset into slot; 0 if dense or not applicable)
    u64  offsets_nbytes   (0 if dense)
    u64  inner_offsets_offset  (used by ragged_variants alt/ref fields; 0 otherwise)
    u64  inner_offsets_nbytes
    u8   name_len         (length of field name in bytes, 0 for non-RaggedVariants)
    char name[name_len]   (UTF-8 field name, variable length)

  Kind=2 (ragged_variants) is a special sentinel written *once* as the sole
  "array" for a RaggedVariants chunk.  The payload encodes each field's buffers
  in sequence; the field names and per-field descriptor info are embedded in the
  single kind=2 descriptor block.  Specifically, for kind=2 the descriptor
  encodes:
    u8  n_fields
    FieldDescriptor[n_fields]:
      u8   field_kind     (0=numeric, 1=alleles)
      char dtype_str[4]   (leaf dtype)
      u64  outer_offsets_offset
      u64  outer_offsets_nbytes
      u64  inner_offsets_offset  (alleles only; 0 for numeric)
      u64  inner_offsets_nbytes
      u64  data_offset
      u64  data_nbytes
      u64  regular_size   (ploidy; RegularArray.size)
      u8   name_len
      char name[name_len]
"""

from __future__ import annotations

import struct
from typing import Sequence

import numpy as np

HEADER_RESERVED = 4096

_PREAMBLE = struct.Struct("<QQB")  # n_instances, payload_bytes, n_arrays


def _align(off: int, align: int = 8) -> int:
    rem = off % align
    return off if rem == 0 else off + (align - rem)


def _dtype_to_bytes(dt: np.dtype) -> bytes:
    """Encode dtype as a 4-byte zero-padded ASCII string (dtype.str)."""
    s = dt.str  # e.g. '<f4', '|S1', '|u1'
    encoded = s.encode("ascii")
    assert len(encoded) <= 4, f"dtype.str {s!r} too long"
    return encoded.ljust(4, b"\x00")


def _dtype_from_bytes(b: bytes) -> np.dtype:
    """Decode a 4-byte dtype descriptor back to np.dtype."""
    s = b.rstrip(b"\x00").decode("ascii")
    return np.dtype(s)


def slot_capacity_for(arrays: Sequence[np.ndarray]) -> int:
    """Worst-case slot bytes for the given dense arrays.

    Used by tests; production sizing uses ChunkPlanner.peak_chunk_bytes + header.
    """
    payload = sum(_align(a.nbytes) for a in arrays)
    return HEADER_RESERVED + payload


def _write_dense(buf: memoryview, a: np.ndarray, cursor: int) -> tuple[dict, int]:
    """Write a dense ndarray into buf at cursor. Returns descriptor dict and new cursor."""
    cursor = _align(cursor)
    data_off = cursor
    a_c = np.ascontiguousarray(a)
    np.frombuffer(buf, dtype=a_c.dtype, count=a_c.size, offset=data_off).reshape(
        a_c.shape
    )[...] = a_c
    cursor += a_c.nbytes
    return {
        "kind": 0,
        "dtype_str": _dtype_to_bytes(a_c.dtype),
        "shape": list(a_c.shape),
        "data_offset": data_off,
        "data_nbytes": a_c.nbytes,
        "offsets_offset": 0,
        "offsets_nbytes": 0,
        "inner_offsets_offset": 0,
        "inner_offsets_nbytes": 0,
        "name": b"",
    }, cursor


def _write_ragged(buf: memoryview, a, cursor: int) -> tuple[dict, int]:
    """Write a seqpro.rag.Ragged into buf. Returns descriptor dict and new cursor."""
    data_arr = np.ascontiguousarray(a.data)
    off_arr = np.ascontiguousarray(a.offsets)

    cursor = _align(cursor)
    data_off = cursor
    np.frombuffer(buf, dtype=data_arr.dtype, count=data_arr.size, offset=data_off)[
        ...
    ] = data_arr.ravel()
    cursor += data_arr.nbytes

    cursor = _align(cursor)
    off_off = cursor
    np.frombuffer(buf, dtype=off_arr.dtype, count=off_arr.size, offset=off_off)[...] = (
        off_arr
    )
    cursor += off_arr.nbytes

    return {
        "kind": 1,
        "dtype_str": _dtype_to_bytes(data_arr.dtype),
        "shape": [data_arr.size],  # flat length in header; offsets carry grouping info
        "data_offset": data_off,
        "data_nbytes": data_arr.nbytes,
        "offsets_offset": off_off,
        "offsets_nbytes": off_arr.nbytes,
        "inner_offsets_offset": 0,
        "inner_offsets_nbytes": 0,
        "name": b"",
    }, cursor


def _write_rag_variants(buf: memoryview, rv, cursor: int) -> tuple[dict, int]:
    """Write a RaggedVariants into buf via _core.Ragged buffers.

    Layout per kind=2 block:
      The descriptor's 'shape' carries [n_fields] so the reader knows how many
      FieldDescriptors follow. Each field is encoded inline as a FieldDescriptor
      (see module docstring). The byte layout is IDENTICAL to _write_flat_variants
      so kind=2 descriptors are interchangeable across the flat and ragged paths.

    For each field of rv:
      - Numeric (field_kind=0): outer_offsets = field.offsets (b*p+1 int64),
        leaf_data = field.data.
      - Alleles (field_kind=1): outer_offsets = field.offsets (b*p+1 int64),
        inner_offsets = field._rl.str_offsets (n_variants+1 int64),
        leaf_data = field.data (S1 bytes).
    ploidy (regular_size) is rv.shape[1].
    """
    fields = rv.fields
    n_fields = len(fields)
    regular_size = int(rv.shape[1])  # ploidy

    field_descs: list[dict] = []

    for fname in fields:
        f = rv[fname]

        outer_offsets = np.ascontiguousarray(f.offsets, dtype=np.int64)
        leaf_data = np.ascontiguousarray(f.data)

        if getattr(f, "is_string", False):
            # Allele field (opaque-string Ragged, shape (b, p, ~v)):
            # _rl.str_offsets holds per-variant char boundaries (n_variants+1,).
            inner_offsets = np.ascontiguousarray(f._rl.str_offsets, dtype=np.int64)
            field_kind = 1
        else:
            # Numeric field (shape (b, p, ~v)):
            inner_offsets = np.empty(0, dtype=np.int64)
            field_kind = 0

        cursor = _align(cursor)
        outer_off = cursor
        np.frombuffer(
            buf, dtype=outer_offsets.dtype, count=outer_offsets.size, offset=outer_off
        )[...] = outer_offsets
        cursor += outer_offsets.nbytes

        if field_kind == 1:  # alleles: inner offsets present; numeric: skip
            cursor = _align(cursor)
            inner_off = cursor
            np.frombuffer(
                buf,
                dtype=inner_offsets.dtype,
                count=inner_offsets.size,
                offset=inner_off,
            )[...] = inner_offsets
            cursor += inner_offsets.nbytes
        else:
            inner_off = 0

        cursor = _align(cursor)
        data_off = cursor
        np.frombuffer(
            buf, dtype=leaf_data.dtype, count=leaf_data.size, offset=data_off
        )[...] = leaf_data.ravel()
        cursor += leaf_data.nbytes

        name_bytes = fname.encode("utf-8")
        field_descs.append(
            {
                "field_kind": field_kind,
                "dtype_str": _dtype_to_bytes(leaf_data.dtype),
                "outer_offsets_offset": outer_off,
                "outer_offsets_nbytes": outer_offsets.nbytes,
                "inner_offsets_offset": inner_off,
                "inner_offsets_nbytes": inner_offsets.nbytes if field_kind == 1 else 0,
                "data_offset": data_off,
                "data_nbytes": leaf_data.nbytes,
                "regular_size": regular_size,
                "name": name_bytes,
            }
        )

    return {
        "kind": 2,
        "dtype_str": b"\x00" * 4,
        "shape": [n_fields],
        "data_offset": 0,
        "data_nbytes": 0,
        "offsets_offset": 0,
        "offsets_nbytes": 0,
        "inner_offsets_offset": 0,
        "inner_offsets_nbytes": 0,
        "name": b"",
        "_field_descs": field_descs,
    }, cursor


def _write_rag_annotated(buf: memoryview, rah, cursor: int) -> tuple[dict, int]:
    """Write a RaggedAnnotatedHaps (kind=3) into buf.

    Stores its three ragged components (haps S1, var_idxs int32, ref_coords
    int32) as data+offsets segments under a single descriptor. The consumer
    re-introduces the (n_inst, ploidy) axis via _reshape_ragged_for_chunk.
    """
    sub_descs: list[dict] = []
    for comp in (rah.haps, rah.var_idxs, rah.ref_coords):
        data_arr = np.ascontiguousarray(comp.data)
        off_arr = np.ascontiguousarray(comp.offsets)

        cursor = _align(cursor)
        data_off = cursor
        np.frombuffer(buf, dtype=data_arr.dtype, count=data_arr.size, offset=data_off)[
            ...
        ] = data_arr.ravel()
        cursor += data_arr.nbytes

        cursor = _align(cursor)
        off_off = cursor
        np.frombuffer(buf, dtype=off_arr.dtype, count=off_arr.size, offset=off_off)[
            ...
        ] = off_arr
        cursor += off_arr.nbytes

        sub_descs.append(
            {
                "dtype_str": _dtype_to_bytes(data_arr.dtype),
                "data_offset": data_off,
                "data_nbytes": data_arr.nbytes,
                "offsets_offset": off_off,
                "offsets_nbytes": off_arr.nbytes,
            }
        )

    return {
        "kind": 3,
        "dtype_str": b"\x00" * 4,
        "shape": [len(sub_descs)],
        "data_offset": 0,
        "data_nbytes": 0,
        "offsets_offset": 0,
        "offsets_nbytes": 0,
        "inner_offsets_offset": 0,
        "inner_offsets_nbytes": 0,
        "name": b"",
        "_sub_descs": sub_descs,
    }, cursor


def _pack_descriptor(d: dict) -> bytes:
    """Pack one top-level array descriptor into bytes."""
    kind = d["kind"]
    ndim = len(d["shape"])
    name = d.get("name", b"")
    out = struct.pack("<BB4s", kind, ndim, d["dtype_str"])
    for dim in d["shape"]:
        out += struct.pack("<Q", int(dim))
    out += struct.pack(
        "<6Q",
        d["data_offset"],
        d["data_nbytes"],
        d["offsets_offset"],
        d["offsets_nbytes"],
        d["inner_offsets_offset"],
        d["inner_offsets_nbytes"],
    )
    out += struct.pack("<B", len(name))
    out += name

    if kind == 2:
        field_descs = d["_field_descs"]
        out += struct.pack("<B", len(field_descs))
        for fd in field_descs:
            fname = fd["name"]
            out += struct.pack(
                "<B4s7QB",
                fd["field_kind"],
                fd["dtype_str"],
                fd["outer_offsets_offset"],
                fd["outer_offsets_nbytes"],
                fd["inner_offsets_offset"],
                fd["inner_offsets_nbytes"],
                fd["data_offset"],
                fd["data_nbytes"],
                fd["regular_size"],
                len(fname),
            )
            out += fname

    if kind == 4:
        field_descs = d["_field_descs"]
        out += struct.pack("<B", len(field_descs))
        for fd in field_descs:
            fname = fd["name"]
            out += struct.pack(
                "<B4s7QB",
                fd["field_kind"],
                fd["dtype_str"],
                fd["outer_offsets_offset"],
                fd["outer_offsets_nbytes"],
                fd["inner_offsets_offset"],
                fd["inner_offsets_nbytes"],
                fd["data_offset"],
                fd["data_nbytes"],
                fd["regular_size"],
                len(fname),
            )
            out += fname

    if kind == 3:
        sub_descs = d["_sub_descs"]
        out += struct.pack("<B", len(sub_descs))
        for sd in sub_descs:
            out += struct.pack(
                "<4s4Q",
                sd["dtype_str"],
                sd["data_offset"],
                sd["data_nbytes"],
                sd["offsets_offset"],
                sd["offsets_nbytes"],
            )

    return out


def write_chunk(
    buf: memoryview,
    arrays: Sequence,
    n_instances: int,
) -> int:
    """Write arrays into the shared-memory slot.

    Supports np.ndarray (kind=0), seqpro.rag.Ragged (kind=1),
    and RaggedVariants (kind=2). The flat containers ``_Flat`` (kind=1),
    ``_FlatVariants`` (kind=2), ``_FlatAnnotatedHaps`` (kind=3), and
    ``RaggedAnnotatedHaps`` (kind=3) serialize into the same on-wire kinds
    as their ragged counterparts; the ``flat`` flag on ``read_chunk`` selects
    which reader reconstructs them.

    Returns total bytes consumed (header + payload).
    """
    from seqpro.rag import Ragged
    from ._dataset._rag_variants import RaggedVariants
    from ._ragged import RaggedAnnotatedHaps
    from ._flat import _Flat, _FlatAnnotatedHaps
    from ._dataset._flat_variants import _FlatVariants, _FlatVariantWindows

    if len(arrays) > 255:
        raise ValueError("at most 255 arrays per chunk")

    descriptors: list[dict] = []
    cursor = HEADER_RESERVED

    for a in arrays:
        if isinstance(a, _FlatVariantWindows):
            desc, cursor = _write_flat_variant_windows(buf, a, cursor)
        elif isinstance(a, _FlatVariants):
            desc, cursor = _write_flat_variants(buf, a, cursor)
        elif isinstance(a, RaggedVariants):
            desc, cursor = _write_rag_variants(buf, a, cursor)
        elif isinstance(a, (RaggedAnnotatedHaps, _FlatAnnotatedHaps)):
            desc, cursor = _write_rag_annotated(buf, a, cursor)
        elif isinstance(a, (Ragged, _Flat)):
            desc, cursor = _write_ragged(buf, a, cursor)
        elif isinstance(a, np.ndarray):
            desc, cursor = _write_dense(buf, a, cursor)
        else:
            raise TypeError(f"write_chunk: unsupported array type {type(a)}")
        descriptors.append(desc)

    payload_bytes = cursor - HEADER_RESERVED

    hdr = bytearray()
    hdr += _PREAMBLE.pack(n_instances, payload_bytes, len(descriptors))
    for d in descriptors:
        hdr += _pack_descriptor(d)

    if len(hdr) > HEADER_RESERVED:
        raise ValueError(
            f"Header too large ({len(hdr)} bytes) for HEADER_RESERVED={HEADER_RESERVED}. "
            "Increase HEADER_RESERVED or reduce the number of arrays/fields."
        )
    buf[: len(hdr)] = bytes(hdr)
    return cursor


def _unpack_one_descriptor(buf_bytes: memoryview, cursor: int) -> tuple[dict, int]:
    """Unpack one array descriptor starting at cursor into the header bytes region."""
    kind, ndim = struct.unpack_from("<BB", buf_bytes, cursor)
    cursor += 2
    dtype_str = bytes(buf_bytes[cursor : cursor + 4])
    cursor += 4
    shape = []
    for _ in range(ndim):
        (dim,) = struct.unpack_from("<Q", buf_bytes, cursor)
        shape.append(int(dim))
        cursor += 8
    (
        data_offset,
        data_nbytes,
        offsets_offset,
        offsets_nbytes,
        inner_offsets_offset,
        inner_offsets_nbytes,
    ) = struct.unpack_from("<6Q", buf_bytes, cursor)
    cursor += 48
    (name_len,) = struct.unpack_from("<B", buf_bytes, cursor)
    cursor += 1
    name = bytes(buf_bytes[cursor : cursor + name_len]).decode("utf-8")
    cursor += name_len

    d = {
        "kind": kind,
        "dtype_str": dtype_str,
        "shape": shape,
        "data_offset": data_offset,
        "data_nbytes": data_nbytes,
        "offsets_offset": offsets_offset,
        "offsets_nbytes": offsets_nbytes,
        "inner_offsets_offset": inner_offsets_offset,
        "inner_offsets_nbytes": inner_offsets_nbytes,
        "name": name,
    }

    if kind == 2:
        (n_fields,) = struct.unpack_from("<B", buf_bytes, cursor)
        cursor += 1
        field_descs = []
        for _ in range(n_fields):
            (field_kind,) = struct.unpack_from("<B", buf_bytes, cursor)
            cursor += 1
            fdtype_str = bytes(buf_bytes[cursor : cursor + 4])
            cursor += 4
            (
                outer_offsets_offset,
                outer_offsets_nbytes,
                inner_offsets_offset_fd,
                inner_offsets_nbytes_fd,
                fd_data_offset,
                fd_data_nbytes,
                regular_size,
            ) = struct.unpack_from("<7Q", buf_bytes, cursor)
            cursor += 56
            (fname_len,) = struct.unpack_from("<B", buf_bytes, cursor)
            cursor += 1
            fname = bytes(buf_bytes[cursor : cursor + fname_len]).decode("utf-8")
            cursor += fname_len
            field_descs.append(
                {
                    "field_kind": field_kind,
                    "dtype_str": fdtype_str,
                    "outer_offsets_offset": outer_offsets_offset,
                    "outer_offsets_nbytes": outer_offsets_nbytes,
                    "inner_offsets_offset": inner_offsets_offset_fd,
                    "inner_offsets_nbytes": inner_offsets_nbytes_fd,
                    "data_offset": fd_data_offset,
                    "data_nbytes": fd_data_nbytes,
                    "regular_size": regular_size,
                    "name": fname,
                }
            )
        d["_field_descs"] = field_descs

    if kind == 4:
        (n_fields,) = struct.unpack_from("<B", buf_bytes, cursor)
        cursor += 1
        field_descs = []
        for _ in range(n_fields):
            (field_kind,) = struct.unpack_from("<B", buf_bytes, cursor)
            cursor += 1
            fdtype_str = bytes(buf_bytes[cursor : cursor + 4])
            cursor += 4
            (
                outer_offsets_offset,
                outer_offsets_nbytes,
                inner_offsets_offset_fd,
                inner_offsets_nbytes_fd,
                fd_data_offset,
                fd_data_nbytes,
                regular_size,
            ) = struct.unpack_from("<7Q", buf_bytes, cursor)
            cursor += 56
            (fname_len,) = struct.unpack_from("<B", buf_bytes, cursor)
            cursor += 1
            fname = bytes(buf_bytes[cursor : cursor + fname_len]).decode("utf-8")
            cursor += fname_len
            field_descs.append(
                {
                    "field_kind": field_kind,
                    "dtype_str": fdtype_str,
                    "outer_offsets_offset": outer_offsets_offset,
                    "outer_offsets_nbytes": outer_offsets_nbytes,
                    "inner_offsets_offset": inner_offsets_offset_fd,
                    "inner_offsets_nbytes": inner_offsets_nbytes_fd,
                    "data_offset": fd_data_offset,
                    "data_nbytes": fd_data_nbytes,
                    "regular_size": regular_size,
                    "name": fname,
                }
            )
        d["_field_descs"] = field_descs

    if kind == 3:
        (n_subs,) = struct.unpack_from("<B", buf_bytes, cursor)
        cursor += 1
        sub_descs = []
        for _ in range(n_subs):
            sdtype_str = bytes(buf_bytes[cursor : cursor + 4])
            cursor += 4
            (
                sd_data_offset,
                sd_data_nbytes,
                sd_offsets_offset,
                sd_offsets_nbytes,
            ) = struct.unpack_from("<4Q", buf_bytes, cursor)
            cursor += 32
            sub_descs.append(
                {
                    "dtype_str": sdtype_str,
                    "data_offset": sd_data_offset,
                    "data_nbytes": sd_data_nbytes,
                    "offsets_offset": sd_offsets_offset,
                    "offsets_nbytes": sd_offsets_nbytes,
                }
            )
        d["_sub_descs"] = sub_descs

    return d, cursor


def _read_dense(buf: memoryview, d: dict, copy: bool = True) -> np.ndarray:
    dtype = _dtype_from_bytes(d["dtype_str"])
    shape = d["shape"]
    count = int(np.prod(shape)) if shape else 1
    view = np.frombuffer(
        buf, dtype=dtype, count=count, offset=d["data_offset"]
    ).reshape(shape)
    return view.copy() if copy else view


def _read_ragged(buf: memoryview, d: dict, copy: bool = True):
    from seqpro.rag import Ragged

    dtype = _dtype_from_bytes(d["dtype_str"])
    n_items = d["shape"][0]
    data = np.frombuffer(buf, dtype=dtype, count=n_items, offset=d["data_offset"])
    n_offsets = d["offsets_nbytes"] // 8
    offsets = np.frombuffer(
        buf, dtype=np.int64, count=n_offsets, offset=d["offsets_offset"]
    )
    if copy:
        data = data.copy()
        offsets = offsets.copy()
    n_groups = len(offsets) - 1
    return Ragged.from_offsets(data, (n_groups, None), offsets)


def _read_rag_variants(buf: memoryview, d: dict, copy: bool = True):
    """Reconstruct a RaggedVariants from a kind=2 descriptor via _core.Ragged.

    For each field:
      - Alleles (field_kind=1): rebuild as Ragged.from_offsets(char_data,
        (b*p, None, None), [outer_offsets, inner_offsets]).to_strings().reshape(b, p, None).
      - Numeric (field_kind=0): rebuild as Ragged.from_offsets(leaf, (b*p, None),
        outer_offsets).reshape(b, p, None).
    Fields share the same outer offsets object (required by Ragged.from_fields).
    """
    from seqpro.rag import Ragged
    from ._dataset._rag_variants import RaggedVariants, _share_offsets

    field_rags: dict[str, Ragged] = {}
    shared_outer: np.ndarray | None = None

    for fd in d["_field_descs"]:
        fname = fd["name"]
        leaf_dtype = _dtype_from_bytes(fd["dtype_str"])
        regular_size = fd["regular_size"]  # ploidy

        n_outer = fd["outer_offsets_nbytes"] // 8
        outer_offsets = np.frombuffer(
            buf, dtype=np.int64, count=n_outer, offset=fd["outer_offsets_offset"]
        )

        leaf_nbytes = fd["data_nbytes"]
        leaf_count = leaf_nbytes // leaf_dtype.itemsize
        leaf_data = np.frombuffer(
            buf, dtype=leaf_dtype, count=leaf_count, offset=fd["data_offset"]
        )

        if copy:
            outer_offsets = outer_offsets.copy()
            leaf_data = leaf_data.copy()

        # b*p = number of (batch, ploidy) cells = len(outer_offsets) - 1
        b_times_p = len(outer_offsets) - 1
        b = b_times_p // regular_size if regular_size else b_times_p

        if fd["field_kind"] == 1:
            # Allele field: outer_offsets (variant-level) + inner_offsets (char-level)
            n_inner = fd["inner_offsets_nbytes"] // 8
            inner_offsets = np.frombuffer(
                buf, dtype=np.int64, count=n_inner, offset=fd["inner_offsets_offset"]
            )
            if copy:
                inner_offsets = inner_offsets.copy()
            # Build S1-char Ragged with two ragged axes: (b*p, ~variants, ~chars)
            # then collapse the char axis into opaque strings and reshape to (b, p, ~v).
            rag = (
                Ragged.from_offsets(
                    leaf_data,
                    (b_times_p, None, None),
                    [outer_offsets, inner_offsets],
                )
                .to_strings()
                .reshape(b, regular_size, None)
            )
        else:
            # Numeric field: (b*p, ~variants) → (b, p, ~variants)
            rag = Ragged.from_offsets(
                leaf_data, (b_times_p, None), outer_offsets
            ).reshape(b, regular_size, None)

        # Share the same outer-offsets object across all fields (required by
        # Ragged.from_fields which checks value equality; sharing avoids O(n) checks).
        if shared_outer is None:
            shared_outer = np.asarray(rag.offsets)
        rag = _share_offsets(rag, shared_outer)
        field_rags[fname] = rag

    return RaggedVariants.from_record(Ragged.from_fields(field_rags))


def _flat_ploidy(shape) -> int:
    """Ploidy (RegularArray.size) for a flat field shape: the last fixed dim when there are >=2 fixed dims, else 1."""
    fixed = [d for d in shape if d is not None]
    return fixed[-1] if len(fixed) >= 2 else 1


def _write_flat_variants(buf: memoryview, fv, cursor: int) -> tuple[dict, int]:
    """Write a _FlatVariants into buf as a kind=2 block.

    Mirrors _write_rag_variants's byte layout but reads each field straight off
    the flat numpy buffers (`_Flat` scalars: outer offsets + leaf data;
    `_FlatAlleles`: var_offsets (outer) + seq_offsets (inner) + byte_data).

    Fields are written in ``fv.fields`` dict-insertion order, which mirrors the
    field order produced by ``_write_rag_variants``, so the descriptor field
    order is consistent across the flat and ragged write paths.
    """
    from ._dataset._flat_variants import _FlatAlleles

    field_descs: list[dict] = []
    for name, f in fv.fields.items():
        if isinstance(f, _FlatAlleles):
            outer_offsets = np.ascontiguousarray(f.var_offsets, np.int64)
            inner_offsets = np.ascontiguousarray(f.seq_offsets, np.int64)
            leaf_data = np.ascontiguousarray(f.byte_data)
            field_kind = 1
            regular_size = _flat_ploidy(f.shape)
        else:  # _Flat scalar field (start / ilen / dosage / info[...])
            outer_offsets = np.ascontiguousarray(f.offsets, np.int64)
            inner_offsets = np.empty(0, dtype=np.int64)
            leaf_data = np.ascontiguousarray(f.data)
            field_kind = 0
            regular_size = _flat_ploidy(f.shape)

        cursor = _align(cursor)
        outer_off = cursor
        np.frombuffer(buf, dtype=np.int64, count=outer_offsets.size, offset=outer_off)[
            ...
        ] = outer_offsets
        cursor += outer_offsets.nbytes

        if field_kind == 1:
            cursor = _align(cursor)
            inner_off = cursor
            np.frombuffer(
                buf, dtype=np.int64, count=inner_offsets.size, offset=inner_off
            )[...] = inner_offsets
            cursor += inner_offsets.nbytes
        else:
            inner_off = 0

        cursor = _align(cursor)
        data_off = cursor
        np.frombuffer(
            buf, dtype=leaf_data.dtype, count=leaf_data.size, offset=data_off
        )[...] = leaf_data.ravel()
        cursor += leaf_data.nbytes

        field_descs.append(
            {
                "field_kind": field_kind,
                "dtype_str": _dtype_to_bytes(leaf_data.dtype),
                "outer_offsets_offset": outer_off,
                "outer_offsets_nbytes": outer_offsets.nbytes,
                "inner_offsets_offset": inner_off,
                "inner_offsets_nbytes": inner_offsets.nbytes if field_kind == 1 else 0,
                "data_offset": data_off,
                "data_nbytes": leaf_data.nbytes,
                "regular_size": regular_size,
                "name": name.encode("utf-8"),
            }
        )

    return {
        "kind": 2,
        "dtype_str": b"\x00" * 4,
        "shape": [len(field_descs)],
        "data_offset": 0,
        "data_nbytes": 0,
        "offsets_offset": 0,
        "offsets_nbytes": 0,
        "inner_offsets_offset": 0,
        "inner_offsets_nbytes": 0,
        "name": b"",
        "_field_descs": field_descs,
    }, cursor


def _write_flat_variant_windows(buf: memoryview, fvw, cursor: int) -> tuple[dict, int]:
    """Write a _FlatVariantWindows into buf as a kind=4 block.

    Reuses the kind=2 FieldDescriptor format: window slots (``ref_window``,
    ``alt_window``, ``ref``, ``alt``) serialize like kind=2 allele fields
    (field_kind=1: outer=var_offsets, inner=seq_offsets, data=tokens); scalar
    ``.fields`` serialize like kind=2 numeric fields (field_kind=0).

    Args:
        buf: The memoryview of the shared-memory slot.
        fvw: The _FlatVariantWindows to serialize.
        cursor: Byte offset at which to begin writing the payload.

    Returns:
        Tuple of (descriptor dict, updated cursor).
    """
    from ._dataset._flat_variants import _WINDOW_FIELD_NAMES

    field_descs: list[dict] = []

    def _emit_two_level(name, data, seq_off, var_off, regular_size):
        nonlocal cursor
        outer = np.ascontiguousarray(var_off, np.int64)
        inner = np.ascontiguousarray(seq_off, np.int64)
        leaf = np.ascontiguousarray(data)
        cursor = _align(cursor)
        outer_off = cursor
        np.frombuffer(buf, np.int64, outer.size, outer_off)[...] = outer
        cursor += outer.nbytes
        cursor = _align(cursor)
        inner_off = cursor
        np.frombuffer(buf, np.int64, inner.size, inner_off)[...] = inner
        cursor += inner.nbytes
        cursor = _align(cursor)
        data_off = cursor
        np.frombuffer(buf, leaf.dtype, leaf.size, data_off)[...] = leaf.ravel()
        cursor += leaf.nbytes
        field_descs.append(
            {
                "field_kind": 1,
                "dtype_str": _dtype_to_bytes(leaf.dtype),
                "outer_offsets_offset": outer_off,
                "outer_offsets_nbytes": outer.nbytes,
                "inner_offsets_offset": inner_off,
                "inner_offsets_nbytes": inner.nbytes,
                "data_offset": data_off,
                "data_nbytes": leaf.nbytes,
                "regular_size": regular_size,
                "name": name.encode("utf-8"),
            }
        )

    # scalar .fields first (numeric, field_kind=0) — mirror _write_flat_variants
    for name, f in fvw.fields.items():
        outer = np.ascontiguousarray(f.offsets, np.int64)
        leaf = np.ascontiguousarray(f.data)
        cursor = _align(cursor)
        outer_off = cursor
        np.frombuffer(buf, np.int64, outer.size, outer_off)[...] = outer
        cursor += outer.nbytes
        cursor = _align(cursor)
        data_off = cursor
        np.frombuffer(buf, leaf.dtype, leaf.size, data_off)[...] = leaf.ravel()
        cursor += leaf.nbytes
        field_descs.append(
            {
                "field_kind": 0,
                "dtype_str": _dtype_to_bytes(leaf.dtype),
                "outer_offsets_offset": outer_off,
                "outer_offsets_nbytes": outer.nbytes,
                "inner_offsets_offset": 0,
                "inner_offsets_nbytes": 0,
                "data_offset": data_off,
                "data_nbytes": leaf.nbytes,
                "regular_size": _flat_ploidy(f.shape),
                "name": name.encode("utf-8"),
            }
        )

    # present window slots (two-level, field_kind=1)
    for slot in _WINDOW_FIELD_NAMES:
        w = getattr(fvw, slot)
        if w is not None:
            _emit_two_level(
                slot, w.data, w.seq_offsets, w.var_offsets, _flat_ploidy(w.shape)
            )

    return {
        "kind": 4,
        "dtype_str": b"\x00" * 4,
        "shape": [len(field_descs)],
        "data_offset": 0,
        "data_nbytes": 0,
        "offsets_offset": 0,
        "offsets_nbytes": 0,
        "inner_offsets_offset": 0,
        "inner_offsets_nbytes": 0,
        "name": b"",
        "_field_descs": field_descs,
    }, cursor


def _read_rag_annotated(buf: memoryview, d: dict, copy: bool = True):
    """Reconstruct a RaggedAnnotatedHaps (kind=3) from its 3 ragged components.

    Components are returned with flat ``(n_groups, None)`` shape; the consumer's
    ``_reshape_ragged_for_chunk`` re-introduces the ploidy axis from n_instances.
    """
    from seqpro.rag import Ragged

    from ._ragged import RaggedAnnotatedHaps

    comps = []
    for sd in d["_sub_descs"]:
        dtype = _dtype_from_bytes(sd["dtype_str"])
        count = sd["data_nbytes"] // dtype.itemsize
        data = np.frombuffer(buf, dtype=dtype, count=count, offset=sd["data_offset"])
        n_offsets = sd["offsets_nbytes"] // 8
        offsets = np.frombuffer(
            buf, dtype=np.int64, count=n_offsets, offset=sd["offsets_offset"]
        )
        if copy:
            data = data.copy()
            offsets = offsets.copy()
        n_groups = len(offsets) - 1
        comps.append(Ragged.from_offsets(data, (n_groups, None), offsets))

    return RaggedAnnotatedHaps(haps=comps[0], var_idxs=comps[1], ref_coords=comps[2])


def _read_flat_ragged(buf: memoryview, d: dict, copy: bool = True):
    from ._flat import _Flat

    dtype = _dtype_from_bytes(d["dtype_str"])
    n_items = d["shape"][0]
    data = np.frombuffer(buf, dtype=dtype, count=n_items, offset=d["data_offset"])
    n_offsets = d["offsets_nbytes"] // 8
    offsets = np.frombuffer(
        buf, dtype=np.int64, count=n_offsets, offset=d["offsets_offset"]
    )
    if copy:
        data = data.copy()
        offsets = offsets.copy()
    n_groups = len(offsets) - 1
    return _Flat(data, offsets, (n_groups, None))


def _read_flat_variants(buf: memoryview, d: dict, copy: bool = True):
    from ._flat import _Flat
    from ._dataset._flat_variants import _FlatAlleles, _FlatVariants

    fields: dict = {}
    for fd in d["_field_descs"]:
        name = fd["name"]
        leaf_dtype = _dtype_from_bytes(fd["dtype_str"])
        regular_size = fd["regular_size"]

        n_var = fd["outer_offsets_nbytes"] // 8
        var_off = np.frombuffer(
            buf, dtype=np.int64, count=n_var, offset=fd["outer_offsets_offset"]
        )
        leaf_count = fd["data_nbytes"] // leaf_dtype.itemsize
        leaf = np.frombuffer(
            buf, dtype=leaf_dtype, count=leaf_count, offset=fd["data_offset"]
        )
        if copy:
            var_off = var_off.copy()
            leaf = leaf.copy()

        n_bp = len(var_off) - 1
        b = n_bp // regular_size if regular_size else n_bp
        shape = (b, regular_size, None)

        if fd["field_kind"] == 1:
            n_seq = fd["inner_offsets_nbytes"] // 8
            seq_off = np.frombuffer(
                buf, dtype=np.int64, count=n_seq, offset=fd["inner_offsets_offset"]
            )
            if copy:
                seq_off = seq_off.copy()
            fields[name] = _FlatAlleles(leaf, seq_off, var_off, shape)
        else:
            fields[name] = _Flat(leaf, var_off, shape)

    return _FlatVariants(fields)


def _read_flat_variant_windows(buf: memoryview, d: dict, copy: bool = True):
    """Reconstruct a _FlatVariantWindows from a kind=4 descriptor.

    Partitions the field descriptors by name: names in ``_WINDOW_FIELD_NAMES``
    (``ref_window``, ``alt_window``, ``ref``, ``alt``) become ``_FlatWindow``
    slots (two-level offsets); the rest become scalar ``_Flat`` fields.

    Args:
        buf: The memoryview of the shared-memory slot.
        d: The unpacked descriptor dict for this array (with ``_field_descs``).
        copy: If True, returned arrays own their data instead of viewing buf.

    Returns:
        The reconstructed _FlatVariantWindows.
    """
    from ._flat import _Flat
    from ._dataset._flat_variants import (
        _FlatWindow,
        _FlatVariantWindows,
        _WINDOW_FIELD_NAMES,
    )

    fields: dict = {}
    windows: dict = {}
    for fd in d["_field_descs"]:
        name = fd["name"]
        leaf_dtype = _dtype_from_bytes(fd["dtype_str"])
        rs = fd["regular_size"]
        n_outer = fd["outer_offsets_nbytes"] // 8
        var_off = np.frombuffer(buf, np.int64, n_outer, fd["outer_offsets_offset"])
        leaf = np.frombuffer(
            buf, leaf_dtype, fd["data_nbytes"] // leaf_dtype.itemsize, fd["data_offset"]
        )
        if copy:
            var_off, leaf = var_off.copy(), leaf.copy()
        n_bp = len(var_off) - 1
        b = n_bp // rs if rs else n_bp
        if name in _WINDOW_FIELD_NAMES:
            n_inner = fd["inner_offsets_nbytes"] // 8
            seq_off = np.frombuffer(buf, np.int64, n_inner, fd["inner_offsets_offset"])
            if copy:
                seq_off = seq_off.copy()
            windows[name] = _FlatWindow(leaf, seq_off, var_off, (b, rs, None, None))
        else:
            fields[name] = _Flat(leaf, var_off, (b, rs, None))
    return _FlatVariantWindows(fields, **windows)


def _read_flat_annotated(buf: memoryview, d: dict, copy: bool = True):
    from ._flat import _Flat, _FlatAnnotatedHaps

    comps = []
    for sd in d["_sub_descs"]:
        dtype = _dtype_from_bytes(sd["dtype_str"])
        count = sd["data_nbytes"] // dtype.itemsize
        data = np.frombuffer(buf, dtype=dtype, count=count, offset=sd["data_offset"])
        n_offsets = sd["offsets_nbytes"] // 8
        offsets = np.frombuffer(
            buf, dtype=np.int64, count=n_offsets, offset=sd["offsets_offset"]
        )
        if copy:
            data = data.copy()
            offsets = offsets.copy()
        n_groups = len(offsets) - 1
        comps.append(_Flat(data, offsets, (n_groups, None)))

    return _FlatAnnotatedHaps(haps=comps[0], var_idxs=comps[1], ref_coords=comps[2])


def read_chunk(
    buf: memoryview, copy: bool = True, flat: bool = False
) -> tuple[int, list]:
    """Read arrays from the shared-memory slot.

    Args:
        buf: The memoryview of the shared-memory slot.
        copy: If True (default), returned arrays own their data (safe to use after
            the slot is released). If False, arrays are zero-copy views into buf
            (valid only while buf remains mapped and unmodified by the producer).
        flat: If True, kinds 1/2/3 reconstruct ``_Flat`` / ``_FlatVariants`` /
            ``_FlatAnnotatedHaps`` instead of the eagerly-materialized types
            (``Ragged`` / ``RaggedVariants`` / ``RaggedAnnotatedHaps``). Kind=4
            (``_FlatVariantWindows``) has no eagerly-materialized counterpart, so
            it always reconstructs flat regardless of this flag.

    Returns:
        (n_instances, [arrays...]) where arrays may be np.ndarray,
        seqpro.rag.Ragged, RaggedVariants, _Flat, _FlatVariants,
        _FlatAnnotatedHaps, or _FlatVariantWindows depending on the ``flat``
        flag and the array's kind.
    """
    n_inst, payload_bytes, n_arrays = _PREAMBLE.unpack_from(buf, 0)
    cursor = _PREAMBLE.size

    views: list = []
    for _ in range(n_arrays):
        d, cursor = _unpack_one_descriptor(buf, cursor)
        kind = d["kind"]
        if kind == 0:
            views.append(_read_dense(buf, d, copy=copy))
        elif kind == 1:
            views.append(
                (_read_flat_ragged if flat else _read_ragged)(buf, d, copy=copy)
            )
        elif kind == 2:
            views.append(
                (_read_flat_variants if flat else _read_rag_variants)(buf, d, copy=copy)
            )
        elif kind == 3:
            views.append(
                (_read_flat_annotated if flat else _read_rag_annotated)(
                    buf, d, copy=copy
                )
            )
        elif kind == 4:
            views.append(_read_flat_variant_windows(buf, d, copy=copy))
        else:
            raise ValueError(f"Unknown descriptor kind {kind}")

    return int(n_inst), views
