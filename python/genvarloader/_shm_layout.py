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
    """Write a RaggedVariants awkward array into buf.

    Layout per kind=2 block:
      The descriptor's 'shape' carries [n_fields, regular_size (ploidy)] so the
      reader can reconstruct the RegularArray wrapper. Each field is encoded
      inline as a FieldDescriptor (see module docstring).
    """
    import awkward as ak
    from awkward.contents import RegularArray, ListOffsetArray, NumpyArray

    fields = rv.fields
    n_fields = len(fields)

    field_descs: list[dict] = []

    for field in fields:
        f_layout = ak.to_layout(rv[field])

        # Level 0: RegularArray(size=ploidy)
        assert isinstance(f_layout, RegularArray), (
            f"Expected RegularArray for field {field!r}"
        )
        regular_size = f_layout.size
        # Level 1: ListOffsetArray (outer: groups of variants per (batch, ploid) cell)
        outer = f_layout.content
        assert isinstance(outer, ListOffsetArray), (
            f"Expected outer ListOffsetArray for {field!r}"
        )
        outer_offsets = np.ascontiguousarray(outer.offsets.data)  # int64

        inner_content = outer.content
        if isinstance(inner_content, ListOffsetArray):
            # Alleles field: (batch, ploidy, ~variants, ~allele_bytes)
            inner = inner_content
            inner_offsets = np.ascontiguousarray(inner.offsets.data)  # int64
            leaf = inner.content
            assert isinstance(leaf, NumpyArray), (
                f"Expected NumpyArray leaf for {field!r} alleles"
            )
            leaf_data = np.ascontiguousarray(leaf.data)
            field_kind = 1  # alleles
        elif isinstance(inner_content, NumpyArray):
            # Numeric field: (batch, ploidy, ~variants)
            inner_offsets = np.empty(0, dtype=np.int64)
            leaf_data = np.ascontiguousarray(inner_content.data)
            field_kind = 0  # numeric
        else:
            raise TypeError(
                f"Unexpected layout for field {field!r}: {type(inner_content)}"
            )

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

        name_bytes = field.encode("utf-8")
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

    return out


def write_chunk(
    buf: memoryview,
    arrays: Sequence,
    n_instances: int,
) -> int:
    """Write arrays into the shared-memory slot.

    Supports np.ndarray (kind=0), seqpro.rag.Ragged (kind=1),
    and RaggedVariants (kind=2).

    Returns total bytes consumed (header + payload).
    """
    from seqpro.rag import Ragged
    from ._dataset._rag_variants import RaggedVariants

    if len(arrays) > 255:
        raise ValueError("at most 255 arrays per chunk")

    descriptors: list[dict] = []
    cursor = HEADER_RESERVED

    for a in arrays:
        if isinstance(a, RaggedVariants):
            desc, cursor = _write_rag_variants(buf, a, cursor)
        elif isinstance(a, Ragged):
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
    import awkward as ak
    from awkward.contents import RegularArray, ListOffsetArray, NumpyArray
    from awkward.index import Index64
    from ._dataset._rag_variants import RaggedVariants

    field_arrays: dict[str, ak.Array] = {}

    for fd in d["_field_descs"]:
        fname = fd["name"]
        leaf_dtype = _dtype_from_bytes(fd["dtype_str"])
        regular_size = fd["regular_size"]

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

        if fd["field_kind"] == 1:
            n_inner = fd["inner_offsets_nbytes"] // 8
            inner_offsets = np.frombuffer(
                buf, dtype=np.int64, count=n_inner, offset=fd["inner_offsets_offset"]
            )
            if copy:
                inner_offsets = inner_offsets.copy()
            inner_loa = ListOffsetArray(
                Index64(inner_offsets),
                NumpyArray(leaf_data, parameters={"__array__": "byte"}),
                parameters={"__array__": "bytestring"},
            )
            outer_loa = ListOffsetArray(Index64(outer_offsets), inner_loa)
            arr = ak.Array(RegularArray(outer_loa, regular_size))
        else:
            outer_loa = ListOffsetArray(Index64(outer_offsets), NumpyArray(leaf_data))
            arr = ak.Array(RegularArray(outer_loa, regular_size))

        field_arrays[fname] = arr

    return RaggedVariants.from_ak(ak.zip(field_arrays, depth_limit=1))


def read_chunk(buf: memoryview, copy: bool = True) -> tuple[int, list]:
    """Read arrays from the shared-memory slot.

    Parameters
    ----------
    buf
        The memoryview of the shared-memory slot.
    copy
        If True (default), returned arrays own their data (safe to use after
        the slot is released). If False, arrays are zero-copy views into buf
        (valid only while buf remains mapped and unmodified by the producer).

    Returns (n_instances, [arrays...]) where arrays may be np.ndarray,
    seqpro.rag.Ragged, or RaggedVariants.
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
            views.append(_read_ragged(buf, d, copy=copy))
        elif kind == 2:
            views.append(_read_rag_variants(buf, d, copy=copy))
        else:
            raise ValueError(f"Unknown descriptor kind {kind}")

    return int(n_inst), views
