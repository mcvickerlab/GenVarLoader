"""Reference flank fetch + byte->int tokenization (sub-project C).

Produces flat token buffers from already-gathered variant fields + the reference
genome. All-numpy hot path.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from .._ragged import Ragged
from .._utils import lengths_to_offsets
from ..genvarloader import get_reference as _get_reference_ffi
from ._flat_variants import _FlatWindow


def build_token_lut(alphabet: bytes, unknown_token: int) -> tuple[NDArray, np.dtype]:
    """Build a 256-entry byte->token lookup table (seqpro-style).

    Every byte value in ``alphabet`` maps to its position; every other byte
    (including ``N`` and padded out-of-bounds positions) maps to ``unknown_token``.
    """
    max_token = max(len(alphabet) - 1, unknown_token)
    dtype = np.uint8 if max_token <= np.iinfo(np.uint8).max else np.int32
    lut = np.full(256, unknown_token, dtype=dtype)
    for i, b in enumerate(alphabet):
        lut[b] = i
    return lut, np.dtype(dtype)


def _slice_flanks(data: NDArray[np.uint8], rw_off: NDArray[np.int64], flank_len: int):
    """Derive per-variant (f5, f3) flanks from a contiguous ref-window read
    ``rw = [start-L, end+L)``. ``f5`` = first ``L`` bytes of each row, ``f3`` =
    last ``L``. Byte-identical to fetching ``[start-L, start)`` / ``[end, end+L)``
    separately: rows are always ``ref_len + 2L >= 2L + 1`` long so the two
    fixed-``L`` windows never overlap, and ``padded_slice`` pads OOB by absolute
    coordinate so boundary padding matches. Both returned arrays are ``(n, L)``.
    """
    cols = np.arange(flank_len)
    f5 = data[rw_off[:-1, None] + cols]
    f3 = data[rw_off[1:, None] - flank_len + cols]
    return f5, f3


def compute_flank_tokens(
    reference,
    v_contigs: NDArray[np.integer],  # (n_var,) contig id per variant
    starts: NDArray[np.integer],  # (n_var,)
    ilens: NDArray[np.integer],  # (n_var,)
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[
        np.int64
    ],  # (b*p + 1,) variant boundaries per (instance, ploid) row
) -> tuple[NDArray, NDArray[np.int64]]:
    """Ride-along flank tokens: ``[flank5 | flank3]`` (``2 * flank_len`` tokens) per
    variant.

    Returns ``(token_data, offsets)``:

    - ``token_data`` is the flat token buffer, length ``n_var * 2 * flank_len``,
      laid out variant-major (each variant contributes ``2 * flank_len``
      contiguous tokens: flank5 then flank3).
    - ``offsets`` is ``row_offsets`` unchanged (length ``b*p + 1``): it groups
      *variants* into the ``b*p`` instance/ploid rows. These are variant-level
      offsets, NOT byte indices into ``token_data`` -- the fixed ``2 * flank_len``
      inner dimension is carried separately as a trailing regular axis when this
      buffer is wrapped as ``_Flat.from_offsets(token_data, (b, p, None, 2*L),
      offsets)`` (see ``get_variants_flat``). The trailing regular axis supplies
      the per-variant stride, so the offsets stay at variant granularity.
    """
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    f5, f3 = _slice_flanks(
        rw.data.view(np.uint8), np.asarray(rw.offsets, np.int64), flank_len
    )  # each (n, L)
    flank_bytes = np.concatenate([f5, f3], axis=1)  # (n, 2L)
    tokens = lut[flank_bytes]  # vectorized 256-LUT gather -> lut.dtype
    return tokens.reshape(-1), np.asarray(row_offsets, np.int64)


@nb.njit(nogil=True, cache=True)  # pragma: no cover - njit
def _assemble_alt_windows(f5, f3, alt_data, alt_seq_off, flank_len):
    """Concatenate flank5 (fixed L) + alt (variable) + flank3 (fixed L) per variant
    into a flat byte buffer. f5/f3 are (n_var, L) row-major flat (n_var*L,)."""
    n = alt_seq_off.shape[0] - 1
    out_off = np.empty(n + 1, np.int64)
    out_off[0] = 0
    for i in range(n):
        alt_len = alt_seq_off[i + 1] - alt_seq_off[i]
        out_off[i + 1] = out_off[i] + 2 * flank_len + alt_len
    out = np.empty(out_off[n], np.uint8)
    for i in range(n):
        dst = out_off[i]
        for k in range(flank_len):
            out[dst] = f5[i * flank_len + k]
            dst += 1
        for k in range(alt_seq_off[i], alt_seq_off[i + 1]):
            out[dst] = alt_data[k]
            dst += 1
        for k in range(flank_len):
            out[dst] = f3[i * flank_len + k]
            dst += 1
    return out, out_off


def compute_ref_window(
    reference,
    v_contigs: NDArray[np.integer],
    starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[np.int64],
) -> "_FlatWindow":
    """ref window = tokenized single contiguous read ``[start-L, end+L)``."""
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    ref_tok = lut[rw.data.view(np.uint8)]
    return _FlatWindow(
        ref_tok,
        np.asarray(rw.offsets, np.int64),
        np.asarray(row_offsets, np.int64),
        (None,),
    )


def compute_alt_window(
    reference,
    v_contigs: NDArray[np.integer],
    starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    alt_data: NDArray[np.uint8],
    alt_seq_off: NDArray[np.int64],
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[np.int64],
) -> "_FlatWindow":
    """alt window = tokenized ``flank5 . alt . flank3`` assembly."""
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    f5, f3 = _slice_flanks(
        rw.data.view(np.uint8), np.asarray(rw.offsets, np.int64), flank_len
    )  # each (n, L)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5).reshape(-1),
        np.ascontiguousarray(f3).reshape(-1),
        np.asarray(alt_data, np.uint8),
        np.asarray(alt_seq_off, np.int64),
        flank_len,
    )
    alt_tok = lut[alt_bytes]
    return _FlatWindow(
        alt_tok,
        alt_off,
        np.asarray(row_offsets, np.int64),
        (None,),
    )


def tokenize_alleles(
    allele_data: NDArray[np.uint8],
    allele_seq_off: NDArray[np.int64],
    lut: NDArray,
    row_offsets: NDArray[np.int64],
) -> "_FlatWindow":
    """Bare tokenized allele (no flanks) as a two-level ``_FlatWindow``: just the
    LUT applied to the gathered allele bytes, with the allele byte offsets."""
    tok = lut[np.asarray(allele_data, np.uint8)]
    return _FlatWindow(
        tok,
        np.asarray(allele_seq_off, np.int64),
        np.asarray(row_offsets, np.int64),
        (None,),
    )


def compute_windows(
    reference,
    v_contigs: NDArray[np.integer],
    starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    alt_data: NDArray[np.uint8],
    alt_seq_off: NDArray[np.int64],
    flank_len: int,
    lut: NDArray,
    row_offsets: NDArray[np.int64],
) -> tuple["_FlatWindow", "_FlatWindow"]:
    """ref_window = [start-L, end+L) read; alt_window = flank5 . alt . flank3.

    Single fused fetch: read the ref window once and derive the alt-window flanks
    by slicing it, instead of the previous 3 separate ``reference.fetch`` calls.
    Byte-identical to ``(compute_ref_window, compute_alt_window)``.
    """
    starts = np.asarray(starts, np.int32)
    ilens = np.asarray(ilens, np.int32)
    ends = starts - np.minimum(ilens, 0) + 1
    rw = reference.fetch(v_contigs, starts - flank_len, ends + flank_len)
    data = rw.data.view(np.uint8)
    rw_off = np.asarray(rw.offsets, np.int64)
    row_off = np.asarray(row_offsets, np.int64)

    # ref window: tokenize the contiguous read directly.
    ref_w = _FlatWindow(lut[data], rw_off, row_off, (None,))

    # alt window: flank5 . alt . flank3 from sliced flanks.
    f5, f3 = _slice_flanks(data, rw_off, flank_len)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5).reshape(-1),
        np.ascontiguousarray(f3).reshape(-1),
        np.asarray(alt_data, np.uint8),
        np.asarray(alt_seq_off, np.int64),
        flank_len,
    )
    alt_w = _FlatWindow(lut[alt_bytes], alt_off, row_off, (None,))
    return ref_w, alt_w


class _RefShim:
    """Minimal reference-object shim wrapping raw (reference, ref_offsets) arrays.

    Implements the ``.fetch(contigs, starts, ends)`` interface used by
    ``compute_flank_tokens``, ``compute_ref_window``, and ``compute_alt_window``,
    backed by the ``get_reference`` FFI call so behavior is byte-identical to a
    ``Reference`` object (same padded-slice logic, same OOB padding).
    """

    def __init__(
        self,
        reference: NDArray[np.uint8],
        ref_offsets: NDArray[np.int64],
        pad_char: int,
    ) -> None:
        self._ref = np.ascontiguousarray(reference, np.uint8)
        self._off = np.ascontiguousarray(ref_offsets, np.int64)
        self._pad = int(pad_char)

    def fetch(
        self,
        contigs: NDArray[np.integer],
        starts: NDArray[np.integer],
        ends: NDArray[np.integer],
    ) -> "Ragged":
        contigs = np.ascontiguousarray(contigs, np.int32)
        starts = np.ascontiguousarray(starts, np.int32)
        ends = np.ascontiguousarray(ends, np.int32)
        n = len(contigs)
        lengths = np.asarray(ends - starts, np.int64)
        out_offsets = lengths_to_offsets(lengths)
        regions = np.stack([contigs, starts, ends], axis=1).astype(np.int32)
        data = _get_reference_ffi(
            regions, out_offsets, self._ref, self._off, self._pad, False
        )
        return Ragged.from_offsets(data.view("S1"), (n, None), out_offsets)


def _assemble_variant_buffers_numba(
    mode: int,
    v_idxs: NDArray[np.int32],
    row_offsets: NDArray[np.int64],
    alt_global: NDArray[np.uint8],
    alt_off_global: NDArray[np.int64],
    ref_global: "NDArray[np.uint8] | None",
    ref_off_global: "NDArray[np.int64] | None",
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: int,
    alt_mode: int,
    flank_len: int,
    lut: "NDArray | None",
    v_contigs: NDArray[np.int32],
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    reference: NDArray[np.uint8],
    ref_offsets: NDArray[np.int64],
    pad_char: int,
) -> "dict[str, tuple[NDArray, NDArray[np.int64]]]":
    """Numba/numpy oracle for assemble_variant_buffers: composes existing helpers.

    Mirrors the Rust ``assemble_variants_mode`` / ``assemble_windows_mode`` logic,
    producing the same ``{name: (data, seq_offsets)}`` dict contract. Used as the
    parity reference in ``assert_kernel_parity_dict``. Does NOT re-implement any
    sub-kernel logic — delegates entirely to the registered helpers.
    """
    from ._flat_variants import _gather_alleles

    v_idxs = np.ascontiguousarray(v_idxs, np.int32)
    row_offsets = np.ascontiguousarray(row_offsets, np.int64)
    alt_global = np.ascontiguousarray(alt_global, np.uint8)
    alt_off_global = np.ascontiguousarray(alt_off_global, np.int64)

    out: dict[str, tuple[NDArray, NDArray[np.int64]]] = {}

    if mode == 0:  # variants mode
        alt_data, alt_seq_off = _gather_alleles(v_idxs, alt_global, alt_off_global)
        out["alt"] = (alt_data, alt_seq_off)

        if want_ref_bytes and ref_global is not None and ref_off_global is not None:
            rg = np.ascontiguousarray(ref_global, np.uint8)
            ro = np.ascontiguousarray(ref_off_global, np.int64)
            ref_data, ref_seq_off = _gather_alleles(v_idxs, rg, ro)
            out["ref"] = (ref_data, ref_seq_off)

        if want_flank:
            # v_starts / ilens are GLOBAL per-variant arrays; gather by v_idxs.
            starts_v = np.asarray(v_starts, np.int32)[v_idxs]
            ilens_v = np.asarray(ilens, np.int32)[v_idxs]
            ref_shim = _RefShim(reference, ref_offsets, pad_char)
            tok, off = compute_flank_tokens(
                ref_shim, v_contigs, starts_v, ilens_v, flank_len, lut, row_offsets
            )
            out["flank_tokens"] = (tok, off)

    else:  # windows mode
        alt_data, alt_seq_off = _gather_alleles(v_idxs, alt_global, alt_off_global)
        # v_starts / ilens are GLOBAL; gather by v_idxs before passing to helpers.
        starts_v = np.asarray(v_starts, np.int32)[v_idxs]
        ilens_v = np.asarray(ilens, np.int32)[v_idxs]
        ref_shim = _RefShim(reference, ref_offsets, pad_char)

        if ref_mode == 1:  # flanked ref window: [start-L, end+L)
            rw = compute_ref_window(
                ref_shim, v_contigs, starts_v, ilens_v, flank_len, lut, row_offsets
            )
            out["ref_window"] = (rw.data, rw.seq_offsets)
        elif ref_mode == 2:  # bare tokenized ref allele (no flanks)
            rg = np.ascontiguousarray(ref_global, np.uint8)
            ro = np.ascontiguousarray(ref_off_global, np.int64)
            ref_data, ref_seq_off = _gather_alleles(v_idxs, rg, ro)
            rw = tokenize_alleles(ref_data, ref_seq_off, lut, row_offsets)
            out["ref"] = (rw.data, rw.seq_offsets)

        if alt_mode == 1:  # flanked alt window: flank5 . alt . flank3
            aw = compute_alt_window(
                ref_shim,
                v_contigs,
                starts_v,
                ilens_v,
                alt_data,
                alt_seq_off,
                flank_len,
                lut,
                row_offsets,
            )
            out["alt_window"] = (aw.data, aw.seq_offsets)
        elif alt_mode == 2:  # bare tokenized alt allele (no flanks)
            aw = tokenize_alleles(alt_data, alt_seq_off, lut, row_offsets)
            out["alt"] = (aw.data, aw.seq_offsets)

    return out
