"""Reference flank fetch + byte->int tokenization (sub-project C).

Produces flat token buffers from already-gathered variant fields + the reference
genome. No awkward on the hot path.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

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
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    n = starts.shape[0]
    f5 = f5.reshape(n, flank_len)
    f3 = f3.reshape(n, flank_len)
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
    f5 = reference.fetch(v_contigs, starts - flank_len, starts).data.view(np.uint8)
    f3 = reference.fetch(v_contigs, ends, ends + flank_len).data.view(np.uint8)
    alt_bytes, alt_off = _assemble_alt_windows(
        np.ascontiguousarray(f5),
        np.ascontiguousarray(f3),
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
) -> tuple[_FlatWindow, _FlatWindow]:
    """ref_window = [start-L, end+L) read; alt_window = flank5 . alt . flank3.
    Thin wrapper over compute_ref_window / compute_alt_window."""
    return (
        compute_ref_window(
            reference, v_contigs, starts, ilens, flank_len, lut, row_offsets
        ),
        compute_alt_window(
            reference,
            v_contigs,
            starts,
            ilens,
            alt_data,
            alt_seq_off,
            flank_len,
            lut,
            row_offsets,
        ),
    )
