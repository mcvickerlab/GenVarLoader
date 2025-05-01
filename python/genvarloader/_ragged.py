from __future__ import annotations

from typing import Any, Optional, Tuple, TypeGuard, TypeVar, Union

import awkward as ak
import numba as nb
import numpy as np
from attrs import define
from einops import repeat
from numpy.typing import NDArray
from seqpro._ragged import Ragged

from ._types import DTYPE, AnnotatedHaps
from ._utils import _lengths_to_offsets

__all__ = ["Ragged", "RaggedIntervals", "INTERVAL_DTYPE", "pad_ragged"]

RDTYPE = TypeVar("RDTYPE", bound=np.generic)


@define
class RaggedAnnotatedHaps:
    haps: Ragged[np.bytes_]
    var_idxs: Ragged[np.int32]
    ref_coords: Ragged[np.int32]

    @property
    def shape(self):
        return self.haps.shape

    def to_padded(self) -> AnnotatedHaps:
        haps = to_padded(self.haps, b"N")
        var_idxs = to_padded(self.var_idxs, -1)
        ref_coords = to_padded(self.ref_coords, -1)
        return AnnotatedHaps(haps, var_idxs, ref_coords)

    def reshape(self, shape: tuple[int, ...]) -> RaggedAnnotatedHaps:
        return RaggedAnnotatedHaps(
            self.haps.reshape(shape),
            self.var_idxs.reshape(shape),
            self.ref_coords.reshape(shape),
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> RaggedAnnotatedHaps:
        return RaggedAnnotatedHaps(
            self.haps.squeeze(axis),
            self.var_idxs.squeeze(axis),
            self.ref_coords.squeeze(axis),
        )

    def to_fixed_shape(self, shape: tuple[int, ...]) -> AnnotatedHaps:
        haps = self.haps.data.reshape(shape)
        var_idxs = self.var_idxs.data.reshape(shape)
        ref_coords = self.ref_coords.data.reshape(shape)
        return AnnotatedHaps(haps, var_idxs, ref_coords)


@define
class RaggedVariants:
    """Typically contains ragged arrays with shape (batch, ploidy, ~variants)"""

    alts: ak.Array  # (batch, ploidy, ~variants, ~length)
    pos: Ragged[np.int32]
    ilens: Ragged[np.int32]
    ccfs: Ragged[np.float32]


def is_rag_dtype(rag: Ragged, dtype: type[DTYPE]) -> TypeGuard[Ragged[DTYPE]]:
    return np.issubdtype(rag.data.dtype, dtype)


INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)
RaggedIntervals = Ragged[np.void]


def to_padded(rag: Ragged[RDTYPE], pad_value: Any) -> NDArray[RDTYPE]:
    """Convert this Ragged array to a rectilinear array by right-padding each entry with a value.
    The final axis will have the maximum length across all entries.

    Parameters
    ----------
    pad_value
        Value to pad the entries with.

    Returns
    -------
        Padded array with shape :code:`(*self.shape, self.lengths.max())`.
    """
    length = rag.lengths.max()
    shape = (*rag.shape, length)
    if rag.data.dtype.str == "|S1":
        if isinstance(pad_value, (str, bytes)):
            if len(pad_value) != 1:
                raise ValueError(
                    "Tried padding an S1 array with a `pad_value` that is multiple characters."
                )
            pad_value = np.uint8(ord(pad_value))
        elif isinstance(pad_value, int):
            if pad_value < 0 or pad_value > 255:
                raise ValueError(
                    "Tried padding an S1 array with an integer `pad_value` outside the ASCII range."
                )
            pad_value = np.uint8(pad_value)
        else:
            raise ValueError(
                "Tried padding an S1 array with a `pad_value` that isn't a string, byte, or integer."
            )
        padded = np.empty((np.prod(shape[:-1]), shape[-1]), dtype=np.uint8)
        pad_ragged(rag.data.view(np.uint8), rag.offsets, pad_value, padded)
        padded = padded.view(rag.data.dtype).reshape(shape)
    else:
        padded = np.empty((np.prod(shape[:-1]), shape[-1]), dtype=rag.data.dtype)
        pad_ragged(rag.data, rag.offsets, pad_value, padded)
        padded = padded.reshape(shape)

    return padded


@nb.njit(parallel=True, nogil=True, cache=True)
def pad_ragged(
    data: NDArray[DTYPE],
    offsets: NDArray[np.integer],
    pad_value: DTYPE,
    out: NDArray[DTYPE],
):
    for i in nb.prange(len(offsets)):
        if offsets.ndim == 1:
            if i == len(offsets) - 1:
                continue
            start, end = offsets[i], offsets[i + 1]
        else:
            start, end = offsets[i]
        entry_len = end - start
        out[i, :entry_len] = data[start:end]
        out[i, entry_len:] = pad_value
    return out


NUCLEOTIDES = b"ACGT"
COMPLEMENTS = b"TGCA"


@nb.njit(parallel=True, nogil=True, cache=True)
def _rc_helper(
    data: NDArray[np.uint8], offsets: NDArray[np.int64], mask: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    out = data.copy()
    for i in nb.prange(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        _data = data[start:end]
        _out = out[start:end]
        if mask[i]:
            for nuc, comp in zip(NUCLEOTIDES, COMPLEMENTS):
                _out[_data == nuc] = comp
            _out[:] = _out[::-1]
        else:
            _out[:] = _data
    return out


def _reverse_complement(
    seqs: Ragged[np.bytes_], mask: NDArray[np.bool_]
) -> Ragged[np.bytes_]:
    # (b [p] ~l), (b)
    if seqs.ndim == 2:
        ploidy = seqs.shape[1]
        mask = repeat(mask, "b -> (b p)", p=ploidy)
    rc_seqs = _rc_helper(seqs.data.view(np.uint8), seqs.offsets, mask)
    return Ragged.from_offsets(rc_seqs.view("S1"), seqs.shape, seqs.offsets)


#! for whatever reason, this causes data corruption with parallel=True?!
@nb.njit(nogil=True, cache=True)
def _reverse_helper(data: NDArray, offsets: NDArray[np.int64], mask: NDArray[np.bool_]):
    for i in nb.prange(len(offsets) - 1):
        if mask[i]:
            start, end = offsets[i], offsets[i + 1]
            data[start:end] = np.flip(data[start:end])


def _reverse(tracks: Ragged, mask: NDArray[np.bool_]):
    """Reverses data along the ragged axis in-place."""
    # (b t [p] ~l), (b)
    if tracks.ndim == 2:
        n_tracks = tracks.shape[1]
        mask = repeat(mask, "b -> (b t)", t=n_tracks)
    elif tracks.ndim == 3:
        n_tracks, ploidy = tracks.shape[1:]
        mask = repeat(mask, "b -> (b t p)", t=n_tracks, p=ploidy)
    _reverse_helper(tracks.data, tracks.offsets, mask)


def _jitter(
    *arrays: Ragged,
    max_jitter: int,
    seed: Optional[Union[int, np.random.Generator]] = None,
    starts: NDArray[np.int64] | None = None,
):
    """Jitter the data along the ragged axis by up to `max_jitter`.

    Assumes only the first axis is to be jittered. In other words, assumes that no user would want to
    jitter independently across ploidy or tracks.
    """
    batch_size = arrays[0].shape[0]

    if starts is None:
        if not isinstance(seed, np.random.Generator):
            seed = np.random.default_rng(seed)
        starts = seed.integers(0, 2 * max_jitter + 1, size=batch_size)
    out_offsets = tuple(_lengths_to_offsets(a.lengths - 2 * max_jitter) for a in arrays)

    jittered_data = _jitter_helper(
        tuple(a.data for a in arrays),
        tuple(a.offsets for a in arrays),
        tuple(a.shape for a in arrays),
        tuple(out_offsets),
        starts,
    )

    out = tuple(
        Ragged.from_offsets(jit_data, rag_arr.shape, out_offsets[i])
        for i, (rag_arr, jit_data) in enumerate(zip(arrays, jittered_data))
    )

    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def _jitter_helper(
    data: Tuple[NDArray, ...],
    offsets: Tuple[NDArray[np.int64], ...],
    shapes: Tuple[Tuple[int, ...], ...],
    out_offsets: Tuple[NDArray[np.int64], ...],
    starts: NDArray[np.int64],
) -> Tuple[NDArray, ...]:
    """Helper to jitter ragged data. Ragged arrays should have shape (batch, ...).

    Parameters
    ----------
    data
        Tuple of ragged data arrays.
    offsets
        Tuple of offsets arrays.
    out_offsets
        Tuple of output offsets arrays.
    starts
        Shape: (batch). Array of starting points for jittering.
    """
    n_arrays = len(data)
    batch_size = len(starts)
    out_data = tuple(
        np.empty(out_offsets[i][-1], dtype=d.dtype) for i, d in enumerate(data)
    )
    for arr in nb.prange(n_arrays):
        arr_data = data[arr]
        arr_offsets = offsets[arr]

        out_arr_data = out_data[arr]
        out_arr_offsets = out_offsets[arr]

        arr_shape = shapes[arr]
        n_per_batch = np.prod(arr_shape[1:])

        for jit in nb.prange(batch_size):
            idx_s = jit * n_per_batch
            idx_e = (jit + 1) * n_per_batch

            jit_s = starts[jit]

            for row in nb.prange(idx_s, idx_e):
                row_s = arr_offsets[row]
                out_row_s, out_row_e = out_arr_offsets[row], out_arr_offsets[row + 1]
                out_len = out_row_e - out_row_s
                out_arr_data[out_row_s:out_row_e] = arr_data[
                    row_s + jit_s : row_s + jit_s + out_len
                ]

    return out_data
