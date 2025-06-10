from __future__ import annotations

from functools import partial
from typing import Any, TypeGuard, TypeVar

import numba as nb
import numpy as np
from attrs import define
from einops import repeat
from numpy.typing import NDArray
from phantom import Phantom
from seqpro._ragged import Ragged

from ._types import DTYPE, AnnotatedHaps

__all__ = ["Ragged", "RaggedIntervals", "pad_ragged"]

RDTYPE = TypeVar("RDTYPE", bound=np.generic)
INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)
RaggedIntervals = Ragged[np.void]


def is_rag_dtype(rag: Any, dtype: type[DTYPE]) -> TypeGuard[Ragged[DTYPE]]:
    return isinstance(rag, Ragged) and np.issubdtype(rag.data.dtype, dtype)


class RaggedSeqs(
    Ragged[np.bytes_], Phantom, predicate=partial(is_rag_dtype, dtype=np.bytes_)
): ...


@define
class RaggedAnnotatedHaps:
    """Ragged version of :class:`AnnotatedHaps`."""

    haps: Ragged[np.bytes_]
    """Haplotypes with dtype S1."""
    var_idxs: Ragged[np.int32]
    """Variant indices for each position in the haplotypes. A value of -1 indicates no variant was applied at the position."""
    ref_coords: Ragged[np.int32]
    """Reference coordinates for each position in haplotypes."""

    @property
    def shape(self):
        """Shape of the haplotypes and all annotations."""
        return self.haps.shape

    def to_padded(self) -> AnnotatedHaps:
        """Convert this Ragged array to a rectilinear array by right-padding each entry with appropriate values.
        The final axis will have the maximum length across all entries."""
        haps = to_padded(self.haps, b"N")
        var_idxs = to_padded(self.var_idxs, -1)
        ref_coords = to_padded(self.ref_coords, np.iinfo(self.ref_coords.dtype).max)
        return AnnotatedHaps(haps, var_idxs, ref_coords)

    def reshape(self, shape: int | tuple[int, ...]) -> RaggedAnnotatedHaps:
        """Reshape the haplotypes and all annotations.

        Parameters
        ----------
        shape
            New shape for the haplotypes and all annotations. The total number of elements
            must remain the same.
        """
        return RaggedAnnotatedHaps(
            self.haps.reshape(shape),
            self.var_idxs.reshape(shape),
            self.ref_coords.reshape(shape),
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> RaggedAnnotatedHaps:
        """Squeeze the haplotypes and all annotations along the specified axis.

        Parameters
        ----------
        axis
            Axis or axes to squeeze. If None, all axes of length 1 are squeezed.
        """
        return RaggedAnnotatedHaps(
            self.haps.squeeze(axis),
            self.var_idxs.squeeze(axis),
            self.ref_coords.squeeze(axis),
        )

    def to_fixed_shape(self, shape: tuple[int, ...]) -> AnnotatedHaps:
        """If all entries in the ragged array have the same shape, convert to a rectilinear shape.

        Parameters
        ----------
        shape
            Shape to convert to, including the length axis. The total number of elements must remain the same.
        """
        haps = self.haps.data.reshape(shape)
        var_idxs = self.var_idxs.data.reshape(shape)
        ref_coords = self.ref_coords.data.reshape(shape)
        return AnnotatedHaps(haps, var_idxs, ref_coords)


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
            start, end = offsets[:, i]
        entry_len = end - start
        out[i, :entry_len] = data[start:end]
        out[i, entry_len:] = pad_value
    return out


NUCLEOTIDES = b"ACGT"
COMPLEMENTS = b"TGCA"


#! for whatever reason, this causes data corruption with parallel=True?!
#! assumes offsets are 1D
@nb.njit(nogil=True, cache=True)
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
    return out


def reverse_complement(
    seqs: Ragged[np.bytes_], mask: NDArray[np.bool_]
) -> Ragged[np.bytes_]:
    # (b [p] ~l), (b)
    if seqs.ndim == 2:
        ploidy = seqs.shape[1]
        mask = repeat(mask, "b -> (b p)", p=ploidy)
    rc_seqs = _rc_helper(seqs.data.view(np.uint8), seqs.offsets, mask)
    return Ragged.from_offsets(rc_seqs.view("S1"), seqs.shape, seqs.offsets)


#! for whatever reason, this causes data corruption with parallel=True?!
#! assumes offsets are 1D
@nb.njit(nogil=True, cache=True)
def _reverse_helper(data: NDArray, offsets: NDArray[np.int64], mask: NDArray[np.bool_]):
    for i in nb.prange(len(offsets) - 1):
        if mask[i]:
            start, end = offsets[i], offsets[i + 1]
            data[start:end] = np.flip(data[start:end])


def reverse(tracks: Ragged, mask: NDArray[np.bool_]):
    """Reverses data along the ragged axis in-place."""
    # (b t [p] ~l), (b)
    if tracks.ndim == 2:
        n_tracks = tracks.shape[1]
        mask = repeat(mask, "b -> (b t)", t=n_tracks)
    elif tracks.ndim == 3:
        n_tracks, ploidy = tracks.shape[1:]
        mask = repeat(mask, "b -> (b t p)", t=n_tracks, p=ploidy)
    _reverse_helper(tracks.data, tracks.offsets, mask)
