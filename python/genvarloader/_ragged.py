from __future__ import annotations

from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import numba as nb
import numpy as np
from attrs import define
from numpy.typing import NDArray

from ._types import Idx

__all__ = ["Ragged", "RaggedIntervals", "INTERVAL_DTYPE", "pad_ragged"]

DTYPE = TypeVar("DTYPE", bound=np.generic)
RDTYPE = TypeVar("RDTYPE", bound=np.generic, contravariant=True)


@define
class Ragged(Generic[RDTYPE]):
    """Ragged array i.e. a rectilinear array where the final axis is ragged. Should not be initialized
    directly, use :meth:`from_offsets()` or :meth:`from_lengths()` instead.
    Does not have a :code:`__getitem__` method currently.

    Examples
    --------

    .. code-block:: python

        r = Ragged.from_lengths(np.arange(10), np.array([3, 2, 5]))
        assert r.offsets ==  np.array([0, 3, 5, 10])
        assert r.data[r.offsets[0]:r.offsets[1]] == np.array([0, 1, 2])
        assert r.data[r.offsets[1]:r.offsets[2]] == np.array([3, 4])
        assert r.data[r.offsets[2]:r.offsets[3]] == np.array([5, 6, 7, 8, 9])

    """

    data: NDArray[RDTYPE]
    """A 1D array of the data."""
    shape: Tuple[int, ...]
    """Shape of the ragged array, excluding the length dimension. For example, if
        the shape is (2, 3), then the j, k-th element can be mapped to an index for
        offsets with :code:`i = np.ravel_multi_index((j, k), shape)`. The number of ragged
        elements corresponds to the product of the shape."""
    maybe_offsets: Optional[NDArray[np.int64]] = None
    maybe_lengths: Optional[NDArray[np.int32]] = None

    def __attrs_post_init__(self):
        if self.shape == ():
            raise ValueError("Ragged array must have at least one element.")
        if self.maybe_offsets is None and self.maybe_lengths is None:
            raise ValueError("Either offsets or lengths must be provided.")

    def __len__(self):
        return self.shape[0]

    @property
    def offsets(self) -> NDArray[np.int64]:
        """Offsets into the data array to get corresponding elements. The i-th element
        is accessible as :code:`data[offsets[i]:offsets[i+1]]`."""
        if self.maybe_offsets is None:
            self.maybe_offsets = np.empty(
                np.prod(self.shape, dtype=np.int64) + 1, dtype=np.int64
            )
            self.maybe_offsets[0] = 0
            np.cumsum(self.lengths.ravel(), out=self.maybe_offsets[1:])
        return self.maybe_offsets

    @property
    def lengths(self) -> NDArray[np.int32]:
        """Array with appropriate shape containing lengths of each element in the ragged array."""
        if self.maybe_lengths is None:
            self.maybe_lengths = np.diff(self.offsets).reshape(self.shape)
        return self.maybe_lengths

    @classmethod
    def from_offsets(
        cls,
        data: NDArray[DTYPE],
        shape: Union[int, Tuple[int, ...]],
        offsets: NDArray[np.int64],
    ) -> "Ragged[DTYPE]":
        """Create a Ragged array from data and offsets.

        Parameters
        ----------
        data
            1D data array.
        shape
            Shape of the ragged array, excluding the length dimension.
        offsets
            Offsets into the data array to get corresponding elements.
        """
        if isinstance(shape, int):
            shape = (shape,)
        return cls(data, shape, maybe_offsets=offsets)

    @classmethod
    def from_lengths(
        cls, data: NDArray[DTYPE], lengths: NDArray[np.int32]
    ) -> "Ragged[DTYPE]":
        """Create a Ragged array from data and lengths. The lengths array should have
        the intended shape of the Ragged array.

        Parameters
        ----------
        data
            1D data array.
        lengths
            Lengths of each element in the ragged array.
        """
        return cls(data, lengths.shape, maybe_lengths=lengths)

    @classmethod
    def empty(
        cls, shape: Union[int, Tuple[int, ...]], dtype: type[DTYPE]
    ) -> "Ragged[DTYPE]":
        """Create an empty Ragged array."""
        if shape == ():
            raise ValueError("Ragged array must have at least one element.")

        if isinstance(shape, int):
            shape = (shape,)

        return cls(
            np.empty(0, dtype=dtype),
            shape,
            maybe_offsets=np.empty(np.prod(shape, dtype=np.int64) + 1, dtype=np.int64),
        )

    @staticmethod
    def concat(*arrays: "Ragged[DTYPE]", axis: int) -> "Ragged[DTYPE]":
        # """Concatenate multiple Ragged arrays along a given axis."""
        # need to check whether this would lead to incorrect indexing
        raise NotImplementedError
        if len(set((*a.shape[:axis], *a.shape[axis + 1 :]) for a in arrays)) != 1:
            raise ValueError(
                f"All arrays must have the same shape except along axis {axis}."
            )

        if len(set(a.data.dtype for a in arrays)) != 1:
            raise ValueError("All arrays must have the same dtype.")

        data = np.concatenate([a.data for a in arrays])
        lengths = np.concatenate([a.lengths for a in arrays], axis=axis)
        return Ragged.from_lengths(data, lengths)

    @staticmethod
    def stack(*arrays: "Ragged[DTYPE]") -> "Ragged[DTYPE]":
        """Stack multiple ragged arrays along a new first axis."""
        if len(set(a.shape for a in arrays)) != 1:
            raise ValueError("All arrays must have the same shape.")

        if len(set(a.data.dtype for a in arrays)) != 1:
            raise ValueError("All arrays must have the same dtype.")

        data = np.concatenate([a.data for a in arrays])
        lengths = np.stack([a.lengths for a in arrays], axis=0)
        return Ragged.from_lengths(data, lengths)

    def to_padded(self, pad_value: Any) -> NDArray[RDTYPE]:
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
        length = self.lengths.max()
        shape = (*self.shape, length)
        return pad_ragged(self.data, self.offsets, shape, pad_value)

    def squeeze(self, axis: Union[int, Tuple[int, ...]] = -1) -> Ragged[RDTYPE]:
        """Squeeze the ragged array along the given non-ragged axis."""
        return Ragged.from_lengths(self.data, np.squeeze(self.lengths, axis=axis))

    def reshape(self, shape: Tuple[int, ...]) -> Ragged[RDTYPE]:
        """Reshape non-ragged axes."""
        return Ragged.from_lengths(self.data, self.lengths.reshape(shape))

    def __str__(self):
        return (
            f"Ragged<shape={self.shape} dtype={self.data.dtype} size={self.data.size}>"
        )

    def __getitem__(self, idx: Idx):
        if isinstance(idx, (int, np.integer)):
            return self.data[self.offsets[idx] : self.offsets[idx + 1]]
        else:
            raise NotImplementedError


INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)
RaggedIntervals = Ragged[np.void]


@nb.njit(parallel=True, nogil=True, cache=True)
def pad_ragged(
    data: NDArray[DTYPE],
    offsets: NDArray[np.integer],
    shape: Tuple[int, ...],
    pad_value: DTYPE,
):
    out = np.empty((np.prod(shape[:-1]), shape[-1]), dtype=data.dtype)
    for i in nb.prange(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        entry_len = end - start
        out[i, :entry_len] = data[start:end]
        out[i, entry_len:] = pad_value
    out = out.reshape(shape)
    return out


COMPLEMENT_MAP = dict(zip(b"ACGT", b"TGCA"))


@nb.njit(parallel=True, nogil=True, cache=True)
def _rc_helper(
    data: NDArray[np.uint8], offsets: NDArray[np.int64], mask: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    out = np.empty_like(data)
    for i in nb.prange(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        _data = data[start:end]
        _out = out[start:end]
        if mask[i]:
            for nuc, comp in COMPLEMENT_MAP.items():
                _out[_data == nuc] = comp
            _out[:] = _out[::-1]
        else:
            _out[:] = _data
    return out


def _reverse_complement(
    seqs: Ragged[np.bytes_], mask: NDArray[np.bool_]
) -> Ragged[np.bytes_]:
    _, mask = np.broadcast_arrays(seqs.lengths, mask)
    rc_seqs = _rc_helper(seqs.data.view(np.uint8), seqs.offsets, mask.ravel())
    return Ragged.from_offsets(rc_seqs.view("S1"), seqs.shape, seqs.offsets)


@nb.njit(parallel=True, nogil=True, cache=True)
def _reverse_helper(
    data: NDArray[np.uint8], offsets: NDArray[np.int64], mask: NDArray[np.bool_]
):
    for i in nb.prange(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        if mask[i]:
            data[start:end] = data[start:end:-1]


def _reverse(tracks: Ragged[np.float32], mask: NDArray[np.bool_]):
    """Reverses data along the ragged axis in-place."""
    _, mask = np.broadcast_arrays(tracks.lengths, mask)
    _reverse_helper(tracks.data.view(np.uint8), tracks.offsets, mask.ravel())
