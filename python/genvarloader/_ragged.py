from __future__ import annotations

from typing import Any, Generic, Optional, Tuple, TypeVar, Union

import awkward as ak
import numba as nb
import numpy as np
from attrs import define
from awkward.contents import ListOffsetArray, NumpyArray, RegularArray
from awkward.index import Index64
from numpy.typing import NDArray

from ._types import Idx
from ._utils import _lengths_to_offsets, idx_like_to_array

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
            self.maybe_offsets = _lengths_to_offsets(self.lengths)
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
        data: NDArray[RDTYPE],
        shape: Union[int, Tuple[int, ...]],
        offsets: NDArray[np.int64],
    ) -> "Ragged[RDTYPE]":
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
        cls, data: NDArray[RDTYPE], lengths: NDArray[np.int32]
    ) -> "Ragged[RDTYPE]":
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
        cls, shape: Union[int, Tuple[int, ...]], dtype: type[RDTYPE]
    ) -> "Ragged[RDTYPE]":
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
        if self.data.dtype.str == "|S1":
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
            pad_ragged(self.data.view(np.uint8), self.offsets, pad_value, padded)
            padded = padded.view(self.data.dtype).reshape(shape)
        else:
            padded = np.empty((np.prod(shape[:-1]), shape[-1]), dtype=self.data.dtype)
            pad_ragged(self.data, self.offsets, pad_value, padded)
            padded = padded.reshape(shape)

        return padded

    def squeeze(
        self, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Ragged[RDTYPE]:
        """Squeeze the ragged array along the given non-ragged axis."""
        return Ragged.from_lengths(self.data, self.lengths.squeeze(axis))

    def reshape(self, shape: Tuple[int, ...]) -> Ragged[RDTYPE]:
        """Reshape non-ragged axes."""
        # this is correct because all reshaping operations preserve the layout i.e. raveled ordered
        return Ragged.from_lengths(self.data, self.lengths.reshape(shape))

    def __str__(self):
        return (
            f"Ragged<shape={self.shape} dtype={self.data.dtype} size={self.data.size}>"
        )

    def __getitem__(self, idx: Union[Idx, Tuple[Idx, ...]]):
        if not isinstance(idx, tuple):
            idx = (idx,)

        if len(idx) < len(self.shape):
            raise IndexError(
                f"Too few indices for array: expected {len(self.shape)}, got {len(idx)}."
            )

        idx = tuple(
            idx_like_to_array(_idx, self.shape[i]) for i, _idx in enumerate(idx)
        )
        idx = np.ravel_multi_index(idx, self.shape).squeeze()

        return self.data[self.offsets[idx] : self.offsets[idx + 1]]

    def to_awkward(self) -> ak.Array:
        """Convert to an `Awkward <https://awkward-array.org/doc/main/>`_ array without copying. Note that this effectively
        returns a view of the data, so modifying the data will modify the original array."""
        layout = ListOffsetArray(
            Index64(self.offsets),
            NumpyArray(self.data),  # type: ignore | NDArray[RDTYPE] is ArrayLike
        )

        for size in reversed(self.shape[1:]):
            layout = RegularArray(layout, size)

        return ak.Array(layout)

    @classmethod
    def from_awkward(cls, awk: "ak.Array") -> "Ragged":
        """Convert from an `Awkward <https://awkward-array.org/doc/main/>`_ array without copying. Note that this effectively
        returns a view of the data, so modifying the data will modify the original array."""
        # parse shape
        shape_str = str(awk.type).split(" * ")
        try:
            shape = tuple(map(int, shape_str[:-2]))
        except ValueError as err:
            raise ValueError(
                f"Only the final axis of an awkward array may be variable to convert to ragged, but got {awk.type}."
            ) from err

        # extract data and offsets
        layout = awk.layout._offsets_and_flattened(0, -1)[1].content
        data = np.asarray(layout.content.data)
        offsets = np.asarray(layout.offsets.data)

        return cls.from_offsets(data, shape, offsets)


INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)
RaggedIntervals = Ragged[np.void]


@nb.njit(parallel=True, nogil=True, cache=True)
def pad_ragged(
    data: NDArray[DTYPE],
    offsets: NDArray[np.integer],
    pad_value: DTYPE,
    out: NDArray[DTYPE],
):
    for i in nb.prange(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
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
    out = np.empty_like(data)
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
    _, mask = np.broadcast_arrays(seqs.lengths, mask)
    rc_seqs = _rc_helper(seqs.data.view(np.uint8), seqs.offsets, mask.ravel())
    return Ragged.from_offsets(rc_seqs.view("S1"), seqs.shape, seqs.offsets)


@nb.njit(parallel=True, nogil=True, cache=True)
def _reverse_helper(
    data: NDArray[np.uint8], offsets: NDArray[np.int64], mask: NDArray[np.bool_]
):
    for i in nb.prange(len(offsets) - 1):
        if mask[i]:
            start, end = offsets[i], offsets[i + 1]
            if start > 0:
                reverse_slice = slice(end - 1, start - 1, -1)
            else:
                reverse_slice = slice(end - 1, None, -1)
            data[start:end] = data[reverse_slice]


def _reverse(tracks: Ragged[np.float32], mask: NDArray[np.bool_]):
    """Reverses data along the ragged axis in-place."""
    _, mask = np.broadcast_arrays(tracks.lengths, mask)
    _reverse_helper(tracks.data.view(np.uint8), tracks.offsets, mask.ravel())


def _jitter(
    *arrays: Ragged,
    max_jitter: int,
    seed: Optional[Union[int, np.random.Generator]] = None,
):
    """Jitter the data along the ragged axis by up to `max_jitter`.

    Assumes only the first axis is to be jittered. In other words, assumes that no user would want to
    jitter independently across ploidy or tracks.
    """
    if not isinstance(seed, np.random.Generator):
        seed = np.random.default_rng(seed)

    batch_size = arrays[0].shape[0]

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
