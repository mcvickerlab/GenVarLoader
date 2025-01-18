from __future__ import annotations

from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from attrs import define
from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = ["Reader", "Ragged"]


DTYPE = TypeVar("DTYPE", bound=np.generic)
Idx = Union[int, np.integer, Sequence[int], NDArray[np.integer], slice]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


class Reader(Protocol):
    """Implements the read() method for returning data aligned to genomic coordinates."""

    name: str
    """Name of the reader, corresponding to the name of the DataArrays it returns."""
    dtype: DTypeLike
    """Data type of what the reader returns."""
    contigs: Mapping[str, int]
    sizes: Dict[str, int]
    """Sizes of the dimensions/axes of what the reader returns."""
    coords: Dict[str, NDArray]
    """Coordinates of what the reader returns, i.e. dimension labels."""
    rev_strand_fn: Callable[[NDArray], NDArray]
    """Function to reverse (and potentially complement) data for a genomic region. This
        is used when the strand is negative."""
    chunked: bool
    """Whether the reader acts like a chunked array store, in which sequential reads
        are far more performant than random access."""

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        **kwargs,
    ) -> NDArray:
        """Read data corresponding to given genomic coordinates, akin to orthogonal indexing.
        The output shape will have length as the final dimension/axis i.e. (..., length).

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based exclusive.
        **kwargs
            Additional keyword arguments. For example, which samples or ploid numbers to
            return.

        Returns
        -------
        NDArray
            Data corresponding to the given genomic coordinates. The final axis is the
            length axis i.e. has length == (ends - starts).sum().

        Notes
        -----
        When multiple regions are provided (i.e. multiple starts and ends) they should
        be concatenated together in the output array along the length dimension.
        """
        ...


class _ToZarr(Protocol):
    """Implements the to_zarr() method."""

    def to_zarr(self, store: Path):
        """Materialize genomes-wide data as a Zarr store.

        Parameters
        ----------
        store : Path
            Directory to write the Zarr store.
        """
        ...


RDTYPE = TypeVar("RDTYPE", bound=np.generic, contravariant=True)


@define
class Ragged(Generic[RDTYPE]):
    """Ragged array with non-length dimensions."""

    data: NDArray[RDTYPE]
    """A 1D array of the data."""
    shape: Tuple[int, ...]
    """Shape of the ragged array, excluding the length dimension. For example, if
        the shape is (2, 3), then the j, k-th element can be mapped to an index for
        offsets with :code:`i = np.ravel_multi_index((j, k), shape)`. The number of ragged
        elements should correspond to the product of the shape."""
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
        """Concatenate multiple Ragged arrays along a given axis."""
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


INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)
RaggedIntervals = Ragged[np.void]
