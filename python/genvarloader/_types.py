from __future__ import annotations

from pathlib import Path
from typing import (
    Callable,
    Dict,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from attrs import define
from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = ["Reader"]


DTYPE = TypeVar("DTYPE", bound=np.generic)
Idx = Union[
    int, np.integer, Sequence[int], slice, NDArray[np.integer], NDArray[np.bool_]
]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


@define
class AnnotatedHaps:
    haps: NDArray[np.bytes_]
    var_idxs: NDArray[np.int32]
    ref_coords: NDArray[np.int32]

    @property
    def shape(self):
        return self.haps.shape

    def reshape(self, *shape: int | tuple[int, ...]):
        return AnnotatedHaps(
            self.haps.reshape(*shape),
            self.var_idxs.reshape(*shape),
            self.ref_coords.reshape(*shape),
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> AnnotatedHaps:
        return AnnotatedHaps(
            self.haps.squeeze(axis),
            self.var_idxs.squeeze(axis),
            self.ref_coords.squeeze(axis),
        )


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
