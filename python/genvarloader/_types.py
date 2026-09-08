from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, DTypeLike, NDArray

if TYPE_CHECKING:
    from ._ragged import RaggedIntervals

__all__ = ["IntervalTrack", "Reader"]


DTYPE = TypeVar("DTYPE", bound=np.generic)
INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)
IIdx = int | np.integer | NDArray[np.integer]
Idx = IIdx | slice | Sequence[int] | Sequence[bool] | NDArray[np.bool_] | pl.Series
StrIdx = Idx | str | Sequence[str] | NDArray[np.str_] | NDArray[np.object_]


@dataclass(slots=True)
class AnnotatedHaps:
    haps: NDArray[np.bytes_]
    """Haplotypes with dtype S1."""
    var_idxs: NDArray[np.int32]
    """Variant indices for each position in the haplotypes. A value of -1 indicates no variant was applied at the position."""
    ref_coords: NDArray[np.int32]
    """Reference coordinates for each position in haplotypes."""

    @property
    def shape(self):
        """Shape of the haplotypes and all annotations."""
        return self.haps.shape

    def reshape(self, shape: int | tuple[int, ...]):
        """Reshape the haplotypes and all annotations.

        Args:
            shape: New shape for the haplotypes and all annotations. The total number of elements
                must remain the same.
        """
        return AnnotatedHaps(
            self.haps.reshape(shape),
            self.var_idxs.reshape(shape),
            self.ref_coords.reshape(shape),
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> AnnotatedHaps:
        """Squeeze the haplotypes and all annotations along the specified axis.

        Args:
            axis: Axis or axes to squeeze. If None, all axes of length 1 will be squeezed.
        """
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
    sizes: dict[str, int]
    """Sizes of the dimensions/axes of what the reader returns."""
    coords: dict[str, NDArray]
    """Coordinates of what the reader returns, i.e. dimension labels."""
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

        Args:
            contig (str): Name of the contig/chromosome.
            starts (ArrayLike): Start coordinates, 0-based.
            ends (ArrayLike): End coordinates, 0-based exclusive.
            **kwargs: Additional keyword arguments. For example, which samples or ploid numbers to
                return.

        Returns:
            NDArray: Data corresponding to the given genomic coordinates. The final axis is the
                length axis i.e. has length == (ends - starts).sum().

        Notes:
            When multiple regions are provided (i.e. multiple starts and ends) they should
            be concatenated together in the output array along the length dimension.
        """
        ...

    @staticmethod
    def rev_strand_fn(data: NDArray) -> NDArray:
        """Function to reverse (and potentially complement) data for a genomic region.

        This is used when the strand is negative.
        """
        ...


class IntervalTrack(Protocol):
    """Structural protocol implemented by interval-valued track readers (e.g. :class:`BigWigs`, :class:`Table`).

    Used by :func:`gvl.write()` to accept either source via the ``tracks=`` parameter.
    """

    name: str
    samples: list[str]
    contigs: Mapping[str, int]

    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        """Return shape ``(regions, samples)`` count of intervals overlapping each ``(region, sample)`` cell."""
        ...

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> RaggedIntervals:
        """Read intervals using pre-computed offsets.

        This is unsafe — if ``offsets`` does not match what
        :meth:`count_intervals` would produce for the same arguments, behaviour
        is undefined. Use :meth:`count_intervals` to obtain valid offsets.
        """
        ...
