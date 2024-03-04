from pathlib import Path
from typing import Callable, Dict, Mapping, Protocol

from numpy.typing import ArrayLike, DTypeLike, NDArray


class Reader(Protocol):
    """Implements the read() method for returning data aligned to genomic coordinates.

    Attributes
    ----------
    name : str
        Name of the reader, corresponding to the name of the DataArrays it returns.
    dtype : np.dtype
        Data type of what the reader returns.
    sizes : Dict[Hashable, int]
        Sizes of the dimensions/axes of what the reader returns.
    coords : Dict[Hashable, NDArray]
        Coordinates of what the reader returns, i.e. dimension labels.
    contig_starts_with_chr : bool, optional
        Whether the contigs start with "chr" or not. Queries to `read()` will
        normalize the contig name to add or remove this prefix to match what the
        underlying file uses.
    rev_strand_fn : Callable[[NDArray], NDArray]
        Function to reverse (and potentially complement) data for a genomic region. This
        is used when the strand is negative.
    chunked : bool
        Whether the reader acts like a chunked array store, in which sequential reads
        are far more performant than random access.
    """

    name: str
    dtype: DTypeLike
    contigs: Mapping[str, int]
    sizes: Dict[str, int]
    coords: Dict[str, NDArray]
    rev_strand_fn: Callable[[NDArray], NDArray]
    chunked: bool

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

    def vidx(
        self, contigs: ArrayLike, starts: ArrayLike, length: int, **kwargs
    ) -> NDArray:
        """Read data akin to vectorized indexing. Output shape will be (queries, length)."""
        ...


class ToZarr(Protocol):
    """Implements the to_zarr() method."""

    def to_zarr(self, store: Path):
        """Materialize genomes-wide data as a Zarr store.

        Parameters
        ----------
        store : Path
            Directory to write the Zarr store.
        """
        ...
