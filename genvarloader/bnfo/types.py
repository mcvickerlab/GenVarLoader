from pathlib import Path
from typing import Optional, Protocol, Sequence, Union

import numpy as np
import xarray as xr
from attrs import define
from numpy.typing import NDArray


class Reader(Protocol):
    """Implements the read() method for returning data aligned to genomic coordinates.

    Attributes
    ----------
    virtual_data : xr.DataArray
        Virtual data describing the type and dimensions of the data yielded by this
        reader. This data includes all dimensions except the length dimension since
        this is determined by the length of the genomic range passed to read().
    """

    virtual_data: xr.DataArray

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        """Read data corresponding to given genomic coordinates. The output shape will
        have length as the final axis i.e. (..., length).

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        start : int
            Start coordinate, 0-based.
        end : int
            End coordinate, 0-based exclusive.
        **kwargs
            Additional keyword arguments. For example, which samples or ploid numbers to
            return.

        Returns
        -------
        NDArray
            Data corresponding to the given genomic coordinates. The final axis is the
            length axis i.e. has length == end - start.
        """
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


@define
class SparseAlleles:
    """Sparse/ragged array of alleles.

    Attributes
    ----------
    offsets : NDArray[np.uint32]
        Shape: (samples + 1). Offsets for the index boundaries of each sample such
        that variants for sample `i` are `positions[offsets[i] : offsets[i+1]]` and
        `alleles[..., offsets[i] : offsets[i+1]]`.
    positions : NDArray[np.int32]
        Shape: (variants). 0-based position of each variant.
    alleles : NDArray[np.bytes_]
        Shape: (ploid, variants). Alleles found at each variant.
    """

    offsets: NDArray[np.uint32]
    positions: NDArray[np.int32]
    alleles: NDArray[np.bytes_]


@define
class DenseAlleles:
    """Dense array of alleles.

    Attributes
    ----------
    positions : NDArray[np.int32]
        Shape: (variants). 0-based position of each variant.
    alleles : NDArray[np.bytes_]
        Shape: (samples, ploid, variants). Alleles found at each variant.
    """

    positions: NDArray[np.int32]
    alleles: NDArray[np.bytes_]


class Variants(Protocol):
    """Implements the read() method for returning variants from a given genomic range."""

    samples: Union[Sequence[str], NDArray[np.str_]]
    n_samples: int
    ploidy: int

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[Union[SparseAlleles, DenseAlleles]]:
        """Read variants found in the given genomic coordinates, optionally for specific
        samples and ploid numbers.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        start : int
            Start coordinate, 0-based.
        end : int
            End coordinate, 0-based exclusive.
        **kwargs
            Additional keyword arguments. May include `samples: Iterable[str]` and
            `ploid: Iterable[int]` to specify samples and ploid numbers.

        Returns
        -------
        If no variants are in the region specified, returns None. Otherwise, either
        SparseAlleles or DenseAlleles depending on the implementation.
        """
        ...
