from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence, Union, overload

import numpy as np
import polars as pl
import xarray as xr
from attrs import define
from loguru import logger
from numpy.typing import NDArray


class Reader(Protocol):
    """Implements the read() method for returning data aligned to genomic coordinates.

    Attributes
    ----------
    virtual_data : xr.DataArray
        Virtual data describing the type and dimensions of the data yielded by this
        reader. This data includes all dimensions except the length dimension since
        this is determined by the length of the genomic range passed to `read()`.
    contig_starts_with_chr : str, optional
        Whether the contigs start with "chr" or not. Queries to `read()` will
        normalize the contig name to add or remove this prefix to match what the
        underlying file uses.
    """

    virtual_data: xr.DataArray
    contig_starts_with_chr: Optional[bool]

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

    def infer_contig_prefix(self, contigs: Iterable[str]) -> bool:
        n_chr_start = sum(1 for c in contigs if c.startswith("chr"))
        if n_chr_start > 0:
            contig_starts_with_chr = True
        else:
            contig_starts_with_chr = False
        return contig_starts_with_chr

    def normalize_contig_name(self, contig: str) -> str:
        """Normalize the contig name to adhere to the convention of the underlying file.
        i.e. remove or add "chr" to the contig name.

        Parameters
        ----------
        contig : str

        Returns
        -------
        str
            Normalized contig name.
        """
        if self.contig_starts_with_chr is None:
            logger.warning(
                """Attempted to normalize a contig name for a reader that has no 
                convention for contig names. Returning contig name as is.
                """
            )
            return contig
        elif self.contig_starts_with_chr and not contig.startswith("chr"):
            contig = "chr" + contig
        elif not self.contig_starts_with_chr and contig.startswith("chr"):
            contig = contig[3:]
        return contig


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
class VLenAlleles:
    offsets: NDArray[np.uint32]
    alleles: NDArray[np.bytes_]

    @overload
    def __getitem__(self, idx: int) -> NDArray[np.bytes_]:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "VLenAlleles":
        ...

    def __getitem__(self, idx: Union[int, slice, np.integer]):
        if isinstance(idx, (int, np.integer)):
            return self.get_idx(idx)
        elif isinstance(idx, slice):
            return self.get_slice(idx)

    def get_idx(self, idx: Union[int, np.integer]):
        if idx >= len(self) or idx < -len(self):
            raise IndexError("Index out of range.")
        if idx < 0:
            idx = len(self) + idx
        return self.alleles[self.offsets[idx] : self.offsets[idx + 1]]

    def get_slice(self, slc: slice):
        start: Optional[int]
        stop: Optional[int]
        start, stop = slc.start, slc.stop
        if start is None:
            start = 0
        if start >= len(self) or (stop is not None and stop <= start):
            return VLenAlleles(np.empty(0, np.uint32), np.empty(0, "|S1"))
        if stop is not None:
            stop += 1
        new_offsets = self.offsets[start:stop] - self.offsets[start]
        _start, _stop = new_offsets[0], new_offsets[-1]
        new_alleles = self.alleles[_start:_stop]
        return VLenAlleles(new_offsets, new_alleles)

    def __len__(self):
        return len(self.offsets) - 1

    @classmethod
    def from_polars(cls, alleles: pl.Series):
        offsets = np.zeros(alleles.len() + 1, np.uint32)
        offsets[1:] = alleles.str.lengths().cumsum().to_numpy()
        flat_alleles = np.frombuffer(
            alleles.str.concat("").to_numpy()[0].encode(), "S1"
        )
        return cls(offsets, flat_alleles)


@define
class SparseAlleles:
    """Sparse/ragged array of single base pair alleles.

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


@define
class DenseGenotypes:
    """Dense array(s) of genotypes.

    Attributes
    ----------
    positions : NDArray[np.int32]
        Shape: (variants)
    ref : VLenAlleles
        Shape: (variants). REF alleles.
    alt : VLenAlleles
        Shape: (variants). ALT alleles.
    genotypes : NDArray[np.int8]
        Shape: (samples, ploid, variants)
    max_end : int
        End of reference to ensure enough is available to pad fixed length haplotypes.
    """

    positions: NDArray[np.int32]
    sizes: NDArray[np.int32]
    ref: VLenAlleles
    alt: VLenAlleles
    genotypes: NDArray[np.int8]
    max_end: int


class Variants(Protocol):
    """
    Implements the read() method for returning variants from a given genomic range.
    """

    samples: Union[Sequence[str], NDArray[np.str_]]
    n_samples: int
    ploidy: int
    contig_starts_with_chr: Optional[bool]

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[Union[SparseAlleles, DenseGenotypes]]:
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
            Additional keyword arguments. May include `sample: Iterable[str]` and
            `ploid: Iterable[int]` to specify sample names and ploid numbers.

        Returns
        -------
        If no variants are in the region specified, returns None. Otherwise, returns
        either SparseAlleles or DenseGenotypes depending on the file format.
        """
        ...

    def infer_contig_prefix(self, contigs: Iterable[str]) -> bool:
        n_chr_start = sum(1 for c in contigs if c.startswith("chr"))
        if n_chr_start > 0:
            contig_starts_with_chr = True
        else:
            contig_starts_with_chr = False
        return contig_starts_with_chr

    def normalize_contig_name(self, contig: str) -> str:
        """Normalize the contig name to adhere to the convention of the variant's
        underlying file. i.e. remove or add "chr" to the contig name.

        Parameters
        ----------
        contig : str

        Returns
        -------
        str
            Normalized contig name.
        """
        if self.contig_starts_with_chr is None:
            logger.warning(
                """Attempted to normalize a contig name for a reader that has no 
                convention for contig names. Returning contig name as is.
                """
            )
            return contig
        elif self.contig_starts_with_chr and not contig.startswith("chr"):
            contig = "chr" + contig
        elif not self.contig_starts_with_chr and contig.startswith("chr"):
            contig = contig[3:]
        return contig
