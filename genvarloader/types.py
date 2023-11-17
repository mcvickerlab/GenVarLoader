from pathlib import Path
from typing import (
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)

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
    rev_strand_fn : Callable[[NDArray], NDArray]
        Function to reverse (and potentially complement) data for a genomic region. This
        is used when the strand is negative.
    """

    name: str
    dtype: np.dtype
    sizes: Dict[Hashable, int]
    coords: Dict[Hashable, NDArray]
    contig_starts_with_chr: Optional[bool]
    rev_strand_fn: Callable[[NDArray], NDArray]

    def read(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        out: Optional[NDArray] = None,
        **kwargs
    ) -> xr.DataArray:
        """Read data corresponding to given genomic coordinates. The output shape will
        have length as the final dimension/axis i.e. (..., length).

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : NDArray[int32]
            End coordinates, 0-based exclusive.
        out : NDArray, optional
            Array to put the result into. Otherwise allocates one.
        **kwargs
            Additional keyword arguments. For example, which samples or ploid numbers to
            return.

        Returns
        -------
        xarray.DataArray
            Data corresponding to the given genomic coordinates. The final axis is the
            length axis i.e. has length == end - start.

        Notes
        -----
        Each call to `read` should correspond to a single disk access for each file
        represented by the Reader by reading from min(starts) to max(ends).
        When multiple regions are provided (i.e. multiple starts and ends) they should
        be concatenated together in the output array along the length dimension.
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
    """Variable length alleles.

    Create VLenAlleles from a polars Series of strings:
    >>> alleles = VLenAlleles.from_polars(pl.Series(["A", "AC", "G"]))

    Create VLenAlleles from offsets and alleles:
    >>> offsets = np.array([0, 1, 3, 4], np.uint32)
    >>> alleles = np.frombuffer(b"AACG", "|S1")
    >>> alleles = VLenAlleles(offsets, alleles)

    Get a single allele:
    >>> alleles[0]
    b'A'

    Get a slice of alleles:
    >>> alleles[1:]
    VLenAlleles(offsets=array([0, 2, 3]), alleles=array([b'AC', b'G'], dtype='|S1'))
    """

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
        new_offsets = self.offsets[start:stop].copy()
        _start, _stop = new_offsets[0], new_offsets[-1]
        new_alleles = self.alleles[_start:_stop]
        new_offsets -= self.offsets[start]
        return VLenAlleles(new_offsets, new_alleles)

    def __len__(self):
        return len(self.offsets) - 1

    @classmethod
    def from_polars(cls, alleles: pl.Series):
        offsets = np.zeros(alleles.len() + 1, np.uint32)
        offsets[1:] = alleles.str.len_bytes().cumsum().to_numpy()
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
class DenseGenotypes:
    """Dense array(s) of genotypes.

    Attributes
    ----------
    positions : NDArray[np.int32]
        Shape: (variants)
    size_diffs : NDArray[np.int32]
        Shape : (variants). Difference in length between the REF and the ALT alleles.
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
    size_diffs: NDArray[np.int32]
    ref: VLenAlleles
    alt: VLenAlleles
    genotypes: NDArray[np.int8]


class Variants(Protocol):
    """
    Implements the read() method for returning variants from a given genomic range.
    """

    samples: Union[Sequence[str], NDArray[np.str_]]
    n_samples: int
    ploidy: int
    contig_starts_with_chr: Optional[bool]

    def read(
        self, contig: str, starts: NDArray[np.int64], ends: NDArray[np.int64], **kwargs
    ) -> Union[Optional[SparseAlleles], List[Optional[DenseGenotypes]]]:
        """Read variants found in the given genomic coordinates, optionally for specific
        samples and ploid numbers.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : int, NDArray[int32]
            End coordinates, 0-based exclusive.
        **kwargs
            Additional keyword arguments. May include `sample: Iterable[str]` and
            `ploid: Iterable[int]` to specify sample names and ploid numbers.

        Returns
        -------
        Returns a list of optional DenseGenotypes, one for each query region.
        """
        ...

    def read_for_haplotype_construction(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        target_length: int,
        **kwargs
    ) -> Tuple[List[Optional[DenseGenotypes]], NDArray[np.int64]]:
        """Read variants sufficient to reconstruct haplotypes of a target length
        spanning the given genomic coordinates. This may necessitate returning variants
        beyond the ranges themselves, since deletions shrink the sequence and require
        information about variants past the end of the query regions.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[int32]
            Start coordinates, 0-based.
        ends : int, NDArray[int32]
            End coordinates, 0-based exclusive.
        target_length : int
            Target length of the reconstructed haplotypes.

        Returns
        -------
        List[Optional[DenseGenotypes]]
            Genotypes for each query region.
        NDArray[np.int64]
            New ends for querying the reference genome such that enough sequence is
            available to get haplotypes of `target_length`.
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
