from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class Reader(Protocol):
    """Protocol for classes that implement the read() method.

    Attributes
    ----------
    name : str
        Name of the reader.
    dtype : np.dtype
        Data type of the arrays returned by the read() method.
    sizes : Dict[str, int]
        Dictionary mapping non-length dimension names to their sizes, in order.
    """

    name: str
    dtype: np.dtype
    sizes: Dict[str, int]

    def read(self, contig: str, start: int, end: int, **kwargs) -> NDArray:
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
    """Protocol for classes that implement the to_zarr() method."""

    def to_zarr(self, store: Path):
        """Materialize genomes-wide data as a Zarr store.

        Parameters
        ----------
        store : Path
            Directory to write the Zarr store.
        """
        ...


class Variants(Protocol):
    samples: List[str]
    n_samples: int
    ploidy: int

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[Tuple[NDArray[np.uint32], NDArray[np.int32], NDArray[np.bytes_]]]:
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
            Additional keyword arguments. May include `samples: list[str]` and
            `ploid: Iterable[int]` to specify samples and ploid numbers.

        Returns
        -------
        If no variants are in the region specified, returns None. Otherwise:

        offsets : NDArray[np.uint32]
            Shape: (samples + 1). Offsets for the index boundaries of each sample such
            that variants for sample `i` are `positions[offsets[i] : offsets[i+1]]` and
            `alleles[..., offsets[i] : offsets[i+1]]`.
        positions : NDArray[np.int32]
            Shape: (variants). 0-based position of each variant.
        alleles : NDArray[np.bytes_]
            Shape: (ploid, variants). Alleles found at each variant.
        """
        ...
