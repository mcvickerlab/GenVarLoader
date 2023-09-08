import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .fasta import Fasta
from .types import DenseAlleles, Reader, SparseAlleles, Variants


@nb.njit(
    "(u1[:, :, :], u4[:], i4[:], u1[:, :])",
    nogil=True,
    parallel=True,
    cache=True,
)
def apply_variants(
    seqs: NDArray[np.uint8],
    offsets: NDArray[np.uint32],
    positions: NDArray[np.int32],
    alleles: NDArray[np.uint8],
):
    # seqs (s, p, l)
    # offsets (s+1)
    # positions (v)
    # alleles (p, v)
    for sample_idx in nb.prange(len(offsets) - 1):
        start = offsets[sample_idx]
        end = offsets[sample_idx + 1]
        sample_positions = positions[start:end]
        sample_alleles = alleles[:, start:end]
        sample_seq = seqs[sample_idx]
        sample_seq[:, sample_positions] = sample_alleles


class FastaVariants(Reader):
    def __init__(self, name: str, fasta: Fasta, variants: Variants) -> None:
        self.fasta = fasta
        self.variants = variants
        self.virtual_data = xr.DataArray(
            da.empty((self.variants.n_samples, self.variants.ploidy), dtype="S1"),
            name=name,
            coords={
                "sample": np.asarray(self.variants.samples),
                "ploid": np.arange(self.variants.ploidy, dtype=np.uint32),
            },
        )

    def read(self, contig: str, start: int, end: int, **kwargs) -> xr.DataArray:
        """_summary_

        Parameters
        ----------
        contig : str
            _description_
        start : int
            _description_
        end : int
            _description_
        **kwargs
            Additional keyword arguments. May include `samples: Iterable[str]` and
            `ploid: Iterable[int]` to specify samples and ploid numbers.

        Returns
        -------
        xr.DataArray
            _description_
        """
        samples = kwargs.get("sample", None)
        if samples is None:
            n_samples = self.variants.n_samples
        else:
            n_samples = len(samples)

        ploid = kwargs.get("ploid", None)
        if ploid is None:
            ploid = self.variants.ploidy
        else:
            ploid = len(ploid)

        ref: NDArray[np.bytes_] = self.fasta.read(contig, start, end).to_numpy()
        result = self.variants.read(contig, start, end, **kwargs)
        seqs = np.tile(ref, (n_samples, ploid, 1))

        if result is None:
            return xr.DataArray(seqs, dims=["sample", "ploid", "length"])
        elif isinstance(result, SparseAlleles):
            apply_variants(
                seqs.view(np.uint8),
                result.offsets,
                result.positions - start,
                result.alleles.view(np.uint8),
            )
        elif isinstance(result, DenseAlleles):
            seqs[..., result.positions - start] = result.alleles

        return xr.DataArray(seqs.view("S1"), dims=["sample", "ploid", "length"])
