import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .fasta import Fasta
from .types import Reader, Variants


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
        sample_pos = positions[start:end]
        sample_alel = alleles[:, start:end]
        sample_seq = seqs[sample_idx]
        sample_seq[:, sample_pos] = sample_alel


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
        ref = self.fasta.read(contig, start, end).to_numpy()
        seqs = np.tile(ref, (self.variants.n_samples, self.variants.ploidy, 1))
        result = self.variants.read(contig, start, end, **kwargs)

        if result is None:
            return xr.DataArray(seqs, dims=["sample", "ploid", "length"])

        offsets, positions, alleles = result
        positions = positions - start

        apply_variants(seqs.view("u1"), offsets, positions, alleles.view("u1"))
        return xr.DataArray(seqs.view("S1"), dims=["sample", "ploid", "length"])
