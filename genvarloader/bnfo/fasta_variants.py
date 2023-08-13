import numba as nb
import numpy as np
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
        self.name = name
        self.fasta = fasta
        self.variants = variants
        self.dtype = np.dtype("S1")
        self.sizes = {"sample": self.variants.n_samples, "ploidy": self.variants.ploidy}
        self.bytes_per_length = variants.n_samples * variants.ploidy

    def read(self, contig: str, start: int, end: int, **kwargs) -> NDArray[np.bytes_]:
        ref = self.fasta.read(contig, start, end)
        seqs = np.tile(ref, (self.variants.n_samples, self.variants.ploidy, 1))
        result = self.variants.read(contig, start, end, **kwargs)

        if result is None:
            return seqs

        offsets, positions, alleles = result
        positions = positions - start

        apply_variants(seqs.view("u1"), offsets, positions, alleles.view("u1"))
        return seqs.view("S1")
