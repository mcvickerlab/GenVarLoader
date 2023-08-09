import numba as nb
import numpy as np
from numpy.typing import NDArray

from .fasta import Fasta
from .types import Reader, Variants


@nb.njit(
    "u1[:, :, :](u1[:, :, :], u4[:], i4[:], u1[:, :])",
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
    for i in nb.prange(len(offsets) - 1):
        s = offsets[i]
        e = offsets[i + 1]
        i_pos = positions[s:e]
        i_ale = alleles[:, s:e]
        i_seq = seqs[i]
        i_seq[:, i_pos] = i_ale
    return seqs


class FastaVariants(Reader):
    def __init__(self, name: str, fasta: Fasta, variants: Variants) -> None:
        self.name = name
        self.fasta = fasta
        self.variants = variants
        self.bytes_per_length = variants.n_samples * variants.ploidy

    def read(self, contig: str, start: int, end: int) -> NDArray[np.bytes_]:
        ref = self.fasta.read(contig, start, end)
        seqs = np.tile(ref, (self.variants.n_samples, self.variants.ploidy, 1))
        result = self.variants.read(contig, start, end)

        if result is None:
            return seqs

        offsets, positions, alleles = result
        positions -= start

        seqs = apply_variants(seqs.view("u1"), offsets, positions, alleles.view("u1"))
        return seqs.view("S1")
