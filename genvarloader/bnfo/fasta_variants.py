from typing import Optional

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .fasta import Fasta
from .types import DenseGenotypes, Reader, Variants


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
        """Read a variant sequence corresponding to a genomic range, sample, and ploid.

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
            `ploid: Iterable[int]` to specify samples and ploid numbers.

        Returns
        -------
        xr.DataArray
            Variant sequences, dimensions: (sample, ploid, length)
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

        variants = self.variants.read(contig, start, end, **kwargs)

        if variants is None:
            ref: NDArray[np.bytes_] = self.fasta.read(contig, start, end).to_numpy()

            # this is faster than np.tile ¯\_(ツ)_/¯
            seqs = np.empty((n_samples, ploid, len(ref)), dtype=ref.dtype)
            seqs[...] = ref
            return xr.DataArray(seqs, dims=["sample", "ploid", "length"])
        elif isinstance(variants, DenseGenotypes):
            ref: NDArray[np.bytes_] = self.fasta.read(
                contig, start, variants.max_end
            ).to_numpy()
            seqs = np.empty((n_samples, ploid, end - start), dtype=ref.dtype)
            shifts = sample_shifts(variants.genotypes, variants.sizes)
            construct_haplotypes_with_indels(
                seqs.view(np.uint8),
                ref.view(np.uint8),
                shifts,
                variants.positions - start,
                variants.sizes,
                variants.genotypes,
                variants.alt.offsets,
                variants.alt.alleles.view(np.uint8),
            )
        else:
            # SparseAlleles
            raise NotImplementedError

        return xr.DataArray(seqs.view("S1"), dims=["sample", "ploid", "length"])


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


def sample_shifts(genotypes, sizes, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    diffs = np.where(genotypes == 1, sizes, 0).cumsum(-1, dtype=np.int32)
    shifts = rng.integers(0, diffs[..., -1].clip(0) + 1, dtype=np.int32)
    return shifts


@nb.njit(nogil=True)
def construct_haplotypes_with_indels(
    out: NDArray[np.uint8],
    ref: NDArray[np.uint8],
    shifts: NDArray[np.int32],
    rel_positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    genotypes: NDArray[np.int8],
    alt_offsets: NDArray[np.uint32],
    alt_alleles: NDArray[np.uint8],
):
    n_samples = out.shape[0]
    ploidy = out.shape[1]
    fixed_length = out.shape[-1]
    n_variants = len(rel_positions)

    for sample in nb.prange(n_samples):
        for hap in nb.prange(ploidy):
            # where to get next reference subsequence
            ref_idx = 0
            # where to put next subsequence
            out_idx = 0
            # total amount to shift by
            shift = shifts[sample, hap]
            # how much we've shifted
            shifted = 0

            # first variant is a DEL spanning start
            v_rel_pos = rel_positions[0]
            v_diff = sizes[0]
            if v_rel_pos < 0 and genotypes[sample, hap, 0] == 1:
                ref_idx = v_rel_pos - v_diff + 1
                # first variant index for this sample, haplotype
                start_idx = 1
            else:
                start_idx = 0

            for variant in range(start_idx, n_variants):
                # UNKNOWN or REF
                if genotypes[sample, hap, variant] != 1:
                    continue

                # position of variant relative to ref from fetch(contig, start, q_end)
                # i.e. put it into same coordinate system as ref_idx
                v_rel_pos = rel_positions[variant]
                v_diff = sizes[variant]
                allele = alt_alleles[alt_offsets[variant] : alt_offsets[variant + 1]]
                v_len = len(allele)

                # handle shift
                if shifted < shift:
                    ref_shift_dist = v_rel_pos - ref_idx
                    # enough distance between ref_idx and variant to finish shift
                    if shifted + ref_shift_dist >= shift:
                        ref_idx += shift - shifted
                        shifted = shift
                        # can still use the variant and whatever ref is left between
                        # ref_idx and the variant
                    # not enough distance to finish the shift even with the variant
                    elif shifted + ref_shift_dist + v_len < shift:
                        ref_idx = v_rel_pos + 1
                        shifted += ref_shift_dist + v_len
                        continue
                    # ref + (some of) variant is enough to finish shift
                    else:
                        # how much left to shift - amount of ref we can use
                        allele_start_idx = shift - shifted - ref_shift_dist
                        allele = allele[allele_start_idx:]
                        v_len = len(allele)
                        # adjust ref_idx so that no reference is written
                        ref_idx = v_rel_pos
                        shifted = shift

                # add reference sequence
                ref_len = v_rel_pos - ref_idx
                out[sample, hap, out_idx : out_idx + ref_len] = ref[ref_idx:v_rel_pos]
                out_idx += ref_len

                # handle insertions + substitions
                # for deletions we simply write reference up to the variant (above)
                # and increment the ref_idx (below)
                # add variant
                if v_diff > 0:
                    writable_length = min(v_len, fixed_length - out_idx)
                    out[sample, hap, out_idx : out_idx + v_len] = allele[
                        :writable_length
                    ]
                    out_idx += v_len

                # non-deletion ALT alleles always replace 1 nt of reference for a
                # normalized VCF
                ref_idx = v_rel_pos + 1

                if out_idx >= fixed_length:
                    break

            # fill rest with reference sequence
            unfilled_length = fixed_length - out_idx
            if unfilled_length > 0:
                out[sample, hap, out_idx:] = ref[ref_idx : ref_idx + unfilled_length]
