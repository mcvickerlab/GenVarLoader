from textwrap import dedent
from typing import Optional

import dask.array as da
import numba as nb
import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import assert_never

from .fasta import Fasta
from .types import DenseGenotypes, Reader, SparseAlleles, Variants
from .util import get_rel_starts


class FastaVariants(Reader):
    def __init__(
        self, name: str, fasta: Fasta, variants: Variants, seed: Optional[int] = None
    ) -> None:
        self.fasta = fasta
        self.variants = variants
        self.virtual_data = xr.DataArray(
            da.empty(
                (self.variants.n_samples, self.variants.ploidy), dtype="S1"
            ),  # pyright: ignore[reportPrivateImportUsage]
            name=name,
            coords={
                "sample": np.asarray(self.variants.samples),
                "ploid": np.arange(self.variants.ploidy, dtype=np.uint32),
            },
        )
        self.contig_starts_with_chr = None
        if self.fasta.contig_starts_with_chr != self.variants.contig_starts_with_chr:
            logger.warning(
                dedent(
                    f"""
                Reference sequence and variant files have different contig naming
                conventions. Contig names in queries will be normalized so that they
                will still run, but this may indicate that the variants were not aligned
                to the reference being used to construct haplotypes. The reference
                file's contigs{"" if self.fasta.contig_starts_with_chr else " don't"}
                start with "chr" whereas the variant file's are the opposite.
                """
                )
                .replace("\n", " ")
                .strip()
            )
        self.rng = np.random.default_rng(seed)

    def read(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        out: Optional[NDArray[np.bytes_]] = None,
        **kwargs,
    ) -> xr.DataArray:
        """Read a variant sequence corresponding to a genomic range, sample, and ploid.

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
            ! Must include...
            target_length : int
                Desired length of reconstructed haplotypes.
            ? May optionally include...
            sample : Iterable[str]
                Specify samples.
            ploid : Iterable[int]
                Specify ploid numbers.
            seed : int
                For deterministic shifting of haplotypes longer than the query.

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

        seed = kwargs.get("seed", None)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        try:
            target_length = kwargs.pop("target_length")
        except KeyError:
            raise RuntimeError(
                "target_length is a required keyword argument for FastaVariants.read()."
            )

        variants, max_ends = self.variants.read_for_haplotype_construction(
            contig, starts, ends, target_length, **kwargs
        )

        lengths = ends - starts
        rel_starts = get_rel_starts(starts, ends)

        ref: NDArray[np.bytes_] = self.fasta.read(contig, starts, max_ends).to_numpy()
        ref_lengths = max_ends - starts
        ref_rel_starts = get_rel_starts(starts, max_ends)

        if out is None:
            # alloc then fill is faster than np.tile ¯\_(ツ)_/¯
            seqs = np.empty((n_samples, ploid, lengths.sum()), dtype=ref.dtype)
        else:
            seqs = out

        for variant, start, length, rel_start, ref_length, ref_rel_start in zip(
            variants, starts, lengths, rel_starts, ref_lengths, ref_rel_starts
        ):
            subseq = seqs[..., rel_start : rel_start + length]
            # subref can be longer than subseq
            subref = ref[ref_rel_start : ref_rel_start + ref_length]
            if variant is None:
                subseq[...] = subref[:length]
            elif isinstance(variant, DenseGenotypes):
                shifts = self.sample_shifts(variant.genotypes, variant.size_diffs)
                construct_haplotypes_with_indels(
                    subseq.view(np.uint8),
                    subref.view(np.uint8),
                    shifts,
                    variant.positions - start,
                    variant.size_diffs,
                    variant.genotypes,
                    variant.alt.offsets,
                    variant.alt.alleles.view(np.uint8),
                )
            elif isinstance(variant, SparseAlleles):
                raise NotImplementedError
            else:
                assert_never(variant)

        return xr.DataArray(seqs.view("S1"), dims=["sample", "ploid", "length"])

    def sample_shifts(self, genotypes: NDArray[np.int8], sizes: NDArray[np.int32]):
        diffs = np.where(genotypes == 1, sizes, 0).sum(-1, dtype=np.int32).clip(0)
        shifts = self.rng.integers(0, diffs + 1, dtype=np.int32)
        return shifts


@nb.njit(nogil=True, cache=True, parallel=True)
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
    length = out.shape[-1]
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
                # diff of v(-1) has been normalized to consider where ref is
                # otherwise, ref_idx = v_rel_pos - v_diff + 1
                # e.g. a -10 diff became -3 if v_rel_pos = -7
                ref_idx = v_rel_pos - v_diff + 1
                # increment the variant index
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

                # overlapping variants
                # v_rel_pos < ref_idx only if we see an ALT at a given position a second
                # time or more. We'll do what bcftools consensus does and only use the
                # first ALT variant we find.
                if v_rel_pos < ref_idx:
                    continue

                v_diff = sizes[variant]
                allele = alt_alleles[alt_offsets[variant] : alt_offsets[variant + 1]]
                v_len = len(allele)

                # handle shift
                if shifted < shift:
                    ref_shift_dist = v_rel_pos - ref_idx
                    # not enough distance to finish the shift even with the variant
                    if shifted + ref_shift_dist + v_len < shift:
                        ref_idx = v_rel_pos + 1
                        shifted += ref_shift_dist + v_len
                        continue
                    # enough distance between ref_idx and variant to finish shift
                    elif shifted + ref_shift_dist >= shift:
                        ref_idx += shift - shifted
                        shifted = shift
                        # can still use the variant and whatever ref is left between
                        # ref_idx and the variant
                    # ref + (some of) variant is enough to finish shift
                    else:
                        # adjust ref_idx so that no reference is written
                        ref_idx = v_rel_pos
                        shifted = shift
                        # how much left to shift - amount of ref we can use
                        allele_start_idx = shift - shifted - ref_shift_dist
                        # NEED THIS CHECK! otherwise parallel=True can cause a SystemError!
                        # parallel jit cannot handle changes in array dimension.
                        # without this, allele can change from a 1D array to a 0D array.
                        if allele_start_idx == v_len:
                            continue
                        allele = allele[allele_start_idx:]
                        v_len = len(allele)

                # add reference sequence
                ref_len = v_rel_pos - ref_idx
                if out_idx + ref_len >= length:
                    # ref will get written by final clause
                    break
                out[sample, hap, out_idx : out_idx + ref_len] = ref[
                    ref_idx : ref_idx + ref_len
                ]
                out_idx += ref_len

                # insertions + substitions
                writable_length = min(v_len, length - out_idx)
                out[sample, hap, out_idx : out_idx + writable_length] = allele[
                    :writable_length
                ]
                out_idx += writable_length
                # +1 because ALT alleles always replace 1 nt of reference for a
                # normalized VCF
                ref_idx = v_rel_pos + 1

                # deletions, move ref to end of deletion
                if v_diff < 0:
                    ref_idx -= v_diff

                if out_idx >= length:
                    break

            # fill rest with reference sequence
            unfilled_length = length - out_idx
            if unfilled_length > 0:
                out[sample, hap, out_idx:] = ref[ref_idx : ref_idx + unfilled_length]


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
