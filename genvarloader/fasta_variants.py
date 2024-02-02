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
    dtype = np.dtype("S1")

    def __init__(
        self,
        name: str,
        reference: Fasta,
        variants: Variants,
        jitter_long: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize a `FastaVariants` reader.

        Parameters
        ----------
        name : str
            Name of the `FastaVariants` reader.
        reference : Fasta
            `Fasta` reader for reading the reference genome.
        variants : Variants
            `Variants` reader for reading variants.
        jitter_long : bool
            Whether to jitter haplotypes that are longer than query regions. If
            False, then haplotypes longer than query regions will always be
            right truncated.
        seed : int, optional
            Seed for jittering.
        """
        self.reference = reference
        self.variants = variants
        self.chunked = self.reference.chunked or self.variants.chunked
        self.jitter_long = jitter_long
        self.contig_starts_with_chr = None

        if self.reference.pad is None:
            raise ValueError(
                "Reference sequence must have a pad character for FastaVariants"
            )

        self.name = name
        self.coords = {
            "sample": np.asarray(self.variants.samples),
            "ploid": np.arange(self.variants.ploidy, dtype=np.uint32),
        }
        self.sizes = {k: len(v) for k, v in self.coords.items()}

        self.virtual_data = xr.DataArray(
            da.empty(  # pyright: ignore[reportPrivateImportUsage]
                (self.variants.n_samples, self.variants.ploidy), dtype="S1"
            ),
            name=name,
            coords=self.coords,
        )

        if (
            self.reference.contig_starts_with_chr
            != self.variants.contig_starts_with_chr
        ):
            logger.warning(
                dedent(
                    f"""
                Reference sequence and variant files have different contig naming
                conventions. Contig names in queries will be normalized so that they
                will still run, but this may indicate that the variants were not aligned
                to the reference being used to construct haplotypes. The reference
                file's contigs{"" if self.reference.contig_starts_with_chr else " don't"}
                start with "chr" whereas the variant file's are the opposite.
                """
                )
                .replace("\n", " ")
                .strip()
            )

        self.rng = np.random.default_rng(seed)
        self.rev_strand_fn = self.reference.rev_strand_fn

    def read(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        out: Optional[NDArray[np.bytes_]] = None,
        **kwargs,
    ) -> NDArray[np.bytes_]:
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
            May optionally include...
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

        starts, ends = np.asarray(starts, dtype=np.int64), np.asarray(
            ends, dtype=np.int64
        )

        variants, max_ends = self.variants.read_for_haplotype_construction(
            contig, starts, ends, **kwargs
        )

        lengths = ends - starts
        total_length = lengths.sum()
        rel_starts = get_rel_starts(starts, ends)

        ref = self.reference.read(contig, starts, max_ends)
        ref_lengths = max_ends - starts
        ref_rel_starts = get_rel_starts(starts, max_ends)

        if out is None:
            # alloc then fill is faster than np.tile ¯\_(ツ)_/¯
            seqs = np.empty((n_samples, ploid, total_length), dtype=ref.dtype)
        else:
            seqs = out

        if variants is None:
            seqs[...] = ref
            return seqs

        if isinstance(variants, DenseGenotypes):
            if self.jitter_long:
                shifts = self.sample_shifts(
                    variants.genotypes, variants.size_diffs, variants.offsets
                )
            else:
                shifts = np.zeros((n_samples, ploid, len(starts)), dtype=np.int32)

            construct_haplotypes(
                seqs.view(np.uint8),
                ref.view(np.uint8),
                shifts,
                variants.positions,
                variants.size_diffs,
                variants.genotypes,
                variants.alt.offsets,
                variants.alt.alleles.view(np.uint8),
                variants.offsets,
                starts,
                rel_starts,
                lengths,
                ref_rel_starts,
                ref_lengths,
                np.uint8(
                    # pad existing is checked on init
                    ord(self.reference.pad)  # type: ignore[arg-type]
                ),
            )
        elif isinstance(variants, SparseAlleles):
            raise NotImplementedError
        else:
            assert_never(variants)

        return seqs

    def sample_shifts(
        self,
        genotypes: NDArray[np.int8],  # (s p v)
        size_diffs: NDArray[np.int32],  # (v)
        offsets: NDArray[np.uint32],  # (r+1)
    ):
        # (s p v)
        diffs = np.where(genotypes == 1, size_diffs, 0)
        # (s p r+1) -> (s p r)
        if offsets[-1] == diffs.shape[-1]:
            total_diffs = np.add.reduceat(diffs, offsets[:-1], axis=-1).clip(0)
        else:
            total_diffs = np.add.reduceat(diffs, offsets, axis=-1)[..., :-1].clip(0)
        no_vars = offsets[1:] == offsets[:-1]
        total_diffs[..., no_vars] = 0
        shifts = self.rng.integers(0, total_diffs + 1, dtype=np.int32)
        return shifts


@nb.njit(nogil=True, cache=True, parallel=True)
def construct_haplotypes(
    out: NDArray[np.uint8],  # (s p o_len)
    ref: NDArray[np.uint8],  # (r_len), r_len >= o_len
    shifts: NDArray[np.int32],  # (s p r)
    positions: NDArray[np.int32],  # (v)
    sizes: NDArray[np.int32],  # (v)
    genotypes: NDArray[np.int8],  # (s p v)
    alt_offsets: NDArray[np.uint32],  # (v + 1)
    alt_alleles: NDArray[np.uint8],  # (v)
    region_offsets: NDArray[np.uint32],  # (r + 1)
    starts: NDArray[np.int64],  # (r)
    rel_starts: NDArray[np.int64],  # (r)
    lengths: NDArray[np.int64],  # (r)
    ref_rel_starts: NDArray[np.int64],  # (r)
    ref_lengths: NDArray[np.int64],  # (r)
    pad_char: np.uint8,
):
    n_samples = out.shape[0]
    ploidy = out.shape[1]

    for region in nb.prange(len(region_offsets) - 1):
        r_s = region_offsets[region]
        r_e = region_offsets[region + 1]
        n_variants = r_e - r_s
        # prepend variables by _ to indicate they are relative to the region
        _length = lengths[region]
        _out = out[..., rel_starts[region] : rel_starts[region] + _length]
        _ref = ref[
            ref_rel_starts[region] : ref_rel_starts[region] + ref_lengths[region]
        ]
        if n_variants == 0:
            _out[...] = _ref[:]
            continue
        _shifts = shifts[..., r_s:r_e]
        _positions = positions[r_s:r_e] - starts[region]
        _sizes = sizes[r_s:r_e]
        _genos = genotypes[..., r_s:r_e]
        _alt_offsets = alt_offsets[r_s : r_e + 1]
        _alt_alleles = alt_alleles[_alt_offsets[0] : _alt_offsets[-1]]
        for sample in nb.prange(n_samples):
            for hap in nb.prange(ploidy):
                # where to get next reference subsequence
                ref_idx = 0
                # where to put next subsequence
                out_idx = 0
                # total amount to shift by
                shift = _shifts[sample, hap, 0]
                # how much we've shifted
                shifted = 0

                # first variant is a DEL spanning start
                v_rel_pos = _positions[0]
                v_diff = _sizes[0]
                if v_rel_pos < 0 and _genos[sample, hap, 0] == 1:
                    # diff of v(-1) has been normalized to consider where ref is
                    # otherwise, ref_idx = v_rel_pos - v_diff + 1
                    # e.g. a -10 diff became -3 if v_rel_pos = -7
                    ref_idx = v_rel_pos - v_diff + 1
                    # increment the variant index
                    start_idx = 0 + 1
                else:
                    start_idx = 0

                for variant in range(start_idx, n_variants):
                    # UNKNOWN -9 or REF 0
                    if _genos[sample, hap, variant] != 1:
                        continue

                    # position of variant relative to ref from fetch(contig, start, q_end)
                    # i.e. put it into same coordinate system as ref_idx
                    v_rel_pos = _positions[variant]

                    # overlapping variants
                    # v_rel_pos < ref_idx only if we see an ALT at a given position a second
                    # time or more. We'll do what bcftools consensus does and only use the
                    # first ALT variant we find.
                    if v_rel_pos < ref_idx:
                        continue

                    v_diff = _sizes[variant]
                    allele = _alt_alleles[
                        _alt_offsets[variant] : _alt_offsets[variant + 1]
                    ]
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
                            #! without if statement, parallel=True can cause a SystemError!
                            # * parallel jit cannot handle changes in array dimension.
                            # * without this, allele can change from a 1D array to a 0D
                            # * array.
                            if allele_start_idx == v_len:
                                continue
                            allele = allele[allele_start_idx:]
                            v_len = len(allele)

                    # add reference sequence
                    ref_len = v_rel_pos - ref_idx
                    if out_idx + ref_len >= _length:
                        # ref will get written by final clause
                        break
                    _out[sample, hap, out_idx : out_idx + ref_len] = _ref[
                        ref_idx : ref_idx + ref_len
                    ]
                    out_idx += ref_len

                    # insertions + substitions
                    writable_length = min(v_len, _length - out_idx)
                    _out[sample, hap, out_idx : out_idx + writable_length] = allele[
                        :writable_length
                    ]
                    out_idx += writable_length
                    # +1 because ALT alleles always replace 1 nt of reference for a
                    # normalized VCF
                    ref_idx = v_rel_pos + 1

                    # deletions, move ref to end of deletion
                    if v_diff < 0:
                        ref_idx -= v_diff

                    if out_idx >= _length:
                        break

                # fill rest with reference sequence and pad with Ns
                unfilled_length = _length - out_idx
                if unfilled_length > 0:
                    writable_ref = min(unfilled_length, len(_ref) - ref_idx)
                    out_end_idx = out_idx + writable_ref
                    ref_end_idx = ref_idx + writable_ref
                    _out[sample, hap, out_idx:out_end_idx] = _ref[ref_idx:ref_end_idx]

                    if out_end_idx < _length:
                        _out[sample, hap, out_end_idx:] = pad_char
