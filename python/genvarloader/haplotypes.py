from typing import Dict, Iterable, List, Optional, Union

import numba as nb
import numpy as np
from einops import rearrange
from numpy.typing import ArrayLike, NDArray
from typing_extensions import assert_never

from .fasta import Fasta
from .types import Reader
from .utils import get_rel_starts
from .variants import DenseGenotypes, Variants


class Haplotypes:
    """Construct haplotypes from a reference and tracks of variants.

    Parameters
    ----------
    variants : Variants
        Variants to use for constructing haplotypes.
    reference : Optional[Fasta], optional
        Reference genome, by default None.
    tracks : Optional[Union[Reader, Iterable[Reader]]], optional
        Tracks of variants, by default None.
    jitter_long : bool, optional
        Whether to jitter long haplotypes, by default True.
    seed : Optional[int], optional
        Seed for deterministic shifting of haplotypes longer than the query, by default None.

    Examples
    --------
    Construct haplotypes from a reference and a track of variants.

    >>> import genvarloader as gvl
    >>> variants = gvl.Variants.from_vcf("variants.vcf")
    >>> reference = gvl.Fasta("seq", "reference.fa")
    >>> tracks = gvl.BigWigs("depth", {"sample1": "sample1.bw", "sample2": "sample2.bw"})
    >>> haplotypes = gvl.Haplotypes(variants, reference, tracks)
    >>> contig = "chr1"
    >>> starts = [0, 100, 200]
    >>> ends = [100, 200, 300]
    >>> data = haplotypes.read(contig, starts, ends)
    >>> data
    {
        'seq': <NDArray (sample, ploid, length)>,
        'depth': <NDArray (sample, ploid, length)>
    }
    """

    def __init__(
        self,
        variants: Variants,
        reference: Optional[Fasta] = None,
        tracks: Optional[Union[Reader, Iterable[Reader]]] = None,
        jitter_long: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.variants = variants
        self.reference = reference
        self.jitter_long = jitter_long
        self.rng = np.random.default_rng(seed)

        if tracks is None:
            tracks = []
        elif not isinstance(tracks, Iterable):
            tracks = [tracks]
        self.tracks = tracks

        self.readers: List[Reader] = []

        if reference is not None:
            self.readers.append(reference)
            if reference.pad is None:
                raise ValueError("Reference must have a pad character.")
            self.pad = np.uint8(ord(reference.pad))

        if tracks is not None:
            self.readers.extend(tracks)

        self.chunked = any(r.chunked for r in self.readers)

        if len(self.readers) == 0:
            raise ValueError(
                "Must provide at least one reader, whether a reference or a track."
            )

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: Optional[ArrayLike] = None,
        ploid: Optional[ArrayLike] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, NDArray]:
        """Read data corresponding to a genomic range, sample, and ploid.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based exclusive.
        **kwargs
            May include...
            sample : Iterable[str]
                Specify samples.
            ploid : Iterable[int]
                Specify ploid numbers.
            seed : int
                For deterministic shifting of haplotypes longer than the query.

        Returns
        -------
        Dict[str, NDArray]
            Variant sequences and re-aligned tracks, each with dimensions: (sample, ploid, length)
            Keys correspond to the names of the readers.
        """
        samples = sample
        if samples is None:
            n_samples = self.variants.n_samples
        else:
            samples = np.atleast_1d(np.asarray(samples, dtype=np.str_))
            n_samples = len(samples)

        if ploid is None:
            ploid = self.variants.ploidy
        else:
            ploid = np.atleast_1d(np.asarray(ploid, dtype=np.intp))
            ploid = len(ploid)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        starts = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends = np.atleast_1d(np.asarray(ends, dtype=np.int64))

        variants, max_ends = self.variants.read_for_haplotype_construction(
            contig, starts, ends, samples, np.arange(ploid, dtype=np.intp)
        )

        lengths = ends - starts
        total_length = lengths.sum()
        rel_starts = get_rel_starts(starts, ends)

        reader_lengths = max_ends - starts
        reader_rel_starts = get_rel_starts(
            starts,
            max_ends,  # pyright: ignore[reportArgumentType]
        )

        if variants is None:
            out = {}
            for r in self.readers:
                data = r.read(
                    contig,
                    starts,
                    max_ends,
                    sample=samples,
                    ploid=np.arange(ploid, dtype=np.intp),
                    **kwargs,
                )
                if isinstance(r, Fasta):
                    # (l) -> (s p l)
                    data = np.broadcast_to(data, (n_samples, ploid, len(data)))
                else:
                    # (... s? p? l) -> (... s p l)
                    broadcast_track_to_haps(r.sizes, data, n_samples, ploid)
                out[r.name] = data
        else:
            out: Dict[str, NDArray] = {}
            if self.jitter_long:
                # (s p r)
                shifts = self.sample_shifts(variants)
            else:
                # (s p r)
                shifts = np.zeros((n_samples, ploid, len(starts)), dtype=np.int32)

            if self.reference is not None:
                # ref shape: (length,)
                ref: NDArray[np.bytes_] = self.reference.read(contig, starts, max_ends)
                out[self.reference.name] = ref_to_haplotypes(
                    reference=ref,
                    variants=variants,
                    starts=starts,
                    ref_lengths=reader_lengths,
                    ref_rel_starts=reader_rel_starts,
                    n_samples=n_samples,
                    ploid=ploid,
                    total_length=total_length,
                    lengths=lengths,
                    rel_starts=rel_starts,
                    shifts=shifts,
                    pad_char=self.pad,
                    out=None,
                )

            if self.tracks is not None:
                for reader in self.tracks:
                    # track shape: (..., length) and in ... is 'sample' and maybe 'ploid'
                    track = reader.read(
                        contig,
                        starts,
                        max_ends,
                        sample=samples,
                        ploid=np.arange(ploid, dtype=np.intp),
                        **kwargs,
                    )
                    track = broadcast_track_to_haps(
                        reader.sizes, track, n_samples, ploid
                    )
                    track = realign(
                        track=track,
                        variants=variants,
                        starts=starts,
                        track_lengths=reader_lengths,
                        track_rel_starts=reader_rel_starts,
                        total_length=total_length,
                        lengths=lengths,
                        rel_starts=rel_starts,
                        shifts=shifts,
                        out=None,
                    )
                    if "sample" in reader.sizes:
                        sample_dim_idx = list(reader.sizes).index("sample")
                        track = track.swapaxes(-3, sample_dim_idx)
                    if "ploid" in reader.sizes:
                        ploid_dim_idx = list(reader.sizes).index("ploid")
                        track = track.swapaxes(-2, ploid_dim_idx)
                    out[reader.name] = track

        return out

    def sample_shifts(self, variants: DenseGenotypes) -> NDArray[np.int32]:
        genotypes = variants.genotypes
        size_diffs = variants.size_diffs
        offsets = variants.offsets
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


def ref_to_haplotypes(
    reference: NDArray[np.bytes_],
    variants: Optional[DenseGenotypes],
    starts: NDArray[np.int64],
    ref_lengths: NDArray[np.int64],
    ref_rel_starts: NDArray[np.int64],
    n_samples: int,
    ploid: int,
    total_length: int,
    lengths: NDArray[np.int64],
    rel_starts: NDArray[np.int64],
    shifts: NDArray[np.int32],
    pad_char: np.uint8,
    out: Optional[NDArray[np.bytes_]] = None,
) -> NDArray:
    if out is None:
        # alloc then fill is faster than np.tile ¯\_(ツ)_/¯
        seqs = np.empty((n_samples, ploid, total_length), dtype=reference.dtype)
    else:
        seqs = out

    if variants is None:
        seqs[:] = reference
    elif isinstance(variants, DenseGenotypes):
        construct_haplotypes(
            seqs.view(np.uint8),
            reference.view(np.uint8),
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
            pad_char,
        )
    else:
        assert_never(variants)

    return seqs


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
        # (s p r) -> (s p)
        _shifts = shifts[..., region]
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
                shift = _shifts[sample, hap]
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
                            # consume ref up to the end of the variant
                            ref_idx = v_rel_pos + 1
                            # add the length of skipped ref and size of the variant to the shift
                            shifted += ref_shift_dist + v_len
                            # skip the variant
                            continue
                        # enough distance between ref_idx and variant to finish shift
                        elif shifted + ref_shift_dist >= shift:
                            ref_idx += shift - shifted
                            shifted = shift
                            # can still use the variant and whatever ref is left between
                            # ref_idx and the variant
                        # ref + (some of) variant is enough to finish shift
                        else:
                            # consume ref up to beginning of variant
                            # ref_idx will be moved to end of variant after using the variant
                            ref_idx = v_rel_pos
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
                            # done shifting
                            shifted = shift

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


def broadcast_track_to_haps(
    dim_sizes: Dict[str, int], track: NDArray, n_samples: int, ploid: int
):
    # tracks may have sample and ploid dimensions and others as well
    # if they do not have sample or ploid dimensions, then
    # they must be broadcasted
    # (... l) -> (... s p l)
    in_dims = " ".join(dim_sizes.keys())
    sample_entry = "sample" if "sample" in dim_sizes else "1"
    ploid_entry = "ploid" if "ploid" in dim_sizes else "1"
    out_dims = " ".join(
        [d for d in dim_sizes.keys() if d not in ("sample", "ploid")]
        + [sample_entry, ploid_entry, "length"]
    )
    track = rearrange(track, f"{in_dims} length -> {out_dims}")
    track = np.broadcast_to(
        track, (*track.shape[:-3], n_samples, ploid, track.shape[-1])
    )
    return track


def realign(
    track: NDArray,  # (... s p l)
    variants: DenseGenotypes,
    starts: NDArray[np.int64],
    track_lengths: NDArray[np.int64],
    track_rel_starts: NDArray[np.int64],
    total_length: int,
    lengths: NDArray[np.int64],
    rel_starts: NDArray[np.int64],
    shifts: NDArray[np.int32],
    out: Optional[NDArray] = None,
) -> NDArray:  # (... s p l)
    if out is None:
        out = np.empty((*track.shape[:-1], total_length), dtype=track.dtype)

    if variants is None:
        out[:] = track
    elif isinstance(variants, DenseGenotypes):
        realign_track_to_haplotype(
            out,
            track,
            shifts,
            variants.positions,
            variants.size_diffs,
            variants.genotypes,
            variants.offsets,
            starts,
            rel_starts,
            lengths,
            track_rel_starts,
            track_lengths,
        )
    else:
        assert_never(variants)

    return out


@nb.njit(nogil=True, cache=True, parallel=True)
def realign_track_to_haplotype(
    out: NDArray,  # (..., s p o_len)
    track: NDArray,  # (..., s p t_len) t_len >= o_len
    shifts: NDArray[np.int32],  # (s p r)
    positions: NDArray[np.int32],  # (v)
    sizes: NDArray[np.int32],  # (v)
    genotypes: NDArray[np.int8],  # (s p v)
    region_offsets: NDArray[np.uint32],  # (r + 1)
    starts: NDArray[np.int64],  # (r)
    rel_starts: NDArray[np.int64],  # (r)
    lengths: NDArray[np.int64],  # (r)
    track_rel_starts: NDArray[np.int64],  # (r)
    track_lengths: NDArray[np.int64],  # (r)
):
    n_samples, ploidy, length = out.shape[-3:]
    n_variants = len(positions)

    for region in nb.prange(len(region_offsets) - 1):
        r_s = region_offsets[region]
        r_e = region_offsets[region + 1]
        n_variants = r_e - r_s
        # prepend variables by _ to indicate they are relative to the region
        _length = lengths[region]
        _out = out[..., rel_starts[region] : rel_starts[region] + _length]
        _track = track[
            track_rel_starts[region] : track_rel_starts[region] + track_lengths[region]
        ]
        if n_variants == 0:
            _out[...] = _track[:]
            continue
        # (s p r) -> (s p)
        _shifts = shifts[..., region]
        _positions = positions[r_s:r_e] - starts[region]
        _sizes = sizes[r_s:r_e]
        _genos = genotypes[..., r_s:r_e]

        for sample in nb.prange(n_samples):
            for hap in nb.prange(ploidy):
                _subtrack = _track[..., sample, hap, :]
                # where to get next reference subsequence
                track_idx = 0
                # where to put next subsequence
                out_idx = 0
                # total amount to shift by
                shift = _shifts[sample, hap]
                # how much we've shifted
                shifted = 0

                # first variant is a DEL spanning start
                v_rel_pos = _positions[0]
                v_diff = _sizes[0]
                if v_rel_pos < 0 and _genos[sample, hap, 0] == 1:
                    # diff of v(-1) has been normalized to consider where ref is
                    # otherwise, ref_idx = v_rel_pos - v_diff + 1
                    # e.g. a -10 diff became -3 if v_rel_pos = -7
                    track_idx = v_rel_pos - v_diff + 1
                    # increment the variant index
                    start_idx = 1
                else:
                    start_idx = 0

                for variant in range(start_idx, n_variants):
                    # UNKNOWN or REF
                    if _genos[sample, hap, variant] != 1:
                        continue

                    # position of variant relative to ref from fetch(contig, start, q_end)
                    # i.e. put it into same coordinate system as ref_idx
                    v_rel_pos = _positions[variant]

                    # overlapping variants
                    # v_rel_pos < ref_idx only if we see an ALT at a given position a second
                    # time or more. We'll do what bcftools consensus does and only use the
                    # first ALT variant we find.
                    if v_rel_pos < track_idx:
                        continue

                    v_diff = _sizes[variant]
                    v_len = max(0, v_diff + 1)
                    value = _subtrack[..., v_rel_pos]

                    # allele = alt_alleles[alt_offsets[variant] : alt_offsets[variant + 1]]

                    # handle shift
                    if shifted < shift:
                        ref_shift_dist = v_rel_pos - track_idx
                        # not enough distance to finish the shift even with the variant
                        if shifted + ref_shift_dist + v_len < shift:
                            track_idx = v_rel_pos + 1
                            shifted += ref_shift_dist + v_len
                            continue
                        # enough distance between ref_idx and variant to finish shift
                        elif shifted + ref_shift_dist >= shift:
                            track_idx += shift - shifted
                            shifted = shift
                            # can still use the variant and whatever ref is left between
                            # ref_idx and the variant
                        # ref + (some of) variant is enough to finish shift
                        else:
                            # adjust ref_idx so that no reference is written
                            track_idx = v_rel_pos
                            # how much left to shift - amount of ref we can use
                            allele_start_idx = shift - shifted - ref_shift_dist
                            if allele_start_idx == v_len:
                                continue
                            v_len -= allele_start_idx
                            # done shifting
                            shifted = shift

                    # add reference sequence
                    track_len = v_rel_pos - track_idx
                    if out_idx + track_len >= length:
                        # ref will get written by final clause
                        break
                    _out[..., sample, hap, out_idx : out_idx + track_len] = _subtrack[
                        ..., track_idx : track_idx + track_len
                    ]
                    out_idx += track_len

                    # insertions + substitions
                    writable_length = min(v_len, length - out_idx)
                    _out[..., sample, hap, out_idx : out_idx + writable_length] = value
                    out_idx += writable_length
                    # +1 because ALT alleles always replace 1 nt of reference for a
                    # normalized VCF
                    track_idx = v_rel_pos + 1

                    # deletions, move ref to end of deletion
                    if v_diff < 0:
                        track_idx -= v_diff

                    if out_idx >= _length:
                        break

                # fill rest with reference sequence
                unfilled_length = _length - out_idx
                if unfilled_length > 0:
                    _out[..., sample, hap, out_idx:] = _subtrack[
                        ..., track_idx : track_idx + unfilled_length
                    ]
