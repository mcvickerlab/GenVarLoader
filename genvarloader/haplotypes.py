from typing import Iterable, List, Optional, Sequence, TypeVar, Union

import numba as nb
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import assert_never

from .types import DenseGenotypes, Dict, Reader, Variants
from .util import get_rel_starts


class Haplotypes:
    def __init__(
        self,
        variants: Variants,
        reference: Optional[Reader] = None,
        tracks: Optional[Union[Reader, Iterable[Reader]]] = None,
        jitter_long: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.variants = variants
        self.reference = reference
        self.jitter_long = jitter_long
        self.seed = seed

        if tracks is not None and not isinstance(tracks, Iterable):
            tracks = [tracks]
        self.tracks = tracks

        self.readers: List[Reader] = []

        if reference is not None:
            self.readers.append(reference)

        if tracks is not None:
            self.readers.extend(tracks)

        if len(self.readers) == 0:
            raise ValueError(
                "Must provide at least one reader, whether a reference or a track."
            )

    def read(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        out: Optional[Dict[str, Optional[NDArray]]] = None,
        **kwargs,
    ) -> Dict[str, xr.DataArray]:
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
            May include...
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
        if out is None:
            out = {reader.name: None for reader in self.readers}

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

        reader_lengths = max_ends - starts
        reader_rel_starts = get_rel_starts(starts, max_ends)

        diffs: List[Optional[NDArray[np.int32]]] = []
        for variant in variants:
            if variant is not None:
                diffs.append(np.where(variant.genotypes == 1, variant.size_diffs, 0))
            else:
                diffs.append(None)

        if self.jitter_long:
            shifts = [
                None if diff is None else self.sample_shifts(diff) for diff in diffs
            ]
        else:
            shifts = [None] * len(variants)

        _out: Dict[str, xr.DataArray] = {}

        if self.reference is not None:
            # ref shape: (length,)
            ref: NDArray[np.bytes_] = self.reference.read(
                contig, starts, max_ends
            ).to_numpy()
            _out[self.reference.name] = ref_to_haplotypes(
                reference=ref,
                variants=variants,
                starts=starts,
                ref_lengths=reader_lengths,
                ref_rel_starts=reader_rel_starts,
                out=out[self.reference.name],
                n_samples=n_samples,
                ploid=ploid,
                total_length=total_length,
                lengths=lengths,
                rel_starts=rel_starts,
                shifts=shifts,
            )

        if self.tracks is not None:
            for reader in self.tracks:
                # track shape: (..., length) and in ... is 'sample' and maybe 'ploid'
                track = reader.read(contig, starts, max_ends, **kwargs)
                _out[reader.name] = realign(
                    track=track,
                    variants=variants,
                    starts=starts,
                    track_lengths=reader_lengths,
                    track_rel_starts=reader_rel_starts,
                    out=out[reader.name],
                    ploid=ploid,
                    total_length=total_length,
                    lengths=lengths,
                    rel_starts=rel_starts,
                    shifts=shifts,
                )

        return _out

    def sample_shifts(self, diffs: NDArray[np.int32]) -> NDArray[np.int32]:
        total_diffs = diffs.sum(-1, dtype=np.int32).clip(0)
        shifts = self.rng.integers(0, total_diffs + 1, dtype=np.int32)
        return shifts


def ref_to_haplotypes(
    reference: NDArray[np.bytes_],
    variants: List[Optional[DenseGenotypes]],
    starts: NDArray[np.int64],
    ref_lengths: NDArray[np.int64],
    ref_rel_starts: NDArray[np.int64],
    out: Optional[NDArray[np.bytes_]],
    n_samples: int,
    ploid: int,
    total_length: int,
    lengths: NDArray[np.int64],
    rel_starts: NDArray[np.int64],
    shifts: Sequence[Optional[NDArray[np.int32]]],
):
    if out is None:
        # alloc then fill is faster than np.tile ¯\_(ツ)_/¯
        seqs = np.empty((n_samples, ploid, total_length), dtype=reference.dtype)
    else:
        seqs = out

    for (variant, start, length, rel_start, ref_length, ref_rel_start, _shifts,) in zip(
        variants, starts, lengths, rel_starts, ref_lengths, ref_rel_starts, shifts
    ):
        subseq = seqs[..., rel_start : rel_start + length]
        # subref can be longer than subseq
        subref = reference[ref_rel_start : ref_rel_start + ref_length]
        if variant is None:
            subseq[...] = subref[:length]
        elif isinstance(variant, DenseGenotypes):
            if _shifts is None:
                _shifts = np.zeros((n_samples, ploid), dtype=np.int32)
            construct_haplotypes_with_indels(
                subseq.view(np.uint8),
                subref.view(np.uint8),
                _shifts,
                variant.positions - start,
                variant.size_diffs,
                variant.genotypes,
                variant.alt.offsets,
                variant.alt.alleles.view(np.uint8),
            )
        else:
            assert_never(variant)

    return xr.DataArray(seqs, dims=["sample", "ploid", "length"])


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


DTYPE = TypeVar("DTYPE", bound=np.generic)


def realign(
    track: xr.DataArray,
    variants: List[Optional[DenseGenotypes]],
    starts: NDArray[np.int64],
    track_lengths: NDArray[np.int64],
    track_rel_starts: NDArray[np.int64],
    out: Optional[NDArray[DTYPE]],
    ploid: int,
    total_length: int,
    lengths: NDArray[np.int64],
    rel_starts: NDArray[np.int64],
    shifts: Sequence[Optional[NDArray[np.int32]]],
) -> xr.DataArray:
    if "ploid" not in track.sizes:
        final_out_dim_order = track.dims[:-1] + ("ploid", "length")
        track = track.transpose(..., "sample", "length")
        _track = track.values.repeat(ploid, axis=-2)
    else:
        final_out_dim_order = track.dims
        track = track.transpose(..., "sample", "ploid", "length")
        _track = track.values

    if out is None:
        out = np.empty((*_track.shape[:-1], total_length), dtype=track.dtype)

    for (
        variant,
        start,
        length,
        rel_start,
        track_length,
        track_rel_start,
        _shifts,
    ) in zip(
        variants, starts, lengths, rel_starts, track_lengths, track_rel_starts, shifts
    ):
        subout = out[..., rel_start : rel_start + length]
        # subtrack can be longer than subseq
        subtrack = _track[..., track_rel_start : track_rel_start + track_length]
        if variant is None:
            subout[...] = subtrack[..., :length]
        elif isinstance(variant, DenseGenotypes):
            realign_track_to_haplotype(
                subout,
                subtrack,
                _shifts,
                variant.positions - start,
                variant.size_diffs,
                variant.genotypes,
            )

    _out = xr.DataArray(out, dims=track.dims, coords=track.coords, name=track.name)
    return _out.transpose(*final_out_dim_order)


@nb.njit(nogil=True, cache=True, parallel=True)
def realign_track_to_haplotype(
    out: NDArray[DTYPE],
    track: NDArray[DTYPE],
    shifts: NDArray[np.int32],
    rel_positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    genotypes: NDArray[np.int8],
):
    n_samples, ploidy, length = out.shape[-3:]
    n_variants = len(rel_positions)

    for sample in nb.prange(n_samples):
        for hap in nb.prange(ploidy):
            _track = track[..., sample, hap, :]
            # where to get next reference subsequence
            track_idx = 0
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
                track_idx = v_rel_pos - v_diff + 1
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
                if v_rel_pos < track_idx:
                    continue

                v_diff = sizes[variant]
                v_len = max(0, v_diff + 1)
                value = _track[..., v_rel_pos]

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
                        shifted = shift
                        # how much left to shift - amount of ref we can use
                        allele_start_idx = shift - shifted - ref_shift_dist
                        if allele_start_idx == v_len:
                            continue
                        v_len -= allele_start_idx

                # add reference sequence
                track_len = v_rel_pos - track_idx
                if out_idx + track_len >= length:
                    # ref will get written by final clause
                    break
                out[..., sample, hap, out_idx : out_idx + track_len] = _track[
                    ..., track_idx : track_idx + track_len
                ]
                out_idx += track_len

                # insertions + substitions
                writable_length = min(v_len, length - out_idx)
                out[..., sample, hap, out_idx : out_idx + writable_length] = value
                out_idx += writable_length
                # +1 because ALT alleles always replace 1 nt of reference for a
                # normalized VCF
                track_idx = v_rel_pos + 1

                # deletions, move ref to end of deletion
                if v_diff < 0:
                    track_idx -= v_diff

                if out_idx >= length:
                    break

            # fill rest with reference sequence
            unfilled_length = length - out_idx
            if unfilled_length > 0:
                out[..., sample, hap, out_idx:] = _track[
                    ..., track_idx : track_idx + unfilled_length
                ]
