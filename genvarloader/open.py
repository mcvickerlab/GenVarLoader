import json
from enum import Enum, auto
from pathlib import Path
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numba as nb
import numpy as np
import polars as pl
import seqpro as sp
from attrs import define
from loguru import logger
from numpy.typing import NDArray

from .fasta import Fasta
from .util import normalize_contig_name
from .variants import VLenAlleles

Idx = Union[int, np.integer, Sequence[int], NDArray[np.integer], slice]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


class GVLDataset:
    class State(Enum):
        HAPS_ITVS = auto()
        REF_ITVS = auto()
        HAPS = auto()
        ITVS = auto()

    def __init__(
        self,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        transform: Optional[Callable] = None,
        ignore_haplotypes: bool = False,
        ignore_tracks: bool = False,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist.")
        self.samples: NDArray[np.str_] = np.load(self.path / "samples.npy")
        self.regions: NDArray[np.int32] = np.load(self.path / "regions.npy")
        self.n_samples = len(self.samples)
        self.n_regions = len(self.regions)
        self.transform = transform

        with open(self.path / "metadata.json") as f:
            metadata = json.load(f)

        self.region_length = int(metadata["region_length"])
        self.ploidy = int(metadata["ploidy"])
        self.n_variants = int(metadata.get("n_variants", 0))
        self.n_intervals = int(metadata.get("n_intervals", 0))
        self.max_jitter = int(metadata.get("max_jitter", 0))
        self.rng = np.random.default_rng(seed)

        logger.info(f"\n{repr(self)}")

        self.has_ref = False
        self.has_genos = False
        self.has_itvs = False

        if reference is None and self.n_variants > 0:
            raise ValueError(
                "Genotypes found but no reference genome provided. This is required to reconstruct haplotypes."
            )
        elif reference is not None:
            logger.info("Loading reference genome into memory.")
            self.init_reference(reference, metadata)
            self.has_ref = True

        if self.n_variants > 0:
            if ignore_haplotypes:
                logger.info("Ignoring existing haplotypes.")
            else:
                # initialize variant records
                variants = pl.read_ipc(self.path / "variants.arrow")
                self.variant_positions = variants["POS"].to_numpy()
                self.variant_sizes = variants["ILEN"].to_numpy()
                self.alts = VLenAlleles.from_polars(variants["ALT"])
                self.has_genos = True

        if self.n_intervals > 0:
            if ignore_tracks:
                logger.info("Ignoring existing tracks.")
            else:
                self.has_itvs = True

        self.genotypes = None
        self.intervals = None

        if self.has_ref and self.has_genos and self.has_itvs:
            self.state = self.State.HAPS_ITVS
        elif self.has_ref and self.has_genos:
            self.state = self.State.HAPS
        elif self.has_ref and self.has_itvs:
            self.state = self.State.REF_ITVS
        elif self.has_itvs:
            self.state = self.State.ITVS
        else:
            raise ValueError(
                "No genotypes or intervals found. At least one of these must be present."
            )

    def init_reference(self, reference: Union[str, Path], metadata: Dict[str, Any]):
        fasta = Fasta("ref", reference, "N")
        fasta.sequences = fasta._get_contigs(metadata["contigs"])
        if TYPE_CHECKING:
            assert fasta.sequences is not None
            assert fasta.pad is not None
        refs: List[NDArray[np.bytes_]] = []
        next_offset = 0
        ref_offsets: Dict[str, int] = {}
        for contig in metadata["contigs"]:
            arr = fasta.sequences[contig]
            refs.append(arr)
            ref_offsets[contig] = next_offset
            next_offset += len(arr)
        self.reference = np.concatenate(refs).view(np.uint8)
        self.pad_char = ord(fasta.pad)
        self.contigs = cast(
            List[str],
            [normalize_contig_name(c, fasta.contigs) for c in metadata["contigs"]],
        )  # type: ignore
        if any(c is None for c in self.contigs):
            raise ValueError("Contig names in metadata do not match reference.")
        self.ref_offsets = np.empty(len(self.contigs) + 1, np.uint64)
        self.ref_offsets[:-1] = np.array(
            [ref_offsets[c] for c in self.contigs], dtype=np.uint64
        )
        self.ref_offsets[-1] = len(self.reference)

    @property
    def shape(self):
        """Return the shape of the dataset. (n_samples, n_regions)"""
        return self.n_samples, self.n_regions

    def __len__(self) -> int:
        return self.n_samples * self.n_regions

    def __repr__(self) -> str:
        return dedent(
            f"""
            GVL store {self.path.name}
            Original region length: {self.region_length - 2*self.max_jitter:,}
            Max jitter: {self.max_jitter:,}
            # of regions: {self.n_regions:,}
            # of samples: {self.n_samples:,}
            Has genotypes: {self.n_variants > 0}
            Has intervals: {self.n_intervals > 0}\
            """
        ).strip()

    def __getitem__(self, idx: Idx) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Get a batch of haplotypes and tracks or intervals and tracks.

        Parameters
        ----------
        s_r_idx : Tuple[Idx, Idx]
            Tuple of sample and region indices, in that order.
        """
        squeeze = False
        if isinstance(idx, (int, np.integer)):
            _idx = [idx]
            squeeze = True
        elif isinstance(idx, slice):
            _idx = np.arange(self.n_samples * self.n_regions, dtype=np.uintp)[idx]
        else:
            _idx = idx
        _idx = cast(ListIdx, _idx)

        if self.genotypes is None and self.has_genos:
            self.init_genotypes()
        if self.intervals is None and self.has_itvs:
            self.init_intervals()

        s_idx, r_idx = np.unravel_index(_idx, self.shape)
        regions = self.regions[r_idx]

        if self.state is self.State.HAPS_ITVS:
            if TYPE_CHECKING:
                assert self.genotypes is not None
                assert self.intervals is not None

            haps, shifts = self.get_haplotypes_and_shifts(
                s_idx, r_idx, self.genotypes, regions
            )
            tracks = self.get_hap_tracks(_idx, self.intervals, regions, shifts)

            if self.max_jitter > 0:
                seed = self.rng.integers(np.iinfo(np.intp).max, dtype=np.intp)
                haps, tracks = sp.jitter(
                    haps,
                    tracks,
                    max_jitter=self.max_jitter,
                    length_axis=-1,
                    jitter_axes=(0, 1),
                    seed=seed,
                )

            if squeeze:
                haps, tracks = haps.squeeze(0), tracks.squeeze(0)

            if self.transform is not None:
                haps, tracks = self.transform(haps, tracks)

            return haps, tracks
        elif self.state is self.State.HAPS:
            if TYPE_CHECKING:
                assert self.genotypes is not None

            # (n_regions, region_length)
            haps, shifts = self.get_haplotypes_and_shifts(
                s_idx, r_idx, self.genotypes, regions
            )
            if self.max_jitter > 0:
                seed = self.rng.integers(np.iinfo(np.intp).max, dtype=np.intp)
                haps = sp.jitter(
                    haps,
                    max_jitter=self.max_jitter,
                    length_axis=-1,
                    jitter_axes=(0, 1),
                    seed=seed,
                )[0]

            if squeeze:
                haps = haps.squeeze(0)

            if self.transform is not None:
                haps = self.transform(haps)

            return haps
        elif self.state is self.State.REF_ITVS:
            if TYPE_CHECKING:
                assert self.intervals is not None

            ref = get_reference(
                regions,
                self.reference,
                self.ref_offsets,
                self.region_length,
                self.pad_char,
            ).view("S1")

            tracks = self.get_tracks(_idx, self.intervals, regions)
            if self.max_jitter > 0:
                seed = self.rng.integers(np.iinfo(np.intp).max, dtype=np.intp)
                ref, tracks = sp.jitter(
                    ref,
                    tracks,
                    max_jitter=self.max_jitter,
                    length_axis=-1,
                    jitter_axes=0,
                    seed=seed,
                )

            if squeeze:
                ref, tracks = ref.squeeze(0), tracks.squeeze(0)

            if self.transform is not None:
                ref, tracks = self.transform(ref, tracks)

            return ref, tracks
        else:
            if TYPE_CHECKING:
                assert self.intervals is not None

            tracks = self.get_tracks(_idx, self.intervals, regions)
            if self.max_jitter > 0:
                seed = self.rng.integers(np.iinfo(np.intp).max, dtype=np.intp)
                tracks = sp.jitter(
                    tracks,
                    max_jitter=self.max_jitter,
                    length_axis=-1,
                    jitter_axes=0,
                    seed=seed,
                )[0]

            if squeeze:
                tracks = tracks.squeeze(0)

            if self.transform is not None:
                tracks = self.transform(tracks)

            return tracks

    def init_genotypes(self):
        self.genotypes = Genotypes(
            np.memmap(
                self.path / "genotypes" / "genotypes.npy",
                shape=(self.n_variants * self.n_samples, self.ploidy),
                dtype=np.int8,
                mode="r",
            ),
            np.memmap(
                self.path / "genotypes" / "first_variant_idxs.npy",
                dtype=np.uint32,
                mode="r",
            ),
            np.memmap(
                self.path / "genotypes" / "offsets.npy", dtype=np.uint32, mode="r"
            ),
            self.n_samples,
        )

    def init_intervals(self):
        self.intervals = Intervals(
            np.memmap(
                self.path / "intervals" / "intervals.npy",
                shape=(self.n_intervals, 2),
                dtype=np.uint32,
                mode="r",
            ),
            np.memmap(
                self.path / "intervals" / "values.npy",
                dtype=np.float32,
                mode="r",
            ),
            np.memmap(
                self.path / "intervals" / "offsets.npy",
                dtype=np.uint32,
                mode="r",
            ),
        )

    def get_haplotypes_and_shifts(
        self,
        s_idx: ListIdx,
        r_idx: ListIdx,
        genos: "Genotypes",
        regions: NDArray[np.int32],
    ):
        genos = genos[s_idx, r_idx]

        diffs = get_diffs(
            genos.first_v_idxs, genos.offsets, genos.genos, self.variant_sizes
        )
        shifts = self.rng.integers(0, diffs + 1, dtype=np.uint32)
        haps = np.empty((len(regions), self.ploidy, self.region_length), np.uint8)
        reconstruct_haplotypes(
            haps,
            regions,
            shifts,
            genos.first_v_idxs,
            genos.offsets,
            genos.genos,
            self.variant_positions,
            self.variant_sizes,
            self.alts.alleles.view(np.uint8),
            self.alts.offsets,
            self.reference,
            self.ref_offsets,
            self.pad_char,
        )
        return haps.view("S1"), shifts

    def get_tracks(
        self, ds_idx: ListIdx, intervals: "Intervals", regions: NDArray[np.int32]
    ):
        intervals = intervals[ds_idx]
        values = intervals_to_values(
            regions,
            intervals.intervals,
            intervals.values,
            intervals.offsets,
            self.region_length,
        )
        return values

    def get_hap_tracks(
        self,
        ds_idx: ListIdx,
        intervals: "Intervals",
        regions: NDArray[np.int32],
        shifts: NDArray[np.uint32],
    ):
        intervals = intervals[ds_idx]
        values = intervals_to_hap_values(
            regions,
            shifts,
            intervals.intervals,
            intervals.values,
            intervals.offsets,
            self.region_length,
        )
        return values

    def get_bed(self):
        bed = regions_to_bed(self.regions, self.contigs)
        bed = bed.with_columns(chromEnd=pl.col("chromStart") + self.region_length)
        return bed


@nb.njit(parallel=True, nogil=True, cache=True)
def get_reference(
    regions: NDArray[np.int32],
    reference: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    region_length: int,
    pad_char: int,
) -> NDArray[np.uint8]:
    out = np.empty((len(regions), region_length), np.uint8)
    for region in nb.prange(len(regions)):
        q = regions[region]
        c_idx = q[0]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        start = q[1]
        end = q[2]
        out[region] = padded_slice(reference[c_s:c_e], start, end, pad_char)
    return out


@define
class Genotypes:
    genos: NDArray[np.int8]  # (n_variants * n_samples, ploidy)
    first_v_idxs: NDArray[np.uint32]  # (n_regions)
    offsets: NDArray[np.uint32]  # (n_regions + 1)
    n_samples: int

    @property
    def n_regions(self) -> int:
        return len(self.first_v_idxs)

    @property
    def n_variants(self) -> int:
        return len(self.genos) // self.n_samples

    def __len__(self) -> int:
        return len(self.first_v_idxs)

    def __getitem__(self, idx: Tuple[ListIdx, ListIdx]) -> "Genotypes":
        s_idx = idx[0]
        r_idx = idx[1]
        genos = []
        first_v_idxs = self.first_v_idxs[r_idx]
        offsets = np.empty(len(r_idx) + 1, dtype=np.uint32)
        offsets[0] = 0
        shifts = np.asarray(s_idx) * self.n_variants
        for output_idx, (shift, region) in enumerate(zip(shifts, r_idx), 1):
            s, e = self.offsets[region] + shift, self.offsets[region + 1] + shift
            offsets[output_idx] = e - s
            if e > s:
                genos.append(self.genos[s:e])
        if len(genos) == 0:
            genos = np.empty((0, self.genos.shape[1]), dtype=self.genos.dtype)
        else:
            genos = np.concatenate(genos)
        offsets = offsets.cumsum(dtype=np.uint32)

        return Genotypes(genos, first_v_idxs, offsets, self.n_samples)


@nb.njit(parallel=True, nogil=True, cache=True)
def get_diffs(
    first_v_idxs: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    genotypes: NDArray[np.int8],
    size_diffs: NDArray[np.int32],
) -> NDArray[np.uint32]:
    """Get difference in length wrt reference genome for given genotypes.

    Parameters
    ----------
    first_v_idxs : NDArray[np.uint32]
        Shape = (n_regions,) First variant index for each query.
    offsets : NDArray[np.uint32]
        Shape = (n_regions + 1,) Offsets into genos.
    genotypes : NDArray[np.int8]
        Shape = (n_variants, ploidy) Genotypes.
    size_diffs : NDArray[np.int32]
        Shape = (total_variants,) Size of variants.
    """
    n_regions = len(first_v_idxs)
    ploidy = genotypes.shape[1]
    diffs = np.empty((n_regions, ploidy), np.uint32)

    for region in nb.prange(n_regions):
        o_s, o_e = offsets[region], offsets[region + 1]
        n_variants = o_e - o_s

        if n_variants == 0:
            diffs[region] = 0
            continue

        v_s = first_v_idxs[region]
        v_e = v_s + n_variants
        # (v p)
        genos = genotypes[o_s:o_e]
        # (v p) -> (p)
        diff = np.where(genos == 1, size_diffs[v_s:v_e, None], 0).sum(0).clip(0)
        diffs[region] = diff
    return diffs


@nb.njit(parallel=True, nogil=True, cache=True)
def reconstruct_haplotypes(
    out: NDArray[np.uint8],
    regions: NDArray[np.int32],
    shifts: NDArray[np.uint32],
    first_v_idxs: NDArray[np.uint32],
    offsets: NDArray[np.uint32],
    genos: NDArray[np.int8],
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.uintp],
    ref: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    pad_char: int,
):
    """_summary_

    Parameters
    ----------
    out : NDArray[np.uint8]
        Shape = (n_regions, ploidy, out_length) Output array.
    regions : NDArray[np.int32]
        Shape = (n_regions, 3) Regions to reconstruct haplotypes.
    shifts : NDArray[np.uint32]
        Shape = (n_regions, ploidy) Shifts for each query.
    first_v_idxs : NDArray[np.uint32]
        Shape = (n_regions,) First variant index for each query.
    offsets : NDArray[np.uint32]
        Shape = (n_regions + 1,) Offsets into genos.
    genos : NDArray[np.int8]
        Shape = (n_variants, ploidy) Genotypes of variants.
    positions : NDArray[np.int32]
        Shape = (total_variants,) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (total_variants,) Sizes of variants.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length,) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (total_variants,) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length,) Reference sequence.
    ref_offsets : NDArray[np.uint64]
        Shape = (n_contigs,) Offsets of reference sequences.
    pad_char : int
        Padding character.
    """
    n_regions = len(first_v_idxs)
    ploidy = genos.shape[1]
    length = out.shape[2]
    for query in nb.prange(n_regions):
        _out = out[query]
        q = regions[query]
        _shifts = shifts[query]

        c_idx = q[0]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        ref_s = q[1]
        ref_e = q[2]
        _ref = padded_slice(ref[c_s:c_e], ref_s, ref_e, pad_char)

        o_s, o_e = offsets[query], offsets[query + 1]
        n_variants = o_e - o_s

        if n_variants == 0:
            _out[:] = _ref[:length]
            continue

        _genos = genos[o_s:o_e]

        v_s = first_v_idxs[query]
        v_e = v_s + n_variants
        # adjust positions to be relative to reference subsequence
        _positions = positions[v_s:v_e] - ref_s
        _sizes = sizes[v_s:v_e]
        _alt_offsets = alt_offsets[v_s : v_e + 1].copy()
        _alt_alleles = alt_alleles[_alt_offsets[0] : _alt_offsets[-1]]
        _alt_offsets -= _alt_offsets[0]

        for hap in nb.prange(ploidy):
            reconstruct_haplotype(
                _positions,
                _sizes,
                _genos[:, hap],
                _shifts[hap],
                _alt_alleles,
                _alt_offsets,
                _ref,
                _out[hap],
                pad_char,
            )


@nb.njit(nogil=True, cache=True)
def padded_slice(arr: NDArray, start: int, stop: int, pad_char: int):
    pad_left = -min(0, start)
    pad_right = max(0, stop - len(arr))

    if pad_left == 0 and pad_right == 0:
        out = arr[start:stop]
        return out

    out = np.empty(stop - start, arr.dtype)

    if pad_left > 0 and pad_right > 0:
        out_stop = len(out) - pad_right
        out[:pad_left] = pad_char
        out[pad_left:out_stop] = arr[:]
        out[out_stop:] = pad_char
    elif pad_left > 0:
        out[:pad_left] = pad_char
        out[pad_left:] = arr[:stop]
    elif pad_right > 0:
        out_stop = len(out) - pad_right
        out[:out_stop] = arr[start:]
        out[out_stop:] = pad_char

    return out


@nb.njit(nogil=True, cache=True)
def reconstruct_haplotype(
    positions: NDArray[np.int32],
    sizes: NDArray[np.int32],
    genos: NDArray[np.int8],
    shift: int,
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.uintp],
    ref: NDArray[np.uint8],
    out: NDArray[np.uint8],
    pad_char: int,
):
    """Reconstruct a haplotype from reference sequence and variants.

    Parameters
    ----------
    positions : NDArray[np.int32]
        Shape = (n_variants) Positions of variants.
    sizes : NDArray[np.int32]
        Shape = (n_variants) Sizes of variants.
    genos : NDArray[np.int8]
        Shape = (n_variants) Genotypes of variants.
    shift : int
        Shift amount.
    alt_alleles : NDArray[np.uint8]
        Shape = (total_alt_length) ALT alleles.
    alt_offsets : NDArray[np.uintp]
        Shape = (n_variants) Offsets of ALT alleles.
    ref : NDArray[np.uint8]
        Shape = (ref_length) Reference sequence. ref_length >= out_length
    out : NDArray[np.uint8]
        Shape = (out_length) Output array.
    pad_char : int
        Padding character.
    """
    length = len(out)
    n_variants = len(positions)
    # where to get next reference subsequence
    ref_idx = 0
    # where to put next subsequence
    out_idx = 0
    # total amount to shift by
    shift = shift
    # how much we've shifted
    shifted = 0

    # first variant is a DEL spanning start
    v_rel_pos = positions[0]
    v_diff = sizes[0]
    if v_rel_pos < 0 and genos[0] == 1:
        # diff of v(-1) has been normalized to consider where ref is
        # otherwise, ref_idx = v_rel_pos - v_diff + 1
        # e.g. a -10 diff became -3 if v_rel_pos = -7
        ref_idx = v_rel_pos - v_diff + 1
        # increment the variant index
        start_idx = 1
    else:
        start_idx = 0

    for variant in range(start_idx, n_variants):
        # UNKNOWN -9 or REF 0
        if genos[variant] != 1:
            continue

        # position of variant relative to ref from fetch(contig, start, q_end)
        # i.e. has been put into same coordinate system as ref_idx
        v_rel_pos = positions[variant]

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
        if out_idx + ref_len >= length:
            # ref will get written by final clause
            # handles case where extraneous variants downstream of the haplotype were provided
            break
        out[out_idx : out_idx + ref_len] = ref[ref_idx : ref_idx + ref_len]
        out_idx += ref_len

        # insertions + substitions
        writable_length = min(v_len, length - out_idx)
        out[out_idx : out_idx + writable_length] = allele[:writable_length]
        out_idx += writable_length
        # +1 because ALT alleles always replace 1 nt of reference for a
        # normalized VCF
        ref_idx = v_rel_pos + 1

        # deletions, move ref to end of deletion
        if v_diff < 0:
            ref_idx -= v_diff

        if out_idx >= length:
            break

    # fill rest with reference sequence and pad with Ns
    unfilled_length = length - out_idx
    if unfilled_length > 0:
        writable_ref = min(unfilled_length, len(ref) - ref_idx)
        out_end_idx = out_idx + writable_ref
        ref_end_idx = ref_idx + writable_ref
        out[out_idx:out_end_idx] = ref[ref_idx:ref_end_idx]

        if out_end_idx < length:
            out[out_end_idx:] = pad_char


@define
class Intervals:
    intervals: NDArray[np.uint32]  # (n_intervals, 2)
    values: NDArray[np.float32]  # (n_intervals)
    offsets: NDArray[np.uint32]  # (n_queries + 1)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, ds_idx: ListIdx) -> "Intervals":
        intervals = []
        values = []
        offsets = np.empty(len(ds_idx) + 1, dtype=np.uint32)
        offsets[0] = 0
        for output_idx, i in enumerate(ds_idx, 1):
            s, e = self.offsets[i], self.offsets[i + 1]
            offsets[output_idx] = e - s
            if e > s:
                intervals.append(self.intervals[s:e])
                values.append(self.values[s:e])

        if len(intervals) == 0:
            intervals = np.empty((0, 2), dtype=self.intervals.dtype)
            values = np.empty(0, dtype=self.values.dtype)
        else:
            intervals = np.concatenate(intervals)
            values = np.concatenate(values)

        offsets = offsets.cumsum(dtype=np.uint32)

        return Intervals(intervals, values, offsets)


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_values(
    regions: NDArray[np.int32],
    intervals: NDArray[np.uint32],
    values: NDArray[np.float32],
    offsets: NDArray[np.uint32],
    query_length: int,
):
    """Convert intervals to values at base-pair resolution.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    intervals : NDArray[np.uint32]
        Shape = (n_intervals, 2) Intervals.
    values : NDArray[np.float32]
        Shape = (n_intervals,) Values.
    offsets : NDArray[np.uint32]
        Shape = (n_queries + 1,) Offsets into intervals and values.
    query_length : int
        Length of each query.
    """
    n_regions = len(regions)
    out = np.zeros((n_regions, query_length), np.float32)
    for region in nb.prange(n_regions):
        q_s = regions[region, 1]
        o_s, o_e = offsets[region], offsets[region + 1]
        n_intervals = o_e - o_s
        if n_intervals == 0:
            out[region] = 0
            continue

        for interval in nb.prange(o_s, o_e):
            i_s, i_e = intervals[interval] - q_s
            out[region, i_s:i_e] = values[interval]
    return out


@nb.njit(parallel=True, nogil=True, cache=True)
def intervals_to_hap_values(
    regions: NDArray[np.int32],
    shifts: NDArray[np.uint32],
    intervals: NDArray[np.uint32],
    values: NDArray[np.float32],
    offsets: NDArray[np.uint32],
    query_length: int,
):
    """Convert intervals to values at base-pair resolution.

    Parameters
    ----------
    regions : NDArray[np.int32]
        Shape = (n_queries, 3) Regions for each query.
    shifts : NDArray[np.uint32]
        Shape = (n_queries, ploidy) Shifts for each query.
    intervals : NDArray[np.uint32]
        Shape = (n_intervals, 2) Intervals.
    values : NDArray[np.float32]
        Shape = (n_intervals,) Values.
    offsets : NDArray[np.uint32]
        Shape = (n_queries + 1,) Offsets into intervals and values.
    query_length : int
        Length of each query.
    """
    n_queries = len(regions)
    ploidy = shifts.shape[1]
    out = np.zeros((n_queries, ploidy, query_length), np.float32)
    for query in nb.prange(n_queries):
        q_s = regions[query, 1]
        o_s, o_e = offsets[query], offsets[query + 1]
        n_intervals = o_e - o_s

        if n_intervals == 0:
            out[query] = 0
            continue

        for hap in nb.prange(ploidy):
            shift = shifts[query, hap]
            for interval in nb.prange(o_s, o_e):
                i_s, i_e = intervals[interval] - q_s + shift
                out[query, hap, i_s:i_e] = values[interval]
    return out


def adjust_multi_index(
    idxs: Tuple[ListIdx, ...], skip_idxs: Tuple[NDArray[np.integer], ...]
):
    adjusted_idxs: List[NDArray[np.integer]] = []
    for idx, skip in zip(idxs, skip_idxs):
        idx = np.asarray(idx)
        if len(skip) > 0:
            idx += (idx >= skip[:, None]).sum(0)
        adjusted_idxs.append(idx.squeeze())
    return tuple(adjusted_idxs)


def regions_to_bed(regions: NDArray, contigs: Sequence[str]) -> pl.DataFrame:
    """Convert regions to a BED3 DataFrame.

    Parameters
    ----------
    regions : NDArray
        Shape = (n_regions, 3) Regions.
    contigs : Sequence[str]
        Contigs.

    Returns
    -------
    pl.DataFrame
        Bed DataFrame.
    """
    bed = pl.DataFrame(
        regions, schema=["chrom", "chromStart", "chromEnd"]
    ).with_columns(pl.all().cast(pl.Int64))
    cmap = dict(enumerate(contigs))
    bed = bed.with_columns(pl.col("chrom").replace(cmap, return_dtype=pl.Utf8))
    return bed
