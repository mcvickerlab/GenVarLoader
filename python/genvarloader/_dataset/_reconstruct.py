from __future__ import annotations

import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numba as nb
import numpy as np
import polars as pl
from attrs import define, evolve
from einops import repeat
from loguru import logger
from numpy.typing import NDArray

from .._fasta import Fasta
from .._ragged import (
    INTERVAL_DTYPE,
    Ragged,
    RaggedAnnotatedHaps,
    RaggedIntervals,
)
from .._utils import _lengths_to_offsets, _normalize_contig_name
from .._variants._records import VLenAlleles
from ._genotypes import (
    SparseGenotypes,
    SparseSomaticGenotypes,
    choose_unphased_variants,
    get_diffs_sparse,
    reconstruct_haplotypes_from_sparse,
)
from ._intervals import intervals_to_tracks
from ._tracks import shift_and_realign_tracks_sparse
from ._utils import padded_slice

T = TypeVar("T", covariant=True)


@define
class Reference:
    reference: NDArray[np.uint8]
    contigs: List[str]
    offsets: NDArray[np.uint64]
    pad_char: int

    @classmethod
    def from_path_and_contigs(cls, fasta: Union[str, Path], contigs: List[str]):
        _fasta = Fasta("ref", fasta, "N")

        if not _fasta.cache_path.exists():
            logger.info("Memory-mapping FASTA file for faster access.")
            _fasta._write_to_cache()

        contigs = cast(
            List[str],
            [_normalize_contig_name(c, _fasta.contigs) for c in contigs],
        )
        _fasta.sequences = _fasta._get_sequences(contigs)
        if TYPE_CHECKING:
            assert _fasta.sequences is not None
            assert _fasta.pad is not None
        refs: List[NDArray[np.bytes_]] = []
        next_offset = 0
        _ref_offsets: Dict[str, int] = {}
        for contig in contigs:
            arr = _fasta.sequences[contig]
            refs.append(arr)
            _ref_offsets[contig] = next_offset
            next_offset += len(arr)
        reference = np.concatenate(refs).view(np.uint8)
        pad_char = ord(_fasta.pad)
        if any(c is None for c in contigs):
            raise ValueError("Contig names in metadata do not match reference.")
        ref_offsets = np.empty(len(contigs) + 1, np.uint64)
        ref_offsets[:-1] = np.array([_ref_offsets[c] for c in contigs], dtype=np.uint64)
        ref_offsets[-1] = len(reference)
        return cls(reference, contigs, ref_offsets, pad_char)


@define
class _Variants:
    positions: NDArray[np.int32]
    sizes: NDArray[np.int32]
    alts: VLenAlleles

    @classmethod
    def from_table(cls, variants: Union[str, Path, pl.DataFrame]):
        if isinstance(variants, (str, Path)):
            variants = pl.read_ipc(variants)
        return cls(
            variants["POS"].to_numpy(),
            variants["ILEN"].to_numpy(),
            VLenAlleles.from_polars(variants["ALT"]),
        )


class Reconstructor(Protocol[T]):
    """Reconstructs data on-the-fly. e.g. personalized sequences, tracks, etc."""

    def __call__(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> T: ...


@define
class Seqs(Reconstructor[Ragged[np.bytes_]]):
    reference: Reference
    """The reference genome. This is kept in memory."""

    def __call__(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> Ragged[np.bytes_]:
        batch_size = len(idx)
        lengths = regions[:, 2] - regions[:, 1]

        if not isinstance(output_length, int):
            # (b)
            out_lengths = lengths
        else:
            out_lengths = np.full(
                batch_size, output_length + 2 * jitter, dtype=np.int32
            )

        if rng is not None and isinstance(output_length, int):
            # (b)
            max_shift = (lengths - output_length).clip(min=0)
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)
            regions = regions.copy()
            regions[:, 1] += shifts
            regions[:, 2] = regions[:, 1] + output_length + 2 * jitter

        # (b+1)
        out_offsets = _lengths_to_offsets(out_lengths)

        # ragged (b)
        ref = _get_reference(
            regions=regions,
            out_offsets=out_offsets,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
        ).view("S1")
        ref = cast(Ragged[np.bytes_], Ragged.from_offsets(ref, batch_size, out_offsets))

        return ref


@nb.njit(parallel=True, nogil=True, cache=True)
def _get_reference(
    regions: NDArray[np.int32],
    out_offsets: NDArray[np.int64],
    reference: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    pad_char: int,
) -> NDArray[np.uint8]:
    out = np.empty(out_offsets[-1], np.uint8)
    for i in nb.prange(len(regions)):
        o_s, o_e = out_offsets[i], out_offsets[i + 1]
        c_idx, start, end = regions[i, :3]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        out[o_s:o_e] = padded_slice(reference[c_s:c_e], start, end, pad_char)
    return out


H = TypeVar("H", Ragged[np.bytes_], RaggedAnnotatedHaps)


@define
class Haps(Reconstructor[H]):
    reference: Reference
    """The reference genome. This is kept in memory."""
    variants: _Variants
    """The variant sites in the dataset. This is kept in memory."""
    genotypes: Union[SparseGenotypes, SparseSomaticGenotypes]
    """Shape: (regions, samples, ploidy). The genotypes in the dataset. This is memory mapped."""
    haplotype_ilens: NDArray[np.int32]
    """Shape: (regions, samples, ploidy). Length of jitter-extended haplotypes, same order as on disk."""
    annotate: bool

    @classmethod
    def from_path(
        cls: type[Haps[Ragged[np.bytes_]]],
        path: Path,
        reference: Reference,
        phased: bool,
        regions: NDArray[np.int32],
        samples: List[str],
        ploidy: int,
    ) -> Haps[Ragged[np.bytes_]]:
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        variants = _Variants.from_table(path / "genotypes" / "variants.arrow")
        if phased:
            genotypes = SparseGenotypes(
                np.memmap(
                    path / "genotypes" / "variant_idxs.npy",
                    dtype=np.int32,
                    mode="r",
                ),
                np.memmap(path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"),
                len(regions),
                len(samples),
                ploidy,
            )
        else:
            genotypes = SparseSomaticGenotypes(
                np.memmap(
                    path / "genotypes" / "variant_idxs.npy",
                    dtype=np.int32,
                    mode="r",
                ),
                np.memmap(
                    path / "genotypes" / "dosages.npy", dtype=np.float32, mode="r"
                ),
                np.memmap(path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"),
                len(regions),
                len(samples),
            )
        haplotype_ilens = cls._compute_haplotype_ilens(
            genotypes=genotypes,
            variants=variants,
            regions=regions,
            jitter=metadata["max_jitter"],
            rng=None,
        )
        return cls(
            reference=reference,
            variants=variants,
            genotypes=genotypes,
            haplotype_ilens=haplotype_ilens,
            annotate=False,
        )

    @classmethod
    def _compute_haplotype_ilens(
        cls,
        genotypes: Union[SparseGenotypes, SparseSomaticGenotypes],
        variants: _Variants,
        regions: NDArray[np.int32],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> NDArray[np.int32]:
        # r s p
        shape = genotypes.effective_shape[:2]
        ds_idx = np.arange(np.prod(shape), dtype=np.intp)
        r_idx, _ = np.unravel_index(ds_idx, shape)
        geno_offset_idxs = cls.get_geno_offset_idx(ds_idx, genotypes)

        jittered_regions = regions.copy()
        jittered_regions[:, 1] -= jitter
        jittered_regions[:, 2] += jitter

        if isinstance(genotypes, SparseSomaticGenotypes):
            keep, keep_offsets = choose_unphased_variants(
                starts=jittered_regions[r_idx, 1],
                ends=jittered_regions[r_idx, 2],
                geno_offset_idxs=geno_offset_idxs,
                geno_v_idxs=genotypes.variant_idxs,
                geno_offsets=genotypes.offsets,
                positions=variants.positions,
                sizes=variants.sizes,
                dosages=genotypes.dosages,
                deterministic=rng is None,
            )
            hap_ilens = get_diffs_sparse(
                geno_offset_idxs=geno_offset_idxs,
                geno_v_idxs=genotypes.variant_idxs,
                geno_offsets=genotypes.offsets,
                size_diffs=variants.sizes,
                keep=keep,
                keep_offsets=keep_offsets,
            )
        else:
            hap_ilens = get_diffs_sparse(
                geno_offset_idxs=geno_offset_idxs,
                geno_v_idxs=genotypes.variant_idxs,
                geno_offsets=genotypes.offsets,
                size_diffs=variants.sizes,
                starts=jittered_regions[r_idx, 1],
                ends=jittered_regions[r_idx, 2],
                positions=variants.positions,
            )

        return hap_ilens.reshape(*shape, genotypes.ploidy)

    @overload
    def with_annot(self, annotations: Literal[False]) -> Haps[Ragged[np.bytes_]]: ...
    @overload
    def with_annot(self, annotations: Literal[True]) -> Haps[RaggedAnnotatedHaps]: ...

    def with_annot(self, annotations: bool) -> Haps:
        return evolve(self, _annotate=annotations)

    def __call__(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> H:
        haps, *_ = self.get_haps_and_shifts(
            idx=idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
        )
        return haps

    def get_haps_and_shifts(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> tuple[
        H,
        NDArray[np.intp],
        NDArray[np.int32],
        Optional[NDArray[np.bool_]],
        Optional[NDArray[np.int64]],
        NDArray[np.int32],
    ]:
        batch_size = len(idx)
        lengths = regions[:, 2] - regions[:, 1]

        r_idx, s_idx = np.unravel_index(idx, self.genotypes.effective_shape[:2])
        geno_offset_idx = self.get_geno_offset_idx(idx, self.genotypes)

        if isinstance(self.genotypes, SparseSomaticGenotypes):
            keep, keep_offsets = choose_unphased_variants(
                starts=regions[:, 1],
                ends=regions[:, 2],
                geno_offset_idxs=geno_offset_idx,
                geno_v_idxs=self.genotypes.variant_idxs,
                geno_offsets=self.genotypes.offsets,
                dosages=self.genotypes.dosages,
                positions=self.variants.positions,
                sizes=self.variants.sizes,
                deterministic=rng is None,
            )
        else:
            keep = None
            keep_offsets = None

        # (b p)
        diffs = self.haplotype_ilens[r_idx, s_idx]
        hap_lengths = lengths[:, None] + diffs

        if rng is None or isinstance(output_length, str):
            # (b p)
            shifts = np.zeros((batch_size, self.genotypes.ploidy), dtype=np.int32)
        else:
            # if the haplotype is longer than the region, shift it randomly
            # by up to:
            # the difference in length between the haplotype and the region
            # PLUS the difference in length between the region and the output_length
            # (b p)
            max_shift = diffs.clip(min=0)
            if isinstance(output_length, int):
                # (b p)
                max_shift += (lengths - output_length).clip(min=0)[:, None]
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)

        if not isinstance(output_length, int):
            # (b p)
            out_lengths = hap_lengths
        else:
            out_lengths = np.full(
                (batch_size, self.genotypes.ploidy),
                output_length + 2 * jitter,
                dtype=np.int32,
            )
        # (b*p+1)
        out_offsets = _lengths_to_offsets(out_lengths)

        # (b p l), (b p l), (b p l)
        if self.annotate:
            haps, maybe_annot_v_idx, maybe_annot_pos = self._get_haplotypes(
                geno_offset_idx=geno_offset_idx,
                regions=regions,
                out_offsets=out_offsets,
                shifts=shifts,
                keep=keep,
                keep_offsets=keep_offsets,
                annotate=self.annotate,
            )
        else:
            haps = self._get_haplotypes(
                geno_offset_idx=geno_offset_idx,
                regions=regions,
                out_offsets=out_offsets,
                shifts=shifts,
                keep=keep,
                keep_offsets=keep_offsets,
                annotate=self.annotate,
            )

        if isinstance(self.genotypes, SparseSomaticGenotypes):
            # (b 1 l) -> (b l) remove ploidy dim
            haps = haps.squeeze(1)

        if self.annotate:
            out = RaggedAnnotatedHaps(haps, maybe_annot_v_idx, maybe_annot_pos)  # type: ignore
        else:
            out = haps

        return (
            out,  # type: ignore | pylance doesn't like this but it's correct behavior for the signature
            geno_offset_idx,
            shifts,
            keep,
            keep_offsets,
            hap_lengths,
        )

    @staticmethod
    def get_geno_offset_idx(
        idx: NDArray[np.intp], genotypes: Union[SparseGenotypes, SparseSomaticGenotypes]
    ) -> NDArray[np.intp]:
        r_idx, s_idx = np.unravel_index(idx, genotypes.effective_shape[:2])
        ploid_idx = np.arange(genotypes.ploidy, dtype=np.intp)
        rsp_idx = (r_idx[:, None], s_idx[:, None], ploid_idx)
        geno_offset_idx = np.ravel_multi_index(rsp_idx, genotypes.effective_shape)
        return geno_offset_idx

    @overload
    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: Optional[NDArray[np.bool_]],
        keep_offsets: Optional[NDArray[np.int64]],
        annotate: Literal[False],
    ) -> Ragged[np.bytes_]: ...
    @overload
    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: Optional[NDArray[np.bool_]],
        keep_offsets: Optional[NDArray[np.int64]],
        annotate: Literal[True],
    ) -> Tuple[Ragged[np.bytes_], Ragged[np.int32], Ragged[np.int32]]: ...

    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: Optional[NDArray[np.bool_]],
        keep_offsets: Optional[NDArray[np.int64]],
        annotate: bool,
    ) -> (
        Ragged[np.bytes_] | Tuple[Ragged[np.bytes_], Ragged[np.int32], Ragged[np.int32]]
    ):
        """Reconstruct haplotypes from sparse genotypes.

        Parameters
        ----------
        geno_offset_idx
            Shape: (queries). The genotype offset indices. i.e. the dataset indices.
        regions
            Shape: (queries). The regions to reconstruct.
        out_offsets
            Shape: (queries+1). Offsets for haplotypes and annotations.
        shifts
            Shape: (queries, ploidy). The shift for each haplotype.
        keep
            Ragged array, shape: (variants). Whether to keep each variant. Implicitly has the same offsets
            as the sparse genotypes corresponding to geno_offset_idx.
        """
        haps = Ragged.from_offsets(
            np.empty(out_offsets[-1], np.uint8), shifts.shape, out_offsets
        )

        if annotate:
            annot_v_idxs = Ragged.from_offsets(
                np.empty(out_offsets[-1], np.int32), shifts.shape, out_offsets
            )
            annot_positions = Ragged.from_offsets(
                np.empty(out_offsets[-1], np.int32), shifts.shape, out_offsets
            )
        else:
            annot_v_idxs = None
            annot_positions = None

        # don't need to pass annot offsets because they are the same as haps offsets
        reconstruct_haplotypes_from_sparse(
            geno_offset_idxs=geno_offset_idx,
            out=haps.data,
            out_offsets=haps.offsets,
            regions=regions,
            shifts=shifts,
            geno_offsets=self.genotypes.offsets,
            geno_v_idxs=self.genotypes.variant_idxs,
            positions=self.variants.positions,
            sizes=self.variants.sizes,
            alt_alleles=self.variants.alts.alleles.view(np.uint8),
            alt_offsets=self.variants.alts.offsets,
            ref=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            keep=keep,
            keep_offsets=keep_offsets,
            annot_v_idxs=annot_v_idxs.data
            if annot_v_idxs is not None
            else annot_v_idxs,
            annot_ref_pos=annot_positions.data
            if annot_positions is not None
            else annot_positions,
        )

        haps.data = haps.data.view("S1")
        haps = cast(Ragged[np.bytes_], haps)

        if annotate:
            return haps, annot_v_idxs, annot_positions  # type: ignore
        else:
            return haps


@define
class Tracks(Reconstructor[Ragged[np.float32]]):
    intervals: Dict[str, RaggedIntervals]
    """The intervals in the dataset. This is memory mapped."""
    active_tracks: List[str]

    def with_tracks(self, tracks: str | List[str]) -> Tracks:
        if isinstance(tracks, str):
            tracks = [tracks]
        if missing := list(set(self.intervals) - set(tracks)):
            raise ValueError(f"Missing tracks: {missing}")
        return evolve(self, active_tracks=tracks)

    @classmethod
    def from_path(cls, path: Path, regions: NDArray[np.int32], samples: List[str]):
        available_tracks: List[str] = []
        for p in (path / "intervals").iterdir():
            if len(list(p.iterdir())) == 0:
                p.rmdir()
            else:
                available_tracks.append(p.name)
        available_tracks.sort()
        active_tracks = available_tracks
        intervals: Optional[Dict[str, RaggedIntervals]] = {}
        for track in available_tracks:
            itvs = np.memmap(
                path / "intervals" / track / "intervals.npy",
                dtype=INTERVAL_DTYPE,
                mode="r",
            )
            offsets = np.memmap(
                path / "intervals" / track / "offsets.npy",
                dtype=np.int64,
                mode="r",
            )
            intervals[track] = RaggedIntervals.from_offsets(
                itvs, (len(regions), len(samples)), offsets
            )
        return cls(intervals, active_tracks)

    def __call__(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> Ragged[np.float32]:
        lengths = regions[:, 2] - regions[:, 1]
        if isinstance(output_length, int):
            out_lengths = track_lengths = np.full_like(
                lengths, output_length + 2 * jitter
            )
            if rng is not None:
                max_shift = (lengths - output_length).clip(min=0)
                shifts = rng.integers(0, max_shift + 1, dtype=np.int32)
                regions = regions.copy()
                regions[:, 1] += shifts
                regions[:, 2] = regions[:, 1] + output_length + 2 * jitter
        else:
            out_lengths = track_lengths = lengths

        # (b [p])
        out_ofsts_per_t = _lengths_to_offsets(out_lengths)
        track_ofsts_per_t = _lengths_to_offsets(track_lengths)
        # caller accounts for ploidy
        n_per_track = out_ofsts_per_t[-1]
        # ragged (b t [p] l)
        out = np.empty(len(self.active_tracks) * n_per_track, np.float32)
        out_lens = repeat(out_lengths, "b -> b t", t=len(self.active_tracks))
        out_offsets = _lengths_to_offsets(out_lens)

        for track_ofst, name in enumerate(self.active_tracks):
            intervals = self.intervals[name]
            # (b t l) ragged
            _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]
            intervals_to_tracks(
                offset_idxs=idx,
                starts=regions[:, 1],
                intervals=intervals.data,
                itv_offsets=intervals.offsets,
                out=_out,
                out_offsets=track_ofsts_per_t,
            )

        out_shape = (len(idx), len(self.active_tracks))

        # ragged (b t [p] l)
        tracks = Ragged.from_offsets(out, out_shape, out_offsets)

        return tracks


@define
class SeqsTracks(Reconstructor[Tuple[Ragged[np.bytes_], Ragged[np.float32]]]):
    seqs: Seqs
    tracks: Tracks

    def __call__(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> Tuple[Ragged[np.bytes_], Ragged[np.float32]]:
        seqs = self.seqs(
            idx=idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
        )
        tracks = self.tracks(
            idx=idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
        )
        return seqs, tracks


@define
class HapsTracks(Reconstructor[tuple[H, Ragged[np.float32]]]):
    haps: Haps[H]
    tracks: Tracks

    @overload
    def with_annot(
        self, annotations: Literal[False]
    ) -> HapsTracks[Ragged[np.bytes_]]: ...
    @overload
    def with_annot(
        self, annotations: Literal[True]
    ) -> HapsTracks[RaggedAnnotatedHaps]: ...

    def with_annot(self, annotations: bool) -> HapsTracks:
        haps = evolve(self.haps, _annotate=annotations)
        return evolve(self, haps=haps)

    def __call__(
        self,
        idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> Tuple[H, Ragged[np.float32]]:
        lengths = regions[:, 2] - regions[:, 1]

        haps, geno_idx, shifts, keep, keep_offsets, hap_lengths = (
            self.haps.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                jitter=jitter,
                rng=rng,
            )
        )

        # (b p), (b)
        # need at least length
        track_lengths = np.maximum(hap_lengths, lengths[:, None])

        if not isinstance(output_length, int):
            # (b [p])
            out_lengths = hap_lengths
        else:
            # (b [p])
            out_lengths = np.full_like(track_lengths, output_length + 2 * jitter)

        # (b [p])
        out_ofsts_per_t = _lengths_to_offsets(out_lengths)
        track_ofsts_per_t = _lengths_to_offsets(track_lengths)
        # caller accounts for ploidy
        n_per_track = out_ofsts_per_t[-1]
        # ragged (b t [p] l)
        out = np.empty(len(self.tracks.active_tracks) * n_per_track, np.float32)
        out_lens = repeat(out_lengths, "b p -> b t p", t=len(self.tracks.active_tracks))
        out_offsets = _lengths_to_offsets(out_lens)

        for track_ofst, name in enumerate(self.tracks.active_tracks):
            intervals = self.tracks.intervals[name]

            # (b p l) ragged
            _tracks = np.empty(track_ofsts_per_t[-1], np.float32)
            intervals_to_tracks(
                starts=regions[:, 1],
                offset_idxs=idx,
                intervals=intervals.data,
                itv_offsets=intervals.offsets,
                out=_tracks,
                out_offsets=track_ofsts_per_t,
            )

            _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]
            shift_and_realign_tracks_sparse(
                geno_offset_idxs=geno_idx,
                geno_v_idxs=self.haps.genotypes.variant_idxs,
                geno_offsets=self.haps.genotypes.offsets,
                regions=regions,
                positions=self.haps.variants.positions,
                sizes=self.haps.variants.sizes,
                shifts=shifts,
                tracks=_tracks,
                track_offsets=track_ofsts_per_t,
                out=_out,
                out_offsets=out_ofsts_per_t,
                keep=keep,
                keep_offsets=keep_offsets,
            )

        out_shape = (
            len(idx),
            len(self.tracks.active_tracks),
            self.haps.genotypes.ploidy,
        )

        # ragged (b t [p] l)
        tracks = Ragged.from_offsets(out, out_shape, out_offsets)

        return haps, tracks
