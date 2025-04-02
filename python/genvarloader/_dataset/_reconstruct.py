from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
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
from tqdm.auto import tqdm

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
from ._indexing import DatasetIndexer
from ._intervals import intervals_to_tracks, tracks_to_intervals
from ._tracks import shift_and_realign_tracks_sparse
from ._utils import padded_slice, splits_sum_le_value

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
        idx: NDArray[np.integer],
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
        idx: NDArray[np.integer],
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
                    path / "genotypes" / "ccfs.npy", dtype=np.float32, mode="r"
                ),
                np.memmap(path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"),
                len(regions),
                len(samples),
            )
        return cls(
            reference=reference,
            variants=variants,
            genotypes=genotypes,
            annotate=False,
        )

    def _haplotype_ilens(
        self,
        idx: NDArray[np.integer],
        jittered_regions: NDArray[np.int32],
        deterministic: bool,
        keep: NDArray[np.bool_] | None = None,
        keep_offsets: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int32]:
        """`idx` must be 1D."""
        # (b p)
        geno_offset_idxs = self.get_geno_offset_idx(idx, self.genotypes)

        if isinstance(self.genotypes, SparseSomaticGenotypes):
            if keep is None or keep_offsets is None:
                keep, keep_offsets = choose_unphased_variants(
                    starts=jittered_regions[:, 1],
                    ends=jittered_regions[:, 2],
                    geno_offset_idxs=geno_offset_idxs,
                    geno_v_idxs=self.genotypes.variant_idxs,
                    geno_offsets=self.genotypes.offsets,
                    positions=self.variants.positions,
                    sizes=self.variants.sizes,
                    ccfs=self.genotypes.ccfs,
                    deterministic=deterministic,
                )
            # (r s p)
            hap_ilens = get_diffs_sparse(
                geno_offset_idxs=geno_offset_idxs,
                geno_v_idxs=self.genotypes.variant_idxs,
                geno_offsets=self.genotypes.offsets,
                size_diffs=self.variants.sizes,
                keep=keep,
                keep_offsets=keep_offsets,
            )
        else:
            # (r s p)
            hap_ilens = get_diffs_sparse(
                geno_offset_idxs=geno_offset_idxs,
                geno_v_idxs=self.genotypes.variant_idxs,
                geno_offsets=self.genotypes.offsets,
                size_diffs=self.variants.sizes,
                starts=jittered_regions[:, 1],
                ends=jittered_regions[:, 2],
                positions=self.variants.positions,
            )

        return hap_ilens.reshape(-1, self.genotypes.ploidy)

    @overload
    def with_annot(self, annotations: Literal[False]) -> Haps[Ragged[np.bytes_]]: ...
    @overload
    def with_annot(self, annotations: Literal[True]) -> Haps[RaggedAnnotatedHaps]: ...

    def with_annot(self, annotations: bool) -> Haps:
        return evolve(self, annotate=annotations)

    def __call__(
        self,
        idx: NDArray[np.integer],
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
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> tuple[
        H,
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        Optional[NDArray[np.bool_]],
        Optional[NDArray[np.int64]],
    ]:
        batch_size = len(idx)
        lengths = regions[:, 2] - regions[:, 1]

        geno_offset_idx = self.get_geno_offset_idx(idx, self.genotypes)

        if isinstance(self.genotypes, SparseSomaticGenotypes):
            keep, keep_offsets = choose_unphased_variants(
                starts=regions[:, 1],
                ends=regions[:, 2],
                geno_offset_idxs=geno_offset_idx,
                geno_v_idxs=self.genotypes.variant_idxs,
                geno_offsets=self.genotypes.offsets,
                ccfs=self.genotypes.ccfs,
                positions=self.variants.positions,
                sizes=self.variants.sizes,
                deterministic=rng is None,
            )
        else:
            keep = None
            keep_offsets = None

        # (b p)
        diffs = self._haplotype_ilens(idx, regions, rng is None, keep, keep_offsets)
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
            diffs,
            hap_lengths,
            keep,
            keep_offsets,
        )

    @staticmethod
    def get_geno_offset_idx(
        idx: NDArray[np.integer],
        genotypes: Union[SparseGenotypes, SparseSomaticGenotypes],
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
        if missing := list(set(tracks) - set(self.intervals)):
            raise ValueError(f"Missing tracks: {missing}")
        return evolve(self, active_tracks=tracks)

    @classmethod
    def from_path(cls, path: Path, regions: NDArray[np.int32], n_samples: int):
        available_tracks: List[str] = []
        for p in (path / "intervals").iterdir():
            if len(list(p.iterdir())) == 0:
                p.rmdir()
            else:
                available_tracks.append(p.name)
        available_tracks.sort()
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
                itvs, (len(regions), n_samples), offsets
            )
        return cls(intervals, available_tracks)

    def __call__(
        self,
        idx: NDArray[np.integer],
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

    def write_transformed_track(
        self,
        new_track: str,
        existing_track: str,
        transform: Callable[
            [NDArray[np.intp], NDArray[np.intp], Ragged[np.float32]],
            Ragged[np.float32],
        ],
        path: Path,
        regions: NDArray[np.int32],
        max_jitter: int,
        idxer: DatasetIndexer,
        haps: Haps | None = None,
        max_mem: int = 2**30,
        overwrite: bool = False,
    ) -> Tracks:
        if new_track == existing_track:
            raise ValueError(
                "New track name must be different from existing track name."
            )

        if existing_track not in self.intervals:
            raise ValueError(
                f"Requested existing track {existing_track} does not exist."
            )

        intervals = self.intervals[existing_track]

        out_dir = path / "intervals" / new_track

        if out_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Track at {out_dir} already exists. Set overwrite=True to overwrite."
            )
        elif out_dir.exists() and overwrite:
            # according to GVL file format, should only have intervals.npy and offsets.npy in here
            for p in out_dir.iterdir():
                p.unlink()
            out_dir.rmdir()

        out_dir.mkdir(parents=True, exist_ok=True)

        # (r)
        n_regions, n_samples = idxer.full_shape
        jittered_regions = regions.copy()
        jittered_regions[:, 1] -= max_jitter
        jittered_regions[:, 2] += max_jitter
        r_idx = np.arange(n_regions)[:, None]
        s_idx = np.arange(n_samples)
        # (r s) -> (r*s)
        ds_idx = np.ravel_multi_index((r_idx, s_idx), idxer.full_shape).ravel()
        r_idx, s_idx = np.unravel_index(ds_idx, idxer.full_shape)
        if haps is not None:
            # extend ends by max hap diff to match write implementation
            jittered_regions[:, 2] += (
                haps._haplotype_ilens(ds_idx, jittered_regions, True)
                .reshape(n_regions, n_samples, haps.genotypes.ploidy)
                .max((1, 2))
                .clip(min=0)
            )
        lengths = jittered_regions[:, 2] - jittered_regions[:, 1]
        # for each region:
        # bytes = (4 bytes / bp) * (bp / sample) * samples
        n_regions, n_samples = intervals.shape
        mem_per_region = 4 * lengths * n_samples
        splits = splits_sum_le_value(mem_per_region, max_mem)
        memmap_intervals_offset = 0
        memmap_offsets_offset = 0
        last_offset = 0
        with tqdm(total=len(splits) - 1) as pbar:
            for offset_s, offset_e in zip(splits[:-1], splits[1:]):
                n_regions = int(offset_e - offset_s)
                ir_idx = repeat(
                    np.arange(offset_s, offset_e, dtype=np.intp),
                    "r -> (r s)",
                    s=n_samples,
                )
                is_idx = repeat(
                    np.arange(n_samples, dtype=np.intp), "s -> (r s)", r=n_regions
                )
                ds_idx = np.ravel_multi_index((ir_idx, is_idx), idxer.full_shape)
                ds_idx = idxer.i2d_map[ds_idx]
                r_idx, s_idx = np.unravel_index(ds_idx, idxer.full_shape)

                pbar.set_description("Writing (decompressing)")
                # (r*s)
                _regions = jittered_regions[r_idx]
                # (r*s+1)
                offsets = _lengths_to_offsets(_regions[:, 2] - _regions[:, 1])
                # layout is (regions, samples) so all samples are local for statistics
                tracks = np.empty(offsets[-1], np.float32)
                intervals_to_tracks(
                    offset_idxs=ds_idx,
                    starts=_regions[:, 1],
                    intervals=intervals.data,
                    itv_offsets=intervals.offsets,
                    out=tracks,
                    out_offsets=offsets,
                )
                tracks = Ragged.from_offsets(tracks, (n_regions, n_samples), offsets)

                pbar.set_description("Writing (transforming)")
                transformed_tracks = transform(ir_idx, is_idx, tracks)
                np.testing.assert_equal(tracks.shape, transformed_tracks.shape)

                pbar.set_description("Writing (compressing)")
                itvs, interval_offsets = tracks_to_intervals(
                    _regions, transformed_tracks.data, transformed_tracks.offsets
                )
                np.testing.assert_equal(
                    len(interval_offsets), n_regions * n_samples + 1
                )

                out = np.memmap(
                    out_dir / "intervals.npy",
                    dtype=itvs.dtype,
                    mode="w+" if memmap_intervals_offset == 0 else "r+",
                    shape=itvs.shape,
                    offset=memmap_intervals_offset,
                )
                out[:] = itvs[:]
                out.flush()
                memmap_intervals_offset += out.nbytes

                interval_offsets += last_offset
                last_offset = interval_offsets[-1]
                out = np.memmap(
                    out_dir / "offsets.npy",
                    dtype=interval_offsets.dtype,
                    mode="w+" if memmap_offsets_offset == 0 else "r+",
                    shape=len(interval_offsets) - 1,
                    offset=memmap_offsets_offset,
                )
                out[:] = interval_offsets[:-1]
                out.flush()
                memmap_offsets_offset += out.nbytes
                pbar.update()

        out = np.memmap(
            out_dir / "offsets.npy",
            dtype=np.int64,
            mode="r+",
            shape=1,
            offset=memmap_offsets_offset,
        )
        out[-1] = last_offset
        out.flush()

        return self.from_path(path, regions, n_samples).with_tracks(self.active_tracks)


@define
class SeqsTracks(Reconstructor[Tuple[Ragged[np.bytes_], Ragged[np.float32]]]):
    seqs: Seqs
    tracks: Tracks

    def __call__(
        self,
        idx: NDArray[np.integer],
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
        haps = self.haps.with_annot(annotations)
        return evolve(self, haps=haps)

    def __call__(
        self,
        idx: NDArray[np.integer],  # (b)
        regions: NDArray[np.int32],  # (b 3)
        output_length: Union[Literal["ragged", "variable"], int],
        jitter: int,
        rng: Optional[np.random.Generator],
    ) -> Tuple[H, Ragged[np.float32]]:
        lengths = regions[:, 2] - regions[:, 1]

        # ragged (b p l), (b p), (b p), (b*p*v), (b*p+1), (b p)
        haps, geno_idx, shifts, diffs, hap_lengths, keep, keep_offsets = (
            self.haps.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                jitter=jitter,
                rng=rng,
            )
        )

        if isinstance(output_length, int):
            # (b p)
            out_lengths = np.full_like(hap_lengths, output_length + 2 * jitter)
        else:
            # (b p)
            out_lengths = hap_lengths

        # (b) = lengths + max deletion length across ploidy
        track_lengths = lengths - diffs.clip(max=0).min(1)

        # (b*p+1)
        out_ofsts_per_t = _lengths_to_offsets(out_lengths)
        # (b+1)
        track_ofsts_per_t = _lengths_to_offsets(track_lengths)
        n_per_track = out_ofsts_per_t[-1]
        # ragged (b t p l)
        out = np.empty(len(self.tracks.active_tracks) * n_per_track, np.float32)
        out_lens = repeat(out_lengths, "b p -> b t p", t=len(self.tracks.active_tracks))
        out_offsets = _lengths_to_offsets(out_lens)

        for track_ofst, name in enumerate(self.tracks.active_tracks):
            intervals = self.tracks.intervals[name]

            # ragged (b l)
            _tracks = np.empty(track_ofsts_per_t[-1], np.float32)
            intervals_to_tracks(
                starts=regions[:, 1],  # (b)
                offset_idxs=idx,  # (b)
                intervals=intervals.data,  # (r*s*l)
                itv_offsets=intervals.offsets,  # (r*s+1)
                out=_tracks,  # (b*l)
                out_offsets=track_ofsts_per_t,  # (b+1)
            )

            _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]
            shift_and_realign_tracks_sparse(
                out=_out,  # (b*p*l)
                out_offsets=out_ofsts_per_t,  # (b*p+1)
                regions=regions,  # (b, 3)
                shifts=shifts,  # (b p)
                geno_offset_idxs=geno_idx,  # (b p)
                geno_v_idxs=self.haps.genotypes.variant_idxs,  # (r*s*p*v)
                geno_offsets=self.haps.genotypes.offsets,  # (r*s*p+1)
                positions=self.haps.variants.positions,  # (tot_v)
                sizes=self.haps.variants.sizes,  # (tot_v)
                tracks=_tracks,  # ragged (b l)
                track_offsets=track_ofsts_per_t,  # (b+1)
                keep=keep,  # (b*p*v)
                keep_offsets=keep_offsets,  # (b*p+1)
            )

        out_shape = (
            len(idx),
            len(self.tracks.active_tracks),
            self.haps.genotypes.ploidy,
        )

        # ragged (b t [p] l)
        tracks = Ragged.from_offsets(out, out_shape, out_offsets)

        return haps, tracks
