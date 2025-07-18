from __future__ import annotations

import enum
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Literal, Protocol, TypeVar, cast, overload

import awkward as ak
import numpy as np
import polars as pl
from attrs import define, evolve, field
from awkward.contents import ListOffsetArray, NumpyArray, RegularArray
from awkward.index import Index
from einops import repeat
from genoray._svar import (
    DOSAGE_TYPE,
    POS_TYPE,
    V_IDX_TYPE,
    SparseDosages,
    SparseGenotypes,
)
from loguru import logger
from numpy.typing import NDArray
from seqpro._ragged import Ragged
from tqdm.auto import tqdm
from typing_extensions import assert_never

from .._ragged import INTERVAL_DTYPE, RaggedAnnotatedHaps, RaggedIntervals, RaggedSeqs
from .._utils import lengths_to_offsets
from .._variants._records import RaggedAlleles
from ._genotypes import get_diffs_sparse, reconstruct_haplotypes_from_sparse
from ._indexing import DatasetIndexer
from ._intervals import intervals_to_tracks, tracks_to_intervals
from ._rag_variants import RaggedVariants
from ._reference import Reference, get_reference
from ._tracks import shift_and_realign_tracks_sparse
from ._utils import splits_sum_le_value

T = TypeVar("T", covariant=True)


@define
class _Variants:
    v_starts: NDArray[POS_TYPE]
    ilens: NDArray[np.int32]
    alts: RaggedAlleles

    @classmethod
    def from_table(cls, variants: str | Path | pl.DataFrame):
        if isinstance(variants, (str, Path)):
            variants = pl.read_ipc(variants)
        return cls(
            variants["POS"].to_numpy(),
            variants["ILEN"].to_numpy(),
            RaggedAlleles.from_polars(variants["ALT"]),
        )


class Reconstructor(Protocol[T]):
    """Reconstructs data on-the-fly. e.g. personalized sequences, tracks, etc."""

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> T: ...


@define
class Ref(Reconstructor[Ragged[np.bytes_]]):
    reference: Reference
    """The reference genome. This is kept in memory."""

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> Ragged[np.bytes_]:
        batch_size = len(idx)

        if isinstance(output_length, int):
            # (b)
            out_lengths = np.full(batch_size, output_length, dtype=np.int32)
            regions = regions.copy()
            regions[:, 2] = regions[:, 1] + out_lengths
        else:
            lengths = regions[:, 2] - regions[:, 1]
            out_lengths = lengths

        # (b+1)
        out_offsets = lengths_to_offsets(out_lengths)

        # ragged (b ~l)
        ref = get_reference(
            regions=regions,
            out_offsets=out_offsets,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
        ).view("S1")

        ref = cast(Ragged[np.bytes_], Ragged.from_offsets(ref, batch_size, out_offsets))

        return ref


_H = TypeVar("_H", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)
_NewH = TypeVar("_NewH", RaggedSeqs, RaggedAnnotatedHaps, RaggedVariants)


@define
class Haps(Reconstructor[_H]):
    reference: Reference | None
    """The reference genome. This is kept in memory."""
    variants: _Variants
    """The variant sites in the dataset. This is kept in memory."""
    genotypes: SparseGenotypes
    """Shape: (regions, samples, ploidy). The genotypes in the dataset. This is memory mapped."""
    dosages: SparseDosages | None
    kind: type[_H]
    n_variants: NDArray[np.int32] = field(init=False)
    """Shape: (regions, samples, ploidy). The number of variants in the dataset."""

    def __attrs_post_init__(self):
        self.n_variants = ak.num(self.genotypes.to_awkward(), -1).to_numpy()

    @classmethod
    def from_path(
        cls: type[Haps[RaggedVariants]],
        path: Path,
        reference: Reference | None,
        regions: NDArray[np.int32],
        samples: list[str],
        ploidy: int,
    ) -> Haps[RaggedVariants]:
        svar_meta_path = path / "genotypes" / "svar_meta.json"
        dosages = None

        if svar_meta_path.exists():
            with open(svar_meta_path) as f:
                metadata = json.load(f)
            # (r s p 2)
            shape: tuple[int, ...] = tuple(metadata["shape"])
            dtype = np.dtype(metadata["dtype"])

            offset_path = path / "genotypes" / "offsets.npy"
            geno_path = path / "genotypes" / "link.svar" / "variant_idxs.npy"
            dosage_path = path / "genotypes" / "link.svar" / "dosages.npy"

            offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
            v_idxs = np.memmap(geno_path, dtype=V_IDX_TYPE, mode="r")
            genotypes = SparseGenotypes.from_offsets(
                v_idxs, shape[1:], offsets.reshape(2, -1)
            )

            if dosage_path.exists():
                dosages = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
                dosages = SparseDosages.from_offsets(
                    dosages, shape[1:], offsets.reshape(2, -1)
                )

            logger.info("Loading variant data.")
            svar_index = (
                pl.scan_ipc(
                    path / "genotypes" / "link.svar" / "index.arrow", memory_map=False
                )
                .select("POS", pl.col("ILEN", "ALT").list.first())
                .collect()
            )
            variants = _Variants(
                svar_index["POS"].to_numpy() - 1,
                svar_index["ILEN"].to_numpy(),
                RaggedAlleles.from_polars(svar_index["ALT"]),
            )
        else:
            logger.info("Loading variant data.")
            variants = _Variants.from_table(path / "genotypes" / "variants.arrow")
            v_idxs = np.memmap(
                path / "genotypes" / "variant_idxs.npy",
                dtype=V_IDX_TYPE,
                mode="r",
            )
            offsets = np.memmap(
                path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"
            )
            shape = (len(regions), len(samples), ploidy)
            genotypes = SparseGenotypes.from_offsets(v_idxs, shape, offsets)

        return cls(
            reference=reference,
            variants=variants,
            genotypes=genotypes,
            dosages=dosages,
            kind=RaggedVariants,
        )

    def _haplotype_ilens(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        deterministic: bool,
        keep: NDArray[np.bool_] | None = None,
        keep_offsets: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int32]:
        """`idx` must be 1D."""
        # (b p)
        geno_offset_idxs = self.get_geno_offset_idx(idx, self.genotypes)

        # (r s p)
        hap_ilens = get_diffs_sparse(
            geno_offset_idxs=geno_offset_idxs,
            geno_v_idxs=self.genotypes.data,
            geno_offsets=self.genotypes.offsets,
            ilens=self.variants.ilens,
            q_starts=regions[:, 1],
            q_ends=regions[:, 2],
            v_starts=self.variants.v_starts,
        )

        return hap_ilens.reshape(-1, self.genotypes.shape[-1])

    def to_kind(self, kind: type[_NewH]) -> Haps[_NewH]:
        if kind != RaggedVariants and self.reference is None:
            raise ValueError(
                f"Cannot return {kind.__name__}: no reference genome was provided."
            )
        return cast(Haps[_NewH], evolve(self, kind=kind))

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> _H:
        if issubclass(self.kind, RaggedVariants):
            ragv = self._get_variants(
                idx=idx,
                regions=None,
                shifts=None,
                keep=None,
                keep_offsets=None,
            )
            ragv = cast(_H, ragv)
            return ragv
        else:
            haps, *_ = self.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                rng=rng,
                deterministic=deterministic,
            )
        return haps

    def get_haps_and_shifts(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> tuple[
        _H,
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
        NDArray[np.int64] | None,
    ]:
        ploidy = self.genotypes.ploidy
        batch_size = len(idx)
        # (b)
        lengths = regions[:, 2] - regions[:, 1]

        geno_offset_idx = self.get_geno_offset_idx(idx, self.genotypes)

        # (b p)
        diffs = self._haplotype_ilens(idx, regions, deterministic)
        hap_lengths = lengths[:, None] + diffs

        if deterministic or isinstance(output_length, str):
            # (b p)
            shifts = np.zeros((batch_size, ploidy), dtype=np.int32)
        else:
            # if the haplotype is longer than the region, shift it randomly
            # by up to:
            # the difference in length between the haplotype and the region
            # PLUS the difference in length between the region and the output_length
            # (b)
            # (b p)
            max_shift = diffs.clip(min=0)
            # (b p) + (b 1)
            max_shift += (lengths - output_length).clip(min=0)[:, None]
            # (b p)
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)

        if not isinstance(output_length, int):
            # (b p)
            out_lengths = hap_lengths
        else:
            out_lengths = np.full(
                (batch_size, self.genotypes.shape[-1]),
                output_length,
                dtype=np.int32,
            )
        # (b*p+1)
        out_offsets = lengths_to_offsets(out_lengths)

        # (b p l), (b p l), (b p l)
        if issubclass(self.kind, RaggedSeqs):
            out = self._get_haplotypes(
                geno_offset_idx=geno_offset_idx,
                regions=regions,
                out_offsets=out_offsets,
                shifts=shifts,
                keep=None,
                keep_offsets=None,
                annotate=False,
            )
        elif issubclass(self.kind, RaggedAnnotatedHaps):
            haps, maybe_annot_v_idx, maybe_annot_pos = self._get_haplotypes(
                geno_offset_idx=geno_offset_idx,
                regions=regions,
                out_offsets=out_offsets,
                shifts=shifts,
                keep=None,
                keep_offsets=None,
                annotate=True,
            )
            out = RaggedAnnotatedHaps(haps, maybe_annot_v_idx, maybe_annot_pos)
        elif issubclass(self.kind, RaggedVariants):
            out = self._get_variants(
                idx=idx,
                regions=regions,
                shifts=shifts,
                keep=None,
                keep_offsets=None,
            )
        else:
            assert_never(self.kind)

        return (
            out,  # type: ignore | pylance doesn't like this but it's correct behavior for the signature
            geno_offset_idx,
            shifts,
            diffs,
            hap_lengths,
            None,
            None,
        )

    @staticmethod
    def get_geno_offset_idx(
        idx: NDArray[np.integer],
        genotypes: SparseGenotypes,
    ) -> NDArray[np.intp]:
        r_idx, s_idx = np.unravel_index(idx, genotypes.shape[:2])
        ploid_idx = np.arange(genotypes.shape[-1], dtype=np.intp)
        rsp_idx = (r_idx[:, None], s_idx[:, None], ploid_idx)
        geno_offset_idx = np.ravel_multi_index(rsp_idx, genotypes.shape)
        return geno_offset_idx

    def _get_variants(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32] | None,
        shifts: NDArray[np.int32] | None,
        keep: NDArray[np.bool_] | None,
        keep_offsets: NDArray[np.int64] | None,
    ) -> RaggedVariants:
        # TODO: maybe filter variants for region, shifts, and keep?
        r, s = np.unravel_index(idx, self.genotypes.shape[:2])
        genos = cast(SparseGenotypes, self.genotypes[r, s])
        v_idxs = ak.flatten(genos.to_awkward(), None).to_numpy()
        geno_offsets = lengths_to_offsets(genos.lengths)

        # (b*p*v ~l)
        alts = cast(RaggedAlleles, self.variants.alts[v_idxs])
        alt_offsets = lengths_to_offsets(alts.lengths)
        node = alts.to_awkward().layout
        while not isinstance(node, NumpyArray):
            node = node.content
        data = ak.with_parameter(node, "__array__", "char", highlevel=False)
        l_content = ListOffsetArray(Index(alt_offsets), data)
        vl_content = ListOffsetArray(Index(geno_offsets), l_content)
        pvl_content = RegularArray(vl_content, genos.ploidy)
        alts = ak.Array(pvl_content)

        v_starts = self.variants.v_starts[v_idxs]
        v_starts = Ragged[v_starts.dtype.type].from_offsets(
            v_starts, genos.shape, geno_offsets
        )

        ilens = self.variants.ilens[v_idxs]
        ilens = Ragged[ilens.dtype.type].from_offsets(ilens, genos.shape, geno_offsets)

        if self.dosages is not None:
            # guaranteed to have same shape as genotypes but need to make it contiguous/copy the data
            dosages = cast(SparseDosages, self.dosages[r, s])
            dosages = Ragged[DOSAGE_TYPE].from_offsets(
                ak.flatten(dosages.to_awkward(), None).to_numpy(),
                genos.shape,
                geno_offsets,
            )
        else:
            dosages = None

        variants = RaggedVariants(alts, v_starts, ilens, dosages)

        return variants

    @overload
    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: NDArray[np.bool_] | None,
        keep_offsets: NDArray[np.int64] | None,
        annotate: Literal[False],
    ) -> Ragged[np.bytes_]: ...
    @overload
    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: NDArray[np.bool_] | None,
        keep_offsets: NDArray[np.int64] | None,
        annotate: Literal[True],
    ) -> tuple[Ragged[np.bytes_], Ragged[np.int32], Ragged[np.int32]]: ...

    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: NDArray[np.bool_] | None,
        keep_offsets: NDArray[np.int64] | None,
        annotate: bool,
    ) -> (
        Ragged[np.bytes_]
        | tuple[Ragged[np.bytes_], Ragged[V_IDX_TYPE], Ragged[np.int32]]
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
        assert self.reference is not None

        haps = Ragged.from_offsets(
            np.empty(out_offsets[-1], np.uint8), shifts.shape, out_offsets
        )

        if annotate:
            annot_v_idxs = Ragged.from_offsets(
                np.empty(out_offsets[-1], V_IDX_TYPE), shifts.shape, out_offsets
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
            geno_v_idxs=self.genotypes.data,
            v_starts=self.variants.v_starts,
            ilens=self.variants.ilens,
            alt_alleles=self.variants.alts.data.view(np.uint8),
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


class TrackType(enum.Enum):
    SAMPLE = enum.auto()
    ANNOT = enum.auto()


@define
class Tracks(Reconstructor[Ragged[np.float32]]):
    intervals: dict[str, RaggedIntervals]
    """The intervals in the dataset. This is memory mapped."""
    active_tracks: dict[str, TrackType]
    available_tracks: dict[str, TrackType]

    def with_tracks(self, tracks: str | Iterable[str] | None) -> Tracks:
        if tracks is None:
            return evolve(self, active_tracks={})

        if isinstance(tracks, str):
            _tracks = [tracks]
        else:
            _tracks = tracks

        if missing := list(set(_tracks) - set(self.intervals)):
            raise ValueError(f"Missing tracks: {missing}")

        tracks = {t: self.available_tracks[t] for t in _tracks}

        return evolve(self, active_tracks=tracks)

    @classmethod
    def from_path(cls, path: Path, n_regions: int, n_samples: int):
        strack_dir = path / "intervals"
        atrack_dir = path / "annot_intervals"

        available_tracks: list[str] = []
        if strack_dir.exists():
            for p in (path / "intervals").iterdir():
                if len(list(p.iterdir())) == 0:
                    p.rmdir()
                else:
                    available_tracks.append(p.name)
            available_tracks.sort()

        available_annots: list[str] = []
        if atrack_dir.exists():
            for p in (path / "annot_intervals").iterdir():
                if len(list(p.iterdir())) == 0:
                    p.rmdir()
                else:
                    available_annots.append(p.name)
            available_annots.sort()

        if name_clash := set(available_tracks) & set(available_annots):
            raise ValueError(
                f"Found sample and annotation tracks with the same name: {name_clash}"
            )

        intervals: dict[str, RaggedIntervals] | None = {}

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
                itvs, (n_regions, n_samples), offsets
            )

        for track in available_annots:
            itvs = np.memmap(
                path / "annot_intervals" / track / "intervals.npy",
                dtype=INTERVAL_DTYPE,
                mode="r",
            )
            offsets = np.memmap(
                path / "annot_intervals" / track / "offsets.npy",
                dtype=np.int64,
                mode="r",
            )
            intervals[track] = RaggedIntervals.from_offsets(itvs, n_regions, offsets)

        all_tracks = dict(
            zip(available_tracks, itertools.repeat(TrackType.SAMPLE))
        ) | dict(zip(available_annots, itertools.repeat(TrackType.ANNOT)))

        return cls(intervals, all_tracks, all_tracks)

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> Ragged[np.float32]:
        batch_size = len(idx)

        if isinstance(output_length, int):
            out_lengths = track_lengths = np.full(batch_size, output_length)
        else:
            lengths = regions[:, 2] - regions[:, 1]
            out_lengths = track_lengths = lengths

        # (b [p])
        out_ofsts_per_t = lengths_to_offsets(out_lengths)
        track_ofsts_per_t = lengths_to_offsets(track_lengths)
        # caller accounts for ploidy
        n_per_track = out_ofsts_per_t[-1]
        # ragged (b t [p] l)
        out = np.empty(len(self.active_tracks) * n_per_track, np.float32)
        out_lens = repeat(out_lengths, "b -> b t", t=len(self.active_tracks))
        out_offsets = lengths_to_offsets(out_lens)

        for track_ofst, (name, tracktype) in enumerate(self.active_tracks.items()):
            intervals = self.intervals[name]
            # (b t l) ragged
            _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]

            if tracktype is TrackType.SAMPLE:
                o_idx = idx
            else:
                o_idx = r_idx

            intervals_to_tracks(
                offset_idxs=o_idx,
                starts=regions[:, 1],
                intervals=intervals.data,
                itv_offsets=intervals.offsets,
                out=_out,
                out_offsets=track_ofsts_per_t,
            )

        out_shape = (len(idx), len(self.active_tracks))

        # ragged (b t l)
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
        regions = regions.copy()
        regions[:, 1] -= max_jitter
        regions[:, 2] += max_jitter
        r_idx = np.arange(n_regions)[:, None]
        s_idx = np.arange(n_samples)
        # (r s) -> (r*s)
        ds_idx = np.ravel_multi_index((r_idx, s_idx), idxer.full_shape).ravel()
        r_idx, s_idx = np.unravel_index(ds_idx, idxer.full_shape)
        if haps is not None:
            # extend ends by max hap diff to match write implementation
            regions[:, 2] += (
                haps._haplotype_ilens(ds_idx, regions, True)
                .reshape(n_regions, n_samples, haps.genotypes.shape[-1])
                .max((1, 2))
                .clip(min=0)
            )
        lengths = regions[:, 2] - regions[:, 1]
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
                ds_idx, _, _ = idxer.parse_idx((ir_idx, is_idx))
                r_idx, s_idx = np.unravel_index(ds_idx, idxer.full_shape)

                pbar.set_description("Writing (decompressing)")
                # (r*s)
                _regions = regions[r_idx]
                # (r*s+1)
                offsets = lengths_to_offsets(_regions[:, 2] - _regions[:, 1])
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

        return self.from_path(path, len(regions), n_samples).with_tracks(
            self.active_tracks
        )


@define
class RefTracks(Reconstructor[tuple[Ragged[np.bytes_], Ragged[np.float32]]]):
    seqs: Ref
    tracks: Tracks

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> tuple[Ragged[np.bytes_], Ragged[np.float32]]:
        seqs = self.seqs(
            idx=idx,
            r_idx=r_idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
        )
        tracks = self.tracks(
            idx=idx,
            r_idx=r_idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
        )
        return seqs, tracks


@define
class HapsTracks(Reconstructor[tuple[_H, Ragged[np.float32]]]):
    haps: Haps[_H]
    tracks: Tracks

    def to_kind(self, kind: type[_NewH]) -> HapsTracks[_NewH]:
        haps = self.haps.to_kind(kind)
        return cast(HapsTracks[_NewH], evolve(self, haps=haps))

    def __call__(
        self,
        idx: NDArray[np.integer],  # (b)
        r_idx: NDArray[np.integer],  # (b)
        regions: NDArray[np.int32],  # (b 3)
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
    ) -> tuple[_H, Ragged[np.float32]]:
        lengths = regions[:, 2] - regions[:, 1]

        # ragged (b p l), (b p), (b p), (b*p*v), (b*p+1), (b p)
        haps, geno_idx, shifts, diffs, hap_lengths, keep, keep_offsets = (
            self.haps.get_haps_and_shifts(
                idx=idx,
                regions=regions,
                output_length=output_length,
                rng=rng,
                deterministic=deterministic,
            )
        )

        if isinstance(output_length, int):
            # (b p)
            out_lengths = np.full_like(hap_lengths, output_length)
        else:
            # (b p)
            out_lengths = hap_lengths

        # (b) = lengths (b) + max deletion length across ploidy (b p) -> (b)
        track_lengths = lengths - diffs.clip(max=0).min(1)

        # (b*p+1)
        out_ofsts_per_t = lengths_to_offsets(out_lengths)
        # (b+1)
        track_ofsts_per_t = lengths_to_offsets(track_lengths)
        n_per_track = out_ofsts_per_t[-1]
        # ragged (b t p l)
        out = np.empty(len(self.tracks.active_tracks) * n_per_track, np.float32)
        out_lens = repeat(out_lengths, "b p -> b t p", t=len(self.tracks.active_tracks))
        out_offsets = lengths_to_offsets(out_lens)

        for track_ofst, (name, tracktype) in enumerate(
            self.tracks.active_tracks.items()
        ):
            intervals = self.tracks.intervals[name]

            # ragged (b l)
            _tracks = np.empty(track_ofsts_per_t[-1], np.float32)

            if tracktype is TrackType.SAMPLE:
                o_idx = idx
            else:
                o_idx = r_idx

            intervals_to_tracks(
                offset_idxs=o_idx,  # (b)
                starts=regions[:, 1],  # (b)
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
                geno_v_idxs=self.haps.genotypes.data,  # (r*s*p*v)
                geno_offsets=self.haps.genotypes.offsets,  # (r*s*p+1)
                v_starts=self.haps.variants.v_starts,  # (tot_v)
                ilens=self.haps.variants.ilens,  # (tot_v)
                tracks=_tracks,  # ragged (b l)
                track_offsets=track_ofsts_per_t,  # (b+1)
                keep=keep,  # (b*p*v)
                keep_offsets=keep_offsets,  # (b*p+1)
            )

        out_shape = (
            len(idx),
            len(self.tracks.active_tracks),
            self.haps.genotypes.shape[-1],
        )

        # ragged (b t [p] l)
        tracks = Ragged.from_offsets(out, out_shape, out_offsets)

        return haps, tracks
