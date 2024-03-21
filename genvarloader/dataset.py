import json
from dataclasses import dataclass, replace
from pathlib import Path
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numba as nb
import numpy as np
import polars as pl
from attrs import define
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from .fasta import Fasta
from .torch import get_dataloader
from .util import normalize_contig_name, read_bedlike, with_length
from .variants import VLenAlleles

try:
    import torch
    import torch.utils.data as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

Idx = Union[int, np.integer, Sequence[int], NDArray[np.integer], slice]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


@define
class _Reference:
    reference: NDArray[np.uint8]
    contigs: List[str]
    offsets: NDArray[np.uint64]
    pad_char: int

    @classmethod
    def from_path_and_contigs(cls, fasta: Union[str, Path], contigs: List[str]):
        _fasta = Fasta("ref", fasta, "N")
        contigs = cast(
            List[str],
            [normalize_contig_name(c, _fasta.contigs) for c in contigs],
        )
        _fasta.sequences = _fasta._get_contigs(contigs)
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
    def from_dataframe(cls, variants: Union[str, Path, pl.DataFrame]):
        if isinstance(variants, (str, Path)):
            variants = pl.read_ipc(variants)
        return cls(
            variants["POS"].to_numpy(),
            variants["ILEN"].to_numpy(),
            VLenAlleles.from_polars(variants["ALT"]),
        )


@dataclass(slots=True)
class Dataset:
    path: Path
    max_jitter: int
    n_variants: int
    n_intervals: int
    region_length: int
    contigs: List[str]
    _samples: NDArray[np.str_]
    _regions: NDArray[np.int32]
    rng: np.random.Generator
    sample_idxs: NDArray[np.intp]
    region_idxs: NDArray[np.intp]
    ploidy: Optional[int] = None
    reference: Optional[_Reference] = None
    variants: Optional[_Variants] = None
    has_intervals: bool = False
    sequence_mode: Optional[Literal["reference", "haplotypes"]] = None
    track_mode: bool = False
    genotypes: Optional["Genotypes"] = None
    intervals: Optional["Intervals"] = None
    transform: Optional[Callable] = None
    _idx_map: Optional[NDArray[np.intp]] = None
    _jitter: Optional[int] = None
    return_indices: bool = False

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]] = None,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        samples: NDArray[np.str_] = np.load(path / "samples.npy")
        regions: NDArray[np.int32] = np.load(path / "regions.npy")

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        contigs = metadata["contigs"]
        region_length = int(metadata["region_length"])
        ploidy = int(metadata.get("ploidy", 0))
        n_variants = int(metadata.get("n_variants", 0))
        n_intervals = int(metadata.get("n_intervals", 0))
        max_jitter = int(metadata.get("max_jitter", 0))
        rng = np.random.default_rng()

        if reference is None and n_variants > 0:
            raise ValueError(
                "Genotypes found but no reference genome provided. This is required to reconstruct haplotypes."
            )
        elif reference is not None:
            logger.info(
                "Loading reference genome into memory. This typically has a modest memory footprint (a few GB) and greatly improves performance."
            )
            _reference = _Reference.from_path_and_contigs(reference, contigs)
            has_reference = True
        else:
            _reference = None
            has_reference = False

        if n_variants > 0:
            variants = _Variants.from_dataframe(path / "variants.arrow")
            has_genotypes = True
        else:
            variants = None
            has_genotypes = False

        if n_intervals > 0:
            has_intervals = True
        else:
            has_intervals = False

        if has_reference and has_genotypes:
            sequence_mode = "haplotypes"
        elif has_reference:
            sequence_mode = "reference"
        else:
            sequence_mode = None

        if has_intervals:
            track_mode = True
        else:
            track_mode = False

        dataset = cls(
            path=path,
            max_jitter=max_jitter,
            n_variants=n_variants,
            n_intervals=n_intervals,
            region_length=region_length,
            contigs=contigs,
            _samples=samples,
            _regions=regions,
            rng=rng,
            sample_idxs=np.arange(len(samples), dtype=np.intp),
            region_idxs=np.arange(len(regions), dtype=np.intp),
            ploidy=ploidy,
            reference=_reference,
            variants=variants,
            has_intervals=has_intervals,
            sequence_mode=sequence_mode,
            track_mode=track_mode,
        )

        logger.info(f"\n{str(dataset)}")

        return dataset

    @classmethod
    def open_with_settings(
        cls,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]] = None,
        samples: Optional[Sequence[str]] = None,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
        sequence_mode: Optional[Literal["reference", "haplotypes"]] = None,
        track_mode: Optional[bool] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
    ):
        ds = cls.open(path, reference).with_settings(
            samples=samples,
            regions=regions,
            sequence_mode=sequence_mode,
            track_mode=track_mode,
            transform=transform,
            seed=seed,
            jitter=jitter,
            return_indices=return_indices,
        )
        return ds

    @property
    def has_reference(self) -> bool:
        return self.reference is not None

    @property
    def has_genotypes(self) -> bool:
        return self.variants is not None

    @property
    def samples(self):
        if self._idx_map is None:
            return self._samples
        else:
            return self._samples[self.sample_idxs]

    @property
    def regions(self):
        if self._idx_map is None:
            return self._regions
        else:
            return self._regions[self.region_idxs]

    @property
    def n_samples(self):
        return len(self.sample_idxs)

    @property
    def n_regions(self):
        return len(self.region_idxs)

    @property
    def shape(self):
        """Return the shape of the dataset. (n_samples, n_regions)"""
        return self.n_samples, self.n_regions

    @property
    def jitter(self):
        if self._jitter is None:
            return self.max_jitter
        else:
            return self._jitter

    @property
    def output_length(self):
        return self.region_length - 2 * self.jitter

    @property
    def _full_shape(self):
        """Return the full shape of the dataset, ignoring any subsetting. (n_samples, n_regions)"""
        return len(self._samples), len(self._regions)

    def __len__(self) -> int:
        return self.n_samples * self.n_regions

    def subset_to(
        self,
        samples: Optional[Sequence[str]] = None,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
    ):
        if regions is None and samples is None:
            return self

        if samples is not None:
            _samples = set(samples)
            if missing := _samples.difference(self._samples):
                raise ValueError(f"Samples {missing} not found in the dataset")
            sample_idxs = np.array(
                [i for i, s in enumerate(self.samples) if s in _samples], np.intp
            )
            if self._idx_map is not None:
                sample_idxs = self.sample_idxs[sample_idxs]
        else:
            sample_idxs = self.sample_idxs

        if regions is not None:
            if isinstance(regions, (str, Path)):
                regions = read_bedlike(regions)
            regions = with_length(regions, self.region_length)
            available_regions = self.get_bed()
            n_query_regions = regions.height
            region_idxs = (
                available_regions.with_row_count()
                .join(regions, on=["chrom", "chromStart", "chromEnd"])
                .get_column("row_nr")
                .sort()
                .to_numpy()
            )
            n_available_regions = len(region_idxs)
            if n_query_regions != len(region_idxs):
                raise ValueError(
                    f"Only {n_available_regions}/{n_query_regions} requested regions exist in the dataset."
                )
            if self._idx_map is not None:
                region_idxs = self.region_idxs[region_idxs]
        else:
            region_idxs = self.region_idxs

        idx_map = subset_to_full_raveled_mapping(
            self._full_shape,
            sample_idxs,
            region_idxs,
        )

        return replace(
            self, sample_idxs=sample_idxs, region_idxs=region_idxs, _idx_map=idx_map
        )

    def to_full_dataset(self):
        sample_idxs = np.arange(len(self._samples), dtype=np.intp)
        region_idxs = np.arange(len(self._regions), dtype=np.intp)
        return replace(
            self, sample_idxs=sample_idxs, region_idxs=region_idxs, _idx_map=None
        )

    def with_sequence_mode(self, mode: Optional[Literal["reference", "haplotypes"]]):
        if mode == "haplotypes" and not self.has_genotypes:
            raise ValueError(
                "No genotypes found. Cannot be set to yield haplotypes since genotypes are required to yield haplotypes."
            )
        if mode == "reference" and not self.has_reference:
            raise ValueError(
                "No reference found. Cannot be set to yield reference sequences since reference is required to yield reference sequences."
            )
        return replace(self, sequence_mode=mode)

    def with_tracks(self):
        if not self.has_intervals:
            raise ValueError(
                "No intervals found. Cannot be set to yield tracks since intervals are required to yield tracks."
            )
        return replace(self, track_mode=True)

    def without_tracks(self):
        return replace(self, track_mode=False)

    def with_transform(self, transform: Callable):
        return replace(self, transform=transform)

    def with_seed(self, seed: Optional[int]):
        return replace(self, rng=np.random.default_rng(seed))

    def with_jitter(self, jitter: int):
        if jitter < 0:
            raise ValueError("Jitter must be a non-negative integer.")
        elif jitter > self.max_jitter:
            raise ValueError(
                f"Jitter must be less than or equal to the maximum jitter of the dataset ({self.max_jitter})."
            )
        return replace(self, _jitter=jitter)

    def with_indices(self):
        return replace(self, return_indices=True)

    def with_settings(
        self,
        samples: Optional[Sequence[str]] = None,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
        sequence_mode: Optional[Literal["reference", "haplotypes"]] = None,
        track_mode: Optional[bool] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
    ):
        ds = self
        if samples is not None or regions is not None:
            ds = ds.subset_to(samples, regions)
        if sequence_mode is not None:
            ds = ds.with_sequence_mode(sequence_mode)
        if track_mode is not None:
            if track_mode:
                ds = ds.with_tracks()
            else:
                ds = ds.without_tracks()
        if transform is not None:
            ds = ds.with_transform(transform)
        if seed is not None:
            ds = ds.with_seed(seed)
        if jitter is not None:
            ds = ds.with_jitter(jitter)
        if return_indices is not None:
            ds = ds.with_indices()
        return ds

    def to_dataset(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )

        return TorchDataset(self)

    def to_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Union[td.Sampler, Iterable]] = None,  # type: ignore
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: Optional[Callable] = None,
        generator: Optional[torch.Generator] = None,  # type: ignore
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )

        return get_dataloader(
            dataset=self.to_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def __str__(self) -> str:
        return dedent(
            f"""
            GVL store {self.path.name}
            Original region length: {self.region_length - 2*self.max_jitter:,}
            Max jitter: {self.max_jitter:,}
            # of samples: {self.n_samples:,}
            # of regions: {self.n_regions:,}
            Has genotypes: {self.n_variants > 0}
            Has intervals: {self.n_intervals > 0}
            Is subset: {self._idx_map is not None}\
            """
        ).strip()

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, idx: Idx) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """Get a batch of haplotypes and tracks or intervals and tracks.

        Parameters
        ----------
        idx: Idx
            The index or indices to get. If a single index is provided, the output will be squeezed.
        """
        if self.genotypes is None and self.sequence_mode == "haplotypes":
            self.init_genotypes()
        if self.intervals is None and self.track_mode:
            self.init_intervals()

        if isinstance(idx, (int, np.integer)):
            _idx = [idx]
            squeeze = True
        elif isinstance(idx, slice):
            _idx = np.arange(self.n_samples * self.n_regions, dtype=np.uintp)[idx]
            squeeze = False
        else:
            _idx = idx
            squeeze = False

        _idx = np.asarray(_idx, dtype=np.intp)
        _idx[_idx < 0] += len(self)
        if self._idx_map is not None:
            _idx = self._idx_map[_idx]
        s_idx, r_idx = np.unravel_index(_idx, self._full_shape)

        regions = self._regions[r_idx]

        out: List[NDArray] = []

        if self.sequence_mode == "haplotypes":
            if TYPE_CHECKING:
                assert self.genotypes is not None
                assert self.variants is not None

            genos = self.genotypes[s_idx, r_idx]
            shifts = self.get_shifts(genos, self.variants.sizes)
            out.append(self.get_haplotypes(genos, regions, shifts))
        elif self.sequence_mode == "reference":
            if TYPE_CHECKING:
                assert self.reference is not None
            shifts = None
            out.append(
                get_reference(
                    regions,
                    self.reference.reference,
                    self.reference.offsets,
                    self.region_length,
                    self.reference.pad_char,
                ).view("S1")
            )
        else:
            shifts = None

        if self.track_mode:
            if TYPE_CHECKING:
                assert self.intervals is not None
            out.append(self.get_tracks(_idx, self.intervals, regions, shifts))

        if self.jitter > 0:
            start = self.rng.integers(0, self.jitter)
            out = [o[..., start : start + self.output_length] for o in out]

        if squeeze:
            out = [o.squeeze() for o in out]

        if self.return_indices:
            out.extend((_idx, s_idx, r_idx))

        if self.transform is not None:
            out = self.transform(*out)

        if len(out) == 1:
            _out = out[0]
        else:
            _out = out

        return _out  # type: ignore

    def init_genotypes(self):
        if TYPE_CHECKING:
            assert self.ploidy is not None
        n_samples = len(self._samples)
        self.genotypes = Genotypes(
            np.memmap(
                self.path / "genotypes" / "genotypes.npy",
                shape=(self.n_variants * n_samples, self.ploidy),
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
            n_samples,
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

    def get_shifts(self, genos: "Genotypes", variant_sizes: NDArray[np.int32]):
        diffs = get_diffs(genos.first_v_idxs, genos.offsets, genos.genos, variant_sizes)
        shifts = self.rng.integers(0, diffs + 1, dtype=np.uint32)
        return shifts

    def get_haplotypes(
        self,
        genos: "Genotypes",
        regions: NDArray[np.int32],
        shifts: NDArray[np.uint32],
    ):
        if TYPE_CHECKING:
            assert self.reference is not None
            assert self.variants is not None
            assert self.ploidy is not None

        haps = np.empty((len(regions), self.ploidy, self.region_length), np.uint8)
        reconstruct_haplotypes(
            haps,
            regions,
            shifts,
            genos.first_v_idxs,
            genos.offsets,
            genos.genos,
            self.variants.positions,
            self.variants.sizes,
            self.variants.alts.alleles.view(np.uint8),
            self.variants.alts.offsets,
            self.reference.reference,
            self.reference.offsets,
            self.reference.pad_char,
        )
        return haps.view("S1")

    def get_tracks(
        self,
        ds_idx: ListIdx,
        intervals: "Intervals",
        regions: NDArray[np.int32],
        shifts: Optional[NDArray[np.uint32]] = None,
    ):
        intervals = intervals[ds_idx]
        if shifts is not None:
            values = intervals_to_hap_values(
                regions,
                shifts,
                intervals.intervals,
                intervals.values,
                intervals.offsets,
                self.region_length,
            )
        else:
            values = intervals_to_values(
                regions,
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


def subset_to_full_raveled_mapping(
    full_shape: Tuple[int, int], ax1_indices: ArrayLike, ax2_indices: ArrayLike
):
    # Generate a grid of indices for the subset array
    row_indices, col_indices = np.meshgrid(ax1_indices, ax2_indices, indexing="ij")

    # Flatten the grid to get all combinations of row and column indices in the subset
    row_indices_flat = row_indices.ravel()
    col_indices_flat = col_indices.ravel()

    # Convert these subset indices to linear indices in the context of the full array
    # This leverages the fact that the linear index in a 2D array is given by: index = row * num_columns + column
    full_array_linear_indices = row_indices_flat * full_shape[1] + col_indices_flat

    return full_array_linear_indices


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


@define
class TorchDataset(td.Dataset):  # type: ignore
    dataset: Dataset

    def __attrs_pre_init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Idx) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        batch = self.dataset[idx]
        return batch
