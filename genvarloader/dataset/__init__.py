import json
from pathlib import Path
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numba as nb
import numpy as np
import polars as pl
from attrs import define, evolve, field
from loguru import logger
from numpy.typing import NDArray

from ..torch import get_dataloader
from ..utils import read_bedlike, with_length
from ..variants import VLenAlleles
from .genotypes import (
    DenseGenotypes,
    get_diffs,
    padded_slice,
    reconstruct_haplotypes_from_dense,
)
from .intervals import Intervals, intervals_to_tracks
from .reference import Reference
from .tracks import GenomeTrack, shift_and_realign_tracks  # noqa: F401
from .utils import regions_to_bed, subset_to_full_raveled_mapping

try:
    import torch
    import torch.utils.data as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

Idx = Union[int, np.integer, Sequence[int], NDArray[np.integer], slice]
ListIdx = Union[Sequence[int], NDArray[np.integer]]


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


@define(frozen=True)
class Dataset:
    """A dataset of genotypes, reference sequences, and intervals. Note: this class is not meant to be instantiated directly.
    Use the `open` or `open_with_settings` class methods to create a dataset after writing the data with genvarloader.write()
    or the genvarloader CLI.
    """

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
    return_sequences: Literal[False, "reference", "haplotypes"]
    ploidy: Optional[int] = None
    _reference: Optional[Reference] = None
    _variants: Optional[_Variants] = None
    has_intervals: bool = False
    return_tracks: bool = False
    _genotypes: Optional["DenseGenotypes"] = None
    _intervals: Optional["Intervals"] = None
    transform: Optional[Callable] = None
    _idx_map: Optional[NDArray[np.intp]] = None
    _jitter: Optional[int] = None
    return_indices: bool = False
    transformed_intervals: Optional[str] = None
    _track_transform: Optional[Callable] = None
    _genome_tracks: Dict[str, GenomeTrack] = field(factory=dict)

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]] = None,
    ):
        """Open a dataset from a path. If no reference genome is provided, the dataset can only yield tracks.

        Parameters
        ----------
        path : Union[str, Path]
            The path to the dataset.
        reference : Optional[Union[str, Path]], optional
            The path to the reference genome, by default None
        """
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
            logger.warning(
                dedent(
                    """
                    Genotypes found but no reference genome provided. This is required to reconstruct haplotypes.
                    No reference or haplotype sequences can be returned by this dataset instance.
                    """
                )
            )
            _reference = None
            has_reference = False
        elif reference is not None:
            logger.info(
                "Loading reference genome into memory. This typically has a modest memory footprint (a few GB) and greatly improves performance."
            )
            _reference = Reference.from_path_and_contigs(reference, contigs)
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
            return_sequences = "haplotypes"
        elif has_reference:
            return_sequences = "reference"
        else:
            return_sequences = False

        if has_intervals:
            return_tracks = True
        else:
            return_tracks = False

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
            _reference=_reference,
            _variants=variants,
            has_intervals=has_intervals,
            return_sequences=return_sequences,
            return_tracks=return_tracks,
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
        return_sequences: Optional[Literal[False, "reference", "haplotypes"]] = None,
        return_tracks: Optional[bool] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
        transformed_intervals: Optional[str] = None,
        extra_tracks: Optional[Dict[str, GenomeTrack]] = None,
    ):
        if return_sequences is False:
            reference = None
        ds = cls.open(path, reference).with_settings(
            samples=samples,
            regions=regions,
            return_sequences=return_sequences,
            return_tracks=return_tracks,
            transform=transform,
            seed=seed,
            jitter=jitter,
            return_indices=return_indices,
            transformed_intervals=transformed_intervals,
            extra_tracks=extra_tracks,
        )
        return ds

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    @property
    def has_genotypes(self) -> bool:
        return self._variants is not None

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

    @property
    def haplotypes(self):
        """Dataset that only returns haplotypes, if available."""
        return self.with_settings(return_sequences="haplotypes", return_tracks=False)

    @property
    def reference(self):
        """Dataset that only returns reference sequences, if available."""
        return self.with_settings(return_sequences="reference", return_tracks=False)

    @property
    def tracks(self):
        """Dataset that only returns tracks, if available."""
        return self.with_settings(return_sequences=False, return_tracks=True)

    def __len__(self) -> int:
        return self.n_samples * self.n_regions

    @property
    def available_transformed_intervals(self):
        if not self.has_intervals:
            avail = None
        else:
            avail = [p.stem[:-1] for p in (self.path / "intervals").glob("*_.npy")]
        return avail

    def add_transformed_intervals(
        self,
        name: str,
        transform: Callable[
            [pl.DataFrame, NDArray[np.intp], NDArray[np.uint32], NDArray[np.float32]],
            NDArray[np.float32],
        ],
        max_mem: int = 2**30,
    ):
        """Add transformed interval data to the dataset, writing them to disk with the given name.

        Note: only transformations that would modify non-zero values of tracks are applicable. For example, if the intent
        is to add a constant to a track, this would not be applicable since zero values are not stored in the intervals, and
        thus would not be transformed.

        Parameters
        ----------
        name : str
            The name of the transformed intervals.
        transform : Callable
            A function that takes a DataFrame of regions, sample indices, interval starts and
            ends, and values and returns new values.
        """

        if not self.has_intervals:
            raise ValueError(
                "No intervals found. Cannot add transformed intervals since intervals are required to add transformed intervals."
            )

        if self._intervals is None:
            all_intervals = self.init_intervals()
        else:
            all_intervals = self._intervals

        # 2 uint32 for positions, 1 float32 for value
        regions = self.get_bed()
        max_intervals_per_batch = max_mem / (2 * 4) / 4
        split_indices = (
            np.diff(
                (all_intervals.offsets[1:] % max_intervals_per_batch).astype(np.int32)
            )
            < 0
        ).nonzero()[0] + 1
        split_indices = np.r_[0, split_indices, len(all_intervals.offsets) - 1]
        last_offset = 0
        for offset_s, offset_e in zip(split_indices[:-1], split_indices[1:]):
            s_idxs, r_idxs = np.unravel_index(
                np.arange(offset_s, offset_e, dtype=np.intp), self._full_shape
            )
            interval_s, interval_e = (
                all_intervals.offsets[offset_s],
                all_intervals.offsets[offset_e],
            )
            intervals = all_intervals.intervals[interval_s:interval_e]
            values = all_intervals.values[interval_s:interval_e]
            transformed_intervals = transform(
                regions[r_idxs], s_idxs, intervals, values
            )
            out = np.memmap(
                self.path / "intervals" / f"{name}_.npy",
                dtype=np.float32,
                mode="w+",
                shape=len(transformed_intervals),
                offset=last_offset,
            )
            out[:] = transformed_intervals
            out.flush()
            last_offset += out.nbytes

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

        return evolve(
            self, sample_idxs=sample_idxs, region_idxs=region_idxs, _idx_map=idx_map
        )

    def to_full_dataset(self):
        sample_idxs = np.arange(len(self._samples), dtype=np.intp)
        region_idxs = np.arange(len(self._regions), dtype=np.intp)
        return evolve(
            self, sample_idxs=sample_idxs, region_idxs=region_idxs, _idx_map=None
        )

    def with_settings(
        self,
        samples: Optional[Sequence[str]] = None,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
        return_sequences: Optional[Literal[False, "reference", "haplotypes"]] = None,
        return_tracks: Optional[bool] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
        transformed_intervals: Optional[str] = None,
        extra_tracks: Optional[Dict[str, GenomeTrack]] = None,
    ):
        """Modify settings of the dataset, returning a new dataset without modifying the old one.

        Parameters
        ----------
        samples : Optional[Sequence[str]], optional
            The samples to subset to, by default None
        regions : Optional[Union[str, Path, pl.DataFrame]], optional
            The regions to subset to, by default None
        sequence_mode : Optional[Literal["reference", "haplotypes"]], optional
            The sequence mode to set, by default None. Set this to False to disable returning sequences entirely.
        track_mode : Optional[bool], optional
            The track mode to set, by default None
        transform : Optional[Callable], optional
            The transform to set, by default None
        seed : Optional[int], optional
            The seed to set, by default None
        jitter : Optional[int], optional
            The jitter to set, by default None
        return_indices : Optional[bool], optional
            Whether to return indices, by default None
        transformed_intervals : Optional[str], optional
            The transformed intervals to set, by default None
        """
        ds = self
        to_evolve: Dict[str, Any] = {}

        if samples is not None or regions is not None:
            ds = ds.subset_to(samples, regions)

        if return_sequences is not None:
            if return_sequences == "haplotypes" and not self.has_genotypes:
                raise ValueError(
                    "No genotypes found. Cannot be set to yield haplotypes since genotypes are required to yield haplotypes."
                )
            if return_sequences == "reference" and not self.has_reference:
                raise ValueError(
                    "No reference found. Cannot be set to yield reference sequences since reference is required to yield reference sequences."
                )
            to_evolve["return_sequences"] = return_sequences

        if return_tracks is not None:
            if return_tracks:
                if not self.has_intervals:
                    raise ValueError(
                        "No intervals found. Cannot be set to yield tracks since intervals are required to yield tracks."
                    )
                to_evolve["return_tracks"] = True
            else:
                to_evolve["return_tracks"] = False

        if transform is not None:
            to_evolve["transform"] = transform

        if seed is not None:
            to_evolve["rng"] = np.random.default_rng(seed)

        if jitter is not None:
            if jitter < 0:
                raise ValueError("Jitter must be a non-negative integer.")
            elif jitter > self.max_jitter:
                raise ValueError(
                    f"Jitter must be less than or equal to the maximum jitter of the dataset ({self.max_jitter})."
                )
            to_evolve["_jitter"] = jitter

        if return_indices is not None:
            to_evolve["return_indices"] = return_indices

        if transformed_intervals is not None:
            if self.available_transformed_intervals is None:
                raise ValueError(
                    "No transformed intervals available. Use the add_transformed_intervals method to add transformed intervals."
                )
            elif transformed_intervals not in self.available_transformed_intervals:
                raise ValueError(
                    f"Transformed intervals {transformed_intervals} not found. Available transformed intervals: {self.available_transformed_intervals}"
                )
            to_evolve["transformed_intervals"] = transformed_intervals

        if extra_tracks is not None:
            to_evolve["_genome_tracks"] = extra_tracks

        return evolve(ds, **to_evolve)

    def to_dataset(self):
        """Convert the dataset to a map-style PyTorch Dataset."""
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
        """Convert the dataset to a PyTorch DataLoader."""
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

    def isel(self, samples: Idx, regions: Idx):
        """Select a subset of samples and regions from the dataset.

        Parameters
        ----------
        samples : ListIdx
            The indices of the samples to select.
        regions : ListIdx
            The indices of the regions to select.
        """
        if isinstance(samples, slice):
            if samples.stop is None:
                samples = slice(samples.start, len(self._samples))
            _samples = np.arange(
                samples.start, samples.stop, samples.step, dtype=np.intp
            )
        else:
            _samples = samples

        if isinstance(regions, slice):
            if regions.stop is None:
                regions = slice(regions.start, len(self._regions))
            _regions = np.arange(
                regions.start, regions.stop, regions.step, dtype=np.intp
            )
        else:
            _regions = regions

        ds_idxs = np.ravel_multi_index((_samples, _regions), self.shape)
        return self[ds_idxs]

    def sel(self, samples: List[str], regions: pl.DataFrame):
        """Select a subset of samples and regions from the dataset.

        Parameters
        ----------
        samples : List[str]
            The names of the samples to select.
        regions : pl.DataFrame
            The regions to select.
        """
        s_to_i = dict(zip(self._samples, range(len(self._samples))))
        sample_idxs = np.array([s_to_i[s] for s in samples], np.intp)
        region_idxs = (
            with_length(regions, self.region_length)
            .join(
                self.get_bed().with_row_count(),
                on=["chrom", "chromStart", "chromEnd"],
                how="left",
            )
            .get_column("row_nr")
        )
        if (n_missing := region_idxs.is_null().sum()) > 0:
            raise ValueError(f"{n_missing} regions not found in the dataset.")
        region_idxs = region_idxs.to_numpy()
        ds_idxs = np.ravel_multi_index((sample_idxs, region_idxs), self.shape)
        return self[ds_idxs]

    def __getitem__(self, idx: Idx) -> Union[NDArray, Tuple[NDArray, ...]]:
        """Get a batch of haplotypes and tracks or intervals and tracks.

        Parameters
        ----------
        idx: Idx
            The index or indices to get. If a single index is provided, the output will be squeezed.
        """
        to_evolve = {}
        if self._genotypes is None and self.return_sequences == "haplotypes":
            to_evolve["genotypes"] = self.init_genotypes()
        if self._intervals is None and self.return_tracks:
            to_evolve["intervals"] = self.init_intervals(self.transformed_intervals)
        self = evolve(self, **to_evolve)

        if isinstance(idx, (int, np.integer)):
            _idx = [idx]
            squeeze = True
        elif isinstance(idx, slice):
            if idx.stop is None:
                idx = slice(idx.start, len(self))
            _idx = np.r_[idx]
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

        if self.return_sequences == "haplotypes":
            if TYPE_CHECKING:
                assert self._genotypes is not None
                assert self._variants is not None

            genos = self._genotypes[s_idx, r_idx]
            shifts = self.get_shifts(genos, self._variants.sizes)
            out.append(self.get_haplotypes(genos, regions, shifts))
        elif self.return_sequences == "reference":
            if TYPE_CHECKING:
                assert self._reference is not None
            shifts = None
            out.append(
                get_reference(
                    regions,
                    self._reference.reference,
                    self._reference.offsets,
                    self.region_length,
                    self._reference.pad_char,
                ).view("S1")
            )
        else:
            shifts = None

        if self.return_tracks:
            out.append(self.get_tracks(_idx, r_idx, shifts))

        if self.jitter > 0:
            start = self.rng.integers(
                self.max_jitter - self.jitter, self.max_jitter + 1
            )
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
        genotypes = DenseGenotypes(
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
        return genotypes

    def init_intervals(self, transform: Optional[str] = None):
        if transform is None:
            values_file = "values.npy"
        else:
            values_file = f"{transform}_.npy"

        intervals = Intervals(
            np.memmap(
                self.path / "intervals" / "intervals.npy",
                shape=(self.n_intervals, 2),
                dtype=np.uint32,
                mode="r",
            ),
            np.memmap(
                self.path / "intervals" / values_file,
                dtype=np.float32,
                mode="r",
            ),
            np.memmap(
                self.path / "intervals" / "offsets.npy",
                dtype=np.uint32,
                mode="r",
            ),
        )

        return intervals

    def get_shifts(self, genos: "DenseGenotypes", variant_sizes: NDArray[np.int32]):
        diffs = get_diffs(genos.first_v_idxs, genos.offsets, genos.genos, variant_sizes)
        shifts = self.rng.integers(0, diffs + 1, dtype=np.uint32)
        return shifts

    def get_haplotypes(
        self,
        genos: "DenseGenotypes",
        regions: NDArray[np.int32],
        shifts: NDArray[np.uint32],
    ):
        if TYPE_CHECKING:
            assert self._reference is not None
            assert self._variants is not None
            assert self.ploidy is not None

        haps = np.empty((len(regions), self.ploidy, self.region_length), np.uint8)
        reconstruct_haplotypes_from_dense(
            haps,
            regions,
            shifts,
            genos.first_v_idxs,
            genos.offsets,
            genos.genos,
            self._variants.positions,
            self._variants.sizes,
            self._variants.alts.alleles.view(np.uint8),
            self._variants.alts.offsets,
            self._reference.reference,
            self._reference.offsets,
            self._reference.pad_char,
        )
        return haps.view("S1")

    def get_tracks(
        self,
        ds_idx: NDArray[np.intp],
        r_idx: NDArray[np.intp],
        shifts: Optional[NDArray[np.uint32]] = None,
        genos: Optional["DenseGenotypes"] = None,
    ):
        if TYPE_CHECKING:
            assert self._intervals is not None

        tracks = intervals_to_tracks(
            ds_idx,
            r_idx,
            self.regions,
            self._intervals.intervals,
            self._intervals.values,
            self._intervals.offsets,
            self.region_length,
        )

        if shifts is not None and genos is not None:
            if TYPE_CHECKING:
                assert self.ploidy is not None
                assert self._variants is not None

            out = np.empty((len(r_idx), self.ploidy, self.region_length), np.float32)
            tracks = shift_and_realign_tracks(
                regions=self.regions[r_idx],
                positions=self._variants.positions,
                sizes=self._variants.sizes,
                first_v_idxs=genos.first_v_idxs,
                offsets=genos.offsets,
                genos=genos.genos,
                shifts=shifts,
                tracks=tracks,
                out=out,
            )

        return tracks

    def get_bed(self):
        """Get a polars DataFrame of the regions in the dataset, corresponding to the coordinates
        used when writing the dataset. In other words, each region will have length output_length
        + 2 * max_jitter."""
        bed = regions_to_bed(self.regions, self.contigs)
        # Need to do this because the regions file currently maps to the maximum length required
        # based on all possible indels, so some regions have length > output_length + 2 * max_jitter.
        # TODO: future implementation that doesn't use max deletion length will not need to do this
        # that implementation will just use the actual deletion length and has access to the entire reference
        # genome since it's being kept in-memory
        # likewise, the intervals corresponding to everything needed will be pre-computed and available
        # important not to prune the intervals though since a "reference" mode should be supported too
        # this also means the intervals -> values function will need to be updated to handle this and
        # only use the intervals that are necessary
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
class TorchDataset(td.Dataset):  # type: ignore
    dataset: Dataset

    def __attrs_pre_init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: Idx) -> Union[NDArray, Tuple[NDArray, ...]]:
        batch = self.dataset[idx]
        return batch
