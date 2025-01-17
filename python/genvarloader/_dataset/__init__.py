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
import seqpro as sp
from attrs import define, evolve
from einops import repeat
from loguru import logger
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .._torch import get_dataloader
from .._types import INTERVAL_DTYPE, Idx, Ragged, RaggedIntervals
from .._utils import _lengths_to_offsets, read_bedlike, with_length
from .._variants._records import VLenAlleles
from ._genotypes import (
    SparseGenotypes,
    SparseSomaticGenotypes,
    get_diffs_sparse,
    mark_keep_variants,
    padded_slice,
    reconstruct_haplotypes_from_sparse,
    reconstruct_haplotypes_from_sparse_somatic,
)
from ._intervals import intervals_to_tracks, tracks_to_intervals
from ._reference import Reference
from ._tracks import shift_and_realign_tracks_sparse
from ._utils import oidx_to_raveled_idx, regions_to_bed, splits_sum_le_value

try:
    import torch
    import torch.utils.data as td

    from ._torch import TorchDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    import torch
    import torch.utils.data as td

    from ._torch import TorchDataset


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


@define(frozen=True)
class Dataset:
    """A dataset of genotypes, reference sequences, and intervals. Note: this class is not meant to be instantiated directly.
    Use the :py:meth:`Dataset.open` class method to create a dataset after writing the data with :py:func:`genvarloader.write()`
    or the genvarloader CLI.
    """

    @classmethod
    def _open(
        cls,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]],
        deterministic: bool,
    ) -> "Dataset":
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
        regions: NDArray[np.int32] = np.load(path / "regions.npy")

        has_intervals = (path / "intervals").exists()
        has_genotypes = (path / "genotypes").exists()

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        samples: List[str] = metadata["samples"]
        contigs: List[str] = metadata["contigs"]
        region_length: int = metadata["region_length"]
        ploidy: Optional[int] = metadata.get("ploidy", None)
        max_jitter: int = metadata.get("max_jitter", 0)
        phased: Optional[bool] = metadata.get("phased", None)
        rng = np.random.default_rng()

        if reference is None and has_genotypes:
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

        if has_genotypes:
            variants = _Variants.from_table(path / "genotypes" / "variants.arrow")
        else:
            variants = None

        if has_intervals:
            tracks: List[str] = []
            for p in (path / "intervals").iterdir():
                if len(list(p.iterdir())) == 0:
                    p.rmdir()
                else:
                    tracks.append(p.name)
            tracks.sort()
            active_tracks = tracks
        else:
            tracks = []
            active_tracks = None

        if has_reference and has_genotypes:
            sequence_type = "haplotypes"
        elif has_reference:
            sequence_type = "reference"
        else:
            sequence_type = None

        dataset = cls(
            path=path,
            max_jitter=max_jitter,
            jitter=max_jitter,
            region_length=region_length,
            contigs=contigs,
            full_samples=samples,
            full_regions=regions,
            rng=rng,
            deterministic=deterministic,
            sample_idxs=np.arange(len(samples), dtype=np.intp),
            region_idxs=np.arange(len(regions), dtype=np.intp),
            ploidy=ploidy,
            reference=_reference,
            variants=variants,
            available_tracks=tracks,
            sequence_type=sequence_type,
            active_tracks=active_tracks,
            phased=phased,
        )

        logger.info(f"\n{str(dataset)}")

        return dataset

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]] = None,
        samples: Optional[Sequence[str]] = None,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
        return_sequences: Optional[Literal[False, "reference", "haplotypes"]] = None,
        return_tracks: Optional[Union[Literal[False], str, List[str]]] = None,
        transform: Optional[Union[Literal[False], Callable]] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
        deterministic: bool = False,
    ) -> "Dataset":
        """Open a dataset from a path. If no reference genome is provided, the dataset can only yield tracks.

        Parameters
        ----------
        path : Union[str, Path]
            The path to the dataset.
        reference : Optional[Union[str, Path]], optional
            The path to the reference genome, by default None
        samples : Optional[Sequence[str]], optional
            The samples to subset to, by default None
        regions : Optional[Union[str, Path, pl.DataFrame]], optional
            The regions to subset to, by default None
        return_sequences : Optional[Literal[False, "reference", "haplotypes"]], optional
            The sequence type to return. Set this to False to disable returning sequences entirely.
        return_tracks : Optional[Union[Literal[False], str, List[str]]], optional
            The tracks to return, by default None. Set this to False to disable returning tracks entirely.
        transform : Optional[Union[Literal[False], Callable]], optional
            The transform to set, by default None
        seed : Optional[int], optional
            The seed to set, by default None
        jitter : Optional[int], optional
            The jitter to set, by default None
        return_indices : Optional[bool], optional
            Whether to return indices, by default None
        deterministic : bool, optional
            Whether to use randomized or deterministic algorithms. If set to True, this will disable random
            shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
            and always apply the highest dosage group. Note that for unphased variants, this will mean not all possible haplotypes
            can be returned.
        """
        if return_sequences is False:
            _reference = None
        else:
            _reference = reference

        ds = cls._open(path, _reference, deterministic).with_settings(
            samples=samples,
            regions=regions,
            return_sequences=return_sequences,
            return_tracks=return_tracks,
            transform=transform,
            seed=seed,
            jitter=jitter,
            return_indices=return_indices,
        )

        if reference is None and ds.has_genotypes:
            logger.warning(
                "Genotypes found but no reference genome provided. This is required to reconstruct haplotypes."
                " No reference or haplotype sequences can be returned by this dataset instance."
            )

        return ds

    def with_settings(
        self,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
        samples: Optional[Sequence[str]] = None,
        return_sequences: Optional[Literal[False, "reference", "haplotypes"]] = None,
        return_tracks: Optional[Union[Literal[False], str, List[str]]] = None,
        transform: Optional[Union[Literal[False], Callable]] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
    ) -> "Dataset":
        """Modify settings of the dataset, returning a new dataset without modifying the old one.

        Parameters
        ----------
        regions : Optional[Union[str, Path, pl.DataFrame]], optional
            The regions to subset to, by default None
        samples : Optional[Sequence[str]], optional
            The samples to subset to, by default None
        return_sequences : Optional[Literal[False, "reference", "haplotypes"]], optional
            The sequence type to return. Set this to False to disable returning sequences entirely.
        return_tracks : Optional[Union[Literal[False], List[str], str]], optional
            The tracks to return, by default None. Set this to False to disable returning tracks entirely.
        transform : Optional[Union[Literal[False], Callable]], optional
            The transform to set, by default None
        seed : Optional[int], optional
            The seed to set, by default None
        jitter : Optional[int], optional
            The jitter to set, by default None
        return_indices : Optional[bool], optional
            Whether to return indices, by default None
        """
        ds = self
        to_evolve: Dict[str, Any] = {}

        if samples is not None or regions is not None:
            ds = ds.subset_to(regions=regions, samples=samples)

        if return_sequences is not None:
            if return_sequences == "haplotypes" and not self.has_genotypes:
                raise ValueError(
                    "No genotypes found. Cannot be set to yield haplotypes since genotypes are required to yield haplotypes."
                )
            if return_sequences == "reference" and not self.has_reference:
                raise ValueError(
                    "No reference found. Cannot be set to yield reference sequences since reference is required to yield reference sequences."
                )
            if return_sequences is False:
                to_evolve["sequence_type"] = None
            else:
                to_evolve["sequence_type"] = return_sequences

            # reset after changing sequence type
            to_evolve["genotypes"] = None

        if return_tracks is not None:
            if return_tracks is False:
                to_evolve["active_tracks"] = None
            else:
                if isinstance(return_tracks, str):
                    return_tracks = [return_tracks]
                if missing := set(return_tracks).difference(self.available_tracks):
                    raise ValueError(
                        f"Intervals {missing} not found. Available intervals: {self.available_tracks}"
                    )
                to_evolve["active_tracks"] = return_tracks

            # reset after changing active tracks
            to_evolve["intervals"] = None

        if transform is not None:
            if transform is False:
                transform = None
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
            to_evolve["jitter"] = jitter

        if return_indices is not None:
            to_evolve["return_indices"] = return_indices

        return evolve(ds, **to_evolve)

    path: Path
    """The path to the dataset."""

    max_jitter: int
    """The maximum jitter allowable by the underlying data written to disk."""

    jitter: int
    """The current jitter."""

    region_length: int
    """The length of the regions in the dataset corresponding to what was written to disk (i.e. output_length + 2 * max_jitter)."""

    contigs: List[str]
    """The unique contigs in the dataset."""

    full_samples: List[str]
    """The full list of samples in the dataset."""

    full_regions: NDArray[np.int32]
    """The full regions in the dataset."""

    rng: np.random.Generator
    """The random number generator used for jittering and shifting haplotypes that are longer than the output length."""

    sample_idxs: NDArray[np.intp]
    """The indices of the samples to subset to."""

    region_idxs: NDArray[np.intp]
    """The indices of the regions to subset to."""

    sequence_type: Optional[Literal["reference", "haplotypes"]]
    """The type of sequence to return."""

    active_tracks: Optional[List[str]]
    """The active tracks to return."""

    available_tracks: List[str]
    """The available tracks in the dataset."""

    deterministic: bool
    """Whether to use randomized or deterministic algorithms. If set to True, this will disable random
    shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
    and always apply the highest dosage group. Note that for unphased variants, this will mean not all possible haplotypes
    can be returned."""

    ploidy: Optional[int] = None
    """The ploidy of the dataset."""

    reference: Optional[Reference] = None
    """The reference genome. This is kept in memory."""

    variants: Optional[_Variants] = None
    """The variant sites in the dataset. This is kept in memory."""

    genotypes: Optional[Union[SparseGenotypes, SparseSomaticGenotypes]] = None
    """The genotypes in the dataset. This is memory mapped."""

    intervals: Optional[Dict[str, RaggedIntervals]] = None
    """The intervals in the dataset. This is memory mapped."""

    transform: Optional[Callable] = None
    """The transform to apply to the data."""

    idx_map: Optional[NDArray[np.intp]] = None
    """The map from the full dataset to the subset dataset."""

    return_indices: bool = False
    """Whether to return indices."""

    phased: Optional[bool] = None
    """Whether the genotypes are phased. Set to None if genotypes are not present."""

    @property
    def has_reference(self) -> bool:
        return self.reference is not None

    @property
    def has_genotypes(self) -> bool:
        return self.variants is not None

    @property
    def has_intervals(self) -> bool:
        return len(self.available_tracks) > 0

    @property
    def samples(self) -> List[str]:
        if self.idx_map is None:
            return self.full_samples
        else:
            return [self.full_samples[i] for i in self.sample_idxs]

    @property
    def regions(self) -> NDArray[np.int32]:
        if self.idx_map is None:
            return self.full_regions
        else:
            return self.full_regions[self.region_idxs]

    @property
    def n_samples(self) -> int:
        return len(self.sample_idxs)

    @property
    def n_regions(self) -> int:
        return len(self.region_idxs)

    @property
    def output_length(self) -> int:
        return self.region_length - 2 * self.jitter

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataset. (n_samples, n_regions)"""
        return self.n_regions, self.n_samples

    @property
    def full_shape(self) -> Tuple[int, int]:
        """Return the full shape of the dataset, ignoring any subsetting. (n_regions, n_samples)"""
        return len(self.full_regions), len(self.full_samples)

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def write_transformed_track(
        self,
        new_track: str,
        existing_track: str,
        transform: Callable[
            [NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], Ragged[np.float32]],
            Ragged[np.float32],
        ],
        max_mem: int = 2**30,
        overwrite: bool = False,
    ) -> "Dataset":
        """Write transformed tracks to the dataset.

        Parameters
        ----------
        new_track : str
            The name of the new track.
        existing_track : str
            The name of the existing track to transform.
        transform
            A function to apply to the existing track to get a new, transformed track.
            This will be done in chunks such that the tracks provided will not exceed :code:`max_mem`.
            The arguments given to the transform will be the dataset indices, region indices, and
            sample indices as numpy arrays and the tracks themselves as a :class:`Ragged` array with
            shape (regions, samples). The tracks must be a :class:`Ragged` array since regions may be
            different lengths to accomodate indels. This function should then return the transformed
            tracks as a :class:`Ragged` array with the same shape and lengths.
        max_mem : int, optional
            The maximum memory to use in bytes, by default 1 GiB (2**30 bytes)
        overwrite : bool, optional
            Whether to overwrite the existing track, by default False
        """
        if new_track == existing_track:
            raise ValueError(
                "New track name must be different from existing track name."
            )

        if existing_track not in self.available_tracks:
            raise ValueError(
                f"Requested existing track {existing_track} does not exist in this dataset."
            )

        if self.intervals is None:
            intervals = self._init_intervals(existing_track)[existing_track]
        else:
            intervals = self.intervals[existing_track]

        out_dir = self.path / "intervals" / new_track

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

        lengths = self.full_regions[:, 2] - self.full_regions[:, 1]
        # for each region:
        # bytes = (4 bytes / bp) * (bp / sample) * samples
        n_samples = len(self.full_samples)
        mem_per_region = 4 * lengths * n_samples
        splits = splits_sum_le_value(mem_per_region, max_mem)
        memmap_intervals_offset = 0
        memmap_offsets_offset = 0
        last_offset = 0
        with tqdm(total=len(splits) - 1) as pbar:
            for offset_s, offset_e in zip(splits[:-1], splits[1:]):
                r_idx = np.arange(offset_s, offset_e, dtype=np.intp)
                n_regions = len(r_idx)
                s_idx = np.arange(n_samples, dtype=np.intp)
                r_idx = repeat(r_idx, "r -> (r s)", s=n_samples)
                s_idx = repeat(s_idx, "s -> (r s)", r=n_regions)
                ds_idx = np.ravel_multi_index((r_idx, s_idx), self.full_shape)

                pbar.set_description("Writing (decompressing)")
                regions = self.full_regions[r_idx]
                # layout is (regions, samples) so all samples are local for statistics
                tracks = intervals_to_tracks(
                    ds_idx,
                    regions,
                    intervals.data,
                    intervals.offsets,
                )
                offsets = _lengths_to_offsets(regions[:, 2] - regions[:, 1], np.int64)
                tracks = Ragged.from_offsets(tracks, (n_regions, n_samples), offsets)

                pbar.set_description("Writing (transforming)")
                transformed_tracks = transform(ds_idx, r_idx, s_idx, tracks)
                np.testing.assert_equal(tracks.shape, transformed_tracks.shape)

                pbar.set_description("Writing (compressing)")
                itvs, interval_offsets = tracks_to_intervals(
                    regions, transformed_tracks.data, transformed_tracks.offsets
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

        return evolve(self, available_tracks=self.available_tracks + [new_track])

    def subset_to(
        self,
        regions: Optional[Union[str, Path, pl.DataFrame]] = None,
        samples: Optional[Sequence[str]] = None,
    ) -> "Dataset":
        """Subset the dataset to specific regions and/or samples."""
        if regions is None and samples is None:
            return self

        if samples is not None:
            _samples = set(samples)
            if missing := _samples.difference(self.full_samples):
                raise ValueError(f"Samples {missing} not found in the dataset")
            sample_idxs = np.array(
                [i for i, s in enumerate(self.full_samples) if s in _samples], np.intp
            )
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
        else:
            region_idxs = self.region_idxs

        idx_map = oidx_to_raveled_idx(
            row_idx=region_idxs,
            col_idx=sample_idxs,
            full_shape=self.full_shape,
        )

        return evolve(
            self, sample_idxs=sample_idxs, region_idxs=region_idxs, idx_map=idx_map
        )

    def to_full_dataset(self) -> "Dataset":
        sample_idxs = np.arange(len(self.full_samples), dtype=np.intp)
        region_idxs = np.arange(len(self.full_regions), dtype=np.intp)
        return evolve(
            self, sample_idxs=sample_idxs, region_idxs=region_idxs, idx_map=None
        )

    def to_dataset(self) -> "td.Dataset":
        """Convert the dataset to a map-style PyTorch Dataset."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )
        return TorchDataset(self)

    def to_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Union["td.Sampler", Iterable]] = None,  # type: ignore
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: Optional[Callable] = None,
        generator: Optional["torch.Generator"] = None,  # type: ignore
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> "td.DataLoader":
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
        if not self.has_genotypes:
            genotype_status = "None"
        elif self.phased:
            genotype_status = "Phased"
        else:
            genotype_status = "Unphased"

        return dedent(
            f"""
            GVL store {self.path.name}
            Is subset: {self.idx_map is not None}
            # of regions: {self.n_regions:,}
            # of samples: {self.n_samples:,}
            Original region length: {self.region_length - 2*self.max_jitter:,}
            Max jitter: {self.max_jitter:,}
            Genotypes available: {genotype_status}
            Tracks available: {self.available_tracks}\
            """
        ).strip()

    def __repr__(self) -> str:
        return str(self)

    def isel(
        self, regions: Idx, samples: Idx
    ) -> Union[NDArray, Tuple[NDArray, ...], Any]:
        """Eagerly select a subset of samples and regions from the dataset.

        Parameters
        ----------
        samples : ListIdx
            The indices of the samples to select.
        regions : ListIdx
            The indices of the regions to select.
        """
        if isinstance(regions, slice):
            start = 0 if regions.start is None else regions.start
            stop = len(self.regions) if regions.stop is None else regions.stop
            _regions = np.arange(start, stop, regions.step, dtype=np.intp)
        elif isinstance(regions, Sequence):
            _regions = np.asarray(regions, np.intp)
        else:
            _regions = regions

        if isinstance(samples, slice):
            start = 0 if samples.start is None else samples.start
            stop = len(self.samples) if samples.stop is None else samples.stop
            _samples = np.arange(start, stop, samples.step, dtype=np.intp)
        elif isinstance(samples, Sequence):
            _samples = np.asarray(samples, np.intp)
        else:
            _samples = samples

        if isinstance(_regions, (int, np.integer)) and _regions < 0:
            _regions += self.n_regions
        if isinstance(_samples, (int, np.integer)) and _samples < 0:
            _samples += self.n_samples

        if isinstance(_regions, np.ndarray) and isinstance(_samples, np.ndarray):
            _regions = _regions[:, None]

        ds_idxs = np.ravel_multi_index((_regions, _samples), self.shape)
        return self[ds_idxs]

    def sel(
        self,
        regions: Union[str, Tuple[str, int, int], pl.DataFrame],
        samples: Union[str, List[str]],
    ) -> Union[NDArray, Tuple[NDArray, ...], Any]:
        """Eagerly select a subset of samples and regions from the dataset.

        Parameters
        ----------
        regions : str, Tuple[str, int, int], pl.DataFrame
            The regions to select.
        samples : str, List[str]
            The names of the samples to select.
        """
        if isinstance(regions, str):
            try:
                idx = regions.rindex(":")
                contig = regions[:idx]
                start, end = map(int, regions[idx + 1 :].split("-"))
                regions = contig, start, end
            except ValueError:
                raise ValueError("Invalid region format. Must be chrom:start-end.")

        if isinstance(regions, tuple):
            contig, start, end = regions
            regions = pl.DataFrame(
                {
                    "chrom": [contig],
                    "chromStart": [start],
                    "chromEnd": [end],
                }
            )

        if isinstance(samples, str):
            samples = [samples]

        s_to_i = dict(zip(self.full_samples, range(len(self.full_samples))))
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
        ds_idxs = np.ravel_multi_index((region_idxs, sample_idxs), self.shape)
        return self[ds_idxs]

    def __getitem__(
        self, idx: Union[Idx, Tuple[Idx, Idx]]
    ) -> Union[NDArray, Tuple[NDArray, ...], Any]:
        """Get a batch of haplotypes and tracks or intervals and tracks.

        Parameters
        ----------
        idx: Idx
            The index or indices to get. If a single index is provided, the output will be squeezed.
        """
        if isinstance(idx, tuple):
            return self.isel(*idx)

        # Since Dataset is a frozen class, we need to use object.__setattr__ to set the attributes
        # per attrs docs (https://www.attrs.org/en/stable/init.html#post-init)
        if self.genotypes is None and self.sequence_type == "haplotypes":
            object.__setattr__(self, "genotypes", self._init_genotypes())
        if self.intervals is None and self.active_tracks:
            object.__setattr__(self, "intervals", self._init_intervals())

        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += len(self)
            _idx = [idx]
            squeeze = True
        elif isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = len(self) if idx.stop is None else idx.stop
            idx = slice(start, stop, idx.step)
            _idx = np.r_[idx]
            squeeze = False
        else:
            _idx = idx
            squeeze = False

        _idx = np.asarray(_idx, dtype=np.intp)

        if _idx.ndim > 1:
            out_reshape = _idx.shape
            _idx = _idx.ravel()
        else:
            out_reshape = None

        _idx[_idx < 0] += len(self)
        if self.idx_map is not None:
            _idx = self.idx_map[_idx]
        r_idx, s_idx = np.unravel_index(_idx, self.full_shape)
        to_rc = self.full_regions[r_idx, 3] == -1
        should_rc = to_rc.any()

        out: List[NDArray] = []

        if self.sequence_type == "haplotypes":
            if TYPE_CHECKING:
                assert self.genotypes is not None
                assert self.variants is not None
                assert self.ploidy is not None

            geno_offset_idx = self._get_geno_offset_idx(r_idx, s_idx)

            if isinstance(self.genotypes, SparseSomaticGenotypes):
                keep = mark_keep_variants(
                    geno_offset_idx,
                    self.full_regions[r_idx],
                    self.genotypes.offsets,
                    self.genotypes.variant_idxs,
                    self.variants.positions,
                    self.variants.sizes,
                    self.genotypes.dosages,
                    self.ploidy,
                    self.output_length,
                    self.deterministic,
                )
            else:
                keep = None

            if not self.deterministic:
                # (b p)
                shifts = self._get_shifts(geno_offset_idx, keep)
            else:
                # (b p)
                shifts = np.zeros((len(geno_offset_idx), self.ploidy), dtype=np.int32)
            # (b p l)
            haps = self._get_haplotypes(geno_offset_idx, r_idx, shifts, keep)
            if should_rc:
                haps[to_rc] = sp.DNA.reverse_complement(haps[to_rc], -1)
            if not self.phased:
                # (b 1 l) -> (b l)
                haps = haps.squeeze(1)
            out.append(haps)
        elif self.sequence_type == "reference":
            if TYPE_CHECKING:
                assert self.reference is not None
            geno_offset_idx = None
            shifts = None
            # (b l)
            ref = _get_reference(
                r_idx,
                self.full_regions,
                self.reference.reference,
                self.reference.offsets,
                self.region_length,
                self.reference.pad_char,
            ).view("S1")
            if should_rc:
                ref[to_rc] = sp.DNA.reverse_complement(ref[to_rc], -1)
            out.append(ref)
        else:
            geno_offset_idx = None
            shifts = None

        if self.active_tracks:
            # [(b p l) ...]
            tracks = self._get_tracks(_idx, r_idx, shifts, geno_offset_idx)
            if should_rc:
                for t in tracks:
                    t[to_rc] = t[to_rc, ..., ::-1]
            out.extend(tracks)

        if self.jitter > 0:
            out = list(
                sp.jitter(
                    *out,
                    max_jitter=self.jitter,
                    length_axis=-1,
                    jitter_axes=0,
                    seed=self.rng,
                )
            )

        if out_reshape is not None:
            out = [o.reshape(out_reshape + o.shape[1:]) for o in out]

        if squeeze:
            # (1 p l) -> (p l)
            out = [o.squeeze() for o in out]

        if self.return_indices:
            out.extend((_idx, s_idx, r_idx))

        _out = tuple(out)

        if len(out) == 1:
            _out = _out[0]

        if self.transform is not None:
            if isinstance(_out, tuple):
                _out = self.transform(*_out)
            else:
                _out = self.transform(_out)

        return _out

    def _init_genotypes(self):
        if TYPE_CHECKING:
            assert self.ploidy is not None
        if not self.phased:
            genotypes = SparseSomaticGenotypes(
                np.memmap(
                    self.path / "genotypes" / "variant_idxs.npy",
                    dtype=np.int32,
                    mode="r",
                ),
                np.memmap(
                    self.path / "genotypes" / "dosages.npy", dtype=np.float32, mode="r"
                ),
                np.memmap(
                    self.path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"
                ),
                len(self.full_regions),
                len(self.full_samples),
            )
        else:
            genotypes = SparseGenotypes(
                np.memmap(
                    self.path / "genotypes" / "variant_idxs.npy",
                    dtype=np.int32,
                    mode="r",
                ),
                np.memmap(
                    self.path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"
                ),
                len(self.full_regions),
                len(self.full_samples),
                self.ploidy,
            )
        return genotypes

    def _init_intervals(self, tracks: Optional[Union[str, List[str]]] = None):
        if TYPE_CHECKING:
            assert self.active_tracks is not False

        if tracks is None:
            tracks = self.available_tracks
        elif isinstance(tracks, str):
            tracks = [tracks]

        intervals: Dict[str, RaggedIntervals] = {}
        for track in tracks:
            itvs = np.memmap(
                self.path / "intervals" / track / "intervals.npy",
                dtype=INTERVAL_DTYPE,
                mode="r",
            )
            offsets = np.memmap(
                self.path / "intervals" / track / "offsets.npy",
                dtype=np.int64,
                mode="r",
            )
            intervals[track] = RaggedIntervals.from_offsets(
                itvs, self.full_shape, offsets
            )

        return intervals

    def _get_geno_offset_idx(
        self, region_idx: NDArray[np.intp], sample_idx: NDArray[np.intp]
    ):
        if TYPE_CHECKING:
            assert self.genotypes is not None
            assert self.ploidy is not None
        ploid_idx = np.arange(self.ploidy, dtype=np.intp)
        rsp_idx = (region_idx[:, None], sample_idx[:, None], ploid_idx)
        geno_offset_idx = np.ravel_multi_index(rsp_idx, self.genotypes.effective_shape)
        return geno_offset_idx

    def _get_shifts(
        self, geno_offset_idx: NDArray[np.intp], keep: Optional[NDArray[np.bool_]]
    ):
        if TYPE_CHECKING:
            assert self.genotypes is not None
            assert self.variants is not None
        # (b p)
        diffs = get_diffs_sparse(
            geno_offset_idx,
            self.genotypes.variant_idxs,
            self.genotypes.offsets,
            self.variants.sizes,
            keep,
        )
        # (b p)
        shifts = self.rng.integers(0, -diffs.clip(max=0) + 1, dtype=np.int32)
        return shifts

    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        region_idx: NDArray[np.intp],
        shifts: NDArray[np.int32],
        keep: Optional[NDArray[np.bool_]],
    ) -> NDArray[np.bytes_]:
        if TYPE_CHECKING:
            assert self.genotypes is not None
            assert self.reference is not None
            assert self.variants is not None
            assert self.ploidy is not None

        n_queries = len(region_idx)
        haps = np.empty((n_queries, self.ploidy, self.region_length), np.uint8)
        if isinstance(self.genotypes, SparseGenotypes):
            reconstruct_haplotypes_from_sparse(
                geno_offset_idx,
                haps,
                self.full_regions[region_idx],
                shifts,
                self.genotypes.offsets,
                self.genotypes.variant_idxs,
                self.variants.positions,
                self.variants.sizes,
                self.variants.alts.alleles.view(np.uint8),
                self.variants.alts.offsets,
                self.reference.reference,
                self.reference.offsets,
                self.reference.pad_char,
            )
        else:
            assert keep is not None
            reconstruct_haplotypes_from_sparse_somatic(
                geno_offset_idx,
                haps,
                self.full_regions[region_idx],
                shifts,
                self.genotypes.offsets,
                self.genotypes.variant_idxs,
                self.variants.positions,
                self.variants.sizes,
                self.variants.alts.alleles.view(np.uint8),
                self.variants.alts.offsets,
                self.reference.reference,
                self.reference.offsets,
                self.reference.pad_char,
                keep,
            )
        return haps.view("S1")

    def _get_tracks(
        self,
        dataset_idx: NDArray[np.intp],
        region_idx: NDArray[np.intp],
        shifts: Optional[NDArray[np.int32]] = None,
        geno_offset_idx: Optional[NDArray[np.intp]] = None,
    ):
        if TYPE_CHECKING:
            assert self.active_tracks is not None
            assert self.intervals is not None

        # fancy indexing makes a copy so safe to mutate
        regions = self.full_regions[region_idx]
        if shifts is None:
            regions[:, 2] = regions[:, 1] + self.region_length

        tracks: List[NDArray[np.float32]] = []
        for name in self.active_tracks:
            intervals = self.intervals[name]
            # (b*l) ragged
            _tracks = intervals_to_tracks(
                dataset_idx,
                regions,
                intervals.data,
                intervals.offsets,
            )
            track_offsets = _lengths_to_offsets(regions[:, 2] - regions[:, 1], np.int64)

            if shifts is not None and geno_offset_idx is not None:
                if TYPE_CHECKING:
                    assert self.ploidy is not None
                    assert self.variants is not None
                    assert self.genotypes is not None

                out = np.empty(
                    (len(region_idx), self.ploidy, self.region_length), np.float32
                )
                shift_and_realign_tracks_sparse(
                    offset_idx=geno_offset_idx,
                    variant_idxs=self.genotypes.variant_idxs,
                    offsets=self.genotypes.offsets,
                    regions=regions,
                    positions=self.variants.positions,
                    sizes=self.variants.sizes,
                    shifts=shifts,
                    tracks=_tracks,
                    track_offsets=track_offsets,
                    out=out,
                )
                _tracks = out
            else:
                # (b*l) ragged -> (b l)
                _tracks = _tracks.reshape(len(region_idx), self.region_length)

            tracks.append(_tracks)

        return tracks

    def get_bed(self) -> pl.DataFrame:
        """Get a polars.DataFrame of the regions in the dataset, corresponding to the coordinates
        used when writing the dataset. In other words, each region will have length
        :code:`self.output_length + 2 * self.max_jitter`."""
        bed = regions_to_bed(self.full_regions, self.contigs)
        bed = bed.with_columns(chromEnd=pl.col("chromStart") + self.region_length)
        return bed

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@nb.njit(parallel=True, nogil=True, cache=True)
def _get_reference(
    r_idxs: NDArray[np.int32],
    regions: NDArray[np.int32],
    reference: NDArray[np.uint8],
    ref_offsets: NDArray[np.uint64],
    region_length: int,
    pad_char: int,
) -> NDArray[np.uint8]:
    out = np.empty((len(r_idxs), region_length), np.uint8)
    for region in nb.prange(len(r_idxs)):
        q = regions[r_idxs[region]]
        c_idx = q[0]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        start = q[1]
        end = q[2]
        out[region] = padded_slice(reference[c_s:c_e], start, end, pad_char)
    return out
