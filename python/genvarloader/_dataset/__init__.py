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
from attrs import define, evolve, field
from einops import repeat
from loguru import logger
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .._torch import get_dataloader
from .._types import INTERVAL_DTYPE, Idx, Ragged, RaggedIntervals
from .._utils import _lengths_to_offsets, with_length
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
from ._indexing import DatasetIndexer
from ._intervals import intervals_to_tracks, tracks_to_intervals
from ._reference import Reference
from ._tracks import shift_and_realign_tracks_sparse
from ._utils import (
    idx_like_to_array,
    oidx_to_raveled_idx,
    regions_to_bed,
    splits_sum_le_value,
)

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
    Use the :py:meth:`Dataset.open() <genvarloader.Dataset.open()>` method to open a dataset after writing the data with :py:func:`genvarloader.write()`
    or the GenVarLoader CLI.
    """

    @classmethod
    def open(
        cls,
        path: Union[str, Path],
        reference: Optional[Union[str, Path]] = None,
        return_sequences: Optional[Literal[False, "reference", "haplotypes"]] = None,
        return_tracks: Optional[Union[Literal[False], str, List[str]]] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
        jitter: Optional[int] = None,
        return_indices: Optional[bool] = None,
        deterministic: bool = False,
    ) -> "Dataset":
        """Open a dataset from a path. If no reference genome is provided, the dataset can only yield tracks.

        Parameters
        ----------
        path
            Path to a dataset.
        reference
            Path to a reference genome.
        return_sequences
            The sequence type to return. Set this to :code:`False` to disable returning sequences. The default depends
            on the presence of genotypes and reference genome. If genotypes are present and a reference genome is provided,
            haplotypes will be returned. If only a reference genome is provided, reference sequences will be returned. Otherwise,
            no DNA sequences will be returned.
        return_tracks
            The tracks to return. Set this to :code:`False` to disable returning tracks. By default all available tracks
            are active and returned in sorted order by name.
        transform
            A transform function to apply to data. The input should correspond to what is returned by the dataset without the
            transform, and the output can be anything.
        seed
            Random seed for any stochastic operations.
        jitter
            Amount of jitter to use, cannot be more than the maximum jitter of the dataset.
        return_indices
            Whether to return indices. Three indices are returned for each instance corresponding to the dataset, region, and sample index.
            For example, the indices for :code:`dataset[0]` might be :code:`(0, 0, 0)` and this would correspond to the first instance
            that exists on disk, first input region, and first sample. Note that due to sorting regions during writing, the dataset index
            may not correspond to the region and sample indices for a C-ordered :code:`(regions, samples)` matrix (e.g. :code:`(1, 0, 0)` is possible).
            Enabling this is useful if you need to map any of the indices to data for particular regions or samples. For example, to use a read
            depth normalization based on the total library size for a sample. Or during inference, to map predictions back to their regions and samples.
        deterministic
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
        deterministic : bool, optional
            Whether to use randomized or deterministic algorithms. If set to True, this will disable random
            shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
            and always apply the highest dosage group. Note that for unphased variants, this will mean not all possible haplotypes
            can be returned.
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        regions: NDArray[np.int32] = np.load(path / "regions.npy")

        has_intervals = (path / "intervals").exists()
        has_genotypes = (path / "genotypes").exists()

        # read metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        samples: List[str] = metadata["samples"]
        contigs: List[str] = metadata["contigs"]
        region_length: int = metadata["region_length"]
        ploidy: Optional[int] = metadata.get("ploidy", None)
        max_jitter: int = metadata.get("max_jitter", 0)
        phased: Optional[bool] = metadata.get("phased", None)

        # read input regions and generate index map
        input_regions = pl.read_ipc(path / "input_regions.arrow")
        r_idx_map = input_regions["r_idx_map"].to_numpy().astype(np.intp)
        idx_map = oidx_to_raveled_idx(
            r_idx_map, np.arange(len(samples)), (len(regions), len(samples))
        )
        input_regions = input_regions.drop("r_idx_map")

        # initialize random number generator
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

        idxer = DatasetIndexer(
            r_idx_map, np.arange(len(samples)), idx_map, len(regions), len(samples)
        )

        dataset = cls(
            path=path,
            max_jitter=max_jitter,
            jitter=max_jitter,
            region_length=region_length,
            contigs=contigs,
            _full_input_regions=input_regions,
            deterministic=deterministic,
            ploidy=ploidy,
            available_tracks=tracks,
            sequence_type=sequence_type,
            active_tracks=active_tracks,
            phased=phased,
            _full_samples=samples,
            _full_regions=regions,
            _idxer=idxer,
            _rng=rng,
            _reference=_reference,
            _variants=variants,
        )

        logger.info(f"\n{str(dataset)}")

        return dataset

    def with_settings(
        self,
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
        return_sequences
            The sequence type to return. Set this to False to disable returning sequences entirely.
        return_tracks
            The tracks to return, by default None. Set this to False to disable returning tracks entirely.
        transform
            The transform to set, by default None
        seed
            The seed to set, by default None
        jitter
            The jitter to set, by default None
        return_indices
            Whether to return indices, by default None
        """
        to_evolve: Dict[str, Any] = {}

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
            to_evolve["_genotypes"] = None

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
            to_evolve["_intervals"] = None

        if transform is not None:
            if transform is False:
                transform = None
            to_evolve["transform"] = transform

        if seed is not None:
            to_evolve["_rng"] = np.random.default_rng(seed)

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

        return evolve(self, **to_evolve)

    def subset_to(
        self,
        regions: Optional[Idx] = None,
        samples: Optional[Union[Idx, Sequence[str]]] = None,
    ) -> "Dataset":
        """Subset the dataset to specific regions and/or samples.

        Parameters
        ----------
        regions
            The regions to subset to.
        samples
            The samples to subset to.
        """
        if regions is None and samples is None:
            return self

        if samples is not None:
            if isinstance(samples, (str, Sequence)):
                _samples = set(samples)
                if missing := _samples.difference(self._full_samples):
                    raise ValueError(f"Samples {missing} not found in the dataset")
                sample_idxs = np.array(
                    [i for i, s in enumerate(self._full_samples) if s in _samples],
                    np.intp,
                )
            else:
                sample_idxs = idx_like_to_array(samples, self.n_samples)
        else:
            sample_idxs = self._idxer.sample_idxs

        if regions is not None:
            region_idxs = idx_like_to_array(regions, self.n_regions)
        else:
            region_idxs = np.arange(self.n_regions, dtype=np.intp)

        idxer = self._idxer.subset_to(regions=region_idxs, samples=sample_idxs)

        return evolve(self, _idxer=idxer)

    def to_full_dataset(self) -> "Dataset":
        """Return a full sized dataset, undoing any subsetting."""
        return evolve(self, _idxer=self._idxer.to_full_dataset())

    def to_dataset(self) -> "td.Dataset":
        """Convert the dataset to a map-style PyTorch :external+torch:class:`Dataset <torch.utils.data.Dataset>`.
        Requires PyTorch to be installed.
        """
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
        """Convert the dataset to a PyTorch :external+torch:class:`DataLoader <torch.utils.data.DataLoader>`. The parameters are the same as a
        :external+torch:class:`DataLoader <torch.utils.data.DataLoader>` with a few omissions e.g. :code:`batch_sampler`.
        Requires PyTorch to be installed.

        Parameters
        ----------
        batch_size
            How many samples per batch to load.
        shuffle
            Set to True to have the data reshuffled at every epoch.
        sampler
            Defines the strategy to draw samples from the dataset. Can be any :py:class:`Iterable <typing.Iterable>` with :code:`__len__` implemented. If specified, shuffle must not be specified.
        num_workers
            How many subprocesses to use for dataloading. :code:`0` means that the data will be loaded in the main process. For GenVarLoader, it is generally best to set this to 0 or 1 since almost everything in
            GVL is multithreaded. However, if you are using a transform that is compute intensive and single threaded, there may
            be a benefit to setting this > 1.
        collate_fn
            Merges a list of samples to form a mini-batch of Tensor(s).
        pin_memory
            If :code:`True`, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your :code:`collate_fn` returns a batch that is a custom type, see the example below.
        drop_last
            Set to :code:`True` to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If :code:`False` and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        timeout
            If positive, the timeout value for collecting a batch from workers. Should always be non-negative.
        worker_init_fn
            If not :code:`None`, this will be called on each worker subprocess with the worker id (an int in :code:`[0, num_workers - 1]`) as input, after seeding and before data loading.
        multiprocessing_context
            If :code:`None`, the default multiprocessing context of your operating system will be used.
        generator
            If not :code:`None`, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate :code:`base_seed` for workers.
        prefetch_factor
            Number of batches loaded in advance by each worker. 2 means there will be a total of 2 * num_workers batches prefetched across all workers. (default value depends on the set value for num_workers. If value of num_workers=0 default is None. Otherwise, if value of num_workers > 0 default is 2).
        persistent_workers
            If :code:`True`, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive.
        pin_memory_device
            The device to :code:`pin_memory` to if :code:`pin_memory` is :code:`True`.
        """
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

    def isel(
        self, regions: Idx, samples: Idx
    ) -> Union[NDArray, Tuple[NDArray, ...], Any]:
        """Eagerly select a subset of regions and samples from the dataset.

        Parameters
        ----------
        regions
            The indices of the regions to select.
        samples
            The indices of the samples to select.
        """
        _regions = idx_like_to_array(regions, self.n_regions)
        _samples = idx_like_to_array(samples, self.n_samples)

        if isinstance(_regions, np.ndarray) and isinstance(_samples, np.ndarray):
            _regions = _regions[:, None]

        ds_idxs = np.ravel_multi_index((_regions, _samples), self.shape)
        return self[ds_idxs]

    def _sel(
        self,
        regions: Union[str, Tuple[str, int, int], pl.DataFrame],
        samples: Union[str, List[str]],
    ) -> Union[NDArray, Tuple[NDArray, ...], Any]:
        """Eagerly select a subset of regions and samples from the dataset.

        Parameters
        ----------
        regions : str, Tuple[str, int, int], pl.DataFrame
            The regions to select.
        samples : str, List[str]
            The names of the samples to select.
        """
        if isinstance(regions, str):
            try:
                split_idx = regions.rindex(":")
                contig = regions[:split_idx]
                start, end = map(int, regions[split_idx + 1 :].split("-"))
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

        s_to_i = dict(zip(self._full_samples, range(len(self._full_samples))))
        sample_idxs = np.array([s_to_i[s] for s in samples], np.intp)
        region_idxs = regions.join(
            with_length(self.input_regions, self.output_length).with_row_index(),
            on=["chrom", "chromStart", "chromEnd"],
            how="left",
        )["index"]
        if (n_missing := region_idxs.is_null().sum()) > 0:
            raise ValueError(f"{n_missing} regions not found in the dataset.")
        region_idxs = region_idxs.to_numpy()
        ds_idxs = np.ravel_multi_index((region_idxs, sample_idxs), self.shape)
        return self[ds_idxs]

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
        new_track
            The name of the new track.
        existing_track
            The name of the existing track to transform.
        transform
            A function to apply to the existing track to get a new, transformed track.
            This will be done in chunks such that the tracks provided will not exceed :code:`max_mem`.
            The arguments given to the transform will be the dataset indices, region indices, and
            sample indices as numpy arrays and the tracks themselves as a :class:`Ragged` array with
            shape (regions, samples). The tracks must be a :class:`Ragged` array since regions may be
            different lengths to accomodate indels. This function should then return the transformed
            tracks as a :class:`Ragged` array with the same shape and lengths.
        max_mem
            The maximum memory to use in bytes, by default 1 GiB (2**30 bytes)
        overwrite
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

        if self._intervals is None:
            intervals = self._init_intervals(existing_track)[existing_track]
        else:
            intervals = self._intervals[existing_track]

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

        lengths = self._full_regions[:, 2] - self._full_regions[:, 1]
        # for each region:
        # bytes = (4 bytes / bp) * (bp / sample) * samples
        n_samples = len(self._full_samples)
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
                regions = self._full_regions[r_idx]
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

    _full_input_regions: pl.DataFrame = field(alias="_full_input_regions")
    """The input regions that were used to write the dataset with no modifications e.g. no length adjustments."""

    _idxer: DatasetIndexer = field(alias="_idxer")

    _full_samples: List[str] = field(alias="_full_samples")
    """The full list of samples in the dataset."""

    _full_regions: NDArray[np.int32] = field(alias="_full_regions")
    """The full regions in the dataset, sorted by contig and start."""

    _rng: np.random.Generator = field(alias="_rng")
    """The random number generator used for jittering and shifting haplotypes that are longer than the output length."""

    ploidy: Optional[int] = None
    """The ploidy of the dataset."""

    transform: Optional[Callable] = None
    """The transform to apply to the data."""

    return_indices: bool = False
    """Whether to return indices."""

    phased: Optional[bool] = None
    """Whether the genotypes are phased. Set to None if genotypes are not present."""

    _reference: Optional[Reference] = field(default=None, alias="_reference")
    """The reference genome. This is kept in memory."""

    _variants: Optional[_Variants] = field(default=None, alias="_variants")
    """The variant sites in the dataset. This is kept in memory."""

    _genotypes: Optional[Union[SparseGenotypes, SparseSomaticGenotypes]] = field(
        default=None, alias="_genotypes"
    )
    """The genotypes in the dataset. This is memory mapped."""

    _intervals: Optional[Dict[str, RaggedIntervals]] = field(
        default=None, alias="_intervals"
    )
    """The intervals in the dataset. This is memory mapped."""

    @property
    def is_subset(self) -> bool:
        return self._idxer.is_subset

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    @property
    def has_genotypes(self) -> bool:
        return self._variants is not None

    @property
    def has_intervals(self) -> bool:
        return len(self.available_tracks) > 0

    @property
    def samples(self) -> List[str]:
        if not self.is_subset:
            return self._full_samples
        else:
            return [self._full_samples[i] for i in self._idxer.sample_idxs]

    @property
    def regions(self) -> pl.DataFrame:
        return regions_to_bed(self._full_regions[self._idxer.region_idxs], self.contigs)

    @property
    def input_regions(self) -> pl.DataFrame:
        return self._full_input_regions[self._idxer.region_idxs]

    @property
    def n_regions(self) -> int:
        """The number of regions in the dataset."""
        return self._idxer.n_regions

    @property
    def n_samples(self) -> int:
        """The number of samples in the dataset."""
        return self._idxer.n_samples

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
        return self._idxer.full_shape

    def __len__(self) -> int:
        return int(np.prod(self.shape))

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
            Is subset: {self.is_subset}
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
        if self._genotypes is None and self.sequence_type == "haplotypes":
            object.__setattr__(self, "_genotypes", self._init_genotypes())
        if self._intervals is None and self.active_tracks:
            object.__setattr__(self, "_intervals", self._init_intervals())

        # check if need to squeeze batch dim at the end
        if isinstance(idx, (int, np.integer)):
            idx = np.array([idx], np.intp)
            squeeze = True
        else:
            idx = idx
            squeeze = False

        _idx = self._idxer[idx]

        if _idx.ndim > 1:
            out_reshape = _idx.shape
            _idx = _idx.ravel()
        else:
            out_reshape = None

        r_idx, s_idx = np.unravel_index(_idx, self.full_shape)
        to_rc = self._full_regions[r_idx, 3] == -1
        should_rc = to_rc.any()

        out: List[NDArray] = []

        if self.sequence_type == "haplotypes":
            if TYPE_CHECKING:
                assert self._genotypes is not None
                assert self._variants is not None
                assert self.ploidy is not None

            geno_offset_idx = self._get_geno_offset_idx(r_idx, s_idx)

            if isinstance(self._genotypes, SparseSomaticGenotypes):
                keep = mark_keep_variants(
                    geno_offset_idx,
                    self._full_regions[r_idx],
                    self._genotypes.offsets,
                    self._genotypes.variant_idxs,
                    self._variants.positions,
                    self._variants.sizes,
                    self._genotypes.dosages,
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
                assert self._reference is not None
            geno_offset_idx = None
            shifts = None
            # (b l)
            ref = _get_reference(
                r_idx,
                self._full_regions,
                self._reference.reference,
                self._reference.offsets,
                self.region_length,
                self._reference.pad_char,
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
                    seed=self._rng,
                )
            )

        if out_reshape is not None:
            out = [o.reshape(out_reshape + o.shape[1:]) for o in out]

        if squeeze:
            # (1 p l) -> (p l)
            out = [o.squeeze(0) for o in out]

        if self.return_indices:
            out.extend((_idx, r_idx, s_idx))

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
                len(self._full_regions),
                len(self._full_samples),
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
                len(self._full_regions),
                len(self._full_samples),
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
            assert self._genotypes is not None
            assert self.ploidy is not None
        ploid_idx = np.arange(self.ploidy, dtype=np.intp)
        rsp_idx = (region_idx[:, None], sample_idx[:, None], ploid_idx)
        geno_offset_idx = np.ravel_multi_index(rsp_idx, self._genotypes.effective_shape)
        return geno_offset_idx

    def _get_shifts(
        self, geno_offset_idx: NDArray[np.intp], keep: Optional[NDArray[np.bool_]]
    ):
        if TYPE_CHECKING:
            assert self._genotypes is not None
            assert self._variants is not None
        # (b p)
        diffs = get_diffs_sparse(
            geno_offset_idx,
            self._genotypes.variant_idxs,
            self._genotypes.offsets,
            self._variants.sizes,
            keep,
        )
        # (b p)
        shifts = self._rng.integers(0, -diffs.clip(max=0) + 1, dtype=np.int32)
        return shifts

    def _get_haplotypes(
        self,
        geno_offset_idx: NDArray[np.intp],
        region_idx: NDArray[np.intp],
        shifts: NDArray[np.int32],
        keep: Optional[NDArray[np.bool_]],
    ) -> NDArray[np.bytes_]:
        if TYPE_CHECKING:
            assert self._genotypes is not None
            assert self._reference is not None
            assert self._variants is not None
            assert self.ploidy is not None

        n_queries = len(region_idx)
        haps = np.empty((n_queries, self.ploidy, self.region_length), np.uint8)
        if isinstance(self._genotypes, SparseGenotypes):
            reconstruct_haplotypes_from_sparse(
                geno_offset_idx,
                haps,
                self._full_regions[region_idx],
                shifts,
                self._genotypes.offsets,
                self._genotypes.variant_idxs,
                self._variants.positions,
                self._variants.sizes,
                self._variants.alts.alleles.view(np.uint8),
                self._variants.alts.offsets,
                self._reference.reference,
                self._reference.offsets,
                self._reference.pad_char,
            )
        else:
            assert keep is not None
            reconstruct_haplotypes_from_sparse_somatic(
                geno_offset_idx,
                haps,
                self._full_regions[region_idx],
                shifts,
                self._genotypes.offsets,
                self._genotypes.variant_idxs,
                self._variants.positions,
                self._variants.sizes,
                self._variants.alts.alleles.view(np.uint8),
                self._variants.alts.offsets,
                self._reference.reference,
                self._reference.offsets,
                self._reference.pad_char,
                keep,
            )
        return haps.view("S1")

    def _get_tracks(
        self,
        dataset_idx: NDArray[np.integer],
        region_idx: NDArray[np.integer],
        shifts: Optional[NDArray[np.int32]] = None,
        geno_offset_idx: Optional[NDArray[np.integer]] = None,
    ):
        if TYPE_CHECKING:
            assert self.active_tracks is not None
            assert self._intervals is not None

        # fancy indexing makes a copy so safe to mutate
        regions = self._full_regions[region_idx]
        if shifts is None:
            regions[:, 2] = regions[:, 1] + self.region_length

        tracks: List[NDArray[np.float32]] = []
        for name in self.active_tracks:
            intervals = self._intervals[name]
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
                    assert self._variants is not None
                    assert self._genotypes is not None

                out = np.empty(
                    (len(region_idx), self.ploidy, self.region_length), np.float32
                )
                shift_and_realign_tracks_sparse(
                    offset_idx=geno_offset_idx,
                    variant_idxs=self._genotypes.variant_idxs,
                    offsets=self._genotypes.offsets,
                    regions=regions,
                    positions=self._variants.positions,
                    sizes=self._variants.sizes,
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
