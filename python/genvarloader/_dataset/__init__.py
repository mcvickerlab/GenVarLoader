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
    cast,
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
from typing_extensions import assert_never

from .._ragged import (
    INTERVAL_DTYPE,
    Ragged,
    RaggedIntervals,
    _reverse,
    _reverse_complement,
)
from .._torch import get_dataloader
from .._types import Idx
from .._utils import _lengths_to_offsets, with_length
from .._variants._records import VLenAlleles
from ._genotypes import (
    SparseGenotypes,
    SparseSomaticGenotypes,
    get_diffs_sparse,
    mark_keep_variants,
    padded_slice,
    reconstruct_haplotypes_from_sparse,
)
from ._indexing import DatasetIndexer
from ._intervals import intervals_to_tracks, tracks_to_intervals
from ._reference import Reference
from ._tracks import shift_and_realign_tracks_sparse
from ._utils import (
    bed_to_regions,
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
    """A dataset of genotypes, reference sequences, and intervals.

    .. note::

        This class is not meant to be instantiated directly. Use the :py:meth:`Dataset.open() <genvarloader.Dataset.open()>`
        method to open a dataset after writing the data with :py:func:`genvarloader.write()` or the GenVarLoader CLI.

    GVL Datasets act like a set of lazy arrays that can be lazily subset or eagerly indexed similar to a NumPy array. They
    have an effective shape of :code:`(n_regions, n_samples, output_length)`, but only the region and sample dimensions
    can be indexed directly since the return value is typically a tuple of arrays.

    **Eager indexing**

    .. code-block:: python

        dataset[99]  # 100-th instance of the raveled dataset
        dataset[0, 9]  # first region, 10th sample
        dataset.isel(regions=0, samples=9)  # equivalent to the above
        dataset[:10]  # first 10 instances
        dataset[:10, :5]  # first 10 regions and 5 samples
        dataset[[3, 0, 1]]  # 4th, 1st, and 2nd instances
        dataset[[2, 2], [0, 1]]  # 3rd region, 1st and 2nd samples

    **Lazy indexing**

    See :py:meth:`Dataset.subset_to() <genvarloader.Dataset.subset_to()>`. This is useful, for example, to create
    splits for training, validation, and testing, or filter out regions or samples after writing a full dataset.

    **Return values**

    The return value depends on the :code:`Dataset` settings, namely :attr:`sequence_type <genvarloader.Dataset.sequence_type>`,
    :attr:`active_tracks <genvarloader.Dataset.active_tracks>`, :attr:`return_annotations <genvarloader.Dataset.return_annotations>`,
    :attr:`return_indices <genvarloader.Dataset.return_indices>`, and :attr:`transform <genvarloader.Dataset.transform>`. These can
    all be modified after opening a :code:`Dataset` using :py:meth:`Dataset.with_settings() <genvarloader.Dataset.with_settings()>`.
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
        return_annotations: bool = False,
        ragged: bool = False,
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
        return_annotations
            Whether to return sequence annotations. See :attr:`Dataset.return_annotations <genvarloader.Dataset.return_annotations>` for more information.
        ragged
            Whether to return sequences and tracks as ragged arrays. See :attr:`Dataset.ragged <genvarloader.Dataset.ragged>` for more information.
        """

        if return_sequences is False:
            _reference = None
        else:
            _reference = reference

        ds = cls._open(path, _reference).with_settings(
            return_sequences=return_sequences,
            return_tracks=return_tracks,
            transform=transform,
            seed=seed,
            jitter=jitter,
            return_indices=return_indices,
            deterministic=deterministic,
            return_annotations=return_annotations,
            ragged=ragged,
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
        input_bed = pl.read_ipc(path / "input_regions.arrow")
        r_idx_map = input_bed["r_idx_map"].to_numpy().astype(np.intp)
        input_bed = input_bed.drop("r_idx_map")
        input_regions = bed_to_regions(input_bed, contigs)
        i2d_map = oidx_to_raveled_idx(
            r_idx_map, np.arange(len(samples)), (len(regions), len(samples))
        )
        d2i_map = oidx_to_raveled_idx(
            np.argsort(r_idx_map), np.arange(len(samples)), (len(regions), len(samples))
        )

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

        idxer = DatasetIndexer(r_idx_map, np.arange(len(samples)), i2d_map, d2i_map)

        dataset = cls(
            path=path,
            max_jitter=max_jitter,
            jitter=max_jitter,
            region_length=region_length,
            contigs=contigs,
            _full_input_bed=input_bed,
            _full_input_regions=input_regions,
            deterministic=False,
            ploidy=ploidy,
            available_tracks=tracks,
            sequence_type=sequence_type,
            return_annotations=False,
            active_tracks=active_tracks,
            phased=phased,
            ragged=False,
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
        deterministic: Optional[bool] = None,
        return_annotations: Optional[bool] = None,
        ragged: Optional[bool] = None,
    ) -> "Dataset":
        """Modify settings of the dataset, returning a new dataset without modifying the old one.

        Parameters
        ----------
        return_sequences
            The sequence type to return. Set this to False to disable returning sequences entirely.
        return_tracks
            The tracks to return. Set this to False to disable returning tracks entirely.
        transform
            Transform to apply to data.
        seed
            Random seed for non-deterministic operations e.g. jittering and shifting longer-than-requested haplotypes.
        jitter
            How much jitter to use. Must be non-negative and <= the :attr:`max_jitter <genvarloader.Dataset.max_jitter>` of the dataset.
        return_indices
            Whether to return indices.
        deterministic
            Whether to use randomized or deterministic algorithms. If set to True, this will disable random
            shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
            and always apply the highest dosage group. Note that for unphased variants, this will mean not all possible haplotypes
            can be returned.
        return_annotations
            Whether to return sequence annotations. See :attr:`Dataset.return_annotations <genvarloader.Dataset.return_annotations>` for more information.
        ragged
            Whether to return tracks as ragged arrays. This is useful for tracks that have variable lengths, such as indel tracks.
            See :attr:`Dataset.ragged <genvarloader.Dataset.ragged>` for more information.
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
                        f"Track(s) {missing} not found. Available track(s): {self.available_tracks}"
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

        if deterministic is not None:
            to_evolve["deterministic"] = deterministic

        if return_annotations is not None:
            to_evolve["return_annotations"] = return_annotations

        if ragged is not None:
            to_evolve["ragged"] = ragged

        return evolve(self, **to_evolve)

    def subset_to(
        self,
        regions: Optional[Union[Idx, NDArray[np.bool_], pl.Series]] = None,
        samples: Optional[Union[Idx, NDArray[np.bool_], Sequence[str]]] = None,
    ) -> "Dataset":
        """Subset the dataset to specific regions and/or samples by index or a boolean mask. If regions or samples
        are not provided, the corresponding dimension will not be subset.

        Parameters
        ----------
        regions
            The regions to subset to.
        samples
            The samples to subset to.

        Examples
        --------
        Subsetting to the first 10 regions:

        .. code-block:: python

            ds.subset_to(slice(10))

        Subsetting to the 2nd and 4th samples:

        .. code-block:: python

            ds.subset_to(samples=[1, 3])


        Subsetting to chromosome 1, assuming it's labeled :code:`"chr1"`:

        .. code-block:: python

            r_idx = ds.input_regions["chrom"] == "chr1"
            ds.subset_to(regions=r_idx)


        Subsetting to regions labeled by a column "split", assuming "split" existed in the input regions:

        .. code-block:: python

            r_idx = ds.input_regions["split"] == "train"
            ds.subset_to(regions=r_idx)


        Subsetting to dataset regions that intersect with another set of regions (requires `PyRanges <https://github.com/pyranges/pyranges>`_):

        .. code-block:: python

            import pyranges as pr

            regions = gvl.read_bedlike("regions.bed")
            renamer = {
                "chrom": "Chromosome",
                "chromStart": "Start",
                "chromEnd": "End",
                "strand": "Strand"
            }
            regions_pr = pr.PyRanges(bed.rename(renamer, strict=False).to_pandas())
            input_regions_pr = pr.PyRanges(
                ds.input_regions
                .with_row_index()
                .rename(renamer, strict=False)
                .to_pandas()
            )
            r_idx = input_regions_pr.overlap(regions_pr).df["index"].to_numpy()
            ds.subset_to(regions=r_idx)
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
            elif isinstance(samples, np.ndarray) and np.issubdtype(
                samples.dtype, np.bool_
            ):
                sample_idxs = np.nonzero(samples)[0]
            else:
                samples = cast(Idx, samples)  # how to narrow dtype? is this possible?
                sample_idxs = idx_like_to_array(samples, self.n_samples)
        else:
            sample_idxs = self._idxer.sample_idxs

        if regions is not None:
            if isinstance(regions, pl.Series):
                region_idxs = regions.to_numpy()
                if np.issubdtype(region_idxs.dtype, np.bool_):
                    region_idxs = np.nonzero(regions)[0]
                elif not np.issubdtype(region_idxs.dtype, np.integer):
                    raise ValueError("`regions` must be index-like or a boolean mask.")
            else:
                regions = cast(Idx, regions)  # how to narrow dtype? is this possible?
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

            .. important::
                Do not provide a :external+torch:class:`BatchSampler <torch.utils.data.BatchSampler>` here. GVL Datasets use multithreading when indexed with batches of indices to avoid the overhead of multi-processing.
                To leverage this, GVL will automatically wrap the :code:`sampler` with a :external+torch:class:`BatchSampler <torch.utils.data.BatchSampler>`
                so that lists of indices are given to the GVL Dataset instead of one index at a time. See `this post <https://discuss.pytorch.org/t/dataloader-sample-by-slices-from-dataset/113005>`_
                for more information.
        num_workers
            How many subprocesses to use for dataloading. :code:`0` means that the data will be loaded in the main process.

            .. tip::
                For GenVarLoader, it is generally best to set this to 0 or 1 since almost everything in
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

    def isel(self, regions: Idx, samples: Idx) -> Any:
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
                offsets = _lengths_to_offsets(regions[:, 2] - regions[:, 1])
                # layout is (regions, samples) so all samples are local for statistics
                tracks = intervals_to_tracks(
                    offset_idxs=ds_idx,
                    starts=regions[:, 1],
                    out_offsets=offsets,
                    intervals=intervals.data,
                    itv_offsets=intervals.offsets,
                )
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

    return_annotations: bool
    """Whether to return sequence annotations for haplotypes. If :code:`True`, haplotypes will instead be returned as a dictionary
    of the form:

    .. code-block:: python
    
        {
            "haplotypes": NDArray[np.bytes_],
            "variant_indices": NDArray[np.int32],
            "positions": NDArray[np.int32],
        }
    
    where :code:`"haplotypes"` are the haplotypes as bytes/S1, and :code:`"variant_indices"` and :code:`"positions"` are
    arrays with the same shape as :code:`"haplotypes"` that annotate every nucleotide with the variant index and
    reference position it corresponds to. A variant index of -1 corresponds to a reference nucleotide, and a reference
    position of -1 corresponds to padded nucleotides that were added for regions beyond the bounds of the reference genome.
    i.e. if the region's start position is negative or the end position is beyond the end of the reference genome.

    For example, a toy result for :code:`chr1:1-10` could be:

    .. code-block:: text

        haplotypes:        A C G  T ...  T T  A ...
        variant_indices:  -1 3 3 -1 ... -1 4 -1 ...
        positions:         1 2 2  3 ...  6 7  9 ...
    
    where variant 3 is a 1 bp :code:`CG` insertion and variant 4 is a 1 bp deletion :code:`T-`. Note that the first nucleotide
    of every indel maps to a reference position since :func:`gvl.write() <genvarloader.write()>` expects that variants
    are all left-aligned.

    .. important::

        The :code:`"variant_indices"` are numbered with respect to their chromosome. So a variant index of 0 corresponds to the first
        variant on the haplotype's chromosome. Thus, if you want to map the variant index to a multi-chromosome VCF/PGEN, you will need to
        add the number of variants on all chromosomes before the variant index to get the correct variant index in the VCF/PGEN. These values
        are available in the dictionary `gvl.Variants.records.contig_offsets`.

    .. note::

        If :attr:`Dataset.sequence_type <genvarloader.Dataset.sequence_type>` is set to :code:`"reference"` or no reference FASTA was
        provided to :meth:`Dataset.open() <genvarloader.Dataset.open()>`, this will be ignored.
    """

    active_tracks: Optional[List[str]]
    """The active tracks to return."""

    available_tracks: List[str]
    """The available tracks in the dataset."""

    deterministic: bool
    """Whether to use randomized or deterministic algorithms. If set to True, this will disable random
    shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
    and always apply the highest dosage group. Note that for unphased variants, this will mean not all possible haplotypes
    can be returned."""

    ragged: bool
    """Whether the dataset returns :class:`~genvarloader.Ragged` arrays of personalized sequences and tracks. If True, each sequence
    in the :class:`~genvarloader.Ragged` arrays corresponds to the exact reference coordinates specified in the input regions, ignoring the `Dataset.output_length`.
    Because of indels, the lengths of the personalized data may vary relative to the reference coordinates. If True, this will also
    ensure there is no random shifting of personalized data. Finally, if the output length given to :func:`gvl.write() <genvarloader.write()>`
    was less than the length of any input region, variants that do not overlap with the """

    _full_input_bed: pl.DataFrame = field(alias="_full_input_bed")
    """The input BED that was used to write the dataset with no modifications e.g. no length adjustments."""

    _full_input_regions: NDArray[np.int32] = field(alias="_full_input_regions")
    """The input regions that were used to write the dataset with no modifications e.g. no length adjustments."""

    _idxer: DatasetIndexer = field(alias="_idxer")

    _full_samples: List[str] = field(alias="_full_samples")
    """The full list of samples in the dataset."""

    _full_regions: NDArray[np.int32] = field(alias="_full_regions")
    """The full regions in the dataset, sorted by contig and start, as they are ordered on-disk. Compared to the input BED,
    the start and end positions may have been expanded to `region_length`, with additional length to accomodate the largest
    deletion of any haplotype in the dataset. The latter step is done to ensure that there is enough track length to accomodate
    all haplotypes in the dataset."""

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
        """Whether the dataset is a subset."""
        return self._idxer.is_subset

    @property
    def has_reference(self) -> bool:
        """Whether the dataset was provided a reference genome."""
        return self._reference is not None

    @property
    def has_genotypes(self) -> bool:
        """Whether the dataset has genotypes."""
        return self._variants is not None

    @property
    def has_intervals(self) -> bool:
        """Whether the dataset has intervals."""
        return len(self.available_tracks) > 0

    @property
    def samples(self) -> List[str]:
        """The samples in the dataset."""
        if not self.is_subset:
            return self._full_samples
        return [self._full_samples[i] for i in self._idxer.sample_idxs]

    @property
    def regions(self) -> pl.DataFrame:
        """The regions in the dataset."""
        return regions_to_bed(self._full_regions[self._idxer.region_idxs], self.contigs)

    @property
    def input_regions(self) -> pl.DataFrame:
        """The input regions in the dataset as they were provided to :func:`gvl.write() <genvarloader.write()>` i.e. with all BED columns plus any
        extra columns that were present."""
        if self._idxer.region_subset_idxs is None:
            return self._full_input_bed
        return self._full_input_bed[self._idxer.region_subset_idxs]

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
        """The output length of the dataset."""
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
            Original region length: {self.region_length - 2 * self.max_jitter:,}
            Max jitter: {self.max_jitter:,}
            Genotypes available: {genotype_status}
            Tracks available: {self.available_tracks}\
            """
        ).strip()

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, idx: Union[Idx, Tuple[Idx, Idx]]) -> Any:
        """Reconstruct some haplotypes and/or tracks.

        Parameters
        ----------
        idx: Idx
            The index or indices to get. If a single index is provided, the output will be squeezed.
        """
        if self.sequence_type is None and self.active_tracks is None:
            raise ValueError(
                "No sequences or tracks are available. Provide a reference genome or activate at least one track."
            )

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
            squeeze = True
        else:
            squeeze = False

        idx = idx_like_to_array(idx, len(self))

        # map the possibly subset input index to:
        # the on-disk index
        ds_idx = self._idxer[idx]
        # the full input index
        inp_idx = self._idxer.d2i_map[ds_idx]

        if ds_idx.ndim > 1:
            out_reshape = ds_idx.shape
            ds_idx = ds_idx.ravel()
        else:
            out_reshape = None

        r_idx, s_idx = np.unravel_index(ds_idx, self.full_shape)
        if self.ragged:
            # (b)
            ir_idx, is_idx = np.unravel_index(inp_idx, self.full_shape)
            regions = self._full_input_regions[ir_idx, :3]
        else:
            regions = self._full_regions[r_idx, :3].copy()
            regions[:, 2] = regions[:, 1] + self.region_length

        # contig, start, end, strand
        to_rc: NDArray[np.bool_] = self._full_regions[r_idx, 3] == -1
        should_rc = to_rc.any()

        ragged_out: List[Ragged] = []

        if self.sequence_type == "haplotypes":
            if TYPE_CHECKING:
                assert self._genotypes is not None
                assert self._variants is not None
                assert self.ploidy is not None

            # (b 1) for broadcasting against ploidy dimension
            to_rc = to_rc[:, None]

            geno_offset_idx = self._get_geno_offset_idx(r_idx, s_idx)

            # (b)
            lengths = repeat(regions[:, 2] - regions[:, 1], "b  -> b p", p=self.ploidy)
            # may need to skip some variants if returning ragged data since
            # the regions are different
            if self.phased is False or self.ragged:
                dosages: Optional[NDArray[np.float32]] = getattr(
                    self._genotypes, "dosages", None
                )
                keep, keep_offsets = mark_keep_variants(
                    geno_offset_idxs=geno_offset_idx,
                    starts=regions[:, 1],
                    offsets=self._genotypes.offsets,
                    sparse_genos=self._genotypes.variant_idxs,
                    positions=self._variants.positions,
                    sizes=self._variants.sizes,
                    dosages=dosages,
                    ploidy=self.ploidy,
                    target_lengths=lengths,
                    deterministic=self.deterministic,
                )
            else:
                keep = None
                keep_offsets = None

            # (b p)
            diffs = get_diffs_sparse(
                geno_offset_idxs=geno_offset_idx,
                geno_v_idxs=self._genotypes.variant_idxs,
                geno_offsets=self._genotypes.offsets,
                size_diffs=self._variants.sizes,
                keep=keep,
                keep_offsets=keep_offsets,
            )

            if self.deterministic or self.ragged:
                # (b p)
                shifts = np.zeros((len(geno_offset_idx), self.ploidy), dtype=np.int32)
            else:
                # if the haplotype is longer than the region, shift it randomly
                # by up to the difference in length between the haplotype and the region
                # (b p)
                shifts = self._rng.integers(0, diffs.clip(min=0) + 1, dtype=np.int32)

            # (b, p)
            if self.ragged:
                lengths += diffs
            # (b*p+1)
            out_offsets = _lengths_to_offsets(lengths)

            # (b p l), (b p l), (b p l)
            haps, maybe_annot_v_idx, maybe_annot_pos = self._get_haplotypes(
                geno_offset_idx=geno_offset_idx,
                regions=regions,
                out_offsets=out_offsets,
                shifts=shifts,
                keep=keep,
                keep_offsets=keep_offsets,
            )

            if should_rc:
                haps = _reverse_complement(haps, to_rc)

            if self.phased is False:
                # (b 1 l) -> (b l) remove ploidy dim
                haps = haps.squeeze(1)

            ragged_out.append(haps)
            if maybe_annot_v_idx is not None and maybe_annot_pos is not None:
                ragged_out.extend((maybe_annot_v_idx, maybe_annot_pos))
        elif self.sequence_type == "reference":
            if TYPE_CHECKING:
                assert self._reference is not None

            geno_offset_idx = None
            diffs = None
            shifts = None
            # (b+1)
            out_offsets = _lengths_to_offsets(np.diff(regions[:, 1:], 1, 1))

            # (b l)
            ref = _get_reference(
                regions=regions,
                out_offsets=out_offsets,
                reference=self._reference.reference,
                ref_offsets=self._reference.offsets,
                pad_char=self._reference.pad_char,
            ).view("S1")
            ref = Ragged.from_offsets(ref, len(ds_idx), out_offsets)

            if should_rc:
                ref = _reverse_complement(ref, to_rc)

            ragged_out.append(ref)
        elif self.sequence_type is None:
            geno_offset_idx = None
            diffs = None
            shifts = None
        else:
            assert_never(self.sequence_type)

        if self.active_tracks is not None:
            # ploidy dim present if sequence_type == "haplotypes"
            # [(b [p] l) ...]
            tracks = self._get_tracks(
                dataset_idx=ds_idx,
                regions=regions,
                geno_offset_idx=geno_offset_idx,
                diffs=diffs,
                shifts=shifts,
            )
            if should_rc:
                for t in tracks:
                    _reverse(t, to_rc)
            ragged_out.extend(tracks)

        # at this point all entires in `out` are Ragged arrays

        if self.ragged:
            out: List[Any] = ragged_out
        else:
            # convert all ragged arrays to fixed length arrays, assuming they need no padding
            if self.sequence_type == "haplotypes" and self.phased:
                if TYPE_CHECKING:
                    assert self.ploidy is not None
                out = [
                    a.data.reshape(len(ds_idx), self.ploidy, self.region_length)
                    for a in ragged_out
                ]
            else:
                out = [
                    a.data.reshape(len(ds_idx), self.region_length) for a in ragged_out
                ]

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
            # (1 [p] l) -> ([p] l)
            out = [o.squeeze(0) for o in out]

        if self.return_indices:
            out.extend((ds_idx, r_idx, s_idx))

        if self.return_annotations:
            haps = dict(zip(("haplotypes", "variant_indices", "positions"), out[:3]))
            _out = (haps, *out[3:])
        else:
            _out = tuple(out)

        if len(_out) == 1:
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
        regions: NDArray[np.int32],
        out_offsets: NDArray[np.int64],
        shifts: NDArray[np.int32],
        keep: Optional[NDArray[np.bool_]],
        keep_offsets: Optional[NDArray[np.int64]] = None,
    ) -> Tuple[
        Ragged[np.bytes_], Optional[Ragged[np.int32]], Optional[Ragged[np.int32]]
    ]:
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
        if TYPE_CHECKING:
            assert self._genotypes is not None
            assert self._reference is not None
            assert self._variants is not None
            assert self.ploidy is not None

        haps = Ragged.from_offsets(
            np.empty(out_offsets[-1], np.uint8), shifts.shape, out_offsets
        )

        if self.return_annotations:
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
            offset_idxs=geno_offset_idx,
            out=haps.data,
            out_offsets=haps.offsets,
            regions=regions,
            shifts=shifts,
            offsets=self._genotypes.offsets,
            sparse_genos=self._genotypes.variant_idxs,
            positions=self._variants.positions,
            sizes=self._variants.sizes,
            alt_alleles=self._variants.alts.alleles.view(np.uint8),
            alt_offsets=self._variants.alts.offsets,
            ref=self._reference.reference,
            ref_offsets=self._reference.offsets,
            pad_char=self._reference.pad_char,
            keep=keep,
            keep_offsets=keep_offsets,
            annot_v_idxs=annot_v_idxs.data
            if annot_v_idxs is not None
            else annot_v_idxs,
            annot_ref_pos=annot_positions.data
            if annot_positions is not None
            else annot_positions,
        )

        haps = cast(Ragged[np.uint8], haps)
        haps.data = haps.data.view("S1")
        haps = cast(Ragged[np.bytes_], haps)

        return haps, annot_v_idxs, annot_positions

    def _get_tracks(
        self,
        dataset_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        geno_offset_idx: Optional[NDArray[np.integer]] = None,
        diffs: Optional[NDArray[np.int32]] = None,
        shifts: Optional[NDArray[np.int32]] = None,
        keep: Optional[NDArray[np.bool_]] = None,
        keep_offsets: Optional[NDArray[np.int64]] = None,
    ):
        if TYPE_CHECKING:
            assert self.active_tracks is not None
            assert self._intervals is not None

        # (b)
        out_lengths = np.diff(regions[:, 1:], 1, 1)
        # (b)
        track_lengths = out_lengths
        starts = regions[:, 1]

        if diffs is not None:
            if TYPE_CHECKING:
                assert self.ploidy is not None
            # (b p)
            out_lengths = np.tile(out_lengths, (1, self.ploidy))
            # (b p) adjust lengths to accomodate indels
            track_lengths = track_lengths[:, None] - diffs
            starts = np.repeat(starts, self.ploidy)

        out_offsets = _lengths_to_offsets(out_lengths)
        track_offsets = _lengths_to_offsets(track_lengths)

        tracks: List[Ragged[np.float32]] = []
        for name in self.active_tracks:
            intervals = self._intervals[name]
            # (b [p] l) ragged
            _tracks = intervals_to_tracks(
                offset_idxs=dataset_idx,
                starts=starts,
                out_offsets=track_offsets,
                intervals=intervals.data,
                itv_offsets=intervals.offsets,
            )

            if geno_offset_idx is not None and shifts is not None:
                if TYPE_CHECKING:
                    assert self.ploidy is not None
                    assert self._variants is not None
                    assert self._genotypes is not None

                out = np.empty(out_offsets[-1], np.float32)
                _tracks = shift_and_realign_tracks_sparse(
                    offset_idx=geno_offset_idx,
                    variant_idxs=self._genotypes.variant_idxs,
                    variant_offsets=self._genotypes.offsets,
                    regions=regions,
                    positions=self._variants.positions,
                    sizes=self._variants.sizes,
                    shifts=shifts,
                    tracks=_tracks,
                    track_offsets=track_offsets,
                    out=out,
                    out_offsets=out_offsets,
                    keep=keep,
                    keep_offsets=keep_offsets,
                )
                _tracks = Ragged.from_offsets(
                    out, (len(regions), self.ploidy), out_offsets
                )
            else:
                _tracks = Ragged.from_offsets(_tracks, len(regions), out_offsets)

            tracks.append(_tracks)

        return tracks

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


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
        c_idx, start, end = regions[i]
        c_s = ref_offsets[c_idx]
        c_e = ref_offsets[c_idx + 1]
        out[o_s:o_e] = padded_slice(reference[c_s:c_e], start, end, pad_char)
    return out
