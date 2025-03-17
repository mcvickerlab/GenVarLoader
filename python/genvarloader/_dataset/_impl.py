from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import polars as pl
from attrs import define, evolve, field
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import NoReturn, assert_never

from .._ragged import (
    Ragged,
    RaggedAnnotatedHaps,
    _jitter,
    _reverse,
    _reverse_complement,
    is_rag_dtype,
)
from .._torch import TorchDataset, get_dataloader
from .._types import DTYPE, AnnotatedHaps, Idx
from .._utils import idx_like_to_array
from ._genotypes import SparseGenotypes
from ._indexing import DatasetIndexer
from ._reconstruct import Haps, HapsTracks, Reference, Seqs, SeqsTracks, Tracks
from ._utils import bed_to_regions

try:
    import torch
    import torch.utils.data as td

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_py_open = open


@define(frozen=True)
class Dataset:
    output_length: Union[Literal["ragged", "variable"], int]
    """The output length. Can be set to :code:`"ragged"` or :code:`"variable"` to allow for variable length sequences.
    If set to an integer, all sequences will be padded or truncated to this length. See the
    `online documentation <https://genvarloader.readthedocs.io/en/latest/dataset.html>`_ for more information."""
    max_jitter: int
    """Maximum jitter."""
    return_indices: bool
    """Whether to return non-subset row and sample indices."""
    contigs: List[str]
    """List of unique contigs."""
    jitter: int
    """How much jitter to use."""
    deterministic: bool
    """Whether to use randomized or deterministic algorithms. If set to True, this will disable random
    shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
    and always apply the highest dosage group. Note that for unphased variants, this will mean not all possible haplotypes
    can be returned."""
    rc_neg: bool
    """Whether to reverse-complement the sequences on negative strands."""
    transform: Callable | None
    """Tranform to apply to what the dataset would otherwise return on its own."""
    _full_bed: pl.DataFrame = field(alias="_full_bed")
    _full_regions: NDArray[np.int32] = field(alias="_full_regions")
    _jittered_regions: NDArray[np.int32] = field(alias="_jittered_regions")
    _idxer: DatasetIndexer = field(alias="_idxer")
    _seqs: Optional[Seqs | Haps[Ragged[np.bytes_]] | Haps[RaggedAnnotatedHaps]] = field(
        alias="_seqs"
    )
    _tracks: Optional[Tracks] = field(alias="_tracks")
    _recon: (
        Seqs
        | Haps[Ragged[np.bytes_]]
        | Haps[RaggedAnnotatedHaps]
        | Tracks
        | SeqsTracks
        | HapsTracks[Ragged[np.bytes_]]
        | HapsTracks[RaggedAnnotatedHaps]
    ) = field(alias="_recon")
    _rng: np.random.Generator = field(alias="_rng")

    @property
    def is_subset(self) -> bool:
        """Whether the dataset is a subset."""
        return self._idxer.is_subset

    @property
    def has_reference(self) -> bool:
        """Whether the dataset was provided a reference genome."""
        return self._seqs is not None

    @property
    def has_genotypes(self) -> bool:
        """Whether the dataset has genotypes."""
        return isinstance(self._seqs, Haps)

    @property
    def has_intervals(self) -> bool:
        """Whether the dataset has intervals."""
        return self._tracks is not None

    @property
    def samples(self) -> List[str]:
        """The samples in the dataset."""
        return self._idxer.samples

    @property
    def regions(self) -> pl.DataFrame:
        """The input regions in the dataset as they were provided to :func:`gvl.write() <genvarloader.write()>` i.e. with all BED columns plus any
        extra columns that were present."""
        if self._idxer.region_subset_idxs is None:
            return self._full_bed
        return self._full_bed[self._idxer.region_subset_idxs]

    @property
    def n_regions(self) -> int:
        """The number of regions in the dataset."""
        return self._idxer.n_regions

    @property
    def n_samples(self) -> int:
        """The number of samples in the dataset."""
        return self._idxer.n_samples

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataset. (n_samples, n_regions)"""
        return self.n_regions, self.n_samples

    @property
    def full_shape(self) -> Tuple[int, int]:
        """Return the full shape of the dataset, ignoring any subsetting. (n_regions, n_samples)"""
        return self._idxer.full_shape

    @property
    def haplotype_lengths(self) -> NDArray[np.int32] | None:
        """The lengths of the jitter-extended haplotypes. Shape: (regions, samples, ploidy). If the dataset is
        not phased or not deterministic, this will return None because the haplotypes are not guaranteed to be
        a consistent length due to randomness in what variants are used. Otherwise, this will return the
        haplotype lengths."""
        if not isinstance(self._seqs, Haps):
            return None

        if (
            not isinstance(self._seqs.genotypes, SparseGenotypes)
            and not self.deterministic
        ):
            return None

        # (r s p)
        r_idx, s_idx = np.unravel_index(self._idxer.i2d_map, self.full_shape)
        hap_lens = (
            self._jittered_regions[:, 2, None, None]
            - self._jittered_regions[:, 1, None, None]
            + self._seqs.haplotype_ilens
        )[r_idx, s_idx].reshape(*self.shape, self._seqs.genotypes.ploidy)
        return hap_lens

    def __len__(self):
        return self.n_regions * self.n_samples

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def open(
        path: str | Path,
        reference: str | Path | None = None,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic=False,
        rc_neg: bool = True,
    ) -> RaggedDataset[None, None, None, None]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        # read metadata
        with _py_open(path / "metadata.json") as f:
            metadata = json.load(f)
        samples: List[str] = metadata["samples"]
        contigs: List[str] = metadata["contigs"]
        ploidy: Optional[int] = metadata.get("ploidy", None)
        max_jitter: int = metadata.get("max_jitter", 0)
        phased: Optional[bool] = metadata.get("phased", None)

        # read input regions and generate index map
        bed = pl.read_ipc(path / "input_regions.arrow")
        r_idx_map = bed["r_idx_map"].to_numpy().astype(np.intp)
        idxer = DatasetIndexer.from_region_and_sample_idxs(
            r_idx_map, np.arange(len(samples)), samples
        )
        bed = bed.drop("r_idx_map")
        with pl.StringCache():
            pl.Series(contigs, dtype=pl.Categorical)
            sorted_bed = bed.sort(
                pl.col("chrom").cast(pl.Categorical),
                pl.col("chromStart"),
                pl.col("chromEnd"),
                maintain_order=True,
            )
        regions = bed_to_regions(sorted_bed, contigs)

        has_genotypes = (path / "genotypes").exists()
        if has_genotypes:
            if phased is None:
                raise ValueError("Malformed dataset: found genotypes but not phase.")
            if ploidy is None:
                raise ValueError("Malformed dataset: found genotypes but not ploidy.")

        has_intervals = (path / "intervals").exists()

        if reference is not None and has_genotypes and has_intervals:
            logger.info(
                "Loading reference genome into memory. This typically has a modest memory footprint (a few GB) and greatly improves performance."
            )
            _reference = Reference.from_path_and_contigs(reference, contigs)
            assert phased is not None
            assert ploidy is not None
            seqs = Haps.from_path(
                path,
                reference=_reference,
                phased=phased,
                regions=regions,
                samples=samples,
                ploidy=ploidy,
            )
            tracks = Tracks.from_path(path, regions, samples)
            reconstructor = HapsTracks(haps=seqs, tracks=tracks)
        elif reference is not None and has_genotypes:
            logger.info(
                "Loading reference genome into memory. This typically has a modest memory footprint (a few GB) and greatly improves performance."
            )
            _reference = Reference.from_path_and_contigs(reference, contigs)
            assert phased is not None
            assert ploidy is not None
            seqs = Haps.from_path(
                path,
                reference=_reference,
                phased=phased,
                regions=regions,
                samples=samples,
                ploidy=ploidy,
            )
            tracks = None
            reconstructor = seqs
        elif reference is not None and has_intervals:
            logger.info(
                "Loading reference genome into memory. This typically has a modest memory footprint (a few GB) and greatly improves performance."
            )
            _reference = Reference.from_path_and_contigs(reference, contigs)
            seqs = Seqs(reference=_reference)
            tracks = Tracks.from_path(path, regions, samples)
            reconstructor = SeqsTracks(seqs=seqs, tracks=tracks)
        elif has_intervals:
            seqs = None
            tracks = Tracks.from_path(path, regions, samples)
            reconstructor = tracks
        else:
            raise RuntimeError("Malformed dataset: no genotypes or intervals found.")

        dataset = RaggedDataset(
            output_length="ragged",
            max_jitter=max_jitter,
            jitter=jitter,
            contigs=contigs,
            return_indices=False,
            rc_neg=rc_neg,
            transform=None,
            deterministic=deterministic,
            _idxer=idxer,
            _full_bed=bed,
            _full_regions=regions,
            _jittered_regions=regions.copy(),
            _seqs=seqs,
            _tracks=tracks,
            _recon=reconstructor,
            _rng=np.random.default_rng(rng),
        )

        logger.info(f"Opened dataset:\n{dataset}")

        return dataset

    def with_settings(
        self,
        jitter: int | None = None,
        rng: int | np.random.Generator | None = None,
        deterministic: bool | None = None,
        rc_neg: bool | None = None,
    ) -> Dataset:
        to_evolve = {}

        if jitter is not None:
            if jitter < 0:
                raise ValueError(f"Jitter ({jitter}) must be a non-negative integer.")
            elif jitter > self.max_jitter:
                raise ValueError(
                    f"Jitter ({jitter}) must be less than or equal to the maximum jitter of the dataset ({self.max_jitter})."
                )

            if jitter != self.jitter:
                if isinstance(self.output_length, int):
                    min_r_len: int = (
                        self._full_regions[:, 2] - self._full_regions[:, 1]
                    ).min()
                    max_output_length = min_r_len + 2 * self.max_jitter
                    eff_length = self.output_length + 2 * jitter

                    if eff_length > max_output_length:
                        raise ValueError(
                            f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({jitter=}) = {eff_length} must be less"
                            f" than or equal to the maximum output length of the dataset ({max_output_length})."
                            f" The maximum output length is the minimum region length ({min_r_len}) + 2 * (max_jitter={self.max_jitter})."
                        )
                jittered_regions = self._full_regions.copy()
                jittered_regions[:, 1] -= jitter
                jittered_regions[:, 2] += jitter

                to_evolve["jitter"] = jitter
                to_evolve["_jittered_regions"] = jittered_regions

        if rng is not None:
            to_evolve["rng"] = np.random.default_rng(rng)

        if deterministic is not None:
            to_evolve["deterministic"] = deterministic

        if rc_neg is not None:
            to_evolve["rc_neg"] = rc_neg

        return evolve(self, **to_evolve)

    def with_output_length(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> ArrayDataset | RaggedDataset:
        if isinstance(output_length, int):
            if output_length < 1:
                raise ValueError(
                    f"Output length ({output_length}) must be a positive integer."
                )
            min_r_len: int = (self._full_regions[:, 2] - self._full_regions[:, 1]).min()
            max_output_length = min_r_len + 2 * self.max_jitter
            eff_length = output_length + 2 * self.jitter

            if eff_length > max_output_length:
                raise ValueError(
                    f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({self.jitter=}) = {eff_length} must be less"
                    f" than or equal to the maximum output length of the dataset ({max_output_length})."
                    f" The maximum output length is the minimum region length ({min_r_len}) + 2 * (max_jitter={self.max_jitter})."
                )

            return ArrayDataset(
                output_length=output_length,
                max_jitter=self.max_jitter,
                jitter=self.jitter,
                contigs=self.contigs,
                return_indices=self.return_indices,
                rc_neg=self.rc_neg,
                transform=self.transform,
                deterministic=self.deterministic,
                _idxer=self._idxer,
                _full_bed=self._full_bed,
                _full_regions=self._full_regions,
                _jittered_regions=self._jittered_regions,
                _seqs=self._seqs,
                _tracks=self._tracks,
                _recon=self._recon,
                _rng=self._rng,
            )
        else:
            return RaggedDataset(
                output_length=output_length,
                max_jitter=self.max_jitter,
                jitter=self.jitter,
                contigs=self.contigs,
                return_indices=self.return_indices,
                rc_neg=self.rc_neg,
                transform=self.transform,
                deterministic=self.deterministic,
                _idxer=self._idxer,
                _full_bed=self._full_bed,
                _full_regions=self._full_regions,
                _jittered_regions=self._jittered_regions,
                _seqs=self._seqs,
                _tracks=self._tracks,
                _recon=self._recon,
                _rng=self._rng,
            )

    def with_indices(self, return_indices: bool):
        return evolve(self, return_indices=return_indices)

    def with_transform(self, transform: Callable | None):
        return evolve(self, transform=transform)

    def with_seqs(self, kind: Literal[False, "reference", "haplotypes", "annotated"]):
        match kind, self._seqs, self._tracks:
            case False, _, None:
                raise ValueError(
                    "Dataset only has sequences available, so returning no sequences is not possible."
                )
            case False, _, tracks:
                return evolve(self, _recon=tracks)
            case "reference" | "haplotypes" | "annotated", None, _:
                raise ValueError(
                    "Dataset has no reference genome to reconstruct sequences from."
                )
            case "haplotypes" | "annotated", Seqs(), _:
                raise ValueError(
                    "Dataset has no genotypes to reconstruct haplotypes from."
                )
            case "reference", Seqs(reference=ref) | Haps(reference=ref), None:
                seqs = Seqs(reference=ref)
                return evolve(self, _recon=seqs)
            case "reference", Seqs(reference=ref) | Haps(reference=ref), tracks:
                seqs = Seqs(reference=ref)
                return evolve(self, _recon=SeqsTracks(seqs=seqs, tracks=tracks))
            case "haplotypes", Haps() as haps, None:
                return evolve(self, _recon=haps.with_annot(False))
            case "haplotypes", Haps() as haps, tracks:
                return evolve(self, _recon=HapsTracks(haps.with_annot(False), tracks))
            case "annotated", Haps() as haps, None:
                return evolve(self, _recon=haps.with_annot(True))
            case "annotated", Haps() as haps, tracks:
                return evolve(self, _recon=HapsTracks(haps.with_annot(True), tracks))
            case k, s, t:
                assert_never(k), assert_never(s), assert_never(t)

    def with_tracks(self, tracks: Union[Literal[False], str, List[str]]):
        match tracks, self._seqs, self._tracks:
            case False, None, _:
                raise ValueError(
                    "Dataset only has tracks available, so returning no tracks is not possible."
                )
            case False, seqs, _:
                return evolve(self, _recon=seqs)
            case t, _, None:
                raise ValueError("Dataset has no tracks.")
            case t, None, tr:
                return evolve(self, _recon=tr.with_tracks(t))
            case t, Seqs() as seqs, tr:
                return evolve(self, _recon=SeqsTracks(seqs, tr.with_tracks(t)))
            case t, Haps() as haps, tr:
                return evolve(
                    self,
                    _recon=HapsTracks(
                        haps,  # type: ignore | pylance weirdly infers HapsTracks[RaggedAnnotatedHaps]
                        tr.with_tracks(t),
                    ),
                )
            case k, s, t:
                assert_never(k), assert_never(s), assert_never(t)

    def subset_to(
        self,
        regions: Idx | NDArray[np.bool_] | pl.Series | None = None,
        samples: Idx | NDArray[np.bool_] | str | Sequence[str] | None = None,
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
                if missing := _samples.difference(self._idxer.full_samples):
                    raise ValueError(f"Samples {missing} not found in the dataset")
                sample_idx = np.array(
                    [
                        i
                        for i, s in enumerate(self._idxer.full_samples)
                        if s in _samples
                    ],
                    np.intp,
                )
            elif isinstance(samples, np.ndarray) and np.issubdtype(
                samples.dtype, np.bool_
            ):
                sample_idx = np.nonzero(samples)[0]
            else:
                samples = cast(Idx, samples)  # how to narrow dtype? is this possible?
                sample_idx = samples
        else:
            sample_idx = None

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
            region_idxs = None

        idxer = self._idxer.subset_to(regions=region_idxs, samples=sample_idx)

        return evolve(self, _idxer=idxer)

    def to_full_dataset(self) -> "Dataset":
        """Return a full sized dataset, undoing any subsetting."""
        return evolve(self, _idxer=self._idxer.to_full_dataset())

    def to_torch_dataset(self) -> "td.Dataset":
        if self.output_length == "ragged":
            raise ValueError(
                """`output_length` is currently set to "ragged" and ragged output cannot be converted to PyTorch Tensors."""
                """ Set `output_length` to "variable" or an integer."""
            )
        return TorchDataset(self)

    def to_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: "td.Sampler" | Iterable | None = None,  # type: ignore
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Callable | None = None,
        generator: "torch.Generator" | None = None,  # type: ignore
        *,
        prefetch_factor: int | None = None,
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
            dataset=self.to_torch_dataset(),
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

    def __getitem__(
        self, idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]]
    ) -> Any:
        if not isinstance(idx, tuple):
            regions = idx
            samples = slice(None)
        elif len(idx) == 1:
            regions = idx[0]
            samples = slice(None)
        else:
            regions, samples = idx

        if isinstance(samples, str):
            samples = [samples]

        if not isinstance(samples, (int, np.integer, slice)) and isinstance(
            samples[0], str
        ):
            _samples = set(samples)
            if missing := _samples.difference(self._idxer.full_samples):
                raise ValueError(f"Samples {missing} not found in the dataset")
            samples = np.array(
                [i for i, s in enumerate(self._idxer.full_samples) if s in _samples],
                np.intp,
            )
        samples = cast(Idx, samples)  # above clause does this, but can't narrow type

        if isinstance(regions, (int, np.integer)) and isinstance(
            samples, (int, np.integer)
        ):
            squeeze = True
        else:
            squeeze = False

        r_idx = idx_like_to_array(regions, self.n_regions)
        s_idx = idx_like_to_array(samples, self.n_samples)
        idx = np.ravel_multi_index((r_idx, s_idx), self.shape)

        ds_idx = self._idxer[idx]

        if ds_idx.ndim > 1:
            out_reshape = ds_idx.shape
            ds_idx = ds_idx.ravel()
        else:
            out_reshape = None

        r_idx, s_idx = np.unravel_index(ds_idx, self.full_shape)
        regions = self._jittered_regions[r_idx]

        recon = self._recon(
            idx,
            regions,
            self.output_length,
            self.jitter,
            None if self.deterministic else self._rng,
        )

        if not isinstance(recon, tuple):
            out = [recon]
        else:
            out = list(recon)

        if self.rc_neg:
            # (b)
            to_rc: NDArray[np.bool_] = self._full_regions[r_idx, 3] == -1
            out = [self._rc(r, to_rc) for r in out]

        if self.jitter > 0:
            out = self._jitter(out)

        if self.output_length == "variable":
            out = [self._pad(r) for r in out]
        elif isinstance(self.output_length, int):
            out = [self._fix_len(r) for r in out]

        if out_reshape is not None:
            out = [o.reshape(out_reshape + o.shape[1:]) for o in out]

        if squeeze:
            # (1 [p] l) -> ([p] l)
            out = [o.squeeze(0) for o in out]

        if self.return_indices:
            inp_idx = self._idxer.d2i_map[ds_idx]
            inp_r_idx, inp_s_idx = np.unravel_index(inp_idx, self.full_shape)
            out.extend((inp_r_idx, inp_s_idx))  # type: ignore

        if self.transform is not None:
            out = self.transform(*out)
        elif len(out) == 1:
            out = out[0]

        return out

    @overload
    def _rc(self, rag: Ragged[DTYPE], to_rc: NDArray[np.bool_]) -> Ragged[DTYPE]: ...
    @overload
    def _rc(
        self, rag: RaggedAnnotatedHaps, to_rc: NDArray[np.bool_]
    ) -> RaggedAnnotatedHaps: ...
    def _rc(
        self, rag: Ragged | RaggedAnnotatedHaps, to_rc: NDArray[np.bool_]
    ) -> Ragged | RaggedAnnotatedHaps:
        if isinstance(rag, Ragged):
            if is_rag_dtype(rag, np.bytes_):
                if len(rag.shape) == 2:
                    rag = _reverse_complement(rag, to_rc[:, None])
                else:
                    rag = _reverse_complement(rag, to_rc)
            elif is_rag_dtype(rag, np.float32):
                _reverse(rag, to_rc)
        elif isinstance(rag, RaggedAnnotatedHaps):
            rag.haps = _reverse_complement(rag.haps, to_rc[:, None])
        else:
            assert_never(rag)
        return rag

    def _jitter(
        self, rags: list[Ragged[np.bytes_] | Ragged[np.float32] | RaggedAnnotatedHaps]
    ) -> list[Ragged[np.bytes_] | Ragged[np.float32] | RaggedAnnotatedHaps]:
        rag0 = rags[0]
        if isinstance(rag0, Ragged):
            batch_size = rag0.shape[0]
        else:
            batch_size = rag0.haps.shape[0]
        starts = self._rng.integers(0, 2 * self.jitter + 1, batch_size)

        jittered = []
        for r in rags:
            if isinstance(r, Ragged):
                jittered.append(_jitter(r, max_jitter=self.jitter, starts=starts))
            else:
                haps, v_idx, r_coord = _jitter(
                    *(r.haps, r.var_idxs, r.ref_coords),
                    max_jitter=self.jitter,
                    starts=starts,
                )
                jittered.append(RaggedAnnotatedHaps(haps, v_idx, r_coord))

        return jittered

    @overload
    def _pad(self, rag: Ragged[DTYPE]) -> NDArray[DTYPE]: ...
    @overload
    def _pad(self, rag: RaggedAnnotatedHaps) -> AnnotatedHaps: ...
    def _pad(self, rag: Ragged | RaggedAnnotatedHaps) -> NDArray | AnnotatedHaps:
        if isinstance(rag, Ragged):
            if is_rag_dtype(rag, np.bytes_):
                return rag.to_padded(b"N")
            elif is_rag_dtype(rag, np.float32):
                return rag.to_padded(0)
            else:
                raise ValueError(f"Unsupported pad dtype: {rag.data.dtype}")
        elif isinstance(rag, RaggedAnnotatedHaps):
            return rag.to_padded()
        else:
            assert_never(rag)

    @overload
    def _fix_len(self, rag: Ragged[DTYPE]) -> NDArray[DTYPE]: ...
    @overload
    def _fix_len(self, rag: RaggedAnnotatedHaps) -> AnnotatedHaps: ...
    def _fix_len(self, rag: Ragged | RaggedAnnotatedHaps) -> NDArray | AnnotatedHaps:
        assert isinstance(self.output_length, int)
        if isinstance(rag, Ragged):
            # (b p) or (b)
            return rag.data.reshape((*rag.shape, self.output_length))
        elif isinstance(rag, RaggedAnnotatedHaps):
            assert isinstance(self._seqs, Haps)
            return rag.to_fixed_shape((*rag.shape, self.output_length))
        else:
            assert_never(rag)


T = TypeVar("T")

SEQ = TypeVar("SEQ", None, NDArray[np.bytes_], AnnotatedHaps)
RSEQ = TypeVar("RSEQ", None, Ragged[np.bytes_], RaggedAnnotatedHaps)
TRK = TypeVar("TRK", None, NDArray[np.float32])
RTRK = TypeVar("RTRK", None, Ragged[np.float32])
IDX = TypeVar("IDX", None, NDArray[np.integer])
TFM = TypeVar("TFM")


class ArrayDataset(Dataset, Generic[SEQ, TRK, IDX, TFM]):
    @overload
    def with_output_length(
        self: ArrayDataset[NDArray[np.bytes_], None, IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[Ragged[np.bytes_], None, IDX, TFM]: ...
    @overload
    def with_output_length(
        self: ArrayDataset[AnnotatedHaps, None, IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedAnnotatedHaps, None, IDX, TFM]: ...
    @overload
    def with_output_length(
        self: ArrayDataset[None, NDArray[np.float32], IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[None, Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_output_length(
        self: ArrayDataset[NDArray[np.bytes_], NDArray[np.float32], IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[Ragged[np.bytes_], Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_output_length(
        self: ArrayDataset[AnnotatedHaps, NDArray[np.float32], IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedAnnotatedHaps, Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_output_length(
        self,
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[SEQ, TRK, IDX, TFM]: ...
    def with_output_length(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> Union[RaggedDataset[RSEQ, RTRK, IDX, TFM], ArrayDataset[SEQ, TRK, IDX, TFM]]:
        return super().with_output_length(output_length)

    @overload
    def with_seqs(self, kind: Literal[False]) -> ArrayDataset[None, TRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["reference", "haplotypes"]
    ) -> ArrayDataset[NDArray[np.bytes_], TRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["annotated"]
    ) -> ArrayDataset[AnnotatedHaps, TRK, IDX, TFM]: ...
    def with_seqs(
        self, kind: Literal[False, "reference", "haplotypes", "annotated"]
    ) -> ArrayDataset:
        return super().with_seqs(kind)

    @overload
    def with_tracks(
        self, tracks: Literal[False]
    ) -> ArrayDataset[SEQ, None, IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: str
    ) -> ArrayDataset[SEQ, NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: List[str]
    ) -> ArrayDataset[SEQ, NDArray[np.float32], IDX, TFM]: ...
    def with_tracks(
        self, tracks: Union[Literal[False], str, List[str]]
    ) -> ArrayDataset:
        return super().with_tracks(tracks)

    @overload
    def with_indices(
        self, return_indices: Literal[False]
    ) -> ArrayDataset[SEQ, TRK, None, TFM]: ...
    @overload
    def with_indices(
        self, return_indices: Literal[True]
    ) -> ArrayDataset[SEQ, TRK, NDArray[np.integer], TFM]: ...
    def with_indices(self, return_indices: bool) -> ArrayDataset:
        return super().with_indices(return_indices)

    @overload
    def with_transform(
        self: ArrayDataset[SEQ, None, None, TFM],
        transform: Callable[[SEQ], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(
        self: ArrayDataset[None, TRK, None, TFM],
        transform: Callable[[TRK], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(
        self: ArrayDataset[None, None, IDX, TFM],
        transform: Callable[[NoReturn], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(
        self: ArrayDataset[SEQ, TRK, None, TFM],
        transform: Callable[[SEQ, TRK], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(
        self: ArrayDataset[SEQ, None, IDX, TFM],
        transform: Callable[[SEQ, IDX], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(
        self: ArrayDataset[None, TRK, IDX, TFM],
        transform: Callable[[TRK, IDX], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(
        self: ArrayDataset[SEQ, TRK, IDX, TFM],
        transform: Callable[[SEQ, TRK, IDX], T],
    ) -> ArrayDataset[SEQ, TRK, IDX, T]: ...
    @overload
    def with_transform(self, transform: None) -> ArrayDataset[SEQ, TRK, IDX, None]: ...
    def with_transform(self, transform: Callable | None) -> ArrayDataset:
        return super().with_transform(transform)

    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, None, None, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> SEQ: ...
    @overload
    def __getitem__(
        self: ArrayDataset[None, TRK, None, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> TRK: ...
    @overload
    def __getitem__(
        self: ArrayDataset[None, None, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> NoReturn: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, TRK, None, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[SEQ, TRK]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, None, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[SEQ, IDX]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[None, TRK, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[TRK, IDX]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, TRK, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[SEQ, TRK, IDX]: ...
    @overload
    def __getitem__(
        self: ArrayDataset[SEQ, TRK, IDX, TFM],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> TFM: ...
    def __getitem__(
        self, idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]]
    ) -> Any:
        return super().__getitem__(idx)


class RaggedDataset(Dataset, Generic[RSEQ, RTRK, IDX, TFM]):
    @overload
    def with_output_length(
        self: RaggedDataset[Ragged[np.bytes_], None, IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[NDArray[np.bytes_], None, IDX, TFM]: ...
    @overload
    def with_output_length(
        self: RaggedDataset[RaggedAnnotatedHaps, None, IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[AnnotatedHaps, None, IDX, TFM]: ...
    @overload
    def with_output_length(
        self: RaggedDataset[None, Ragged[np.float32], IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[None, NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_output_length(
        self: RaggedDataset[None, RTRK, IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[None, TRK, IDX, TFM]: ...
    @overload
    def with_output_length(
        self: RaggedDataset[Ragged[np.bytes_], Ragged[np.float32], IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[NDArray[np.bytes_], NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_output_length(
        self: RaggedDataset[RaggedAnnotatedHaps, Ragged[np.float32], IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[AnnotatedHaps, NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_output_length(
        self,
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, TFM]: ...
    def with_output_length(
        self, output_length: Union[Literal["ragged", "variable"], int]
    ) -> Union[RaggedDataset[RSEQ, RTRK, IDX, TFM], ArrayDataset[SEQ, TRK, IDX, TFM]]:
        return super().with_output_length(output_length)

    @overload
    def with_seqs(
        self, kind: Literal[False]
    ) -> RaggedDataset[None, RTRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["reference", "haplotypes"]
    ) -> RaggedDataset[Ragged[np.bytes_], RTRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["annotated"]
    ) -> RaggedDataset[RaggedAnnotatedHaps, RTRK, IDX, TFM]: ...
    def with_seqs(
        self, kind: Literal[False, "reference", "haplotypes", "annotated"]
    ) -> RaggedDataset:
        return super().with_seqs(kind)

    @overload
    def with_tracks(
        self, tracks: Literal[False]
    ) -> RaggedDataset[RSEQ, None, IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: str
    ) -> RaggedDataset[RSEQ, Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: List[str]
    ) -> RaggedDataset[RSEQ, Ragged[np.float32], IDX, TFM]: ...
    def with_tracks(
        self, tracks: Union[Literal[False], str, List[str]]
    ) -> RaggedDataset:
        return super().with_tracks(tracks)

    @overload
    def with_indices(
        self, return_indices: Literal[False]
    ) -> RaggedDataset[RSEQ, RTRK, None, TFM]: ...
    @overload
    def with_indices(
        self, return_indices: Literal[True]
    ) -> RaggedDataset[RSEQ, RTRK, NDArray[np.integer], TFM]: ...
    def with_indices(self, return_indices: bool) -> RaggedDataset:
        return super().with_indices(return_indices)

    @overload
    def with_transform(
        self: RaggedDataset[RSEQ, None, None, TFM],
        transform: Callable[[RSEQ], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self: RaggedDataset[None, RTRK, None, TFM],
        transform: Callable[[RTRK], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self: RaggedDataset[None, None, IDX, TFM],
        transform: Callable[[NoReturn], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self: RaggedDataset[RSEQ, RTRK, None, TFM],
        transform: Callable[[RSEQ, RTRK], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self: RaggedDataset[RSEQ, None, IDX, TFM],
        transform: Callable[[RSEQ, IDX], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self: RaggedDataset[None, RTRK, IDX, TFM],
        transform: Callable[[RTRK, IDX], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self: RaggedDataset[RSEQ, RTRK, IDX, TFM],
        transform: Callable[[RSEQ, RTRK, IDX], T],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, T]: ...
    @overload
    def with_transform(
        self, transform: None
    ) -> RaggedDataset[RSEQ, RTRK, IDX, None]: ...
    def with_transform(self, transform: Callable | None) -> RaggedDataset:
        return super().with_transform(transform)

    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, None, None, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> RSEQ: ...
    @overload
    def __getitem__(
        self: RaggedDataset[None, RTRK, None, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> RTRK: ...
    @overload
    def __getitem__(
        self: RaggedDataset[None, None, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> NoReturn: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, RTRK, None, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[RSEQ, RTRK]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, None, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[RSEQ, IDX]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[None, RTRK, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[RTRK, IDX]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, RTRK, IDX, None],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> Tuple[RSEQ, RTRK, IDX]: ...
    @overload
    def __getitem__(
        self: RaggedDataset[RSEQ, RTRK, IDX, TFM],
        idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]],
    ) -> TFM: ...
    def __getitem__(
        self, idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]]
    ) -> Any:
        return super().__getitem__(idx)
