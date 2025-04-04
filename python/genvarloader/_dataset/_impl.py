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
    """A dataset of genotypes, reference sequences, and intervals.

    .. note::

        This class is not meant to be instantiated directly. Use the :py:meth:`Dataset.open() <genvarloader.Dataset.open()>`
        method to open a dataset after writing the data with :py:func:`genvarloader.write()` or the GenVarLoader CLI.

    GVL Datasets act like a collection of lazy ragged arrays that can be lazily subset or eagerly indexed as a 2D NumPy array. They
    have an effective shape of :code:`(n_regions, n_samples, [tracks], [ploidy], output_length)`, but only the region and sample
    dimensions can be indexed directly since the return value is generally a tuple of arrays.

    **Eager indexing**

    .. code-block:: python

        dataset[0, 9]  # first region, 10th sample
        dataset[:10]  # first 10 regions and all samples
        dataset[:10, :5]  # first 10 regions and 5 samples
        dataset[[2, 2], [0, 1]]  # 3rd region, 1st and 2nd samples

    **Lazy indexing**

    See :meth:`Dataset.subset_to() <Dataset.subset_to()>`. This is useful, for example, to create
    splits for training, validation, and testing, or filter out regions or samples after writing a full dataset.
    This is also necessary if you intend to create a Pytorch :external+torch:class:`DataLoader <torch.utils.data.DataLoader>`
    from the Dataset using :meth:`Dataset.to_dataloader() <Dataset.to_dataloader()>`.

    **Return values**

    The return value depends on the :code:`Dataset` state, namely :attr:`sequence_type <Dataset.sequence_type>`,
    :attr:`active_tracks <Dataset.active_tracks>`, :attr:`return_indices <Dataset.return_indices>`, and :attr:`transform <Dataset.transform>`.
    These can all be modified after opening a :code:`Dataset` using the following methods:
    - :meth:`Dataset.with_seqs() <Dataset.with_seqs()>`
    - :meth:`Dataset.with_tracks() <Dataset.with_tracks()>`
    - :meth:`Dataset.with_indices() <Dataset.with_indices()>`
    - :meth:`Dataset.with_transform() <Dataset.with_transform()>`
    """

    @overload
    @staticmethod
    def open(
        path: str | Path,
        reference: None = None,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
    ) -> RaggedDataset[None, RTRK, None, None]: ...
    @overload
    @staticmethod
    def open(
        path: str | Path,
        reference: str | Path,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
    ) -> RaggedDataset[Ragged[np.bytes_], RTRK, None, None]: ...
    @staticmethod
    def open(
        path: str | Path,
        reference: str | Path | None = None,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
    ) -> RaggedDataset[RSEQ, RTRK, None, None]:
        """Open a dataset from a path. If no reference genome is provided, the dataset cannot yield sequences.
        Will initialize the dataset such that it will return tracks and haplotypes (reference sequences if no genotypes) if possible.
        If tracks are available, they will be set to be returned in alphabetical order.

        Parameters
        ----------
        path
            Path to a dataset.
        reference
            Path to a reference genome.
        jitter
            Amount of jitter to use, cannot be more than the maximum jitter of the dataset.
        rng
            Random seed or np.random.Generator for any stochastic operations.
        deterministic
            Whether to use randomized or deterministic algorithms. If set to True, this will disable random
            shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
            and always apply the highest CCF group. Note that for unphased variants, this will mean not all possible haplotypes
            can be returned.
        rc_neg
            Whether to reverse-complement sequences and reverse tracks on negative strands.
        """
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

        match reference, has_genotypes, has_intervals:
            case _, False, False:
                raise RuntimeError(
                    "Malformed dataset: neither genotypes nor intervals found."
                )
            case None, True, False:
                raise RuntimeError(
                    "No reference: dataset only has genotypes but no reference was given."
                    " Resulting dataset would have nothing to return."
                )
            case None, _, True:
                seqs = None
                tracks = Tracks.from_path(path, regions, len(samples))
                tracks = tracks.with_tracks(list(tracks.intervals))
                reconstructor = tracks
            case reference, False, True:
                logger.info(
                    "Loading reference genome into memory. This typically has a modest memory footprint (a few GB) and greatly improves performance."
                )
                _reference = Reference.from_path_and_contigs(reference, contigs)
                seqs = Seqs(reference=_reference)
                tracks = Tracks.from_path(path, regions, len(samples))
                tracks = tracks.with_tracks(list(tracks.intervals))
                reconstructor = SeqsTracks(seqs=seqs, tracks=tracks)
            case reference, True, False:
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
            case reference, True, True:
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
                tracks = Tracks.from_path(path, regions, len(samples))
                tracks = tracks.with_tracks(list(tracks.intervals))
                reconstructor = HapsTracks(haps=seqs, tracks=tracks)
            case reference, has_genotypes, has_intervals:
                assert_never(reference)
                assert_never(has_genotypes)
                assert_never(has_intervals)

        dataset = RaggedDataset(
            path=path,
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
        """Modify settings of the dataset, returning a new dataset without modifying the old one.

        Parameters
        ----------
        jitter
            How much jitter to use. Must be non-negative and <= the :attr:`max_jitter <genvarloader.Dataset.max_jitter>` of the dataset.
        rng
            Random seed or np.random.Generator for non-deterministic operations e.g. jittering and shifting longer-than-requested haplotypes.
        deterministic
            Whether to use randomized or deterministic algorithms. If set to True, this will disable random
            shifting of longer-than-requested haplotypes and, for unphased variants, will enable deterministic variant assignment
            and always apply the highest CCF group. Note that for unphased variants, this will mean not all possible haplotypes
            can be returned.
        rc_neg
            Whether to reverse-complement sequences and reverse tracks on negative strands.
        """
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

    def with_len(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> ArrayDataset | RaggedDataset:
        """Modify the output length of the dataset, returning a new dataset without modifying the old one.

        Parameters
        ----------
        output_length
            The output length. Can be set to :code:`"ragged"` or :code:`"variable"` to allow for variable length sequences.
            If set to an integer, all sequences will be padded or truncated to this length. See the
            `online documentation <https://genvarloader.readthedocs.io/en/latest/dataset.html>`_ for more information.
        """
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
                path=self.path,
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
                path=self.path,
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
        """Whether to return indices, returning a new dataset without modifying the old one.
        Set to :code:`True` to have the dataset return the row and sample indices of the non-subset dataset. This allows you to
        combine arbitrary row- and sample-specific data with dataset output on-the-fly with a transform.

        Parameters
        ----------
        return_indices
            Whether to return indices.
        """
        return evolve(self, return_indices=return_indices)

    def with_transform(self, transform: Callable | None):
        """Modify the transform to apply to the dataset output, returning a new dataset without modifying the old one.
        The transform should take input matching the output of the dataset and can return arbitrary output. In combination
        with indices, this allows you to combine arbitrary row- and sample-specific data with dataset output on-the-fly.

        .. note::
            Depending on how transforms are implemented, they can easily introduce a dataloading bottleneck. If you find
            dataloading is slow, it's often a good idea to try disabling your transform to see if it's impacting throughput.

        Parameters
        ----------
        transform
            The transform to apply to the dataset.
        """
        return evolve(self, transform=transform)

    def with_seqs(self, kind: Literal["reference", "haplotypes", "annotated"] | None):
        """Return a new dataset with the specified sequence type. The sequence type can be one of the following:

        - :code:`"reference"`: reference sequences.
        - :code:`"haplotypes"`: personalized haplotype sequences.
        - :code:`"annotated"`: annotated haplotype sequences, which includes personalized haplotypes along with annotations.

        Annotated haplotypes are returned as the :class:`~genvarloader._types.AnnotatedHaps` class which is roughly:

        .. code-block:: python

            class AnnotatedHaps:
                haps: NDArray[np.bytes_]
                var_idxs: NDArray[np.int32]
                ref_coords: NDArray[np.int32]

        where :code:`haps` are the haplotypes as bytes/S1, and :code:`var_idxs` and :code:`ref_coords` are
        arrays with the same shape as :code:`haps` that annotate every nucleotide with the variant index and
        reference coordinate it corresponds to. A variant index of -1 corresponds to a reference nucleotide, and a reference
        coordinate of -1 corresponds to padded nucleotides that were added for regions beyond the bounds of the reference genome.
        i.e. if the region's start position is negative or the end position is beyond the end of the reference genome.

        For example, a toy result for :code:`chr1:1-10` could be:

        .. code-block:: text

            haps:        A C G  T ...  T T  A ...
            var_idxs:   -1 3 3 -1 ... -1 4 -1 ...
            ref_coords:  1 2 2  3 ...  6 7  9 ...

        where variant 3 is a 1 bp :code:`CG` insertion and variant 4 is a 1 bp deletion :code:`T-`. Note that the first nucleotide
        of every indel maps to a reference position since :func:`gvl.write() <genvarloader.write()>` expects that variants
        are all left-aligned.

        .. important::

            The :code:`var_idxs` are numbered with respect to the full set of variants even if the variants were extracted from per-chromosome VCFs/PGENs.
            So a variant index of 0 corresponds to the first variant across all chromosomes. Thus, if you want to map the variant index to per-chromosome VCFs/PGENs, you will
            need to subtract the number of variants on all other chromosomes before the variant index to get the correct variant index in the VCF/PGEN. Relevant values
            can be obtained by instantiating a `gvl.Variants` class from the VCFs/PGENs and accessing the `Variants.records.contig_offsets` attribute.

        Parameters
        ----------
        kind
            The type of sequences to return. Can be one of :code:`"reference"`, :code:`"haplotypes"`, :code:`"annotated"` or :code:`None` to return no sequences.
        """
        match kind, self._seqs, self._tracks, self._recon:
            case None, _, None, _:
                raise ValueError(
                    "Dataset only has sequences available, so returning no sequences is not possible."
                )
            case None, _, _, Haps() | Seqs():
                raise RuntimeError(
                    "Dataset is set to only return sequences, so setting sequence_type to None would"
                    " result in a Dataset that cannot return anything."
                )
            case None, _, _, (Tracks() as t) | SeqsTracks(tracks=t) | HapsTracks(
                tracks=t
            ):
                return evolve(self, _recon=t)
            case "reference" | "haplotypes" | "annotated", None, _, _:
                raise ValueError(
                    "Dataset has no reference genome to reconstruct sequences from."
                )
            case "haplotypes" | "annotated", Seqs(), _, _:
                raise ValueError(
                    "Dataset has no genotypes to reconstruct haplotypes from."
                )
            case "reference", _, _, Seqs(reference=r) | Haps(reference=r):
                seqs = Seqs(reference=r)
                return evolve(self, _recon=seqs)
            case "reference", Seqs(reference=ref) | Haps(reference=ref), _, (
                (Tracks() as tracks)
                | SeqsTracks(tracks=tracks)
                | HapsTracks(tracks=tracks)
            ):
                seqs = Seqs(reference=ref)
                return evolve(self, _recon=SeqsTracks(seqs=seqs, tracks=tracks))
            case "haplotypes", Haps() as haps, _, Seqs() | Haps():
                return evolve(self, _recon=haps.with_annot(False))
            case "haplotypes", Haps() as haps, _, (
                (Tracks() as tracks)
                | SeqsTracks(tracks=tracks)
                | HapsTracks(tracks=tracks)
            ):
                return evolve(self, _recon=HapsTracks(haps.with_annot(False), tracks))
            case "annotated", Haps() as haps, _, Seqs() | Haps():
                return evolve(self, _recon=haps.with_annot(True))
            case "annotated", Haps() as haps, _, (
                (Tracks() as tracks)
                | SeqsTracks(tracks=tracks)
                | HapsTracks(tracks=tracks)
            ):
                return evolve(self, _recon=HapsTracks(haps.with_annot(True), tracks))
            case k, s, t, r:
                assert_never(k), assert_never(s), assert_never(t), assert_never(r)

    def with_tracks(self, tracks: str | List[str] | None):
        """Modify which tracks to return, returning a new dataset without modifying the old one.

        Parameters
        ----------
        tracks
            The tracks to return. Can be a (list of) track names or :code:`False` to return no tracks."""
        match tracks, self._seqs, self._tracks, self._recon:
            case None, None, _, _:
                raise ValueError(
                    "Dataset only has tracks available, so returning no tracks would"
                    " result in a Dataset that cannot return anything."
                )
            case None, Seqs() | Haps(), _, Tracks():
                raise RuntimeError(
                    "Dataset is set to only return tracks, so setting tracks to None would"
                    " result in a Dataset that cannot return anything."
                )
            case None, _, _, ((Seqs() | Haps()) as seqs) | SeqsTracks(
                seqs=seqs
            ) | HapsTracks(haps=seqs):
                return evolve(self, _recon=seqs)
            case t, _, None, _:
                raise ValueError(
                    "Can't set dataset to return tracks because it has none to begin with."
                )
            case t, _, _, Tracks() as tr:
                return evolve(self, _recon=tr.with_tracks(t))
            case t, _, tr, (Seqs() as seqs) | SeqsTracks(seqs=seqs):
                return evolve(self, _recon=SeqsTracks(seqs, tr.with_tracks(t)))
            case t, _, tr, (Haps() as haps) | HapsTracks(haps=haps):
                return evolve(
                    self,
                    _recon=HapsTracks(
                        haps,  # type: ignore | pylance weirdly infers HapsTracks[RaggedAnnotatedHaps]
                        tr.with_tracks(t),
                    ),
                )
            case k, s, t, r:
                assert_never(k), assert_never(s), assert_never(t), assert_never(r)

    path: Path
    """Path to the dataset."""
    output_length: Literal["ragged", "variable"] | int
    """The output length. Can be set to :code:`"ragged"` or :code:`"variable"` to allow for variable length sequences.
    If set to an integer, all sequences will be padded or truncated to this length. See the
    `online documentation <https://genvarloader.readthedocs.io/en/latest/dataset.html>`_ for more information."""
    max_jitter: int
    """Maximum jitter."""
    return_indices: bool
    """Whether to return row and sample indices corresponding to the full dataset (no subsetting)."""
    contigs: List[str]
    """List of unique contigs."""
    jitter: int
    """How much jitter to use."""
    deterministic: bool
    """Whether to use randomized or deterministic algorithms. If set to :code:`False`, this will enable random
    shifting of longer-than-requested haplotypes and, for unphased variants, enable choosing sets of compatible variants proportional to their CCF;
    otherwise the dataset will always apply compatible sets with the highest CCF.
    
    .. note::
        This setting is independent of :attr:`~Dataset.jitter`, if you want no :attr:`~Dataset.jitter` you should set it to 0.
    """
    rc_neg: bool
    """Whether to reverse-complement the sequences on negative strands."""
    transform: Callable | None
    """Tranform to apply to what the dataset would otherwise return on its own."""
    _full_bed: pl.DataFrame = field(alias="_full_bed")
    _full_regions: NDArray[np.int32] = field(alias="_full_regions")
    """Unjittered, sorted regions matching order on-disk."""
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
        """Return the shape of the dataset. :code:`(n_samples, n_regions)`"""
        return self.n_regions, self.n_samples

    @property
    def full_shape(self) -> Tuple[int, int]:
        """Return the full shape of the dataset, ignoring any subsetting. :code:`(n_samples, n_regions)`"""
        return self._idxer.full_shape

    @property
    def available_tracks(self) -> List[str] | None:
        """The available tracks in the dataset."""
        if self._tracks is None:
            return
        return list(self._tracks.intervals)

    @property
    def active_tracks(self) -> List[str] | None:
        """The active tracks in the dataset."""
        if self._tracks is None:
            return
        return list(self._tracks.active_tracks)

    @property
    def _available_sequences(self) -> list[str] | None:
        """The available sequences in the dataset."""
        match self._seqs:
            case None:
                return None
            case Seqs():
                return ["reference"]
            case Haps():
                return ["reference", "haplotypes", "annotated"]
            case s:
                assert_never(s)

    @property
    def sequence_type(self) -> Literal["haplotypes", "reference", "annotated"] | None:
        """The type of sequences in the dataset."""
        match self._recon:
            case Tracks():
                return
            case (Haps() as haps) | HapsTracks(haps=haps):
                if haps.annotate:
                    return "annotated"
                return "haplotypes"
            case Seqs() | SeqsTracks():
                return "reference"
            case r:
                assert_never(r)

    def __len__(self):
        return self.n_regions * self.n_samples

    def __str__(self) -> str:
        if self._available_sequences is None or self.sequence_type is None:
            seq_type = "None"
        else:
            seqs = self._available_sequences
            seqs[seqs.index(self.sequence_type)] = f"[{self.sequence_type}]"
            seq_type = " ".join(seqs)

        if self.available_tracks is None:
            tracks = None
        else:
            tracks = f"{', '.join(self.available_tracks[:5])}"
            if len(self.available_tracks) > 5:
                tracks += f" + {len(self.available_tracks) - 5} more"

        if self.active_tracks is None:
            act_tracks = None
        else:
            act_tracks = f"{', '.join(self.active_tracks[:5])}"
            if len(self.active_tracks) > 5:
                act_tracks += f" + {len(self.active_tracks) - 5} more"
        return (
            f"GVL store at {self.path}\n"
            f"Is subset: {self.is_subset}\n"
            f"# of regions: {self.n_regions}\n"
            f"# of samples: {self.n_samples}\n"
            f"Output length: {self.output_length}\n"
            f"Jitter: {self.jitter} (max: {self.max_jitter})\n"
            f"Deterministic: {self.deterministic}\n"
            f"Sequence type: {seq_type}\n"
            f"Active tracks: {act_tracks}\n"
            f"Tracks available: {tracks}\n"
        )

    def __repr__(self) -> str:
        return str(self)

    def subset_to(
        self,
        regions: Idx | pl.Series | None = None,
        samples: Idx | str | Sequence[str] | None = None,
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
            if isinstance(samples, np.ndarray) and np.issubdtype(
                samples.dtype, np.bool_
            ):
                sample_idx = np.nonzero(samples)[0]
            elif isinstance(
                samples, (int, np.integer, slice, np.ndarray)
            ) or isinstance(samples[0], int):
                sample_idx = idx_like_to_array(samples, self.n_samples)  # type: ignore
            else:  # str | Sequence
                if isinstance(samples, str):
                    samples = [samples]
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
        else:
            sample_idx = None

        if regions is not None:
            if isinstance(regions, pl.Series):
                region_idxs = regions.to_numpy()
                if np.issubdtype(region_idxs.dtype, np.bool_):
                    region_idxs = np.nonzero(region_idxs)[0]
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

    def haplotype_lengths(
        self,
        regions: Idx | None = None,
        samples: Idx | str | Sequence[str] | None = None,
    ) -> NDArray[np.int32] | None:
        """The lengths of jitter-extended haplotypes for specified regions and samples. If the dataset is
        not phased or not deterministic, this will return :code:`None` because the haplotypes are not guaranteed to be
        a consistent length due to randomness in what variants are used.

        Parameters
        ----------
        regions
            Regions to compute haplotype lengths for.
        samples
            Samples to compute haplotype lengths for.
        """
        if (
            not isinstance(self._seqs, Haps)
            or not isinstance(self._seqs.genotypes, SparseGenotypes)
            or not self.deterministic
        ):
            return None

        if regions is None:
            regions = slice(None)
        if samples is None:
            samples = slice(None)
        idx = (regions, samples)

        ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)

        r_idx, _ = np.unravel_index(ds_idx, self.full_shape)

        # (b)
        regions = self._jittered_regions[r_idx]

        # (b p)
        hap_lens = (
            regions[:, 2, None]  # (b 1)
            - regions[:, 1, None]  # (b 1)
            + self._seqs._haplotype_ilens(ds_idx, regions, self.deterministic)  # (b p)
        )

        if squeeze:
            hap_lens = hap_lens.squeeze(0)

        if out_reshape is not None:
            hap_lens = hap_lens.reshape(*out_reshape, self._seqs.genotypes.ploidy)

        return hap_lens

    def write_transformed_track(
        self,
        new_track: str,
        existing_track: str,
        transform: Callable[
            [NDArray[np.intp], NDArray[np.intp], Ragged[np.float32]],
            Ragged[np.float32],
        ],
        max_mem: int = 2**30,
        overwrite: bool = False,
    ) -> ArrayDataset | RaggedDataset:
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
            The arguments given to the transform will be the region and sample indices as numpy arrays
            and the tracks themselves as a :class:`Ragged` array with
            shape (regions, samples). The tracks must be a :class:`Ragged` array since regions may be
            different lengths to accomodate indels. This function should then return the transformed
            tracks as a :class:`Ragged` array with the same shape and lengths.
        max_mem
            The maximum memory to use in bytes, by default 1 GiB (2**30 bytes)
        overwrite
            Whether to overwrite the existing track, by default False
        """
        if self._tracks is None:
            raise ValueError("Dataset has no tracks to transform.")

        new_tracks = self._tracks.write_transformed_track(
            new_track,
            existing_track,
            transform,
            self.path,
            self._full_regions,
            self.max_jitter,
            self._idxer,
            self._seqs if isinstance(self._seqs, Haps) else None,
            max_mem=max_mem,
            overwrite=overwrite,
        )

        return evolve(self, _tracks=new_tracks)  # type: ignore

    def to_torch_dataset(self) -> "td.Dataset":
        """Convert the dataset to a PyTorch :external+torch:class:`Dataset <torch.utils.data.Dataset>`. Requires PyTorch to be installed."""
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
        # (b)
        ds_idx, squeeze, out_reshape = self._idxer.parse_idx(idx)
        r_idx, s_idx = np.unravel_index(ds_idx, self.full_shape)
        regions = self._jittered_regions[r_idx]

        recon = self._recon(
            ds_idx,
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
                rag = _reverse_complement(rag, to_rc)
            elif is_rag_dtype(rag, np.float32):
                _reverse(rag, to_rc)
        elif isinstance(rag, RaggedAnnotatedHaps):
            rag.haps = _reverse_complement(rag.haps, to_rc)
            _reverse(rag.var_idxs, to_rc)
            _reverse(rag.ref_coords, to_rc)
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
IDX = TypeVar("IDX", None, tuple[NDArray[np.integer], NDArray[np.integer]])
TFM = TypeVar("TFM")


class ArrayDataset(Dataset, Generic[SEQ, TRK, IDX, TFM]):
    """Only for type checking purposes, you should never instantiate this class directly."""

    @overload
    def with_len(
        self: ArrayDataset[NDArray[np.bytes_], None, IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[Ragged[np.bytes_], None, IDX, TFM]: ...
    @overload
    def with_len(
        self: ArrayDataset[AnnotatedHaps, None, IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedAnnotatedHaps, None, IDX, TFM]: ...
    @overload
    def with_len(
        self: ArrayDataset[None, NDArray[np.float32], IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[None, Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_len(
        self: ArrayDataset[NDArray[np.bytes_], NDArray[np.float32], IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[Ragged[np.bytes_], Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_len(
        self: ArrayDataset[AnnotatedHaps, NDArray[np.float32], IDX, TFM],
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RaggedAnnotatedHaps, Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_len(
        self,
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[SEQ, TRK, IDX, TFM]: ...
    def with_len(
        self, output_length: Literal["ragged", "variable"] | int
    ) -> Union[RaggedDataset[RSEQ, RTRK, IDX, TFM], ArrayDataset[SEQ, TRK, IDX, TFM]]:
        return super().with_len(output_length)

    @overload
    def with_seqs(self, kind: None) -> ArrayDataset[None, TRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["reference", "haplotypes"]
    ) -> ArrayDataset[NDArray[np.bytes_], TRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["annotated"]
    ) -> ArrayDataset[AnnotatedHaps, TRK, IDX, TFM]: ...
    def with_seqs(
        self, kind: Literal["reference", "haplotypes", "annotated"] | None
    ) -> ArrayDataset:
        return super().with_seqs(kind)

    @overload
    def with_tracks(self, tracks: None) -> ArrayDataset[SEQ, None, IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: str
    ) -> ArrayDataset[SEQ, NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: List[str]
    ) -> ArrayDataset[SEQ, NDArray[np.float32], IDX, TFM]: ...
    def with_tracks(self, tracks: str | List[str] | None) -> ArrayDataset:
        return super().with_tracks(tracks)

    @overload
    def with_indices(
        self, return_indices: Literal[False]
    ) -> ArrayDataset[SEQ, TRK, None, TFM]: ...
    @overload
    def with_indices(
        self, return_indices: Literal[True]
    ) -> ArrayDataset[
        SEQ, TRK, tuple[NDArray[np.integer], NDArray[np.integer]], TFM
    ]: ...
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
    """Only for type checking purposes, you should never instantiate this class directly."""

    @overload
    def with_len(
        self: RaggedDataset[Ragged[np.bytes_], None, IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[NDArray[np.bytes_], None, IDX, TFM]: ...
    @overload
    def with_len(
        self: RaggedDataset[RaggedAnnotatedHaps, None, IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[AnnotatedHaps, None, IDX, TFM]: ...
    @overload
    def with_len(
        self: RaggedDataset[None, Ragged[np.float32], IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[None, NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_len(
        self: RaggedDataset[None, RTRK, IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[None, TRK, IDX, TFM]: ...
    @overload
    def with_len(
        self: RaggedDataset[Ragged[np.bytes_], Ragged[np.float32], IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[NDArray[np.bytes_], NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_len(
        self: RaggedDataset[RaggedAnnotatedHaps, Ragged[np.float32], IDX, TFM],
        output_length: Union[Literal["variable"], int],
    ) -> ArrayDataset[AnnotatedHaps, NDArray[np.float32], IDX, TFM]: ...
    @overload
    def with_len(
        self,
        output_length: Literal["ragged"],
    ) -> RaggedDataset[RSEQ, RTRK, IDX, TFM]: ...
    def with_len(
        self, output_length: Union[Literal["ragged", "variable"], int]
    ) -> Union[RaggedDataset[RSEQ, RTRK, IDX, TFM], ArrayDataset[SEQ, TRK, IDX, TFM]]:
        return super().with_len(output_length)

    @overload
    def with_seqs(self, kind: None) -> RaggedDataset[None, RTRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["reference", "haplotypes"]
    ) -> RaggedDataset[Ragged[np.bytes_], RTRK, IDX, TFM]: ...
    @overload
    def with_seqs(
        self, kind: Literal["annotated"]
    ) -> RaggedDataset[RaggedAnnotatedHaps, RTRK, IDX, TFM]: ...
    def with_seqs(
        self, kind: Literal["reference", "haplotypes", "annotated"] | None
    ) -> RaggedDataset:
        return super().with_seqs(kind)

    @overload
    def with_tracks(self, tracks: None) -> RaggedDataset[RSEQ, None, IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: str
    ) -> RaggedDataset[RSEQ, Ragged[np.float32], IDX, TFM]: ...
    @overload
    def with_tracks(
        self, tracks: List[str]
    ) -> RaggedDataset[RSEQ, Ragged[np.float32], IDX, TFM]: ...
    def with_tracks(self, tracks: str | List[str] | None) -> RaggedDataset:
        return super().with_tracks(tracks)

    @overload
    def with_indices(
        self, return_indices: Literal[False]
    ) -> RaggedDataset[RSEQ, RTRK, None, TFM]: ...
    @overload
    def with_indices(
        self, return_indices: Literal[True]
    ) -> RaggedDataset[
        RSEQ, RTRK, tuple[NDArray[np.integer], NDArray[np.integer]], TFM
    ]: ...
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
