"""``OpenRequest`` — staged construction for :meth:`Dataset.open`.

``Dataset.open`` is a thin facade that packages its arguments into an
:class:`OpenRequest` and calls :meth:`OpenRequest.resolve`. The resolution
stages live as small helpers on this class so each step can be read and
reasoned about in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import polars as pl
import seqpro as sp
from genoray._utils import ContigNormalizer
from loguru import logger
from numpy.typing import NDArray

from ._indexing import DatasetIndexer
from ._reconstruct import Haps, Ref, Tracks, _build_reconstructor
from ._reference import Reference
from ._utils import bed_to_regions
from ._validate import validate_dataset
from ._write import Metadata

if TYPE_CHECKING:
    from ._impl import RaggedDataset


_py_open = open


SeqsKind = Literal["haplotypes", "reference", "annotated", "variants"] | None


@dataclass(frozen=True, slots=True)
class OpenRequest:
    """Parsed, validated arguments for opening a dataset.

    Construct directly or via :meth:`Dataset.open`. Call :meth:`resolve` to
    produce the dataset.
    """

    path: Path
    reference: str | Path | Reference | None = None
    jitter: int = 0
    rng: int | np.random.Generator | None = False
    deterministic: bool = True
    rc_neg: bool = True
    min_af: float | None = None
    max_af: float | None = None
    region_names: str | None = None
    splice_info: str | tuple[str, str] | None = None
    var_filter: Literal["exonic"] | None = None
    svar: str | Path | None = None
    var_fields: list[str] | None = None

    def resolve(self) -> RaggedDataset:
        """Resolve the request into a :class:`Dataset`."""
        self._validate_path()
        metadata = self._load_metadata()
        idxer, bed, regions = self._build_indexer(metadata)
        reference = self._resolve_reference(metadata.contigs)
        seqs = self._build_seqs(metadata, reference, regions)
        tracks = self._build_tracks(len(regions), len(metadata.samples))

        if seqs is None and tracks is None:
            raise RuntimeError(
                "Malformed dataset: neither genotypes nor intervals found."
            )

        seqs_kind = self._initial_seqs_kind(seqs)
        recon = _build_reconstructor(seqs, tracks, seqs_kind)

        self._check_reference_bounds(bed, seqs, reference)

        dataset = self._assemble_dataset(
            metadata=metadata,
            bed=bed,
            regions=regions,
            idxer=idxer,
            seqs=seqs,
            tracks=tracks,
            seqs_kind=seqs_kind,
            recon=recon,
        )

        dataset = self._apply_post_settings(dataset)

        logger.info(f"Opened dataset:\n{dataset}")
        return dataset

    # ---- stages ----

    def _validate_path(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist.")

    def _load_metadata(self) -> Metadata:
        with _py_open(self.path / "metadata.json") as f:
            metadata = Metadata.model_validate_json(f.read())
        validate_dataset(metadata, self.path)
        return metadata

    def _build_indexer(
        self, metadata: Metadata
    ) -> tuple[DatasetIndexer, pl.DataFrame, NDArray[np.int32]]:
        bed = pl.read_ipc(self.path / "input_regions.arrow")
        if self.region_names is not None:
            _region_names = bed[self.region_names].to_list()
        else:
            _region_names = None
        r_idx_map = bed["r_idx_map"].to_numpy().astype(np.intp)
        idxer = DatasetIndexer.from_region_and_sample_idxs(
            r_idxs=r_idx_map,
            s_idxs=np.arange(len(metadata.samples)),
            samples=metadata.samples,
            regions=_region_names,
        )
        bed = bed.drop("r_idx_map")
        sorted_bed = sp.bed.sort(bed)
        regions = bed_to_regions(sorted_bed, ContigNormalizer(metadata.contigs))
        return idxer, bed, regions

    def _resolve_reference(self, contigs: list[str]) -> Reference | None:
        if isinstance(self.reference, (str, Path)):
            return Reference.from_path(self.reference, contigs)
        return self.reference

    def _has_genotypes(self) -> bool:
        return (self.path / "genotypes").exists()

    def _has_intervals(self) -> bool:
        return (self.path / "intervals").exists() or (
            self.path / "annot_intervals"
        ).exists()

    def _build_seqs(
        self,
        metadata: Metadata,
        reference: Reference | None,
        regions: NDArray[np.int32],
    ) -> Haps | Ref | None:
        if self._has_genotypes():
            if metadata.ploidy is None:
                raise ValueError("Malformed dataset: found genotypes but not ploidy.")
            seqs = Haps.from_path(
                path=self.path,
                reference=reference,
                regions=regions,
                samples=metadata.samples,
                ploidy=metadata.ploidy,
                version=metadata.version,
                svar_link=metadata.svar_link,
                svar_override=self.svar,
                min_af=self.min_af,
                max_af=self.max_af,
                var_fields=self.var_fields,
            )
            if reference is None:
                logger.warning(
                    "No reference: dataset only has genotypes but no reference was given."
                    " Resulting dataset can only support :code:`.with_seqs('variants')` to return RaggedVariants."
                )
            return seqs
        if reference is not None:
            return Ref(reference=reference)
        return None

    def _build_tracks(self, n_regions: int, n_samples: int) -> Tracks | None:
        if not self._has_intervals():
            return None
        tracks = Tracks.from_path(self.path, n_regions, n_samples)
        return tracks.with_tracks(list(tracks.intervals))

    @staticmethod
    def _initial_seqs_kind(seqs: Haps | Ref | None) -> SeqsKind:
        # Default view kind for each storage shape.
        if isinstance(seqs, Haps):
            # Without a reference we can't reconstruct haplotypes; the only
            # sequence view Haps.to_kind allows is RaggedVariants.
            return "haplotypes" if seqs.reference is not None else "variants"
        if isinstance(seqs, Ref):
            return "reference"
        return None

    @staticmethod
    def _check_reference_bounds(
        bed: pl.DataFrame, seqs: Haps | Ref | None, reference: Reference | None
    ) -> None:
        if seqs is None or reference is None:
            return
        cnorm = ContigNormalizer(reference.contigs)
        contig_lengths = dict(zip(reference.contigs, np.diff(reference.offsets)))
        ds_contigs = bed["chrom"].unique().to_list()
        normed_contigs = cnorm.norm(ds_contigs)
        if any(c is None for c in normed_contigs):
            raise ValueError(
                "Some regions in the dataset can not be mapped to a contig in the reference genome."
            )
        normed_contigs = cast(list[str], normed_contigs)
        replacer = {
            c: contig_lengths[norm_c] for c, norm_c in zip(ds_contigs, normed_contigs)
        }
        out_of_bounds = bed.select(
            (pl.col("chromStart") >= pl.col("chrom").replace_strict(replacer)).any()
        ).item()
        if out_of_bounds:
            logger.warning(
                "Some regions in the dataset have a start coordinate that is out"
                " of bounds for the reference genome provided. This may happen if"
                " the dataset's regions are for a different reference genome."
            )

    def _assemble_dataset(
        self,
        metadata: Metadata,
        bed: pl.DataFrame,
        regions: NDArray[np.int32],
        idxer: DatasetIndexer,
        seqs: Haps | Ref | None,
        tracks: Tracks | None,
        seqs_kind: SeqsKind,
        recon,
    ) -> RaggedDataset:
        from ._impl import RaggedDataset

        return RaggedDataset(
            path=self.path,
            output_length="ragged",
            max_jitter=metadata.max_jitter,
            jitter=self.jitter,
            contigs=metadata.contigs,
            return_indices=False,
            rc_neg=self.rc_neg,
            deterministic=self.deterministic,
            _idxer=idxer,
            _sp_idxer=None,
            _full_bed=bed,
            _spliced_bed=None,
            _full_regions=regions,
            _seqs=seqs,
            _tracks=tracks,
            _seqs_kind=seqs_kind,
            _recon=recon,
            _rng=np.random.default_rng(self.rng),
        )

    def _apply_post_settings(self, dataset: RaggedDataset) -> RaggedDataset:
        if self.splice_info is None and self.var_filter is None:
            return dataset
        # splice_info is only valid with haplotypes/reference sequence types.
        # If the dataset still has the default "variants" sequence type (i.e.
        # the caller hasn't called with_seqs yet), promote to "haplotypes" so
        # _check_valid_state() doesn't reject splice_info=... up front.
        if (
            self.splice_info is not None
            and isinstance(dataset._seqs, Haps)
            and dataset.sequence_type == "variants"
        ):
            dataset = dataset.with_seqs("haplotypes")
        return dataset.with_settings(
            splice_info=self.splice_info,
            var_filter=self.var_filter,
        )
