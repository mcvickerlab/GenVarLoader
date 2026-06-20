import gc
import json
import warnings
from collections.abc import Callable, Iterator, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .._bigwig import BigWigs
    from .._table import Table
    from .._types import IntervalTrack
    from .._ragged import RaggedIntervals
    from ._impl import Dataset

import awkward as ak
import numpy as np
import polars as pl
import seqpro as sp
from genoray import PGEN, VCF, Reader, SparseVar
from genoray import exprs as _gexprs
from genoray._svar import dense2sparse
from genoray._svar import _dense2sparse_with_length  # type: ignore[missing-module-attribute]
from genoray._types import V_IDX_TYPE
from genoray._utils import ContigNormalizer, format_memory, parse_memory
from joblib import Parallel, delayed
from loguru import logger
from more_itertools import mark_ends
from natsort import natsorted
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion
from seqpro.rag import Ragged
from tqdm.auto import tqdm

from .._atomic import atomic_dir
from .._ragged import INTERVAL_DTYPE
from .._utils import lengths_to_offsets, normalize_contig_name
from .._variants._utils import path_is_pgen, path_is_vcf
from ._svar_link import SvarLink
from ._utils import bed_to_regions, regions_to_bed, splits_sum_le_value


DATASET_FORMAT_VERSION = SemanticVersion.parse("1.0.0")
"""On-disk layout version for a gvl.write dataset directory. Bump MAJOR only when
an existing dataset can no longer be read correctly by new code."""


def _run_jobs(jobs: "list[Callable[[int], None]]", max_mem: int) -> None:
    """Run track/annot writer jobs, each called with a per-job max_mem budget.

    0/1 real jobs run inline; otherwise jobs run concurrently on the loky
    backend with the budget divided evenly so total peak stays under max_mem.
    None entries in *jobs* are silently filtered out.
    """
    jobs = [j for j in jobs if j is not None]
    if len(jobs) <= 1:
        for j in jobs:
            j(max_mem)
        return
    per = max(max_mem // len(jobs), 1)
    Parallel(n_jobs=len(jobs), backend="loky")(delayed(j)(per) for j in jobs)


class Metadata(BaseModel, arbitrary_types_allowed=True):
    samples: list[str]
    contigs: list[str]
    n_regions: int
    ploidy: int | None = None
    max_jitter: int = 0
    version: SemanticVersion | None = None
    format_version: SemanticVersion | None = None
    svar_link: SvarLink | None = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def write(
    path: str | Path,
    bed: str | Path | pl.DataFrame,
    variants: str | Path | Reader | None = None,
    tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
    annot_tracks: "dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None" = None,
    samples: list[str] | None = None,
    max_jitter: int | None = None,
    overwrite: bool = False,
    max_mem: int | str = "4g",
    extend_to_length: bool = True,
):
    """Write a GVL dataset.

    Parameters
    ----------
    path
        Path to write the dataset to.
    bed
        :func:`BED-like <genvarloader.read_bedlike()>` file or DataFrame of regions satisfying the BED3+ specification.
        Specifically, it must have columns 'chrom', 'chromStart', and 'chromEnd'. If 'strand' is present, its values must be either '+' or '-'.
        Negative stranded regions will be reverse complemented during sequence and/or track reconstruction.
    variants
        A :code:`genoray` VCF or PGEN instance (:code:`genoray` is a GVL dependency so it will be import-able). All variants must be
        left-aligned, bi-allelic, and atomized. Multi-allelic variants can be included by splitting
        them into bi-allelic half-calls. For VCFs, the `bcftools norm <https://samtools.github.io/bcftools/bcftools.html#norm>`_
        command can do all of this normalization. Likewise, see the `PLINK2 documentation <https://www.cog-genomics.org/plink/2.0>`_
        for PGEN files. Commands of interest include :code:`--make-bpgen` for splitting variants,
        :code:`--normalize` for left-aligning and atomizing overlapping variants, and :code:`--ref-from-fa` for REF allele correction.
    tracks
        An :class:`IntervalTrack` (e.g. :class:`BigWigs`, :class:`Table`) or a
        sequence of them. Each track must have a unique ``name``; the on-disk
        layout writes to ``<path>/intervals/<track.name>/``.
    annot_tracks
        Sample-independent annotation tracks, as a mapping of track name to source.
        Each source is a path to an interval table, a path to a bigWig, or a polars
        DataFrame/LazyFrame interpreted as a BED-like interval table (columns ``chrom``,
        ``chromStart``, ``chromEnd``, ``score``). Table/DataFrame sources are served by
        the Rust COITrees overlap backend. Written to ``<path>/annot_intervals/<name>/``.
    samples
        Samples to include in the dataset
    max_jitter
        Maximum jitter to add to the regions
    overwrite
        Whether to overwrite an existing dataset
    max_mem
        Approximate maximum total memory to use, including the genoray variant
        index. The reader's index is loaded eagerly at the start of
        :func:`write` (for :class:`~genoray.VCF` and :class:`~genoray.PGEN`)
        so that :attr:`~genoray.VCF.nbytes` reflects its true size; that value
        is subtracted from ``max_mem`` to determine the budget available for
        genotype chunking. A :class:`ValueError` is raised if the remaining
        budget is too small to fit even a single variant chunk. Otherwise
        ``max_mem`` is a soft limit on overall usage and may be exceeded by
        a small amount.
    extend_to_length
        Whether to continue reading/writing variants until all haplotypes have a length at least as long as the intervals in `bed`.
        Otherwise, deletions can cause the length of haplotypes to be less than the intervals in `bed`. This can be disabled if having
        haplotypes shorter than the intervals is acceptable, in which case they will be padded with reference bases when appropriate.
        Disabling this also reduces the amount of data read/written and is faster to run.

    Notes
    -----
    The dataset directory is built atomically: all data is written to a private sibling
    temp directory and published via :func:`os.replace`. A best-effort ``filelock``
    prevents redundant parallel rebuilds, but correctness relies on the atomic rename —
    the lock is advisory only.

    Out of scope: ``genoray`` ``.gvi`` index files and ``pysam`` ``.fai``/``.gzi`` index
    files are created by those libraries and are not covered by gvl's atomic/locked
    creation. Concurrent jobs that trigger index creation for those files depend on the
    upstream libraries' behavior.
    """
    # ignore polars warning about os.fork which is caused by using joblib's loky backend
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        if variants is None and tracks is None and annot_tracks is None:
            raise ValueError(
                "At least one of `variants`, `tracks`, or `annot_tracks` must be provided."
            )

        if tracks is not None and not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        elif tracks is not None:
            tracks = list(tracks)

        if tracks is not None:
            names = [t.name for t in tracks]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"Duplicate track names: {names}. Each track must have a unique `name`."
                )

        logger.info(f"Writing dataset to {path}")

        max_mem = parse_memory(max_mem)

        metadata: dict[str, Any] = {
            "version": SemanticVersion.parse(version("genvarloader")),
            "format_version": DATASET_FORMAT_VERSION,
        }
        dest = Path(path)
        with atomic_dir(dest, overwrite=overwrite) as path:
            if isinstance(bed, (str, Path)):
                bed = sp.bed.read(bed)

            gvl_bed, contigs, input_to_sorted_idx_map = _prep_bed(bed, max_jitter)
            bed.with_columns(r_idx_map=pl.Series(input_to_sorted_idx_map)).write_ipc(
                path / "input_regions.arrow"
            )
            metadata["contigs"] = contigs
            if max_jitter is not None:
                metadata["max_jitter"] = max_jitter

            available_samples: set[str] | None = None
            if variants is not None:
                if isinstance(variants, (str, Path)):
                    variants = Path(variants)
                    if path_is_pgen(variants):
                        if variants.suffix == "":
                            variants = variants.with_suffix(".pgen")
                        variants = PGEN(variants)
                    elif path_is_vcf(variants):
                        variants = VCF(variants)
                    elif variants.is_dir() and variants.suffix == ".svar":
                        variants = SparseVar(variants)
                    else:
                        raise ValueError(
                            f"File {variants} has an unrecognized file extension. Please provide either a VCF or PGEN file.`"
                        )

                if available_samples is None:
                    available_samples = set(variants.available_samples)

                # Eagerly load the variant index so max_mem accounting is honest.
                # VCF and PGEN both support lazy-index construction; without this,
                # variants.nbytes returns 0 and the budget overcounts memory.
                if isinstance(variants, VCF):
                    if variants._index is None:
                        if not variants._valid_index():
                            logger.info("VCF genoray index is invalid, writing")
                            variants._write_gvi_index()
                        variants._load_index()
                elif isinstance(variants, PGEN):
                    variants._init_index()

            if tracks is not None:
                unavail = []
                for tr in tracks:
                    if unavailable_contigs := set(contigs) - {
                        normalize_contig_name(c, contigs) for c in tr.contigs
                    }:
                        unavail.append(unavailable_contigs)
                    if available_samples is None:
                        available_samples = set(tr.samples)
                    else:
                        available_samples.intersection_update(tr.samples)
                if unavail:
                    logger.warning(
                        f"Contigs in queries {set().union(*unavail)} are not found in one or more tracks."
                    )

            if available_samples is None:
                raise ValueError(
                    "No samples available across all variant file(s) and/or tracks."
                )

            if samples is None:
                samples = list(available_samples)
            elif missing := (set(samples) - available_samples):
                raise ValueError(f"Samples {missing} not found in variants or tracks.")

            samples.sort()

            if len(samples) == 0:
                raise ValueError(
                    "No samples remain after intersecting variant samples with track"
                    " samples. Check that sample IDs match across variants and tracks."
                )

            logger.info(f"Using {len(samples)} samples.")
            metadata["samples"] = samples
            metadata["n_regions"] = gvl_bed.height

            if variants is not None:
                logger.info("Writing genotypes.")

                effective_max_mem = max_mem
                if isinstance(variants, (VCF, PGEN)):
                    idx_bytes = variants.nbytes
                    effective_max_mem = max_mem - idx_bytes
                    logger.info(
                        f"Variant reader resident size: {format_memory(idx_bytes)}; "
                        f"max_mem budget: {format_memory(max_mem)}; "
                        f"available for chunking: {format_memory(max(effective_max_mem, 0))}"
                    )
                    if isinstance(variants, VCF):
                        bytes_per_var = (
                            variants.n_samples * variants.ploidy
                        )  # Genos8: 1 byte
                    else:
                        bytes_per_var = (
                            variants.n_samples * variants.ploidy * 4
                        )  # int32

                    if effective_max_mem < bytes_per_var:
                        raise ValueError(
                            f"max_mem ({format_memory(max_mem)}) is too small: the variant "
                            f"index alone consumes {format_memory(idx_bytes)}, leaving "
                            f"{format_memory(max(effective_max_mem, 0))} for chunking, but "
                            f"at least {format_memory(bytes_per_var)} is needed per variant. "
                            f"Increase max_mem."
                        )

                if isinstance(variants, VCF):
                    variants.set_samples(samples)
                    gvl_bed = _write_from_vcf(
                        path, gvl_bed, variants, effective_max_mem, extend_to_length
                    )
                elif isinstance(variants, PGEN):
                    variants.set_samples(samples)
                    gvl_bed = _write_from_pgen(
                        path, gvl_bed, variants, effective_max_mem, extend_to_length
                    )
                elif isinstance(variants, SparseVar):
                    gvl_bed, _svar_link = _write_from_svar(
                        path, gvl_bed, variants, samples, extend_to_length
                    )
                    metadata["svar_link"] = _svar_link
                metadata["ploidy"] = variants.ploidy
                # free memory
                del variants
                gc.collect()

            _write_regions(path, gvl_bed, contigs)

            jobs: list[Callable[[int], None]] = []
            if tracks is not None:
                _tracks = list(tracks)
                _bed = gvl_bed

                def _tracks_job(
                    mm: int, _tracks: list = _tracks, _bed: pl.DataFrame = _bed
                ) -> None:
                    for tr in _tracks:
                        _write_track(
                            path / "intervals" / tr.name, _bed, tr, samples, mm
                        )

                jobs.append(_tracks_job)

            if annot_tracks is not None:
                annot_bed = regions_to_bed(
                    np.load(path / "regions.npy"), contigs
                ).select("chrom", "chromStart", "chromEnd")
                _annots = dict(annot_tracks)

                def _annot_job(
                    mm: int, _annots: dict = _annots, _bed: pl.DataFrame = annot_bed
                ) -> None:
                    for name, source in _annots.items():
                        _write_annot_track(
                            path / "annot_intervals" / name, _bed, source, mm
                        )

                jobs.append(_annot_job)

            if jobs:
                logger.info(f"Writing {len(jobs)} track categor(ies).")
                _run_jobs(jobs, max_mem)

            _metadata = Metadata(**metadata)
            with open(path / "metadata.json", "w") as f:
                f.write(_metadata.model_dump_json())

        logger.info("Finished writing.")
    finally:
        warnings.simplefilter("default")


def update(
    dataset: "str | Path | Dataset",
    tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
    annot_tracks: "dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None" = None,
    *,
    overwrite: bool = False,
    max_mem: int | str = "4g",
) -> None:
    """Add tracks to an existing on-disk GVL dataset, analogous to :func:`write`.

    Parameters
    ----------
    dataset
        Path to a dataset directory, or an opened :class:`Dataset` (its ``.path`` is used).
        A live dataset can be read while it is being updated; it will not observe the new
        track until reopened.
    tracks
        Per-sample :class:`IntervalTrack` source(s) (:class:`BigWigs`, :class:`Table`),
        written to ``<path>/intervals/<name>/``. The track's sample set must match the
        dataset's exactly (no missing, no extra); samples are reordered to the dataset
        order automatically.
    annot_tracks
        Sample-independent sources, identical to :func:`write`'s ``annot_tracks``, written
        to ``<path>/annot_intervals/<name>/``.
    overwrite
        Replace a track of the same name if present; otherwise adding a duplicate name
        raises ``FileExistsError``.
    max_mem
        Approximate memory budget, divided across concurrently-running categories.
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        from ._impl import Dataset

        path = Path(dataset.path if isinstance(dataset, Dataset) else dataset)
        if not (path / "metadata.json").exists():
            raise FileNotFoundError(f"{path} is not a GVL dataset (no metadata.json).")

        if tracks is None and annot_tracks is None:
            raise ValueError(
                "At least one of `tracks` or `annot_tracks` must be provided."
            )

        meta = Metadata.model_validate_json((path / "metadata.json").read_text())
        contigs = meta.contigs
        ds_samples = meta.samples
        max_mem_b = parse_memory(max_mem)

        if tracks is not None and not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        _tracks = list(tracks) if tracks is not None else []

        names = [tr.name for tr in _tracks]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate track names: {names}. Each track must have a unique `name`."
            )

        # validate strict sample-set agreement for per-sample tracks
        for tr in _tracks:
            if set(tr.samples) != set(ds_samples):
                missing = set(ds_samples) - set(tr.samples)
                extra = set(tr.samples) - set(ds_samples)
                raise ValueError(
                    f"Track {tr.name!r} samples must exactly match the dataset's. "
                    f"missing={missing or '{}'} extra={extra or '{}'}"
                )

        bed = regions_to_bed(np.load(path / "regions.npy"), contigs)
        sample_bed = bed.select("chrom", "chromStart", "chromEnd")
        annot_bed = sample_bed

        jobs: list[Callable[[int], None]] = []

        if _tracks:
            (path / "intervals").mkdir(exist_ok=True)
            _tr = _tracks

            def _tracks_job(
                mm: int, _tr: list = _tr, _bed: pl.DataFrame = sample_bed
            ) -> None:
                for tr in _tr:
                    with atomic_dir(
                        path / "intervals" / tr.name, overwrite=overwrite
                    ) as tmp:
                        _write_track(tmp, _bed, tr, ds_samples, mm)

            jobs.append(_tracks_job)

        if annot_tracks is not None:
            (path / "annot_intervals").mkdir(exist_ok=True)
            _annots = dict(annot_tracks)

            def _annot_job(
                mm: int, _annots: dict = _annots, _bed: pl.DataFrame = annot_bed
            ) -> None:
                for name, source in _annots.items():
                    with atomic_dir(
                        path / "annot_intervals" / name, overwrite=overwrite
                    ) as tmp:
                        _write_annot_track(tmp, _bed, source, mm)

            jobs.append(_annot_job)

        _run_jobs(jobs, max_mem_b)
    finally:
        warnings.simplefilter("default")


def get_splice_bed(
    gtf: str | Path,
    contigs: list[str] | None = None,
    transcript_support_level: str | None = "1",
    require_multiple_of_3: bool = True,
) -> pl.DataFrame:
    """Process a GTF into a BED-compatible DataFrame for splicing datasets.

    The result has columns ``chrom``, ``chromStart`` (0-based), ``chromEnd``,
    ``strand``, ``gene_name``, ``transcript_id``, and ``exon_number``, sorted by
    chromosome (natural order) and ``chromStart``. Pass it directly to
    :func:`gvl.write` for splicing datasets.

    Parameters
    ----------
    gtf
        Path to a GTF file (gzipped or plain) accepted by :func:`seqpro.gtf.scan`.
    contigs
        If provided, keep only rows whose ``seqname`` is in this list.
    transcript_support_level
        If a string, require the GTF ``transcript_support_level`` attribute to
        equal it. ``None`` disables the filter.
    require_multiple_of_3
        If ``True``, keep only transcripts whose summed CDS length is a
        multiple of 3.
    """
    lf = sp.gtf.scan(gtf)

    if contigs is not None:
        lf = lf.filter(pl.col("seqname").is_in(contigs))

    lf = lf.filter(pl.col("feature") == "CDS").rename(
        {
            "seqname": "chrom",
            "start": "chromStart",
            "end": "chromEnd",
        }
    )

    lf = lf.with_columns(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("chromStart") - 1,
        pl.col("strand").cast(pl.Utf8),
        sp.gtf.attr("gene_name"),
        sp.gtf.attr("transcript_id"),
        sp.gtf.attr("exon_number").cast(pl.Int32),
    )

    drop_cols = ["source", "score", "frame", "feature", "attribute"]

    if require_multiple_of_3:
        lf = lf.with_columns(
            transcript_len=(pl.col("chromEnd") - pl.col("chromStart"))
            .sum()
            .over("transcript_id")
        ).filter(pl.col("transcript_len") % 3 == 0)
        drop_cols.append("transcript_len")

    if transcript_support_level is not None:
        lf = lf.filter(
            sp.gtf.attr("transcript_support_level") == transcript_support_level
        )

    df = lf.drop(drop_cols).collect()
    return sp.bed.sort(df)


def _prep_bed(
    bed: pl.DataFrame,
    max_jitter: int | None = None,
) -> tuple[pl.DataFrame, list[str], NDArray[np.intp]]:
    if bed.height == 0:
        raise ValueError("No regions found in the BED file.")

    with pl.StringCache():
        if "strand" not in bed:
            bed = bed.with_columns(strand=pl.lit(1, pl.Int32))
        else:
            bed = bed.with_columns(
                pl.col("strand")
                .cast(pl.Utf8)
                .replace_strict({"+": 1, "-": -1, ".": 1}, return_dtype=pl.Int32)
            )

    bed = bed.select("chrom", "chromStart", "chromEnd", "strand")
    contigs = natsorted(bed["chrom"].unique())
    bed = sp.bed.sort(bed.with_row_index())

    input_to_sorted_idx_map = np.argsort(bed["index"])
    bed = bed.drop("index")

    if max_jitter is not None:
        bed = bed.with_columns(
            chromStart=pl.col("chromStart") - max_jitter,
            chromEnd=pl.col("chromEnd") + max_jitter,
        )

    return bed, contigs, input_to_sorted_idx_map


def _write_regions(path: Path, bed: pl.DataFrame, contigs: list[str]):
    regions = bed_to_regions(bed, ContigNormalizer(contigs))
    np.save(path / "regions.npy", regions)


def _reject_unsupported_variants(index: pl.DataFrame, source: str) -> None:
    """Raise if the variant index contains alleles gvl cannot reconstruct.

    gvl expands each variant's ALT into literal haplotype sequence, so it
    requires bi-allelic, non-symbolic, non-breakend records. This runs over the
    FULL index (post any user-supplied filter), matching the "valid inputs only"
    contract. ``source`` names the input for the error message (e.g. "VCF").
    """
    n_multi, n_sym, n_bnd = index.select(
        n_multi=(pl.col("ALT").list.len() > 1).cast(pl.Int64).sum(),
        n_symbolic=_gexprs.is_symbolic.cast(pl.Int64).sum(),
        n_breakend=_gexprs.is_breakend.cast(pl.Int64).sum(),
    ).row(0)
    if n_multi or n_sym or n_bnd:
        raise ValueError(
            f"{source} contains unsupported variants: {n_multi} multi-allelic, "
            f"{n_sym} symbolic (e.g. <DEL>/<INS>), {n_bnd} breakend. gvl can only "
            f"reconstruct bi-allelic, non-symbolic, non-breakend variants. Remove "
            f"them upstream (bcftools/plink2 — split multi-allelics, drop SVs), or "
            f"construct the genoray reader with a filter such as "
            f"`filter=genoray.exprs.is_biallelic & ~genoray.exprs.is_symbolic & "
            f"~genoray.exprs.is_breakend`."
        )


def _write_from_vcf(
    path: Path, bed: pl.DataFrame, vcf: VCF, max_mem: int, extend_to_length: bool
):
    assert vcf._index is not None, (
        "caller must load the VCF index before _write_from_vcf"
    )

    _reject_unsupported_variants(vcf._index, "VCF")

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "variants.arrow").hardlink_to(vcf._index_path())

    return _write_phased_chunked(
        out_dir, bed, _vcf_region_chunks(bed, vcf, max_mem, extend_to_length)
    )


def _window_to_sparse(
    genos: NDArray[np.integer],
    var_idxs: NDArray[V_IDX_TYPE],
    q_start: int,
    q_end: int,
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    extend_to_length: bool,
) -> Ragged:
    """Convert a full dense region window into per-haplotype sparse genotypes.

    ``genos`` has shape ``(samples, ploidy, variants)`` and must cover the
    entire region window (all genoray memory-chunks concatenated along the
    variant axis). ``var_idxs`` are the window's global variant indices.
    ``v_starts`` (``POS - 1``) and ``ilens`` (``ILEN``) are window-aligned,
    positionally aligned with ``var_idxs``.

    When ``extend_to_length`` is ``True`` this defers to genoray's
    ``_dense2sparse_with_length``, which walks each haplotype's length and keeps
    only the variants it needs to reach ``q_end`` (per-haplotype-minimal,
    identical to ``SparseVar.read_ranges_with_length``). When ``False`` it falls
    back to plain ``dense2sparse`` (every haplotype keeps exactly the variants it
    carries within the window, with no length extension).
    """
    if extend_to_length:
        return _dense2sparse_with_length(
            genos, var_idxs, q_start, q_end, v_starts, ilens
        )
    return dense2sparse(genos, var_idxs)


def _region_end(rag: Ragged, v_ends: NDArray, fallback_end: int) -> int:
    """Per-region chromEnd, floored at the input window so tracks are never
    stored over a truncated region.

    ``rag`` is a sparse ``(samples, ploidy, ~variants)`` Ragged of global
    variant indices. Returns ``max(fallback_end, v_ends[max idx])`` across all
    haplotypes (the furthest retained variant end, but never below the input
    window ``fallback_end``), or ``fallback_end`` when no variant is retained
    (mirrors _write_from_svar).
    """
    if rag.data.size == 0:
        return int(fallback_end)
    return max(int(fallback_end), int(v_ends[int(rag.data.max())]))


def _region_ends_from_list(
    ls_sparse: list[Ragged], v_ends: NDArray, fallback_end: int
) -> int:
    """Same as `_region_end` but over a list of per-chunk Ragged arrays."""
    max_idx = -1
    for rag in ls_sparse:
        if rag.data.size:
            max_idx = max(max_idx, int(rag.data.max()))
    if max_idx < 0:
        return int(fallback_end)
    return max(int(fallback_end), int(v_ends[max_idx]))


def _vcf_region_chunks(
    bed: pl.DataFrame, vcf: VCF, max_mem: int, extend_to_length: bool
) -> Iterator[tuple[list[Ragged], Any, str | None]]:
    assert vcf._index is not None
    pos = vcf._index["POS"].to_numpy()
    ilen_all = vcf._index["ILEN"].list.first().to_numpy()
    # end position of each variant = POS + deletion length (matches _write_from_svar)
    v_ends = pos - np.clip(ilen_all, a_min=None, a_max=0)

    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        # unextended in-range variant indices, split per region
        v_idx, v_offsets = vcf._var_idxs(contig, starts, ends)
        unextended_idxs = np.array_split(v_idx.astype(V_IDX_TYPE), v_offsets[1:-1])

        contig_desc = f"Processing genotypes for {df.height} regions on contig {contig}"
        first_in_contig = True

        if extend_to_length:
            region_iter = vcf._chunk_ranges_with_length(
                contig, starts, ends, max_mem, VCF.Genos8
            )
        else:
            # one generator per region; VCF.chunk takes a single range
            region_iter = (
                vcf.chunk(contig, s, e, max_mem, VCF.Genos8)
                for s, e in zip(starts, ends)
            )

        for ri, range_ in enumerate(region_iter):
            q_start = int(starts[ri])
            q_end = int(ends[ri])
            reg_unext = unextended_idxs[ri]
            desc = contig_desc if first_in_contig else None
            first_in_contig = False

            if extend_to_length:
                # assemble the full window across memory-chunks
                chunk_genos_list: list[NDArray] = []
                n_ext_total = 0
                for _, is_last, (chunk_genos, _chunk_end, n_ext) in mark_ends(range_):
                    chunk_genos_list.append(chunk_genos)
                    if is_last:
                        n_ext_total = n_ext
                genos = np.concatenate(chunk_genos_list, axis=-1)

                if reg_unext.size == 0 and n_ext_total == 0:
                    # empty region: no variants for any sample
                    yield [dense2sparse(genos, reg_unext)], q_end, desc
                    continue

                if n_ext_total > 0:
                    ext_start = int(reg_unext[-1]) + 1
                    ext_idxs = np.arange(
                        ext_start, ext_start + n_ext_total, dtype=V_IDX_TYPE
                    )
                    var_idxs = np.concatenate([reg_unext, ext_idxs])
                else:
                    var_idxs = reg_unext

                v_starts = (pos[var_idxs] - 1).astype(np.int32)
                ilens = ilen_all[var_idxs].astype(np.int32)
                rag = _window_to_sparse(
                    genos, var_idxs, q_start, q_end, v_starts, ilens, True
                )
                region_end = _region_end(rag, v_ends, q_end)
                yield [rag], region_end, desc
            else:
                # no extension: convert each chunk independently with plain
                # dense2sparse; var_idxs are exactly the unextended in-range ones
                ls_sparse: list[Ragged] = []
                offset = 0
                for genos in range_:
                    n_vars = genos.shape[-1]
                    chunk_idxs = reg_unext[offset : offset + n_vars]
                    offset += n_vars
                    ls_sparse.append(dense2sparse(genos, chunk_idxs))
                assert offset == reg_unext.size, (
                    f"VCF.chunk variant count ({offset}) != _var_idxs count ({reg_unext.size}) "
                    f"for region [{q_start}, {q_end})"
                )
                if not ls_sparse:
                    empty_genos = np.empty(
                        (vcf.n_samples, vcf.ploidy, 0), dtype=np.int8
                    )
                    ls_sparse = [dense2sparse(empty_genos, reg_unext)]
                region_end = _region_ends_from_list(ls_sparse, v_ends, q_end)
                yield ls_sparse, region_end, desc


def _write_from_pgen(
    path: Path, bed: pl.DataFrame, pgen: PGEN, max_mem: int, extend_to_length: bool
):
    assert pgen._index is not None, (
        "caller must init the PGEN index before _write_from_pgen"
    )
    _reject_unsupported_variants(pgen._index, "PGEN")
    # _sei is genoray's sparse-extraction index; it is None iff some record is
    # not bi-allelic (ALT count != 1). The validator above rejects records with
    # ALT count > 1, and real PGEN sites always carry >= 1 ALT, so once
    # validation passes _sei is non-None for any genuine PGEN input. A None here
    # therefore signals a genoray-internal failure, not unhandled bad input.
    assert pgen._sei is not None, (
        "PGEN sparse-extraction index is None despite passing variant validation"
    )

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "variants.arrow").hardlink_to(pgen._index_path())

    return _write_phased_chunked(
        out_dir, bed, _pgen_region_chunks(bed, pgen, max_mem, extend_to_length)
    )


def _pgen_region_chunks(
    bed: pl.DataFrame, pgen: PGEN, max_mem: int, extend_to_length: bool
) -> Iterator[tuple[list[Ragged], Any, str | None]]:
    assert pgen._index is not None
    pos = pgen._index["POS"].to_numpy()
    ilen_all = pgen._index["ILEN"].list.first().to_numpy()
    v_ends = pos - np.clip(ilen_all, a_min=None, a_max=0)

    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        contig_desc = f"Processing genotypes for {df.height} regions on contig {contig}"
        first_in_contig = True

        unextended_idxs: list[NDArray] = []
        if extend_to_length:
            region_iter = pgen._chunk_ranges_with_length(contig, starts, ends, max_mem)
        else:
            v_idx, v_offsets = pgen.var_idxs(contig, starts, ends)
            unextended_idxs = np.array_split(v_idx.astype(V_IDX_TYPE), v_offsets[1:-1])
            region_iter = (
                pgen.chunk(contig, int(s), int(e), max_mem)
                for s, e in zip(starts, ends)
            )

        for ri, range_ in enumerate(region_iter):
            q_start = int(starts[ri])
            q_end = int(ends[ri])
            desc = contig_desc if first_in_contig else None
            first_in_contig = False

            if extend_to_length:
                genos_list: list[NDArray] = []
                idx_list: list[NDArray] = []
                for genos, _chunk_end, chunk_idxs in range_:
                    genos_list.append(genos.astype(np.int8))
                    idx_list.append(chunk_idxs.astype(V_IDX_TYPE))
                genos = np.concatenate(genos_list, axis=-1)
                var_idxs = (
                    np.concatenate(idx_list)
                    if idx_list
                    else np.empty(0, dtype=V_IDX_TYPE)
                )

                if var_idxs.size == 0:
                    yield [dense2sparse(genos, var_idxs)], q_end, desc
                    continue

                v_starts = (pos[var_idxs] - 1).astype(np.int32)
                ilens = ilen_all[var_idxs].astype(np.int32)
                rag = _window_to_sparse(
                    genos, var_idxs, q_start, q_end, v_starts, ilens, True
                )
                region_end = _region_end(rag, v_ends, q_end)
                yield [rag], region_end, desc
            else:
                reg_unext = unextended_idxs[ri]
                ls_sparse: list[Ragged] = []
                offset = 0
                for genos in range_:
                    n_vars = genos.shape[-1]
                    chunk_idxs = reg_unext[offset : offset + n_vars]
                    offset += n_vars
                    ls_sparse.append(dense2sparse(genos.astype(np.int8), chunk_idxs))
                assert offset == reg_unext.size, (
                    f"PGEN.chunk variant count ({offset}) != var_idxs count "
                    f"({reg_unext.size}) for region [{q_start}, {q_end})"
                )
                if not ls_sparse:
                    empty_genos = np.empty(
                        (pgen.n_samples, pgen.ploidy, 0), dtype=np.int8
                    )
                    ls_sparse = [dense2sparse(empty_genos, reg_unext)]
                region_end = _region_ends_from_list(ls_sparse, v_ends, q_end)
                yield ls_sparse, region_end, desc


def _write_phased_chunked(
    out_dir: Path,
    bed: pl.DataFrame,
    region_iter: Iterator[tuple[list[Ragged], Any, str | None]],
) -> pl.DataFrame:
    """Aggregate per-region sparse genotype chunks and write them to ``out_dir``.

    ``region_iter`` yields one ``(ls_sparse, region_end, pbar_desc)`` per region
    in ``bed`` order. ``pbar_desc`` updates the progress bar description (used at
    the first region of a new contig); ``None`` leaves it unchanged.
    """
    v_idx_memmap_offsets = 0
    offset_memmap_offsets = 0
    last_offset = 0
    max_ends: list[Any] = []
    pbar = tqdm(total=bed.height, unit=" region")
    first_no_variant_warning = True

    for ls_sparse, region_end, desc in region_iter:
        if desc is not None:
            pbar.set_description(desc)
        max_ends.append(region_end)

        var_idxs = ak.flatten(
            ak.concatenate(ls_sparse, -1),
            None,
        ).to_numpy()
        # (s p)
        lengths = np.stack([a.lengths for a in ls_sparse], 0).sum(0)

        if first_no_variant_warning and (lengths == 0).all():
            first_no_variant_warning = False
            logger.warning(
                "A region has no variants for any sample. This could be expected depending on the region lengths"
                " and source of variants. However, this can also be caused by a mismatch between the"
                " reference genome used for the BED file coordinates and the one used for the variants."
            )

        sp_genos = Ragged.from_lengths(var_idxs, lengths)
        (
            v_idx_memmap_offsets,
            offset_memmap_offsets,
            last_offset,
        ) = _write_phased_variants_chunk(
            out_dir,
            sp_genos,
            v_idx_memmap_offsets,
            offset_memmap_offsets,
            last_offset,
        )
        pbar.update()
    pbar.close()

    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=np.int64,
        mode="r+",
        shape=1,
        offset=offset_memmap_offsets,
    )
    out[-1] = last_offset
    out.flush()

    return bed.with_columns(chromEnd=pl.Series(max_ends))


def _write_from_svar(
    path: Path,
    bed: pl.DataFrame,
    svar: SparseVar,
    samples: list[str],
    extend_to_length: bool,
) -> tuple[pl.DataFrame, SvarLink]:
    _reject_unsupported_variants(svar.index, "SVAR")

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = np.memmap(
        out_dir / "offsets.npy",
        np.int64,
        "w+",
        shape=(2, bed.height, len(samples), svar.ploidy),
    )

    with open(out_dir / "svar_meta.json", "w") as f:
        json.dump({"shape": offsets.shape, "dtype": offsets.dtype.str}, f)

    v_ends = svar.index.select(
        end=pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0)
    )["end"].to_numpy()
    max_ends = np.empty(bed.height, np.int32)
    contig_offset = 0
    pbar = tqdm(total=bed.height, unit=" region")
    first_no_variant_warning = True

    for (c,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        c = cast(str, c)
        pbar.set_description(
            f"Processing genotypes for {df.height} regions on contig {c}"
        )
        # set offsets
        # (2 r s p)
        out = offsets[:, contig_offset : contig_offset + df.height]
        if extend_to_length:
            svar._find_starts_ends_with_length(
                c, df["chromStart"], df["chromEnd"], samples=samples, out=out
            )
        else:
            svar._find_starts_ends(
                c, df["chromStart"], df["chromEnd"], samples=samples, out=out
            )

        if first_no_variant_warning and (out == 0).all((1, 2, 3)).any():
            first_no_variant_warning = False
            logger.warning(
                "Some regions have no variants for any sample. This could be expected depending on the region lengths"
                " and source of variants. However, this can also be caused by a mismatch between the"
                " reference genome used for the BED file coordinates and the one used for the variants."
            )

        # compute max_ends for the bed
        shape = (df.height, len(samples), svar.ploidy, None)
        # (r s p ~v)
        sp_genos = Ragged.from_offsets(svar.genos.data, shape, out.reshape(2, -1))
        # this is fine if there aren't any overlapping variants that could make a v_idx < -1
        # have a further end than v_idx == -1
        # * calling ak.max() means v_idxs is not a view of svar.genos.data
        # (r s p ~v) -> (r)
        v_idxs = ak.max(sp_genos, -1).to_numpy().max((1, 2))
        c_max_ends = max_ends[contig_offset : contig_offset + df.height]
        if v_idxs.mask is np.ma.nomask:
            c_max_ends[:] = v_ends[v_idxs.data]
        else:
            c_max_ends[~v_idxs.mask] = v_ends[v_idxs.data[~v_idxs.mask]]
            c_max_ends[v_idxs.mask] = df.filter(v_idxs.mask)["chromEnd"]
        contig_offset += df.height
        pbar.update(df.height)

    pbar.close()
    offsets.flush()

    import os

    from ._svar_link import SvarFingerprint

    svar_resolved = svar.path.resolve()
    variant_idxs_path = svar_resolved / "variant_idxs.npy"
    svar_link = SvarLink(
        relative_path=os.path.relpath(svar_resolved, start=path).replace(os.sep, "/"),
        absolute_path=str(svar_resolved),
        fingerprint=SvarFingerprint(
            n_variants=svar.index.height,
            variant_idxs_bytes=variant_idxs_path.stat().st_size,
        ),
    )

    return bed.with_columns(
        chromEnd=pl.max_horizontal(pl.Series(max_ends), pl.col("chromEnd"))
    ), svar_link


def _write_phased_variants_chunk(
    out_dir: Path,
    genos: Ragged,
    v_idx_memmap_offset: int,
    offsets_memmap_offset: int,
    last_offset: int,
):
    if not genos.is_empty:
        out = np.memmap(
            out_dir / "variant_idxs.npy",
            dtype=genos.data.dtype,
            mode="w+" if v_idx_memmap_offset == 0 else "r+",
            shape=genos.data.shape,
            offset=v_idx_memmap_offset,
        )
        out[:] = genos.data[:]
        out.flush()
        v_idx_memmap_offset += out.nbytes

    offsets = genos.offsets
    offsets += last_offset
    last_offset = offsets[-1]
    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=offsets.dtype,
        mode="w+" if offsets_memmap_offset == 0 else "r+",
        shape=len(offsets) - 1,
        offset=offsets_memmap_offset,
    )
    out[:] = offsets[:-1]
    out.flush()
    offsets_memmap_offset += out.nbytes
    return v_idx_memmap_offset, offsets_memmap_offset, last_offset


def _write_ragged_intervals(out_dir: Path, itvs: "RaggedIntervals") -> None:
    """Write a RaggedIntervals (values/starts/ends share offsets) to out_dir as
    intervals.npy + offsets.npy. Single-chunk writer used for annotation tracks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = np.memmap(
        out_dir / "intervals.npy",
        dtype=INTERVAL_DTYPE,
        mode="w+",
        shape=itvs.values.data.shape,
    )
    out["start"] = itvs.starts.data
    out["end"] = itvs.ends.data
    out["value"] = itvs.values.data
    out.flush()

    offsets = itvs.values.offsets
    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=offsets.dtype,
        mode="w+",
        shape=len(offsets),
    )
    out[:] = offsets
    out.flush()


def _annot_intervals(
    regions: pl.DataFrame,
    source: "str | Path | pl.DataFrame | pl.LazyFrame",
    max_mem: int,
) -> "RaggedIntervals":
    """Build a sample-less RaggedIntervals (n_regions, None) from an annotation source.

    - bigwig path -> Rust per-region extraction (BigWigs), squeezed sample-less.
    - table path / DataFrame / LazyFrame (BED-like: chrom, chromStart, chromEnd, score)
      -> Rust COITrees overlap.
    """
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() in (
        ".bw",
        ".bigwig",
    ):
        return _annot_intervals_from_bigwig(regions, Path(source), max_mem)

    if isinstance(source, pl.LazyFrame):
        annot = source.collect()
    elif isinstance(source, pl.DataFrame):
        annot = source
    else:
        annot = sp.bed.read(str(source))

    from .._table import annot_overlap

    return annot_overlap(regions, annot)


def _annot_intervals_from_bigwig(
    regions: pl.DataFrame, path: Path, max_mem: int
) -> "RaggedIntervals":
    from seqpro.rag import Ragged

    from .._bigwig import BigWigs
    from .._ragged import RaggedIntervals

    # single pseudo-sample; collapse its sample axis to produce a sample-less track
    bw = BigWigs(name="__annot__", paths={"__annot__": str(path)})
    out_starts, out_ends, out_values, lengths = [], [], [], []
    for (contig,), part in regions.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()
        # (regions, 1)
        itvs = bw.intervals(contig, starts, ends, sample="__annot__")
        for r in range(part.height):
            s = itvs.starts[r, 0]
            out_starts.append(np.asarray(s, dtype=np.int32))
            out_ends.append(np.asarray(itvs.ends[r, 0], dtype=np.int32))
            out_values.append(np.asarray(itvs.values[r, 0], dtype=np.float32))
            lengths.append(len(s))
    flat_starts = np.concatenate(out_starts) if out_starts else np.empty(0, np.int32)
    flat_ends = np.concatenate(out_ends) if out_ends else np.empty(0, np.int32)
    flat_values = np.concatenate(out_values) if out_values else np.empty(0, np.float32)
    offsets = lengths_to_offsets(np.asarray(lengths, np.int32))
    shape = (regions.height, None)
    return RaggedIntervals(
        Ragged.from_offsets(flat_starts, shape, offsets),
        Ragged.from_offsets(flat_ends, shape, offsets),
        Ragged.from_offsets(flat_values, shape, offsets),
    )


def _write_annot_track_rust(
    out_dir: Path,
    regions: pl.DataFrame,
    path: Path,
    max_mem: int,
) -> None:
    from .._bigwig import BigWigs
    from ..genvarloader import bigwig_write_track

    out_dir.mkdir(parents=True, exist_ok=True)
    bw = BigWigs(name="__annot__", paths={"__annot__": str(path)})
    contigs: list[str] = []
    starts_l: list[int] = []
    ends_l: list[int] = []
    for chrom, s, e in zip(
        regions["chrom"].to_list(),
        regions["chromStart"].to_list(),
        regions["chromEnd"].to_list(),
    ):
        norm = normalize_contig_name(chrom, bw.contigs)
        if norm is None:
            raise ValueError(f"Contig {chrom!r} not found in bigWig {path}.")
        contigs.append(norm)
        starts_l.append(int(s))
        ends_l.append(int(e))
    bigwig_write_track(
        [str(path)],
        contigs,
        np.asarray(starts_l, dtype=np.int32),
        np.asarray(ends_l, dtype=np.int32),
        int(max_mem),
        str(out_dir),
        True,
    )


def _write_annot_track(
    out_dir: Path,
    regions: pl.DataFrame,
    source: "str | Path | pl.DataFrame | pl.LazyFrame",
    max_mem: int,
) -> None:
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() in (
        ".bw",
        ".bigwig",
    ):
        return _write_annot_track_rust(out_dir, regions, Path(source), max_mem)
    itvs = _annot_intervals(regions, source, max_mem)
    _write_ragged_intervals(out_dir, itvs)


def _write_track_legacy(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "IntervalTrack",
    samples: list[str] | None,
    max_mem: int,
):
    if samples is None:
        _samples = track.samples
    else:
        if missing := (set(samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        _samples = samples

    MEM_PER_INTERVAL = (
        12 * 2
    )  # start u32, end u32, value f32, times 2 for intermediate copies
    chunk_labels = np.empty(bed.height, np.uint32)
    chunk_offsets: dict[int, NDArray[np.int64]] = {}
    n_chunks = 0
    last_chunk_offset = 0
    pbar = tqdm(total=bed["chrom"].n_unique())
    for (contig,), part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        pbar.set_description(f"Calculating memory usage for {part.height} regions")
        contig = cast(str, contig)
        _contig = normalize_contig_name(contig, track.contigs)
        if _contig is not None:
            starts = part["chromStart"].to_numpy()
            ends = part["chromEnd"].to_numpy()

            # (regions, samples)
            n_per_query = track.count_intervals(contig, starts, ends, sample=_samples)
            # (regions)
            mem_per_r = n_per_query.sum(1) * MEM_PER_INTERVAL

            if np.any(mem_per_r > max_mem):
                # TODO subset by samples as well if needed
                raise NotImplementedError(
                    f"""Memory usage per region exceeds maximum of {max_mem / 1e9} GB.
                    Largest amount needed for a single region is {mem_per_r.max() / 1e9} GB, set
                    `max_mem` to this value or higher. Otherwise, chunking by region and sample is
                    not yet implemented."""
                )

            split_offsets = splits_sum_le_value(mem_per_r, max_mem)
            split_lengths = np.diff(split_offsets)
            for i in range(len(split_lengths)):
                o_s, o_e = split_offsets[i], split_offsets[i + 1]
                chunk_idx = n_chunks + i
                chunk_offsets[chunk_idx] = lengths_to_offsets(
                    n_per_query[o_s:o_e].ravel()
                )
            first_chunk_idx = n_chunks
            last_chunk_idx = n_chunks + len(split_lengths)
            _chunk_labels = np.arange(
                first_chunk_idx, last_chunk_idx, dtype=np.uint32
            ).repeat(split_lengths)
            chunk_labels[last_chunk_offset : last_chunk_offset + len(_chunk_labels)] = (
                _chunk_labels
            )
            n_chunks += len(split_lengths)
            last_chunk_offset += len(_chunk_labels)
        pbar.update()
    pbar.close()
    bed = bed.with_columns(chunk=pl.lit(chunk_labels))

    out_dir.mkdir(parents=True, exist_ok=True)

    interval_offset = 0
    offset_offset = 0
    last_offset = 0
    pbar = tqdm(total=bed["chunk"].n_unique())
    for (chunk_idx,), part in bed.partition_by(
        "chunk", as_dict=True, include_key=False, maintain_order=True
    ).items():
        chunk_idx = cast(int, chunk_idx)
        contig = cast(str, part[0, "chrom"])
        pbar.set_description(f"Reading intervals for {part.height} regions on {contig}")
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()
        _offsets = chunk_offsets[chunk_idx]

        intervals = track._intervals_from_offsets(
            contig, starts, ends, _offsets, sample=_samples
        )

        pbar.set_description(f"Writing intervals for {part.height} regions on {contig}")
        out = np.memmap(
            out_dir / "intervals.npy",
            dtype=INTERVAL_DTYPE,
            mode="w+" if interval_offset == 0 else "r+",
            shape=intervals.values.data.shape,
            offset=interval_offset,
        )
        out["start"] = intervals.starts.data
        out["end"] = intervals.ends.data
        out["value"] = intervals.values.data
        out.flush()
        interval_offset += out.nbytes

        offsets = intervals.values.offsets
        offsets += last_offset
        last_offset = offsets[-1]
        out = np.memmap(
            out_dir / "offsets.npy",
            dtype=offsets.dtype,
            mode="w+" if offset_offset == 0 else "r+",
            shape=len(offsets) - 1,
            offset=offset_offset,
        )
        out[:] = offsets[:-1]
        out.flush()
        offset_offset += out.nbytes
        pbar.update()
    pbar.close()

    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=offsets.dtype,
        mode="r+",
        shape=1,
        offset=offset_offset,
    )
    out[-1] = offsets[-1]
    out.flush()


def _write_track_rust(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "BigWigs",
    samples: list[str],
    max_mem: int,
) -> None:
    from ..genvarloader import bigwig_write_track

    out_dir.mkdir(parents=True, exist_ok=True)
    # ordered sample paths (dataset/sample order)
    paths = [track.paths[s] for s in samples]
    # vectorized contig normalization (equivalent to per-row normalize_contig_name)
    track_contigs = list(track.contigs)
    cnorm = ContigNormalizer(track_contigs)
    norm = cnorm.norm(bed["chrom"].to_list())
    if any(n is None for n in norm):
        bad = next(c for n, c in zip(norm, bed["chrom"].to_list()) if n is None)
        raise ValueError(
            f"Contig {bad!r} not found in bigWig track {track.name!r}."
        )
    contigs = [str(n) for n in norm]
    starts = np.ascontiguousarray(bed["chromStart"].to_numpy(), dtype=np.int32)
    ends = np.ascontiguousarray(bed["chromEnd"].to_numpy(), dtype=np.int32)
    bigwig_write_track(
        paths,
        contigs,
        starts,
        ends,
        int(max_mem),
        str(out_dir),
        False,
    )


def _write_track_table(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "Table",
    samples: list[str],
    max_mem: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # bed is contig-grouped (sp.bed.sort). Map per-region chrom -> Table contig code.
    # Use norm() to detect absent contigs (returns None); force those to -1.
    norm = track._cnorm.norm(bed["chrom"].to_list())
    chrom_codes = track._cnorm.c_idxs(bed["chrom"].to_numpy())
    chrom_codes = np.where(
        np.array([n is None for n in norm]), -1, chrom_codes
    ).astype(np.int32)
    starts = np.ascontiguousarray(bed["chromStart"].to_numpy(), dtype=np.int32)
    ends = np.ascontiguousarray(bed["chromEnd"].to_numpy(), dtype=np.int32)
    track._rust.write_track(
        str(out_dir),
        np.ascontiguousarray(chrom_codes, dtype=np.int32),
        starts,
        ends,
        track._sample_codes(samples),
        int(max_mem),
    )


def _write_track(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "IntervalTrack",
    samples: list[str] | None,
    max_mem: int,
):
    from .._bigwig import BigWigs
    from .._table import Table

    if isinstance(track, BigWigs):
        _samples = samples if samples is not None else track.samples
        if missing := (set(_samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        return _write_track_rust(out_dir, bed, track, _samples, max_mem)
    if isinstance(track, Table):
        _samples = samples if samples is not None else track.samples
        if missing := (set(_samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        return _write_track_table(out_dir, bed, track, _samples, max_mem)
    return _write_track_legacy(out_dir, bed, track, samples, max_mem)
