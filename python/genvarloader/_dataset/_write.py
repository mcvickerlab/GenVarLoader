import gc
import json
import shutil
import warnings
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

if TYPE_CHECKING:
    from .._types import IntervalTrack

import awkward as ak
import numpy as np
import polars as pl
import seqpro as sp
from genoray import PGEN, VCF, Reader, SparseVar
from genoray._svar import dense2sparse
from genoray._types import V_IDX_TYPE
from genoray._utils import ContigNormalizer, format_memory, parse_memory
from loguru import logger
from more_itertools import mark_ends
from natsort import natsorted
from numpy.typing import NDArray
from packaging.version import Version
from pydantic import BaseModel, BeforeValidator, PlainSerializer, WithJsonSchema
from seqpro.rag import Ragged
from tqdm.auto import tqdm

from .._ragged import INTERVAL_DTYPE
from .._utils import lengths_to_offsets, normalize_contig_name
from .._variants._utils import path_is_pgen, path_is_vcf
from ._utils import bed_to_regions, splits_sum_le_value


class Metadata(BaseModel, arbitrary_types_allowed=True):
    samples: list[str]
    contigs: list[str]
    n_regions: int
    ploidy: int | None = None
    max_jitter: int = 0
    version: (
        Annotated[
            Version,
            BeforeValidator(lambda v: Version(v) if isinstance(v, str) else v),
            PlainSerializer(lambda v: str(v), return_type=str),
            WithJsonSchema({"type": "string"}, mode="serialization"),
        ]
        | None
    ) = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def write(
    path: str | Path,
    bed: str | Path | pl.DataFrame,
    variants: str | Path | Reader | None = None,
    tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
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
    """
    # ignore polars warning about os.fork which is caused by using joblib's loky backend
    warnings.simplefilter("ignore", RuntimeWarning)

    if variants is None and tracks is None:
        raise ValueError("At least one of `variants` or `tracks` must be provided.")

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

    metadata: dict[str, Any] = {"version": Version(version("genvarloader"))}
    path = Path(path)
    if path.exists() and overwrite:
        logger.info("Found existing GVL store, overwriting.")
        shutil.rmtree(path)
    elif path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")
    path.mkdir(parents=True, exist_ok=True)

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
                bytes_per_var = variants.n_samples * variants.ploidy  # Genos8: 1 byte
            else:
                bytes_per_var = variants.n_samples * variants.ploidy * 4  # int32

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
            gvl_bed = _write_from_svar(
                path, gvl_bed, variants, samples, extend_to_length
            )
        metadata["ploidy"] = variants.ploidy
        # free memory
        del variants
        gc.collect()

    _write_regions(path, gvl_bed, contigs)

    if tracks is not None:
        logger.info("Writing track intervals.")
        for tr in tracks:
            _write_track(path, gvl_bed, tr, samples, max_mem)

    _metadata = Metadata(**metadata)
    with open(path / "metadata.json", "w") as f:
        json.dump(_metadata.model_dump(), f)

    logger.info("Finished writing.")
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
        {"seqname": "chrom", "start": "chromStart", "end": "chromEnd"}
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


def _write_from_vcf(
    path: Path, bed: pl.DataFrame, vcf: VCF, max_mem: int, extend_to_length: bool
):
    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert vcf._index is not None, "caller must load the VCF index before _write_from_vcf"

    if vcf._index.select((pl.col("ALT").list.len() > 1).any()).item():
        raise ValueError(
            "VCF with filtering applied still contains multi-allelic variants. Please filter or split them."
        )

    (out_dir / "variants.arrow").hardlink_to(vcf._index_path())

    unextended_var_idxs: dict[str, list[NDArray[V_IDX_TYPE]]] = {}
    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        v_idx, offsets = vcf._var_idxs(contig, starts, ends)
        unextended_var_idxs[contig] = np.array_split(
            v_idx.astype(V_IDX_TYPE), offsets[1:-1]
        )

    v_idx_memmap_offsets = 0
    offset_memmap_offsets = 0
    last_offset = 0
    max_ends: list[int] = []
    pbar = tqdm(total=bed.height, unit=" region")
    first_no_variant_warning = True
    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        pbar.set_description(
            f"Processing genotypes for {df.height} regions on contig {contig}"
        )
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy().copy()
        ends = df["chromEnd"].to_numpy().copy()
        for range_, unextended_idxs in zip(
            vcf._chunk_ranges_with_length(contig, starts, ends, max_mem, VCF.Genos8),
            unextended_var_idxs[contig],
        ):
            ls_sparse: list[Ragged] = []
            offset = 0
            for _, is_last, (chunk_genos, chunk_end, n_ext) in mark_ends(range_):
                n_vars = chunk_genos.shape[-1]
                chunk_idxs = unextended_idxs[offset : offset + n_vars]
                offset += n_vars

                if (
                    n_ext > 0
                ):  # also means is_last is True based on implementation of _chunk_ranges_with_length
                    # indices in chunk_idxs are inclusive
                    ext_s_idx = chunk_idxs[-1] + 1
                    ext_idxs = np.arange(ext_s_idx, ext_s_idx + n_ext, dtype=np.int32)
                    chunk_idxs = np.concatenate([chunk_idxs, ext_idxs])

                sp_genos = dense2sparse(chunk_genos, chunk_idxs)
                ls_sparse.append(sp_genos)

                if is_last:
                    max_ends.append(chunk_end)

            var_idxs = ak.flatten(
                ak.concatenate(ls_sparse, -1),
                None,  # type: ignore
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

    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=np.int64,
        mode="r+",
        shape=1,
        offset=offset_memmap_offsets,
    )
    out[-1] = last_offset
    out.flush()

    pbar.close()

    bed = bed.with_columns(chromEnd=pl.Series(max_ends))
    return bed


def _write_from_pgen(
    path: Path, bed: pl.DataFrame, pgen: PGEN, max_mem: int, extend_to_length: bool
):
    if pgen._sei is None:
        raise ValueError(
            "PGEN with filtering has multi-allelic variants. Please filter or split them."
        )
    assert pgen._sei is not None

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "variants.arrow").hardlink_to(pgen._index_path())

    pbar = tqdm(total=bed.height, unit=" region")

    v_idx_memmap_offsets = 0
    offset_memmap_offsets = 0
    last_offset = 0
    max_ends: list[np.integer] = []
    first_no_variant_warning = True
    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        pbar.set_description(
            f"Processing genotypes for {df.height} regions on contig {contig}"
        )
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy().copy()
        ends = df["chromEnd"].to_numpy().copy()
        for range_ in pgen._chunk_ranges_with_length(contig, starts, ends, max_mem):
            ls_sparse: list[Ragged] = []
            for _, is_last, (genos, chunk_end, chunk_idxs) in mark_ends(range_):
                chunk_idxs = chunk_idxs.astype(V_IDX_TYPE)
                sp_genos = dense2sparse(genos.astype(np.int8), chunk_idxs)
                ls_sparse.append(sp_genos)

                if is_last:
                    max_ends.append(chunk_end)

            var_idxs = ak.flatten(
                ak.concatenate(ls_sparse, -1),
                None,  # type: ignore
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

    bed = bed.with_columns(chromEnd=pl.Series(max_ends))
    return bed


def _write_from_svar(
    path: Path,
    bed: pl.DataFrame,
    svar: SparseVar,
    samples: list[str],
    extend_to_length: bool,
):
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

    (out_dir / "link.svar").symlink_to(svar.path.resolve(), target_is_directory=True)

    return bed.with_columns(chromEnd=pl.Series(max_ends))


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


def _write_track(
    path: Path,
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

    out_dir = path / "intervals" / track.name
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
        dtype=offsets.dtype,  # type: ignore
        mode="r+",
        shape=1,
        offset=offset_offset,
    )
    out[-1] = offsets[-1]  # type: ignore
    out.flush()
