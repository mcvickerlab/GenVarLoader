import gc
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, cast

import awkward as ak
import numpy as np
import polars as pl
from genoray import PGEN, VCF, Reader, SparseVar
from genoray._svar import V_IDX_TYPE, SparseGenotypes
from genoray._utils import parse_memory
from loguru import logger
from more_itertools import mark_ends
from natsort import natsorted
from numpy.typing import NDArray
from seqpro._ragged import OFFSET_TYPE
from tqdm.auto import tqdm

from .._bigwig import BigWigs
from .._utils import _lengths_to_offsets, _normalize_contig_name, read_bedlike
from .._variants._utils import path_is_pgen, path_is_vcf
from ._genotypes import SparseSomaticGenotypes
from ._utils import splits_sum_le_value


def write(
    path: Union[str, Path],
    bed: Union[str, Path, pl.DataFrame],
    variants: str | Path | Reader | None = None,
    bigwigs: Optional[Union[BigWigs, List[BigWigs]]] = None,
    samples: Optional[List[str]] = None,
    max_jitter: Optional[int] = None,
    overwrite: bool = False,
    max_mem: int | str = "4g",
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
    bigwigs
        BigWigs object or list of BigWigs objects containing intervals
    samples
        Samples to include in the dataset
    max_jitter
        Maximum jitter to add to the regions
    overwrite
        Whether to overwrite an existing dataset
    max_mem
        Approximate maximum memory to use. This is a soft limit and may be exceeded by a small amount.
    """
    # ignore polars warning about os.fork which is caused by using joblib's loky backend
    warnings.simplefilter("ignore", RuntimeWarning)

    if variants is None and bigwigs is None:
        raise ValueError("At least one of `vcf` or `bigwigs` must be provided.")

    if isinstance(bigwigs, BigWigs):
        bigwigs = [bigwigs]

    logger.info(f"Writing dataset to {path}")

    max_mem = parse_memory(max_mem)

    metadata = {}
    path = Path(path)
    if path.exists() and overwrite:
        logger.info("Found existing GVL store, overwriting.")
        shutil.rmtree(path)
    elif path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(bed, (str, Path)):
        bed = read_bedlike(bed)

    gvl_bed, contigs, input_to_sorted_idx_map = _prep_bed(bed, max_jitter)
    bed.with_columns(r_idx_map=pl.lit(input_to_sorted_idx_map)).write_ipc(
        path / "input_regions.arrow"
    )
    metadata["contigs"] = contigs
    if max_jitter is not None:
        metadata["max_jitter"] = max_jitter

    available_samples: Optional[Set[str]] = None
    if variants is not None:
        if isinstance(variants, (str, Path)):
            variants = Path(variants)
            if path_is_pgen(variants):
                if variants.suffix == "":
                    variants = variants.with_suffix(".pgen")
                variants = PGEN(variants)
            elif path_is_vcf(variants):
                variants = VCF(variants)
            else:
                raise ValueError(
                    f"File {variants} has an unrecognized file extension. Please provide either a VCF or PGEN file.`"
                )

        if available_samples is None:
            available_samples = set(variants.available_samples)

    if bigwigs is not None:
        unavail = []
        for bw in bigwigs:
            if unavailable_contigs := set(contigs) - set(
                _normalize_contig_name(c, contigs) for c in bw.contigs
            ):
                unavail.append(unavailable_contigs)
            if available_samples is None:
                available_samples = set(bw.samples)
            else:
                available_samples.intersection_update(bw.samples)
        if unavail:
            logger.warning(
                f"Contigs in queries {set(unavail)} are not found in the BigWigs."
            )

    if available_samples is None:
        raise ValueError(
            "No samples available across all variant file(s) and/or BigWigs."
        )

    if samples is not None:
        _samples = set(samples)
        if missing := (_samples - available_samples):
            raise ValueError(f"Samples {missing} not found in VCF or BigWigs.")
        samples = list(_samples)
    else:
        samples = list(available_samples)

    samples.sort()

    logger.info(f"Using {len(samples)} samples.")
    metadata["samples"] = samples
    metadata["n_samples"] = len(samples)
    metadata["n_regions"] = gvl_bed.height

    if variants is not None:
        logger.info("Writing genotypes.")
        if isinstance(variants, VCF):
            variants.set_samples(samples)
            gvl_bed = _write_from_vcf(path, gvl_bed, variants, max_mem)
        elif isinstance(variants, PGEN):
            variants.set_samples(samples)
            gvl_bed = _write_from_pgen(path, gvl_bed, variants, max_mem)
        elif isinstance(variants, SparseVar):
            gvl_bed = _write_from_svar(path, gvl_bed, variants, samples)
        metadata["ploidy"] = variants.ploidy
        metadata["phased"] = True
        # free memory
        del variants
        gc.collect()

    _write_regions(path, gvl_bed, contigs)

    if bigwigs is not None:
        logger.info("Writing BigWig intervals.")
        for bw in bigwigs:
            _write_bigwigs(path, gvl_bed, bw, samples, max_mem)

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info("Finished writing.")
    warnings.simplefilter("default")


def _prep_bed(
    bed: pl.DataFrame,
    max_jitter: Optional[int] = None,
) -> Tuple[pl.DataFrame, List[str], NDArray[np.intp]]:
    if bed.height == 0:
        raise ValueError("No regions found in the BED file.")

    contigs = natsorted(bed["chrom"].unique().to_list())
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
    with pl.StringCache():
        pl.Series(contigs, dtype=pl.Categorical)
        bed = bed.with_row_index().sort(
            pl.col("chrom").cast(pl.Categorical),
            pl.col("chromStart"),
            pl.col("chromEnd"),
            maintain_order=True,
        )

    input_to_sorted_idx_map = np.argsort(bed["index"])
    bed = bed.drop("index")

    if max_jitter is not None:
        bed = bed.with_columns(
            chromStart=pl.col("chromStart") - max_jitter,
            chromEnd=pl.col("chromEnd") + max_jitter,
        )

    return bed, contigs, input_to_sorted_idx_map


def _write_regions(path: Path, bed: pl.DataFrame, contigs: List[str]):
    with pl.StringCache():
        pl.Series(contigs, dtype=pl.Categorical)
        regions = bed.with_columns(
            pl.col("chrom").cast(pl.Categorical).to_physical()
        ).with_columns(pl.all().cast(pl.Int32))
    regions = regions.to_numpy()
    np.save(path / "regions.npy", regions)


def _write_from_vcf(path: Path, bed: pl.DataFrame, vcf: VCF, max_mem: int):
    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    if vcf._index is None:
        if not vcf._valid_index():
            vcf._write_gvi_index()

        vcf._load_index()

    assert vcf._index is not None

    if vcf._index.df.select((pl.col("ALT").list.len() > 1).any()).item():
        raise ValueError(
            "VCF with filtering applied still contains multi-allelic variants. Please filter or split them."
        )

    pl.DataFrame(
        {
            "POS": vcf._index.gr.df["Start"],
            "ALT": vcf._index.df["ALT"].list.first(),
            "ILEN": vcf._index.df.select(
                pl.col("ALT").list.first().str.len_bytes().cast(pl.Int32)
                - pl.col("REF").str.len_bytes().cast(pl.Int32)
            ),
        }
    ).write_ipc(out_dir / "variants.arrow")

    unextended_var_idxs: dict[str, list[NDArray[np.integer]]] = {}
    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        v_idx, offsets = vcf._var_idxs(contig, starts, ends)
        unextended_var_idxs[contig] = np.array_split(v_idx, offsets[1:-1])

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
        for range_, var_idxs, e in zip(
            vcf._chunk_ranges_with_length(contig, starts, ends, max_mem, VCF.Genos8),
            unextended_var_idxs[contig],
            ends,
        ):
            var_idxs = var_idxs.astype(V_IDX_TYPE)
            if range_ is None:
                if first_no_variant_warning:
                    first_no_variant_warning = False
                    logger.warning(
                        "A region has no variants for any sample. This could be expected depending on the region lengths"
                        " and source of variants. However, this can also be caused by a mismatch between the"
                        " reference genome used for the BED file coordinates and the one used for the variants."
                        " This warning will not be shown again."
                    )

                max_ends.append(e)
                sp_genos = SparseGenotypes.empty(
                    (1, vcf.n_samples, vcf.ploidy), np.int32
                )
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
                continue

            offset = 0
            ls_sparse: list[SparseGenotypes] = []
            for _, is_last, (genos, chunk_end, n_ext) in mark_ends(range_):
                n_variants = genos.shape[-1]
                chunk_idxs = var_idxs[offset : offset + n_variants]

                if n_ext > 0:  # also means is_last is True
                    ext_s_idx = chunk_idxs[-1] + 1
                    ext_idxs = np.arange(ext_s_idx, ext_s_idx + n_ext, dtype=np.int32)
                    chunk_idxs = np.concatenate([chunk_idxs, ext_idxs])

                sp_genos = SparseGenotypes.from_dense(genos, chunk_idxs)
                ls_sparse.append(sp_genos)

                if is_last:
                    max_ends.append(chunk_end)

            if len(ls_sparse) == 0:
                max_ends.append(e)
                sp_genos = SparseGenotypes.empty(
                    (1, vcf.n_samples, vcf.ploidy), np.int32
                )
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
            else:
                var_idxs = ak.flatten(
                    ak.concatenate([a.to_awkward() for a in ls_sparse], -1), None
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

                sp_genos = SparseGenotypes.from_lengths(var_idxs, lengths)
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


def _write_from_pgen(path: Path, bed: pl.DataFrame, pgen: PGEN, max_mem: int):
    if pgen._sei is None:
        raise ValueError(
            "PGEN with filtering has multi-allelic variants. Please filter or split them."
        )
    assert pgen._sei is not None

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "POS": pgen._sei.v_starts,
            "ALT": pgen._sei.alt,
            "ILEN": pgen._sei.ilens,
        }
    ).write_ipc(out_dir / "variants.arrow")

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
        for range_, e in zip(
            pgen._chunk_ranges_with_length(contig, starts, ends, max_mem), ends
        ):
            if range_ is None:
                if first_no_variant_warning:
                    first_no_variant_warning = False
                    logger.warning(
                        "A region has no variants for any sample. This could be expected depending on the region lengths"
                        " and source of variants. However, this can also be caused by a mismatch between the"
                        " reference genome used for the BED file coordinates and the one used for the variants."
                    )

                max_ends.append(e)
                sp_genos = SparseGenotypes.empty(
                    (1, pgen.n_samples, pgen.ploidy), np.int32
                )
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
                continue

            ls_sparse: list[SparseGenotypes] = []
            for _, is_last, (genos, chunk_end, chunk_idxs) in mark_ends(range_):
                chunk_idxs = chunk_idxs.astype(V_IDX_TYPE)
                sp_genos = SparseGenotypes.from_dense(genos.astype(np.int8), chunk_idxs)
                ls_sparse.append(sp_genos)

                if is_last:
                    max_ends.append(chunk_end)

            if len(ls_sparse) == 0:
                max_ends.append(e)
                sp_genos = SparseGenotypes.empty(
                    (1, pgen.n_samples, pgen.ploidy), np.int32
                )
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
            else:
                sp_genos = SparseGenotypes.from_awkward(
                    ak.concatenate([a.to_awkward() for a in ls_sparse], -1)
                )

                var_idxs = ak.flatten(
                    ak.concatenate([a.to_awkward() for a in ls_sparse], -1), None
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

                sp_genos = SparseGenotypes.from_lengths(var_idxs, lengths)

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
    path: Path, bed: pl.DataFrame, svar: SparseVar, samples: list[str]
):
    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets = np.memmap(
        out_dir / "offsets.npy",
        np.int64,
        "w+",
        shape=(bed.height, len(samples), svar.ploidy, 2),
    )

    with open(out_dir / "svar_meta.json", "w") as f:
        json.dump({"shape": offsets.shape, "dtype": offsets.dtype.str}, f)

    v_ends = svar.granges.End
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
        # (r s p 2)
        out = offsets[contig_offset : contig_offset + df.height]
        svar._find_starts_ends_with_length(
            c, df["chromStart"], df["chromEnd"], samples=samples, out=out
        )

        if (
            first_no_variant_warning
            and (out == np.iinfo(OFFSET_TYPE).max).all((1, 2, 3)).any()
        ):
            first_no_variant_warning = False
            logger.warning(
                "Some regions have no variants for any sample. This could be expected depending on the region lengths"
                " and source of variants. However, this can also be caused by a mismatch between the"
                " reference genome used for the BED file coordinates and the one used for the variants."
            )

        # compute max_ends for the bed
        shape = (df.height, len(samples), svar.ploidy)
        # (r s p ~v)
        sp_genos = SparseGenotypes.from_offsets(
            svar.genos.data, shape, out.reshape(-1, 2)
        )
        # this is fine if there aren't any overlapping variants that could make a v_idx < -1
        # have a further end than v_idx == -1
        # * calling ak.max() means v_idxs is not a view of svar.genos.data
        # (r s p ~v) -> (r)
        v_idxs = ak.max(sp_genos.to_awkward(), -1).to_numpy().max((1, 2))  # type: ignore
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

    (out_dir / "link.svar").symlink_to(svar.path, True)

    return bed.with_columns(chromEnd=pl.Series(max_ends))


def _write_phased_variants_chunk(
    out_dir: Path,
    genos: SparseGenotypes,
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


def _write_somatic_variants_chunk(
    out_dir: Path,
    genos: SparseSomaticGenotypes,
    v_idx_memmap_offset: int,
    ccf_memmap_offset: int,
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

        out = np.memmap(
            out_dir / "ccfs.npy",
            dtype=genos.ccfs.dtype,
            mode="w+" if ccf_memmap_offset == 0 else "r+",
            shape=genos.ccfs.shape,
            offset=ccf_memmap_offset,
        )
        out[:] = genos.ccfs[:]
        out.flush()
        ccf_memmap_offset += out.nbytes

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
    return v_idx_memmap_offset, ccf_memmap_offset, offsets_memmap_offset, last_offset


def _write_bigwigs(
    path: Path,
    bed: pl.DataFrame,
    bigwigs: BigWigs,
    samples: Optional[List[str]],
    max_mem: int,
):
    if samples is None:
        _samples = cast(List[str], bigwigs.samples)
    else:
        if missing := (set(samples) - set(bigwigs.samples)):
            raise ValueError(f"Samples {missing} not found in bigwigs.")
        _samples = samples

    MEM_PER_INTERVAL = (
        12 * 2
    )  # start u32, end u32, value f32, times 2 for intermediate copies
    chunk_labels = np.empty(bed.height, np.uint32)
    chunk_offsets: Dict[int, NDArray[np.int64]] = {}
    n_chunks = 0
    last_chunk_offset = 0
    pbar = tqdm(total=bed["chrom"].n_unique())
    for (contig,), part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        pbar.set_description(f"Calculating memory usage for {part.height} regions")
        contig = cast(str, contig)
        _contig = _normalize_contig_name(contig, bigwigs.contigs)
        if _contig is not None:
            starts = part["chromStart"].to_numpy()
            ends = part["chromEnd"].to_numpy()

            # (regions, samples)
            n_per_query = bigwigs.count_intervals(contig, starts, ends, sample=samples)
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
                chunk_offsets[chunk_idx] = _lengths_to_offsets(
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

    out_dir = path / "intervals" / bigwigs.name
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

        intervals = bigwigs._intervals_from_offsets(
            contig, starts, ends, _offsets, sample=_samples
        )

        pbar.set_description(f"Writing intervals for {part.height} regions on {contig}")
        out = np.memmap(
            out_dir / "intervals.npy",
            dtype=intervals.data.dtype,
            mode="w+" if interval_offset == 0 else "r+",
            shape=intervals.data.shape,
            offset=interval_offset,
        )
        out[:] = intervals.data[:]
        out.flush()
        interval_offset += out.nbytes

        offsets = intervals.offsets
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
