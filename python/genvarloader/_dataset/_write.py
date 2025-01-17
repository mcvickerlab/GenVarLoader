import gc
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import polars as pl
from loguru import logger
from natsort import natsorted
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .._bigwig import BigWigs
from .._utils import (
    _lengths_to_offsets,
    _normalize_contig_name,
    read_bedlike,
    with_length,
)
from .._variants import Variants
from .._variants._genotypes import PgenGenos, VCFGenos
from ._genotypes import (
    SparseGenotypes,
    SparseSomaticGenotypes,
    get_haplotype_region_ilens,
)
from ._utils import splits_sum_le_value

__all__ = ["write"]


INITIAL_END_EXTENSION = 1000
EXTEND_END_MULTIPLIER = 1.1


def write(
    path: Union[str, Path],
    bed: Union[str, Path, pl.DataFrame],
    variants: Optional[Union[str, Path, Variants]] = None,
    bigwigs: Optional[Union[BigWigs, List[BigWigs]]] = None,
    samples: Optional[List[str]] = None,
    length: Optional[int] = None,
    max_jitter: Optional[int] = None,
    overwrite: bool = False,
    max_mem: int = 4 * 2**30,
    phased: bool = True,
    dosage_field: Optional[str] = None,
):
    """Write a GVL dataset.

    Parameters
    ----------
    path : Union[str, Path]
        Path to write the dataset to.
    bed : Union[str, Path, pl.DataFrame]
        BED-like file or DataFrame containing regions to query.
    variants : Optional[Union[str, Path, Variants]], optional
        VCF, PGEN, or :py:class:`Variants` instances containing genotypes, by default None. All variants must be
        left-aligned, bi-allelic, and atomized. Multi-allelic variants can be included by splitting
        them into bi-allelic half-calls. For VCFs, the [bcftools norm](https://samtools.github.io/bcftools/bcftools.html#norm)
        command can do all of this normalization. Likewise, see the [PLINK2 documentation](https://www.cog-genomics.org/plink/2.0)
        for more PGEN files. Commands of interest include --make-bpgen for splitting variants,
        --normalize for left-aligning and atomizing overlapping variants, and --ref-from-fa for REF allele correction.
    bigwigs : Optional[Union[BigWigs, List[BigWigs]]], optional
        BigWigs object or list of BigWigs objects containing intervals, by default None
    samples : Optional[List[str]], optional
        Samples to include in the dataset, by default None
    length : Optional[int], optional
        Length of the regions to query, by default None. Provided regions will be expanded or contracted
        to this length + 2 x `max_jitter` with the center of the region remaining the same.
    max_jitter : Optional[int], optional
        Maximum jitter to add to the regions, by default None
    overwrite : bool, optional
        Whether to overwrite an existing dataset, by default False
    max_mem : int, optional
        Maximum memory to use per region, by default 4 GiB (4 * 2**30 bytes)
    phased : bool, optional
        Whether to treat the genotypes as phased, by default True. If phased=False and using a VCF,
        a dosage FORMAT field must be provided and must have Number = '1' or 'A' in the VCF header.
        All variants that overlap with the BED regions must also have this field present or else
        the write will fail partway and raise a :py:class:`~genvarloader._variants._genotypes.DosageFieldError`.
        For PGEN files, if dosages are not present the write will silently fail with all missing dosages.
        Ostensibly, there is a flag in PGEN files for whether dosages are present. However, the Python
        interface to PGEN, pgenlib, does not currently expose a way to check this flag. Thus, a workaround
        is to use [plink2 --pgen-info](https://www.cog-genomics.org/plink/2.0/basic_stats#pgen_info) to check
        if dosages are present before you write the dataset.
    dosage_field : Optional[str], optional
        Field in the VCF to use as dosage, by default None. Ignored if phased=True.
    """
    # ignore polars warning about os.fork which is caused by using joblib's loky backend
    warnings.simplefilter("ignore", RuntimeWarning)

    if variants is None and bigwigs is None:
        raise ValueError("At least one of `vcf` or `bigwigs` must be provided.")

    if isinstance(bigwigs, BigWigs):
        bigwigs = [bigwigs]

    logger.info(f"Writing dataset to {path}")

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

    gvl_bed, contigs, region_length, src_to_sorted_idx_map = _prep_bed(
        bed, length, max_jitter
    )
    bed.with_columns(r_idx_map=pl.lit(src_to_sorted_idx_map)).write_ipc(
        path / "input_regions.arrow"
    )
    metadata["region_length"] = region_length
    metadata["contigs"] = contigs
    if max_jitter is not None:
        metadata["max_jitter"] = max_jitter

    available_samples: Optional[Set[str]] = None
    if variants is not None:
        if isinstance(variants, Variants) and phased != variants.phased:
            raise ValueError(
                f"Phased argument ({phased}) does not match phased status of Variants object ({variants.phased})."
            )

        if isinstance(variants, (str, Path)):
            variants = Variants.from_file(variants, phased, dosage_field)

        if unavailable_contigs := set(contigs) - {
            _normalize_contig_name(c, contigs) for c in variants.records.contigs
        }:
            logger.warning(
                f"Contigs in queries {unavailable_contigs} are not found in the VCF."
            )

        if available_samples is None:
            available_samples = set(variants.samples)

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
        gvl_bed = _write_variants(
            path,
            gvl_bed,
            variants,
            region_length,
            samples,
            max_mem,
        )
        if isinstance(variants.genotypes, VCFGenos):
            variants.genotypes.close()
        metadata["ploidy"] = variants.ploidy
        metadata["phased"] = phased
        # free memory
        del variants
        gc.collect()

    logger.info("Writing regions.")
    _write_regions(path, gvl_bed, contigs)

    if bigwigs is not None:
        logger.info("Writing BigWig intervals.")
        for bw in bigwigs:
            _write_bigwigs(path, gvl_bed, bw, samples)

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info("Finished writing.")
    warnings.simplefilter("default")


def _prep_bed(
    bed: pl.DataFrame,
    length: Optional[int] = None,
    max_jitter: Optional[int] = None,
) -> Tuple[pl.DataFrame, List[str], int, NDArray[np.intp]]:
    if bed.height == 0:
        raise ValueError("No regions found in the BED file.")

    contigs = natsorted(bed["chrom"].unique().to_list())
    if "strand" not in bed:
        bed = bed.with_columns(strand=pl.lit(1, pl.Int32))
    else:
        bed = bed.with_columns(
            pl.col("strand").replace({"+": 1, "-": -1}, return_dtype=pl.Int32)
        )
    bed = bed.select("chrom", "chromStart", "chromEnd", "strand")
    with pl.StringCache():
        pl.Series(contigs, dtype=pl.Categorical)
        bed = bed.with_row_index().sort(
            pl.col("chrom").cast(pl.Categorical), pl.col("chromStart")
        )

    src_to_sorted_idx_map = np.argsort(bed["index"])
    bed = bed.drop("index")

    if length is None:
        length = cast(
            int, bed.select((pl.col("chromEnd") - pl.col("chromStart")).max()).item()
        )

    if max_jitter is not None:
        length += 2 * max_jitter

    bed = with_length(bed, length)

    return bed, contigs, length, src_to_sorted_idx_map


def _write_regions(path: Path, bed: pl.DataFrame, contigs: List[str]):
    with pl.StringCache():
        pl.Series(contigs, dtype=pl.Categorical)
        regions = bed.with_columns(
            pl.col("chrom").cast(pl.Categorical).to_physical()
        ).with_columns(pl.all().cast(pl.Int32))
    regions = regions.to_numpy()
    np.save(path / "regions.npy", regions)


def _write_variants(
    path: Path,
    bed: pl.DataFrame,
    variants: Variants,
    region_length: int,
    samples: Optional[List[str]] = None,
    max_mem: int = 4 * 2**30,
):
    if samples is None:
        len(variants.samples)
        sample_idx = None
        _samples = cast(List[str], variants.samples.tolist())
    else:
        _, key_idx, query_idx = np.intersect1d(
            variants.samples, samples, return_indices=True
        )
        if missing := (set(samples) - set(variants.samples)):
            raise ValueError(f"Samples {missing} not found in VCF.")
        sample_idx = key_idx[query_idx]
        len(sample_idx)
        _samples = samples

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "POS": np.concatenate(list(variants.records.v_starts.values())),
            "ALT": pl.concat([a.to_polars() for a in variants.records.alt.values()]),
            "ILEN": np.concatenate(list(variants.records.v_diffs.values())),
        }
    ).write_ipc(out_dir / "variants.arrow")

    rel_start_idxs: Dict[str, NDArray[np.int32]] = {}
    rel_end_idxs: Dict[str, NDArray[np.int32]] = {}
    chunk_offsets: Dict[str, NDArray[np.intp]] = {}
    n_chunks = 0
    for (contig,), part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        contig = cast(str, contig)
        _contig = _normalize_contig_name(contig, variants.records.contigs)
        if _contig is not None:
            starts = part["chromStart"].to_numpy()
            ends = starts + region_length + INITIAL_END_EXTENSION
            rel_s_idxs = variants.records.find_relative_start_idx(_contig, starts)
            rel_e_idxs = variants.records.find_relative_end_idx(_contig, ends)

            # variants * ploidy * samples * (4 bytes per genotype + 4 bytes per ilen)
            # up to 4 bytes due to pgenlib reading as i32, then reduced 1 byte
            # cyvcf2 reads slower, but genotypes are cast to i8 per variant
            # + 1 region * ploidy * samples * (4 bytes per ilen)
            n_variants = rel_e_idxs - rel_s_idxs
            n_regions = len(starts)
            if isinstance(variants.genotypes, VCFGenos):
                bytes_per_variant = 1
            elif isinstance(variants.genotypes, PgenGenos):
                bytes_per_variant = 4
            else:
                # NOTE: this should never run unless user provides a custom genotype reader
                # in that case this is a safe-ish upper bound
                bytes_per_variant = 4

            if not variants.phased:
                bytes_per_variant += 4  # dosage is float 32

            mem_per_r = (n_variants * (bytes_per_variant + 4) + 4) * len(_samples)
            if variants.phased:
                mem_per_r *= variants.ploidy

            if np.any(mem_per_r > max_mem):
                # TODO subset by samples as well if needed
                # sketch:
                # 1. for 1 region, read subset of variants -> SparseGenotypes
                # 2. repeat 1 for next subset of variants
                raise ValueError(
                    f"""Memory usage per region exceeds maximum of {max_mem / 1e9} GB.
                    Largest amount needed for a single region is {mem_per_r.max() / 1e9} GB, set
                    `max_mem` to this value or higher."""
                )

            offsets = splits_sum_le_value(mem_per_r, max_mem)
            rel_start_idxs[contig] = rel_s_idxs
            rel_end_idxs[contig] = rel_e_idxs
            chunk_offsets[contig] = offsets
            n_chunks += len(offsets) - 1
        else:
            rel_start_idxs[contig] = np.zeros(part.height, dtype=np.int32)
            rel_end_idxs[contig] = np.zeros(part.height, dtype=np.int32)
            chunk_offsets[contig] = np.array([0, part.height], dtype=np.intp)

    v_idx_memmap_offsets = 0
    dosage_memmap_offsets = 0
    offset_memmap_offsets = 0
    last_offset = 0
    max_ends = np.empty(bed.height, dtype=np.int32)
    last_max_end_idx = 0
    with tqdm(total=n_chunks) as pbar:
        for (contig,), part in bed.partition_by(
            "chrom", as_dict=True, include_key=False, maintain_order=True
        ).items():
            contig = cast(str, contig)
            c_offsets = chunk_offsets[contig]
            for o_s, o_e in zip(c_offsets[:-1], c_offsets[1:]):
                rel_s_idxs = rel_start_idxs[contig][o_s:o_e]
                rel_e_idxs = rel_end_idxs[contig][o_s:o_e]
                starts = part[o_s:o_e, "chromStart"].to_numpy()
                ends = starts + region_length + INITIAL_END_EXTENSION
                n_regions = len(rel_s_idxs)
                pbar.set_description(
                    f"Reading genotypes for {n_regions} regions on chromosome {contig}"
                )

                _contig = _normalize_contig_name(contig, variants.records.contigs)

                genos, chunk_max_ends = _read_variants_chunk(
                    _contig,
                    starts,
                    ends,
                    rel_s_idxs,
                    rel_e_idxs,
                    variants,
                    region_length,
                    _samples,
                    sample_idx,
                )

                max_ends[last_max_end_idx : last_max_end_idx + n_regions] = (
                    chunk_max_ends
                )
                last_max_end_idx += n_regions

                pbar.set_description(
                    f"Writing genotypes for {n_regions} regions on chromosome {contig}"
                )
                if isinstance(genos, SparseGenotypes):
                    (
                        v_idx_memmap_offsets,
                        offset_memmap_offsets,
                        last_offset,
                    ) = _write_phased_variants_chunk(
                        out_dir,
                        genos,
                        v_idx_memmap_offsets,
                        offset_memmap_offsets,
                        last_offset,
                    )
                else:
                    (
                        v_idx_memmap_offsets,
                        dosage_memmap_offsets,
                        offset_memmap_offsets,
                        last_offset,
                    ) = _write_somatic_variants_chunk(
                        out_dir,
                        genos,
                        v_idx_memmap_offsets,
                        dosage_memmap_offsets,
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

    bed = bed.with_columns(chromEnd=pl.lit(max_ends))
    return bed


def _read_variants_chunk(
    contig: Optional[str],
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
    rel_s_idxs: NDArray[np.int32],
    rel_e_idxs: NDArray[np.int32],
    variants: Variants,
    region_length: int,
    samples: List[str],
    sample_idxs: Optional[NDArray[np.intp]],
):
    if contig is None:
        if variants.phased:
            genos = SparseGenotypes.empty(
                n_regions=len(rel_s_idxs),
                n_samples=len(samples),
                ploidy=variants.ploidy,
            )
        else:
            genos = SparseSomaticGenotypes.empty(
                n_regions=len(rel_s_idxs), n_samples=len(samples)
            )

        chunk_max_ends = ends
        return genos, chunk_max_ends

    first = True
    while True:
        logger.debug(f"region length {ends[0] - starts[0]}")
        if not first:
            rel_e_idxs = variants.records.find_relative_end_idx(contig, ends)
        s_idx = variants.records.contig_offsets[contig] + rel_s_idxs
        e_idx = variants.records.contig_offsets[contig] + rel_e_idxs
        n_per_region = e_idx - s_idx

        if n_per_region.sum() == 0:
            if variants.phased:
                genos = SparseGenotypes.empty(
                    n_regions=len(rel_s_idxs),
                    n_samples=len(samples),
                    ploidy=variants.ploidy,
                )
            else:
                genos = SparseSomaticGenotypes.empty(
                    n_regions=len(rel_s_idxs), n_samples=len(samples)
                )
            chunk_max_ends = ends
            return genos, chunk_max_ends

        offsets = _lengths_to_offsets(n_per_region)

        # (s p v)
        logger.debug("read genotypes")
        if variants.phased:
            genos = variants.genotypes.multiprocess_read(
                contig,
                s_idx,
                e_idx,
                sample_idx=sample_idxs,
                n_jobs=len(os.sched_getaffinity(0)),
            )
        else:
            assert variants.dosage_field is not None
            genos, dosages = variants.genotypes.multiprocess_read_genos_and_dosages(
                contig,
                s_idx,
                e_idx,
                variants.dosage_field,
                sample_idx=sample_idxs,
                n_jobs=len(os.sched_getaffinity(0)),
            )

        logger.debug("get haplotype region ilens")
        # (s p r)
        haplotype_ilens = get_haplotype_region_ilens(
            genos, rel_s_idxs, offsets, variants.records.v_diffs[contig]
        )
        haplotype_lengths = ends - starts + haplotype_ilens
        del haplotype_ilens
        logger.debug(f"average haplotype length {haplotype_lengths.mean()}")
        # (s p r)
        missing_length = region_length - haplotype_lengths
        logger.debug(f"max missing length {missing_length.max()}")

        if np.all(missing_length <= 0):
            break

    # (r)
    ends += np.ceil(
        EXTEND_END_MULTIPLIER * missing_length.max((0, 1)).clip(min=0)
    ).astype(np.int32)

    logger.debug("sparsify genotypes")
    if variants.phased:
        genos, chunk_max_ends = SparseGenotypes.from_dense_with_length(
            genos=genos,
            first_v_idxs=rel_s_idxs,
            offsets=offsets,
            ilens=variants.records.v_diffs[contig],
            positions=variants.records.v_starts[contig],
            starts=starts,
            length=region_length,
        )
    else:
        genos, chunk_max_ends = SparseSomaticGenotypes.from_dense_with_length(
            genos=genos,
            first_v_idxs=rel_s_idxs,
            offsets=offsets,
            ilens=variants.records.v_diffs[contig],
            positions=variants.records.v_starts[contig],
            starts=starts,
            length=region_length,
            dosages=dosages,  # type: ignore | guaranteed bound by read_genos_and_dosages
        )
    logger.debug(f"maximum needed length {(chunk_max_ends - starts).max()}")
    logger.debug(f"minimum needed length {(chunk_max_ends - starts).min()}")

    # make indices absolute
    genos.variant_idxs += variants.records.contig_offsets[contig]

    return genos, chunk_max_ends


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
            dtype=genos.variant_idxs.dtype,
            mode="w+" if v_idx_memmap_offset == 0 else "r+",
            shape=genos.variant_idxs.shape,
            offset=v_idx_memmap_offset,
        )
        out[:] = genos.variant_idxs[:]
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
    dosage_memmap_offset: int,
    offsets_memmap_offset: int,
    last_offset: int,
):
    if not genos.is_empty:
        out = np.memmap(
            out_dir / "variant_idxs.npy",
            dtype=genos.variant_idxs.dtype,
            mode="w+" if v_idx_memmap_offset == 0 else "r+",
            shape=genos.variant_idxs.shape,
            offset=v_idx_memmap_offset,
        )
        out[:] = genos.variant_idxs[:]
        out.flush()
        v_idx_memmap_offset += out.nbytes

        out = np.memmap(
            out_dir / "dosages.npy",
            dtype=genos.dosages.dtype,
            mode="w+" if dosage_memmap_offset == 0 else "r+",
            shape=genos.dosages.shape,
            offset=dosage_memmap_offset,
        )
        out[:] = genos.dosages[:]
        out.flush()
        dosage_memmap_offset += out.nbytes

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
    return v_idx_memmap_offset, dosage_memmap_offset, offsets_memmap_offset, last_offset


def _write_bigwigs(
    path: Path, bed: pl.DataFrame, bigwigs: BigWigs, samples: Optional[List[str]]
) -> int:
    if samples is None:
        _samples = cast(List[str], bigwigs.samples)
    else:
        if missing := (set(samples) - set(bigwigs.samples)):
            raise ValueError(f"Samples {missing} not found in bigwigs.")
        _samples = samples

    out_dir = path / "intervals" / bigwigs.name
    out_dir.mkdir(parents=True, exist_ok=True)

    interval_offset = 0
    offset_offset = 0
    last_offset = 0
    n_intervals = 0
    pbar = tqdm(total=bed["chrom"].n_unique())
    for (contig,), part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        contig = cast(str, contig)
        pbar.set_description(f"Reading intervals for {part.height} regions on {contig}")
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()

        intervals = bigwigs.intervals(contig, starts, ends, sample=_samples)

        pbar.set_description(f"Writing intervals for {part.height} regions on {contig}")
        n_intervals += len(intervals.data)
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

    return n_intervals
