import gc
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
from loguru import logger
from natsort import natsorted
from numpy.typing import NDArray
from tqdm.auto import tqdm

from ..bigwig import BigWigs
from ..utils import lengths_to_offsets, normalize_contig_name, read_bedlike, with_length
from ..variants import Variants
from ..variants.genotypes import VCFGenos
from .genotypes import SparseGenotypes, get_haplotype_ilens
from .utils import splits_sum_le_value

EXTEND_END_MULTIPLIER = 1.2


def write(
    path: Union[str, Path],
    bed: Union[str, Path, pl.DataFrame],
    variants: Optional[Union[str, Path]] = None,
    bigwigs: Optional[Union[BigWigs, List[BigWigs]]] = None,
    samples: Optional[List[str]] = None,
    length: Optional[int] = None,
    max_jitter: Optional[int] = None,
    overwrite: bool = False,
    max_mem: int = 4 * 2**30,
):
    if variants is None and bigwigs is None:
        raise ValueError("At least one of `vcf` or `bigwigs` must be provided.")

    if isinstance(bigwigs, BigWigs):
        bigwigs = [bigwigs]

    logger.info(f"Writing to {path}")

    metadata = {}
    path = Path(path)
    if path.exists() and overwrite:
        logger.info("Found existing GVL store, overwriting.")
        shutil.rmtree(path)
    elif path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")
    path.mkdir(parents=True, exist_ok=True)

    bed, contigs, region_length = prep_bed(bed, length, max_jitter)
    metadata["region_length"] = region_length
    metadata["contigs"] = contigs
    if max_jitter is not None:
        metadata["max_jitter"] = max_jitter

    all_samples: List[str] = []
    if variants is not None:
        variants = Path(variants)
        variants = Variants.from_file(variants)

        if unavailable_contigs := set(contigs) - {
            normalize_contig_name(c, contigs) for c in variants.records.contigs
        }:
            logger.warning(
                f"Contigs in queries {unavailable_contigs} are not found in the VCF."
            )

        all_samples.extend(variants.samples)
    else:
        variants = None

    if bigwigs is not None:
        unavail = []
        for bw in bigwigs:
            if unavailable_contigs := set(contigs) - set(
                normalize_contig_name(c, contigs) for c in bw.contigs
            ):
                unavail.append(unavailable_contigs)
            all_samples.extend(bw.samples)
        if unavail:
            logger.warning(
                f"Contigs in queries {set(unavail)} are not found in the BigWigs."
            )

    available_samples = set(all_samples)

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
    metadata["n_regions"] = bed.height

    if variants is not None:
        if TYPE_CHECKING:
            assert variants is not None

        logger.info("Writing genotypes.")
        bed = write_variants(
            path,
            bed,
            variants,
            variants,
            region_length,
            samples,
            max_mem,
        )
        if isinstance(variants.genotypes, VCFGenos):
            variants.genotypes.close()
        metadata["ploidy"] = variants.ploidy
        # free memory
        del variants
        gc.collect()

    write_regions(path, bed, contigs)

    if bigwigs is not None:
        logger.info("Writing BigWig intervals.")
        for bw in bigwigs:
            write_bigwigs(path, bed, bw, samples)

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info("Finished writing.")


def prep_bed(
    bed: Union[str, Path, pl.DataFrame],
    length: Optional[int] = None,
    max_jitter: Optional[int] = None,
) -> Tuple[pl.DataFrame, List[str], int]:
    if isinstance(bed, (str, Path)):
        bed = read_bedlike(bed)

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
        bed = bed.sort(pl.col("chrom").cast(pl.Categorical), pl.col("chromStart"))

    if length is None:
        length = cast(
            int, bed.select((pl.col("chromEnd") - pl.col("chromStart")).max()).item()
        )

    if max_jitter is not None:
        length += 2 * max_jitter

    bed = with_length(bed, length)

    return bed, contigs, length


def write_regions(path: Path, bed: pl.DataFrame, contigs: List[str]):
    with pl.StringCache():
        pl.Series(contigs, dtype=pl.Categorical)
        regions = bed.with_columns(
            pl.col("chrom").cast(pl.Categorical).to_physical()
        ).with_columns(pl.all().cast(pl.Int32))
    regions = regions.to_numpy()
    np.save(path / "regions.npy", regions)


def write_variants(
    path: Path,
    bed: pl.DataFrame,
    vcf: Path,
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

    gvl_arrow = re.sub(r"\.[bv]cf(\.gz)?$", ".gvl.arrow", vcf.name)
    recs = pl.read_ipc(vcf.parent / gvl_arrow)

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs.select("POS", "ALT", "ILEN").write_ipc(out_dir / "variants.arrow")

    rel_start_idxs: Dict[str, NDArray[np.int32]] = {}
    rel_end_idxs: Dict[str, NDArray[np.int32]] = {}
    chunk_offsets: Dict[str, NDArray[np.intp]] = {}
    n_chunks = 0
    for contig, part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        _contig = normalize_contig_name(contig, variants.records.contigs)
        if _contig is not None:
            starts = part["chromStart"].to_numpy()
            ends = starts + round(region_length * EXTEND_END_MULTIPLIER)
            rel_s_idxs = variants.records.find_relative_start_idx(_contig, starts)
            rel_e_idxs = variants.records.find_relative_end_idx(_contig, ends)
            mem_per_r = (rel_e_idxs - rel_s_idxs) * variants.ploidy * len(_samples)
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
    offset_memmap_offsets = 0
    last_offset = 0
    max_ends = np.empty(bed.height, dtype=np.int32)
    last_max_end_idx = 0
    with tqdm(total=n_chunks) as pbar:
        for contig, part in bed.partition_by(
            "chrom", as_dict=True, include_key=False, maintain_order=True
        ).items():
            c_offsets = chunk_offsets[contig]
            for o_s, o_e in zip(c_offsets[:-1], c_offsets[1:]):
                rel_s_idxs = rel_start_idxs[contig][o_s:o_e]
                rel_e_idxs = rel_end_idxs[contig][o_s:o_e]
                starts = part[o_s:o_e, "chromStart"].to_numpy()
                ends = part[o_s:o_e, "chromEnd"].to_numpy(writable=True)
                n_regions = len(rel_s_idxs)
                pbar.set_description(
                    f"Reading genotypes for {n_regions} regions on {contig}"
                )

                _contig = normalize_contig_name(contig, variants.records.contigs)

                genos, chunk_max_ends = read_variants_chunk(
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

                if _contig is not None:
                    # make indices absolute
                    genos.variant_idxs += variants.records.contig_offsets[_contig]
                max_ends[last_max_end_idx : last_max_end_idx + n_regions] = (
                    chunk_max_ends
                )
                last_max_end_idx += n_regions

                pbar.set_description(
                    f"Writing genotypes for {part.height} regions on {contig}"
                )
                (
                    v_idx_memmap_offsets,
                    offset_memmap_offsets,
                    last_offset,
                ) = write_variants_chunk(
                    out_dir,
                    genos,
                    v_idx_memmap_offsets,
                    offset_memmap_offsets,
                    last_offset,
                )
                pbar.update()

    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=np.int32,
        mode="r+",
        shape=1,
        offset=offset_memmap_offsets,
    )
    out[-1] = last_offset
    out.flush()

    bed = bed.with_columns(chromEnd=pl.lit(max_ends))
    return bed


def read_variants_chunk(
    contig: Optional[str],
    starts: NDArray[np.int32],
    ends: NDArray[np.int32],
    rel_s_idx: NDArray[np.int32],
    rel_e_idx: NDArray[np.int32],
    variants: Variants,
    region_length: int,
    samples: List[str],
    sample_idx: Optional[NDArray[np.intp]],
):
    if contig is None:
        genos = SparseGenotypes.empty(
            n_regions=len(rel_s_idx), n_samples=len(samples), ploidy=variants.ploidy
        )
        chunk_max_ends = ends
    else:
        first = True
        while True:
            if not first:
                rel_e_idx = variants.records.find_relative_end_idx(contig, ends)
            s_idx = variants.records.contig_offsets[contig] + rel_s_idx
            e_idx = variants.records.contig_offsets[contig] + rel_e_idx
            # (s p v)
            genos = variants.genotypes.read(contig, s_idx, e_idx, sample_idx=sample_idx)
            n_per_region = e_idx - s_idx
            offsets = lengths_to_offsets(n_per_region)

            # (s p r)
            _, haplotype_ilens = get_haplotype_ilens(
                genos, rel_s_idx, offsets, variants.records.v_diffs[contig]
            )
            haplotype_lengths = rel_e_idx - rel_s_idx + haplotype_ilens
            # (s p r)
            missing_length = region_length - haplotype_lengths
            if np.all(missing_length <= 0):
                break
            # (r)
            ends += np.ceil(EXTEND_END_MULTIPLIER * missing_length.max((0, 1))).astype(
                np.int32
            )

        genos, chunk_max_ends = SparseGenotypes.from_dense_with_length(
            genos=genos,
            first_v_idxs=rel_s_idx,
            offsets=lengths_to_offsets(e_idx - s_idx),
            ilens=variants.records.v_diffs[contig],
            positions=variants.records.v_starts[contig],
            starts=starts,
            length=region_length,
        )

        # make indices absolute
        genos.variant_idxs += variants.records.contig_offsets[contig]
    return genos, chunk_max_ends


def write_variants_chunk(
    out_dir: Path,
    genos: SparseGenotypes,
    v_idx_memmap_offset: int,
    offsets_memmap_offset: int,
    last_offset: int,
):
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


def write_bigwigs(
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
    for contig, part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
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
