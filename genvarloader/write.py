import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
from einops import rearrange
from loguru import logger
from tqdm.auto import tqdm

from .bigwig import BigWigs
from .util import normalize_contig_name, read_bedlike, with_length
from .variants import Variants
from .variants.records import RecordInfo


def write(
    path: Union[str, Path],
    bed: Union[str, Path, pl.DataFrame],
    vcf: Optional[Union[str, Path]] = None,
    bigwigs: Optional[BigWigs] = None,
    samples: Optional[List[str]] = None,
    length: Optional[int] = None,
    max_jitter: Optional[int] = None,
):
    if vcf is None and bigwigs is None:
        raise ValueError("At least one of `vcf` or `bigwigs` must be provided.")

    logger.info(f"Writing to {path}")

    metadata = {}
    path = Path(path)
    if path.exists():
        logger.info("Found existing GVL store, overwriting.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    bed, contigs, region_length = prep_bed(bed, length, max_jitter)
    metadata["region_length"] = region_length
    metadata["contigs"] = contigs
    if max_jitter is not None:
        metadata["max_jitter"] = max_jitter

    available_samples = None
    if vcf is not None:
        vcf = Path(vcf)
        variants = Variants.from_vcf(vcf, use_cache=False)
        if available_samples is None:
            available_samples = set(variants.samples)
        else:
            available_samples &= set(variants.samples)

    if bigwigs is not None:
        if available_samples is None:
            available_samples = set(bigwigs.samples)
        else:
            available_samples &= set(bigwigs.samples)

    if samples is not None:
        _samples = set(samples)
        if missing := (_samples - available_samples):
            raise ValueError(f"Samples {missing} not found in VCF or BigWigs.")
        samples = list(_samples)
    else:
        samples = list(available_samples)

    logger.info(f"Using {len(samples)} samples: {samples}")

    if vcf is not None:
        logger.info("Writing genotypes.")
        bed, ploidy, n_variants, samples = write_variants(
            path,
            bed,
            vcf,
            variants,  # type: ignore
            samples,
        )
        metadata["ploidy"] = ploidy
        metadata["n_variants"] = n_variants

    write_regions(path, bed, contigs)

    if bigwigs is not None:
        logger.info("Writing BigWig intervals.")
        n_intervals = write_bigwigs(path, bed, bigwigs, samples)
        metadata["n_intervals"] = n_intervals

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

    bed = bed.select("chrom", "chromStart", "chromEnd").sort(
        "chrom", "chromStart", "chromEnd"
    )

    if length is None:
        length = cast(
            int, bed.select((pl.col("chromEnd") - pl.col("chromStart")).max()).item()
        )

    if max_jitter is not None:
        length += 2 * max_jitter

    bed = with_length(bed, length)
    contigs = bed["chrom"].unique(maintain_order=True).to_list()

    return bed, contigs, length


def write_regions(path: Path, bed: pl.DataFrame, contigs: List[str]):
    with pl.StringCache():
        pl.Series(contigs, dtype=pl.Categorical)
        regions = bed.with_columns(pl.col("chrom").cast(pl.Categorical)).with_columns(
            pl.all().cast(pl.Int32)
        )
    regions = regions.to_numpy()
    np.save(path / "regions.npy", regions)


def write_variants(
    path: Path,
    bed: pl.DataFrame,
    vcf: Path,
    variants: Variants,
    samples: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, int, int, List[str]]:
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
    np.save(path / "samples.npy", _samples)

    gvl_arrow = re.sub(r"\.[bv]cf(\.gz)?$", ".gvl.arrow", vcf.name)
    recs = pl.read_ipc(vcf.parent / gvl_arrow)
    recs.select("POS", "ALT", "ILEN").write_ipc(path / "variants.arrow")

    (path / "genotypes").mkdir(parents=True, exist_ok=True)

    memmap_offsets = {
        "genotypes": 0,
        "fv_idxs": 0,
        "offsets": 0,
    }
    last_offset = 0
    n_variants = 0
    pbar = tqdm(total=bed["chrom"].n_unique())
    all_max_ends = []
    for contig, part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        pbar.set_description(f"Writing genotypes for {contig}")
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()
        n_regions = part.height

        _contig = normalize_contig_name(contig, variants.records.contigs)
        if _contig is None:
            records = RecordInfo.empty(n_regions)
            max_ends = ends.astype(np.int32)
            genos = np.empty((0, variants.ploidy), np.int8)
        else:
            (
                records,
                max_ends,
            ) = variants.records.vars_in_range_for_haplotype_construction(
                _contig, starts, ends
            )
            genos = variants.genotypes.read(
                _contig, records.start_idxs, records.end_idxs, sample_idx=sample_idx
            )
            genos = rearrange(genos, "s p v -> (s v) p")

        all_max_ends.append(max_ends)
        n_variants += len(records.positions)

        out = np.memmap(
            path / "genotypes" / "genotypes.npy",
            dtype=genos.dtype,
            mode="w+" if memmap_offsets["genotypes"] == 0 else "r+",
            shape=genos.shape,
            offset=memmap_offsets["genotypes"],
        )
        out[:] = genos[:]
        out.flush()
        memmap_offsets["genotypes"] += out.nbytes

        out = np.memmap(
            path / "genotypes" / "first_variant_idxs.npy",
            dtype=records.start_idxs.dtype,
            mode="w+" if memmap_offsets["fv_idxs"] == 0 else "r+",
            shape=records.start_idxs.shape,
            offset=memmap_offsets["fv_idxs"],
        )
        out[:] = records.start_idxs[:]
        out.flush()
        memmap_offsets["fv_idxs"] += out.nbytes

        offsets = records.offsets.copy()
        offsets += last_offset
        last_offset = offsets[-1]
        out = np.memmap(
            path / "genotypes" / "offsets.npy",
            dtype=offsets.dtype,
            mode="w+" if memmap_offsets["offsets"] == 0 else "r+",
            shape=len(offsets) - 1,
            offset=memmap_offsets["offsets"],
        )
        out[:] = offsets[:-1]
        out.flush()
        memmap_offsets["offsets"] += out.nbytes
        pbar.update()
    pbar.close()

    out = np.memmap(
        path / "genotypes" / "offsets.npy",
        dtype=offsets.dtype,  # type: ignore
        mode="r+",
        shape=1,
        offset=memmap_offsets["offsets"],
    )
    out[-1] = offsets[-1]  # type: ignore
    out.flush()

    max_ends = np.concatenate(all_max_ends)
    bed = bed.with_columns(chromEnd=pl.lit(max_ends))

    return bed, variants.ploidy, int(n_variants), _samples


def write_bigwigs(
    path: Path, bed: pl.DataFrame, bigwigs: BigWigs, samples: Optional[List[str]]
) -> int:
    if samples is None:
        _samples = cast(List[str], bigwigs.samples)
    else:
        if missing := (set(samples) - set(bigwigs.samples)):
            raise ValueError(f"Samples {missing} not found in bigwigs.")
        _samples = samples
    np.save(path / "samples.npy", _samples)

    (path / "intervals").mkdir(parents=True, exist_ok=True)

    memmap_offsets = {
        "intervals": 0,
        "values": 0,
        "offsets": 0,
    }
    last_offset = 0
    n_intervals = 0
    pbar = tqdm(total=bed["chrom"].n_unique())
    for contig, part in bed.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        pbar.set_description(f"Writing intervals for {contig}")
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()

        intervals = bigwigs.intervals(contig, starts, ends, sample=_samples)
        n_intervals += len(intervals.intervals)

        out = np.memmap(
            path / "intervals" / "intervals.npy",
            dtype=intervals.intervals.dtype,
            mode="w+" if memmap_offsets["intervals"] == 0 else "r+",
            shape=intervals.intervals.shape,
            offset=memmap_offsets["intervals"],
        )
        out[:] = intervals.intervals[:]
        out.flush()
        memmap_offsets["intervals"] += intervals.intervals.nbytes

        out = np.memmap(
            path / "intervals" / "values.npy",
            dtype=intervals.values.dtype,
            mode="w+" if memmap_offsets["values"] == 0 else "r+",
            shape=intervals.values.shape,
            offset=memmap_offsets["values"],
        )
        out[:] = intervals.values[:]
        out.flush()
        memmap_offsets["values"] += intervals.values.nbytes

        offsets = np.empty(len(intervals.n_per_query) + 1, dtype=np.uint32)
        offsets[0] = 0
        intervals.n_per_query.cumsum(out=offsets[1:])
        offsets += last_offset
        last_offset = offsets[-1]
        out = np.memmap(
            path / "intervals" / "offsets.npy",
            dtype=offsets.dtype,
            mode="w+" if memmap_offsets["offsets"] == 0 else "r+",
            shape=len(offsets) - 1,
            offset=memmap_offsets["offsets"],
        )
        out[:] = offsets[:-1]
        out.flush()
        memmap_offsets["offsets"] += offsets[:-1].nbytes
        pbar.update()
    pbar.close()

    out = np.memmap(
        path / "intervals" / "offsets.npy",
        dtype=np.uint32,
        mode="r+",
        shape=1,
        offset=memmap_offsets["offsets"],
    )
    out[-1] = n_intervals
    out.flush()

    return n_intervals
