import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import joblib
import numpy as np
import zarr
from pysam import AlignmentFile
from typing_extensions import assert_never

from genvarloader.types import Tn5CountMethod


def init_zarr(
    in_bams: Dict[str, Path],
    out_zarr: Path,
    contigs: Optional[List[str]],
    overwrite: bool,
    extra_attrs: Dict[str, Any],
):
    store_existed = out_zarr.exists()
    root = zarr.open_group(out_zarr)
    root.attrs.update(**extra_attrs)

    samples = list(in_bams.keys())
    existing_samples = cast(List[str], root.attrs.get("samples", []))
    old_samples = [s for s in samples if s in set(existing_samples)]
    new_samples = [s for s in samples if s not in set(existing_samples)]

    if len(new_samples) == 0 and not overwrite:
        return root, np.array([], "u1"), []
    elif len(old_samples) > 0 and not overwrite:
        raise ValueError(
            "Got samples that are already in the output zarr:", old_samples
        )
    else:
        total_samples = existing_samples + new_samples
        root.attrs["samples"] = total_samples

    if not store_existed:
        with AlignmentFile(str(in_bams[new_samples[0]])) as bam:
            if contigs is None:
                contigs = list(bam.references)
            contig_lens = [bam.get_reference_length(c) for c in contigs]
            contig_lengths = dict(zip(contigs, contig_lens))
        root.attrs["contig_lengths"] = contig_lengths
        for c, c_len in contig_lengths.items():
            root.require_dataset(
                c,
                shape=(len(new_samples), c_len),
                chunks=(1, int(1e6)),
                compressor=zarr.Blosc(cname="zstd", clevel=7, shuffle=-1),
            )
    elif len(new_samples) > 0:

        def resize_to_fit_new_samples(arr: Union[zarr.Group, zarr.Array]):
            if isinstance(arr, zarr.Array):
                new_shape = (len(total_samples), *arr.shape[1:])
                arr.resize(new_shape)

        root.visitvalues(resize_to_fit_new_samples)

    if overwrite:
        samples_to_write = in_bams
    else:
        samples_to_write = {s: p for s, p in in_bams.items() if s in new_samples}

    *_, sample_idx = np.intersect1d(
        list(samples_to_write.keys()), total_samples, return_indices=True
    )

    return root, sample_idx, list(samples_to_write.values())


def coverage(
    idx: int, bam_path: Path, out_zarr: zarr.Group, contigs: Optional[List[str]]
) -> None:
    with AlignmentFile(str(bam_path), "r") as bam:
        if not contigs:
            contigs = list(bam.header.references)

        for contig in contigs:
            if bam.get_reference_length(contig) != len(out_zarr[contig]):
                raise RuntimeError(
                    f"Length of contig {contig} in BAM != length of contig in Zarr store."
                )

            cover_array = np.array(bam.count_coverage(contig), dtype=np.uint16)
            cover_array[cover_array > 255] = 255
            cover_array = cover_array.astype(np.uint8)

            out_zarr[contig][idx] = cover_array

            del cover_array
            gc.collect()


def write_coverages(
    in_bams: Dict[str, Path],
    out_zarr: Path,
    contigs: Optional[List[str]] = None,
    n_jobs=None,
    overwrite=False,
):
    extra_attrs = {}
    root, sample_idx, samples_to_write = init_zarr(
        in_bams, out_zarr, contigs, overwrite, extra_attrs
    )

    with joblib.Parallel(n_jobs, prefer="threads") as exe:
        writer = joblib.delayed(coverage)
        tasks = [
            writer(idx, bam_path, out_zarr=root, contigs=contigs)
            for idx, bam_path in zip(sample_idx, samples_to_write)
        ]
        exe(tasks)


def tn5_coverage(
    idx: int,
    bam_path: Path,
    out_zarr: zarr.Group,
    contigs: Optional[List[str]],
    offset_tn5: bool,
    count_method: Tn5CountMethod,
):
    with AlignmentFile(str(bam_path), "r") as bam:

        if contigs is None:
            contigs = list(bam.references)

        # Parse through chroms
        for contig in contigs:
            curr_len = bam.get_reference_length(contig)
            if curr_len != len(out_zarr[contig]):
                raise RuntimeError(
                    f"Length of contig {contig} in BAM != length of contig in Zarr store."
                )

            out_array = np.zeros(curr_len, dtype=np.uint16)

            read_cache: Dict[str, Any] = {}

            for read in bam.fetch(contig):

                if not read.is_proper_pair or read.is_secondary:
                    continue

                if read.query_name not in read_cache:
                    read_cache[read.query_name] = read
                    continue

                # Forward and Reverse w/o r1 and r2
                if read.is_reverse:
                    forward_read = read_cache.pop(read.query_name)
                    reverse_read = read
                else:
                    forward_read = read
                    reverse_read = read_cache.pop(read.query_name)

                # Shift read if accounting for offset
                if offset_tn5:
                    forward_start: int = forward_read.reference_start + 4
                    # 0 based, 1 past aligned
                    reverse_end: int = reverse_read.reference_end - 5
                else:
                    forward_start = forward_read.reference_start
                    reverse_end = reverse_read.reference_end

                # Check count method
                if count_method is Tn5CountMethod.CUTSITE:
                    # Add cut sites to out_array
                    out_array[[forward_start, (reverse_end - 1)]] += 1
                elif count_method is Tn5CountMethod.MIDPOINT:
                    # Add midpoint to out_array
                    out_array[int((forward_start + (reverse_end - 1)) / 2)] += 1
                elif count_method is Tn5CountMethod.FRAGMENT:
                    # Add range to out array
                    out_array[forward_start:reverse_end] += 1
                else:
                    assert_never(count_method)

            # Find int8 overflows
            out_array[out_array > 255] = 255
            out_array = out_array.astype(np.uint8)

            out_zarr[contig][idx] = out_array

            del out_array
            gc.collect()


def write_tn5_coverages(
    in_bams: Dict[str, Path],
    out_zarr: Path,
    contigs: Optional[List[str]] = None,
    n_jobs=None,
    overwrite=False,
    offset_tn5=True,
    count_method: Union[Tn5CountMethod, str] = Tn5CountMethod.CUTSITE,
):
    count_method = Tn5CountMethod(count_method)
    extra_attrs = {"offset_tn5": offset_tn5, "count_method": count_method}

    root, sample_idx, samples_to_write = init_zarr(
        in_bams, out_zarr, contigs, overwrite, extra_attrs
    )

    with joblib.Parallel(n_jobs, prefer="threads") as exe:
        writer = joblib.delayed(tn5_coverage)
        tasks = [
            writer(
                idx,
                bam_path,
                out_zarr=root,
                contigs=contigs,
                offset_tn5=offset_tn5,
                count_method=count_method,
            )
            for idx, bam_path in zip(sample_idx, samples_to_write)
        ]
        exe(tasks)
