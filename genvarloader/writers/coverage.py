import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import zarr
from numpy.typing import NDArray
from pysam import AlignmentFile
from typing_extensions import assert_never

from genvarloader.types import Tn5CountMethod


def _init_zarr(
    in_bams: Dict[str, Path],
    out_zarr: Path,
    feature: Optional[str] = None,
    contigs: Optional[List[str]] = None,
    overwrite_samples: bool = False,
    extra_attrs: Optional[Dict[str, Any]] = None,
    n_threads: Optional[int] = None,
) -> Tuple[zarr.Group, NDArray, List[Path]]:
    """Initialize a coverage zarr file.

    Parameters
    ----------
    in_bams : Dict[str, Path]
        A dictionary mapping sample names to their BAM files.
    out_zarr : Path
        Path to coverage Zarr.
    feature : str
        What feature should be written if writing a new coverage Zarr.
    contigs : Optional[List[str]], optional
        Which contigs to write, by default None
    overwrite : bool, optional
        Whether to overwrite an existing coverage Zarr, by default False
    extra_attrs : Optional[Dict[str, Any]], optional

    Returns
    -------
    root : zarr.Group
        Root of the zarr store.
    sample_idx : ndarray
        Indices of the samples to write. May be non-trivial if overwriting samples.
    samples_to_write : list[Path]
        Paths for samples to write, in the order they are appended to the Zarr store.
    """

    if n_threads is None:
        n_threads = 1

    store_existed = out_zarr.exists()
    root = zarr.open_group(out_zarr)

    # add feature attribute
    if store_existed:
        feature = root.attrs["feature"]
    elif feature is None:
        raise ValueError("Need to specify feature to initialize a Zarr from scratch.")
    else:
        root.attrs["feature"] = feature

    # add extra attributes
    if extra_attrs is None:
        extra_attrs = {}
    root.attrs.update(**extra_attrs)

    # figure out what samples we're adding
    samples = np.array(list(in_bams.keys()))
    logging.info(f"Got samples: {' '.join(samples)}")

    existing_samples = np.array(root.attrs.get("samples", []))
    if len(np.unique(samples)) != len(samples):
        raise ValueError("Got duplicate samples")
    in_existing = np.isin(samples, existing_samples, assume_unique=True)
    old_samples = samples[in_existing]
    new_samples = samples[~in_existing]

    if len(new_samples) == 0 and not overwrite_samples:
        return root, np.array([], "u1"), []
    elif len(old_samples) > 0 and not overwrite_samples:
        raise ValueError(
            "Got samples that are already in the output zarr:", old_samples
        )
    else:
        total_samples = np.concatenate([existing_samples, new_samples])
        root.attrs["samples"] = total_samples.tolist()

    if not store_existed:
        # initialize Zarr for the first time, get contig lengths and create arrays
        with AlignmentFile(str(in_bams[new_samples[0]]), threads=n_threads) as bam:
            if contigs is None:
                contigs = list(bam.references)
            contig_lens = [bam.get_reference_length(c) for c in contigs]
            contig_lengths = dict(zip(contigs, contig_lens))
        root.attrs["contig_lengths"] = contig_lengths
        for c, c_len in contig_lengths.items():
            root.require_dataset(
                c,
                shape=(len(new_samples), c_len),
                dtype="u2",
                chunks=(1, int(1e6)),
                compressor=zarr.Blosc(cname="zstd", clevel=7, shuffle=-1),
            )
        root.require_dataset(
            "read_count", shape=len(new_samples), dtype="u8", chunks=1, compressor=None
        )
    elif len(new_samples) > 0:
        # already existed and we need to add new samples
        def resize_to_fit_new_samples(arr: Union[zarr.Group, zarr.Array]):
            if isinstance(arr, zarr.Array):
                new_shape = (len(total_samples), *arr.shape[1:])
                arr.resize(new_shape)

        root.visitvalues(resize_to_fit_new_samples)

    if overwrite_samples:
        samples_to_write = in_bams
    else:
        samples_to_write = {s: p for s, p in in_bams.items() if s in new_samples}

    logging.info(f"Samples to write: {' '.join(samples_to_write.keys())}")

    samples_to_write_idx = np.flatnonzero(
        np.isin(total_samples, list(samples_to_write.keys()), assume_unique=True)
    )

    return root, samples_to_write_idx, list(samples_to_write.values())


def coverage(
    sample_idx: int, bam_path: Path, out_zarr: zarr.Group, contigs: Optional[List[str]]
) -> None:
    with AlignmentFile(str(bam_path), "r") as bam:
        if not contigs:
            contigs = list(bam.header.references)

        out_zarr["read_count"][sample_idx] = bam.count(read_callback="all")

        for contig in contigs:
            if bam.get_reference_length(contig) != out_zarr[contig].shape[1]:
                raise RuntimeError(
                    f"Length of contig {contig} in BAM != length of contig in Zarr store."
                )

            acgt_covers = bam.count_coverage(contig)
            cover_array = np.stack(acgt_covers, axis=1).sum(1).astype(np.uint16)

            out_zarr[contig][sample_idx] = cover_array

            del cover_array
            gc.collect()


def write_coverages(
    in_bams: Dict[str, Path],
    out_zarr: Path,
    contigs: Optional[List[str]] = None,
    overwrite_samples=False,
    n_jobs=None,
):
    feature = "depth"
    root, sample_idx, samples_to_write = _init_zarr(
        in_bams, out_zarr, feature, contigs, overwrite_samples, n_threads=n_jobs
    )

    writer = joblib.delayed(coverage)
    tasks = [
        writer(s_idx, bam_path, out_zarr=root, contigs=contigs)
        for s_idx, bam_path in zip(sample_idx, samples_to_write)
    ]

    joblib.Parallel(n_jobs, prefer="threads")(tasks)


def tn5_coverage(
    sample_idx: int,
    bam_path: Path,
    out_zarr: zarr.Group,
    contigs: Optional[List[str]],
    offset_tn5: bool,
    count_method: Tn5CountMethod,
):
    with AlignmentFile(str(bam_path), "r") as bam:

        if contigs is None:
            contigs = list(bam.references)

        out_zarr["read_count"][sample_idx] = bam.count(read_callback="all")

        # Parse through chroms
        for contig in contigs:
            curr_len = bam.get_reference_length(contig)
            if curr_len != out_zarr[contig].shape[1]:
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

            out_zarr[contig][sample_idx] = out_array

            del out_array
            gc.collect()


def write_tn5_coverages(
    in_bams: Dict[str, Path],
    out_zarr: Path,
    contigs: Optional[List[str]] = None,
    n_jobs=None,
    overwrite_samples=False,
    offset_tn5=True,
    count_method: Union[Tn5CountMethod, str] = Tn5CountMethod.CUTSITE,
):
    feature = "tn5"
    count_method = Tn5CountMethod(count_method)
    extra_attrs = {
        "offset_tn5": offset_tn5,
        "count_method": count_method.value,
    }

    root, sample_idx, samples_to_write = _init_zarr(
        in_bams,
        out_zarr,
        feature,
        contigs,
        overwrite_samples,
        extra_attrs,
        n_threads=n_jobs,
    )

    writer = joblib.delayed(tn5_coverage)
    tasks = [
        writer(
            s_idx,
            bam_path,
            out_zarr=root,
            contigs=contigs,
            offset_tn5=offset_tn5,
            count_method=count_method,
        )
        for s_idx, bam_path in zip(sample_idx, samples_to_write)
    ]

    if n_jobs is None:
        n_jobs = -2  # use all but 1 CPU

    joblib.Parallel(n_jobs, prefer="threads")(tasks)
