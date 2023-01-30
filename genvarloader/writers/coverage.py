import gc
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
import zarr
from pysam import AlignmentFile
from typing_extensions import assert_never

from genvarloader.types import PathType, Tn5CountMethod


def tn5_coverage(
    in_bam: PathType,
    out_zarr: PathType,
    contigs: Optional[List[str]] = None,
    contig_lens: Optional[Union[List[int], Dict[str, int]]] = None,
    offset_tn5=True,
    count_method: Union[Tn5CountMethod, str] = Tn5CountMethod.CUTSITE,
):
    """Write Tn5 coverage from BAM to Zarr

    Parameters
    ----------
    in_bam : Path
    out_zarr : Path
    contigs : str, optional
        List of contigs to write, defaults to all contigs.
    offset_tn5 : bool, optional
        Whether to offset read lengths for Tn5.
    count_method : Tn5CountMethod, optional
        What to count for coverage.
    """
    count_method = Tn5CountMethod(count_method)

    z = zarr.open_group(out_zarr)
    cover = z.create_group("tn5_coverage")
    cover.attrs["offset_tn5"] = offset_tn5
    cover.attrs["count_method"] = count_method.value

    start_time = perf_counter()
    with AlignmentFile(str(in_bam), "r") as bam:

        # Check valid chrom list and lengths
        if contigs is None:
            contigs = list(bam.references)

        if contig_lens is None:
            _chrom_lens = {c: bam.get_reference_length(c) for c in contigs}
        elif isinstance(contig_lens, list):
            if len(contigs) != len(contig_lens):
                raise ValueError("Number of contigs and contig lengths are different.")
            _chrom_lens = dict(zip(contigs, contig_lens))
        else:
            _chrom_lens = contig_lens

        # Parse through chroms
        for contig in contigs:
            start_chrom = perf_counter()

            curr_len = _chrom_lens[contig]
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

            cover.create_dataset(contig, data=out_array)

            del out_array
            gc.collect()

            print(
                f"Counted {contig} fragments in {perf_counter() - start_chrom:.2f} seconds!"
            )

    print(f"Processed fragment data in {perf_counter() - start_time:.2f} seconds!")


def coverage(in_bam: Path, out_zarr: Path, contigs: Optional[List[str]] = None):
    """Write per-allele coverage from BAM to Zarr

    Parameters
    ----------
    in_bam : Path, BAM file to find read depths for
    out_zarr : Path, Zarr file to write depth
    contigs: list[str]
        contigs to parse, defaults to ALL contigs
    """
    start_time = perf_counter()

    z = zarr.open_group(str(out_zarr))
    cover = z.create_group("coverage")
    with AlignmentFile(str(in_bam), "r") as bam:

        if not contigs:
            contigs = list(bam.header.references)

        for contig in contigs:
            start_chrom = perf_counter()

            cover_arrays = np.array(bam.count_coverage(contig), dtype=np.uint16)
            cover_arrays[cover_arrays > 255] = 255
            cover_arrays = cover_arrays.astype(np.uint8)

            cover.create_dataset(contig, data=cover_arrays)

            del cover_arrays
            gc.collect()

            print(
                f"Counted & wrote contig {contig} coverage in {perf_counter() - start_chrom:.2f} seconds!"
            )

    print(f"Processed allele coverage in {perf_counter() - start_time:.2f} seconds!")
