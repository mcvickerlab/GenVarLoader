import timeit
import tempfile

import numpy as np
import pandas as pd

import pysam
from pysam.libcalignmentfile import AlignmentFile


def get_read_depth(in_bam, chrom_list=None, chrom_lens=None):
    start_time = timeit.default_timer()

    depth_dict = {}
    # Check if inputs valid
    if not chrom_list or not chrom_lens:
        with AlignmentFile(in_bam, "r") as bam:
            if not chrom_list:
                chrom_list = list(bam.header.references)

            chrom_lens = {chrom: bam.header.get_reference_length(chrom)
                          for chrom in chrom_list}

    elif not isinstance(chrom_lens, dict):
        chrom_lens = {chrom: clen for chrom,
                      clen in zip(chrom_list, chrom_lens)}

    # Create temp file for stdout processing
    with tempfile.NamedTemporaryFile() as temp_file:

        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            pysam.depth("-r", chrom, "-o", temp_file.name,
                        in_bam, catch_stdout=False)

            depth_df = pd.read_csv(temp_file.name, sep="\t", header=None,
                                   names=["chrom", "pos", "count"], usecols=["pos", "count"],
                                   dtype={"pos": np.uint32, "count": np.uint16})

            pos_array = depth_df["pos"].to_numpy(copy=False)
            count_array = depth_df["count"].to_numpy(copy=False)

            pos_array -= 1  # convert to 0 base index
            count_array[np.where(count_array > 255)] = 255  # fit within 1 byte

            # make output array
            out_array = np.zeros(chrom_lens[chrom], dtype=np.uint8)
            out_array[pos_array] += count_array

            depth_dict[chrom] = out_array

            print(
                f"Counted {chrom} read-depth in {timeit.default_timer() - start_chrom:.2f} seconds!")

    print(
        f"Processed read-depth data in {timeit.default_timer() - start_time:.2f} seconds!")
    return depth_dict


def get_allele_coverage(in_bam, chrom_list=None):
    start_time = timeit.default_timer()

    coverage_dict = {}
    with AlignmentFile(in_bam, "r") as bam:

        if not chrom_list:
            chrom_list = list(bam.header.references)

        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            cover_arrays = np.array(bam.count_coverage(chrom), dtype=np.uint16)
            cover_arrays[np.where(cover_arrays > 255)] = 255
            cover_arrays = cover_arrays.astype(np.uint8)

            coverage_dict[chrom] = cover_arrays

            print(
                f"Counted {chrom} coverage in {timeit.default_timer() - start_chrom:.2f} seconds!")

    print(
        f"Processed allele coverage in {timeit.default_timer() - start_time:.2f} seconds!")
    return coverage_dict
