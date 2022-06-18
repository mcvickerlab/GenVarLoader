import timeit
import tempfile

import numpy as np
import pandas as pd

import pysam
from pysam.libcalignmentfile import AlignmentFile


# NEW METHOD TO REPLACE get_read_depth
def get_frag_depth(
    in_bam, 
    chrom_list=None, chrom_lens=None, 
    offset_tn5=True, count_method="cutsite"):

    start_time = timeit.default_timer()
    
    depth_dict = {}
    
    with AlignmentFile(in_bam, "r") as bam:

        # Check valid chrom list and lengths
        if not chrom_list or not chrom_lens:
            if not chrom_list:
                chrom_list = list(bam.header.references)
            
            chrom_lens = {chrom:bam.header.get_reference_length(chrom)
                          for chrom in chrom_list}

        elif not isinstance(chrom_lens, dict):
            chrom_lens = {chrom:clen for chrom, clen in zip(chrom_list, chrom_lens)}
        
        # Parse through chroms
        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            curr_len = chrom_lens[chrom]
            out_array = np.zeros(curr_len, dtype=np.uint16)
            
            read_cache = {}

            for read in bam.fetch(chrom):

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
                    forward_start = forward_read.reference_start + 4
                    reverse_end = reverse_read.reference_end - 5 # 0 based, 1 past aligned
                else:
                    forward_start = forward_read.reference_start
                    reverse_end = reverse_read.reference_end
                
                # Check count method
                if count_method == "cutsite":
                    # Add cut sites to out_array
                    out_array[[forward_start, (reverse_end-1)]] += 1
                elif count_method == "midpoint":
                    # Add midpoint to out_array
                    out_array[int((forward_start+(reverse_end-1))/2)] += 1
                elif count_method == "fragment":
                    # Add range to out array
                    out_array[forward_start:reverse_end] += 1
                else:
                    # Default method, currently cutsite
                    out_array[forward_start:reverse_end] += 1 
            
            # Find int8 overflows
            out_array[np.where(out_array > 255)] = 255
            out_array = out_array.astype(np.uint8)
            
            depth_dict[chrom] = out_array
            
            print(f"Counted {chrom} fragments in {timeit.default_timer() - start_chrom} seconds!")

    print(f"Processed fragment data in {timeit.default_timer() - start_time} seconds!")
    return depth_dict


def get_allele_coverage(in_bam, chrom_list=None):
    """Retrieve per-allele coverage from BAM file

    :param in_bam: BAM file to find read depths for
    :type in_bam: str
    :param chrom_list: Chromosomes to parse, defaults to ALL chroms
    :type chrom_list: list of str, optional
    :return: Dictionary with keys: [chrom] and 4xN matrix with row order A, C, G, T 
             containing allelic coverage per pos(0-based)
    :rtype: dict of np.ndarray
    """
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
