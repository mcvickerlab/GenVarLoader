from pathlib import Path
import timeit

import h5py
import numpy as np
import pandas as pd
from pysam import FastaFile

from .encode_data import parse_encode_list, encode_from_fasta
from .get_data import get_read_depth, get_allele_coverage

# FASTA to H5 Writers
def write_genome_seq(in_fasta, out_dir, h5_name=None, chrom_list=None):

    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name)
    else:
        out_h5 = str(Path(out_dir) / "genome_sequence.h5")

    start_time = timeit.default_timer()
    with FastaFile(in_fasta) as fasta, h5py.File(out_h5, "w") as h5_file:

        if not chrom_list:
            chrom_list = fasta.references

        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            seq_array = np.fromiter(fasta.fetch(chrom),
                                    count=fasta.get_reference_length(chrom),
                                    dtype="|S1")

            chrom_group = h5_file.require_group(chrom)
            chrom_group.create_dataset(
                "sequence", data=seq_array, compression="gzip")

            print(
                f"Created {chrom} data in {timeit.default_timer() - start_chrom:.2f} seconds!")

        h5_file.attrs["id"] = "sequence"

    print(f"Finished in {timeit.default_timer() - start_time:.2f} seconds!")
    print(f"Genome character-arrays written to {out_h5}")


def write_encoded_genome(in_fasta, out_dir, h5_name=None, chrom_list=None, encode_spec=None):

    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name)
    else:
        out_h5 = str(Path(out_dir) / "genome_onehot.h5")

    # Get data using encoding function
    onehot_dict = encode_from_fasta(
        in_fasta, chrom_list=chrom_list, encode_spec=encode_spec)

    start_write = timeit.default_timer()
    with h5py.File(out_h5, "w") as h5_file:

        for chrom, onehot in onehot_dict.items():
            chrom_group = h5_file.require_group(chrom)
            chrom_group.create_dataset(
                "onehot", data=onehot, compression="gzip")

        h5_file.attrs["id"] = "onehot"
        h5_file.attrs["encode_spec"] = [base.decode()
                                        for base in parse_encode_list(encode_spec)]

    print(
        f"Finished writing in {timeit.default_timer() - start_write:.2f} seconds!")
    print(f"One-Hot encoded genome written to {out_h5}")


# BAM to H5 Writers
def write_read_depth(in_bam, out_dir, h5_name=None, chrom_list=None, chrom_lens=None):

    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name)
    else:
        out_h5 = str(Path(out_dir) / "read_depths.h5")

    # Get data using read depth function
    depth_dict = get_read_depth(
        in_bam, chrom_list=chrom_list, chrom_lens=chrom_lens)

    start_write = timeit.default_timer()
    with h5py.File(out_h5, "w") as h5_file:

        for chrom, depth_array in depth_dict.items():
            chrom_group = h5_file.require_group(chrom)
            chrom_group.create_dataset(
                "depth", data=depth_array, compression="gzip")

        h5_file.attrs["id"] = "depth"

    print(
        f"Finished writing in {timeit.default_timer() - start_write:.2f} seconds!")
    print(f"Read-Depths written to {out_h5}")


def write_allele_coverage(in_bam, out_dir, h5_name=None, chrom_list=None):

    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name)
    else:
        out_h5 = str(Path(out_dir) / "allele_coverage.h5")

    # Get data using allele coverage function
    coverage_dict = get_allele_coverage(in_bam, chrom_list=chrom_list)

    start_write = timeit.default_timer()
    with h5py.File(out_h5, "w") as h5_file:

        for chrom, cover_matrix in coverage_dict.items():
            chrom_group = h5_file.require_group(chrom)
            chrom_group.create_dataset(
                "coverage", data=cover_matrix, compression="gzip")

        h5_file.attrs["id"] = "coverage"

    print(
        f"Finished writing in {timeit.default_timer() - start_write:.2f} seconds!")
    print(f"Allele Coverage written to {out_h5}")
