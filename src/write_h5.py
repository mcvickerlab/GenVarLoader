from pathlib import Path
from random import choice

import tempfile
import timeit

import h5py
import numpy as np
import pandas as pd

from pysam import FastaFile


def one_hot_array(seq_array):
    """Function called by chrom_genome()

    :param seq_array: _description_
    :type seq_array: _type_
    :return: _description_
    :rtype: _type_
    """
    base_dict = {b"A": 0, b"C": 1, b"G": 2, b"T": 3,
                 b"M": [0, 1], b"R": [0, 2], b"W": [0, 3],
                 b"S": [1, 2], b"Y": [1, 3], b"K": [2, 3],
                 b"V": [0, 1, 2], b"H": [0, 1, 3],
                 b"D": [0, 2, 3], b"B": [1, 2, 3]
                }

    onehot_matrix = np.zeros((len(seq_array), 4), dtype=np.uint8)

    for row, base in zip(onehot_matrix, seq_array):
        if base == b"N":
            row[choice((0, 1, 2, 3))] = 1
        else:
            row[base_dict[base]] = 1

    return onehot_matrix


def chrom_genome(in_fasta, chrom, out_h5):
    """Function called by write_genome_h5()

    :param in_fasta: _description_
    :type in_fasta: _type_
    :param chrom: _description_
    :type chrom: _type_
    :param out_h5: _description_
    :type out_h5: _type_
    """
    start_genome = timeit.default_timer()

    with FastaFile(in_fasta) as file:
        seq_array = np.fromiter(file.fetch(chrom),
                                count=file.get_reference_length(chrom),
                                dtype="|S1")

    with h5py.File(out_h5, "a") as file:
        chrom_group = file.require_group(chrom)
        chrom_group.create_dataset("sequence", data=one_hot_array(seq_array), compression="gzip")

    print(f"Created {chrom} data in {timeit.default_timer() - start_genome} seconds!")


def write_genome_h5(in_fasta, out_dir, h5_name=None, chrom_list=None):
    """Creates onehot-encoded genomme in h5 format

    :param in_fasta: _description_
    :type in_fasta: _type_
    :param out_dir: _description_
    :type out_dir: _type_
    :param h5_name: _description_, defaults to None
    :type h5_name: _type_, optional
    :param chrom_list: _description_, defaults to None
    :type chrom_list: _type_, optional
    """
    
    if not chrom_list:
        with FastaFile(in_fasta) as fasta:
            chrom_list = fasta.references
    
    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name) # add check later to make sure not dir
    else:
        out_h5 = str(Path(out_dir) / "genome.h5")

    start_time = timeit.default_timer()
    
    for chrom in chrom_list:
        chrom_genome(in_fasta, chrom, out_h5)

    print(f"Finished in {timeit.default_timer() - start_time} seconds!")
    print(f"One-Hot encoded genome written to {out_h5}")

