import timeit

import h5py
import numpy as np
import pandas as pd

from pysam import FastaFile


def array_to_onehot(seq_array, base_list):
    seq_array[np.isin(seq_array,
                      [b"A", b"C", b"G", b"T"], invert=True)] = b"N"  # Convert ambiguous

    return pd.get_dummies(seq_array).reindex(
        columns=base_list, fill_value=0).to_numpy()


def parse_encode_list(encode_spec):
    if not encode_spec:
        encode_spec = [b"A", b"C", b"G", b"T", b"N"]

    elif isinstance(encode_spec, (list, tuple)):
        encode_spec = [base.encode() for base in encode_spec]

    elif isinstance(encode_spec, str):
        encode_spec = [base.encode() for base in list(encode_spec)]

    else:
        raise TypeError("Please input string or list of strings!")

    return encode_spec


def encode_sequence(seq_data, encode_spec=None):

    # Process sequence input
    if isinstance(seq_data, str):  # sequence input as string
        seq_data = np.fromiter(seq_data, count=len(seq_data), dtype="|S1")

    elif isinstance(seq_data, np.ndarray):  # seq data is numpy array
        if seq_data.dtype != "|S1":
            seq_data = seq_data.astype("|S1")

    else:
        raise TypeError("Please input as string or numpy array!")

    encode_spec = parse_encode_list(encode_spec)

    return array_to_onehot(seq_data, encode_spec)


def encode_from_fasta(in_fasta, chrom_list=None, encode_spec=None):

    onehot_dict = {}
    start_time = timeit.default_timer()
    print(
        f"Encoding w. Specs: {[base.decode() for base in parse_encode_list(encode_spec)]}")

    with FastaFile(in_fasta) as fasta:

        if not chrom_list:
            chrom_list = fasta.references

        for chrom in chrom_list:
            start_chrom = timeit.default_timer()

            seq_array = np.fromiter(fasta.fetch(chrom),
                                    count=fasta.get_reference_length(chrom),
                                    dtype="|S1")

            onehot_dict[chrom] = encode_sequence(seq_array, encode_spec)

            print(
                f"Encoded {chrom} in {timeit.default_timer() - start_chrom:.2f} seconds!")

    print(f"Finished in {timeit.default_timer() - start_time:.2f} seconds!")
    return onehot_dict


def encode_from_h5(in_h5, chrom_list=None, encode_spec=None):

    if not h5py.is_hdf5(in_h5):
        raise ValueError("File is not valid HDF5")

    onehot_dict = {}
    start_time = timeit.default_timer()
    print(
        f"Encoding w. Specs: {[base.decode() for base in parse_encode_list(encode_spec)]}")

    with h5py.File(in_h5, "r") as file:

        if not chrom_list:
            chrom_list = list(file.keys())

        for chrom in chrom_list:

            if f"{chrom}/sequence" not in file:
                print(f"{chrom} sequence not found!")
                continue

            start_chrom = timeit.default_timer()

            onehot_dict[chrom] = encode_sequence(
                file[chrom]["sequence"][:], encode_spec)

            print(
                f"Encoded {chrom} in {timeit.default_timer() - start_chrom:.2f} seconds!")

        print(
            f"Finished in {timeit.default_timer() - start_time:.2f} seconds!")
        return onehot_dict
