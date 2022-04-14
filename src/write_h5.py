from pathlib import Path
import timeit

import h5py
import numpy as np
import pandas as pd
from pysam import FastaFile

from encode_data import parse_encode_list, encode_from_fasta


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
            chrom_group.create_dataset("sequence", data=seq_array, compression="gzip")
            
            print(f"Created {chrom} data in {timeit.default_timer() - start_chrom} seconds!")

    print(f"Finished in {timeit.default_timer() - start_time} seconds!")
    print(f"Genome character-arrays written to {out_h5}")


def write_encoded_genome(in_fasta, out_dir, h5_name=None, chrom_list=None, encode_spec=None):
    
    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name)
    else:
        out_h5 = str(Path(out_dir) / "genome_onehot.h5")
    
    # Get data using encoding function
    onehot_dict = encode_from_fasta(in_fasta, chrom_list=chrom_list, encode_spec=encode_spec)
    
    start_write = timeit.default_timer()
    with h5py.File(out_h5, "w") as h5_file:

        for chrom, onehot in onehot_dict.items():
            chrom_group = h5_file.require_group(chrom)
            chrom_group.create_dataset("onehot", data=onehot, compression="gzip")

        h5_file.attrs["encode_spec"] = [base.decode() for base in parse_encode_list(encode_spec)]

    print(f"Finished writing in {timeit.default_timer() - start_write} seconds!")
    print(f"One-Hot encoded genome written to {out_h5}")

