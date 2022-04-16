import timeit

import numpy as np
import pandas as pd

from load_data import load_vcf


def parse_encode_dict(encode_spec):
    if not encode_spec:
        encode_spec = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

    elif isinstance(encode_spec, (list, tuple, str)):
        encode_spec = {base:i for i, base in enumerate(encode_spec)}

    elif not isinstance(encode_spec, dict):
        raise TypeError("Please input as dict, list or string!")
    
    return encode_spec


def get_chrom_hap(seq_matrix, snp_df, chrom, allele_dict):
    start_hap = timeit.default_timer()
    
    pos_df = snp_df.replace({"ref": allele_dict, "alt": allele_dict})
    
    pos_df["p1_pos"] = np.where(pos_df["phase1"] == 1, pos_df["alt"], pos_df["ref"])
    pos_df["p2_pos"] = np.where(pos_df["phase2"] == 1, pos_df["alt"], pos_df["ref"])
    
    # get arrays with snp positions
    pos_array = pos_df["start"].to_numpy()
    p1_array = pos_df["p1_pos"].to_numpy()
    p2_array = pos_df["p2_pos"].to_numpy()
    
    # Set positions to 0
    seq_matrix[pos_array] = 0
    p2_matrix = seq_matrix.copy()

    seq_matrix[pos_array, p1_array] = 1
    p2_matrix[pos_array, p2_array] = 1
    
    print(f"Created {chrom} data in {timeit.default_timer() - start_hap:.2f} seconds!")
    return seq_matrix, p2_matrix


def get_encoded_haps(onehot_dict, in_vcf, sample, chrom_list=None, encode_spec=None):
    start_time = timeit.default_timer()

    encode_spec = parse_encode_dict(encode_spec) # Process encode spec as dict

    if not chrom_list:
        chrom_list = list(onehot_dict.keys())

    hap1_dict = {}
    hap2_dict = {}
    for chrom in chrom_list:
        if chrom in onehot_dict:
            snp_df = load_vcf(in_vcf, chrom=chrom, sample=sample)

            hap1_matrix, hap2_matrix = get_chrom_hap(onehot_dict[chrom], snp_df,
                                                     chrom=chrom, allele_dict=encode_spec)
            hap1_dict[chrom] = hap1_matrix
            hap2_dict[chrom] = hap2_matrix

        else:
            print(f"{chrom} not found in onehot data!")
    print(f"Processed haplotype data in {timeit.default_timer() - start_time:.2f} seconds!")
    
    return hap1_dict, hap2_dict

