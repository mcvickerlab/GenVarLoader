import timeit

import numpy as np
import pandas as pd

from .encode_data import parse_encode_spec
from .load_data import load_vcf


def validate_snp_chroms(snp_df, bam_chroms):
    
    vcf_chroms = snp_df["chrom"].unique()
    
    # Create set from input data
    bam_set = set(bam_chroms)
    vcf_set = set(vcf_chroms)
    
    missing_chroms = vcf_set - bam_set
    
    # Check if all vcf chroms match
    if not missing_chroms:
        return snp_df
    else:
        matched_chroms = vcf_set - missing_chroms
        matched_dict = {chrom: chrom if chrom in matched_chroms else None
                        for chrom in vcf_chroms}
    
    print(f"Failed to match {len(missing_chroms)} chroms in VCF")

    # Check if chrom prefix formats match,
    if any([chrom.startswith("chr") for chrom in missing_chroms]):
        print("Trying again w/o 'chr' prefix")
        prefix_dict = {chrom: chrom.replace("chr", "") for chrom in missing_chroms}
    else:
        print("Trying again with 'chr' prefix")
        prefix_dict = {chrom: f"chr{chrom}" for chrom in missing_chroms}

    # Try again with prefix matches
    missing_chroms = set(list(prefix_dict.values())) - bam_set
    
    # Update dict of matched values
    matched_dict.update({key: None if val in missing_chroms else val
                         for key, val in prefix_dict.items()})
    
    # Process renamed and skipped chroms
    rename_dict = {}
    skip_list = []
    for chrom, match in matched_dict.items():
        if match is None:
            skip_list.append(chrom)
        else:
            rename_dict[chrom] = match

    # Update snp_df chroms to match
    snp_df["chrom"] = snp_df["chrom"].map(rename_dict)
    
    if skip_list:
        print("Skipping unmatched chroms...")
        print(*skip_list, sep=", ")
        snp_df = snp_df.dropna(subset="chrom").reset_index(drop=True)

    return snp_df


def get_chrom_hap(seq_matrix, snp_df, chrom, allele_dict):
    """HELPER CALLED BY get_encoded_haps()
    parses haplotypes per chromosome
    """
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


def get_encoded_haps(onehot_dict, in_vcf, sample, chrom_list=None, encode_spec=None, remove_ambiguity=True):
    start_time = timeit.default_timer()
    
    encode_spec = parse_encode_spec(encode_spec)  # Process encode spec as dict
    
    # Validate encode_spec against OHE data
    col_shapes = {value.shape[1] for key, value in onehot_dict.items()}

    if len(col_shapes) != 1:
        raise ValueError("Num cols in OHE Chroms don't match!")
    
    num_cols = col_shapes.pop()
    
    if num_cols != len(encode_spec):
        raise IndexError(
            f"Num cols in OHE({num_cols}) and encode_spec({len(encode_spec)}) don't match!")
    
    # Create DF with SNP's
    load_vcf_out = load_vcf(in_vcf, chrom_list=chrom_list, sample=sample)
    snp_df = validate_snp_chroms(load_vcf_out.copy(), onehot_dict.keys())
    
    if snp_df.empty:
        raise KeyError("Could not find matching chroms. "
            f"VCF Chroms: {list(load_vcf_out.keys())}. "
            f"OHE Chroms: {list(onehot_dict.keys())}")

    # Remove genotypes with ambiguous alleles
    if remove_ambiguity or ("N" not in encode_spec):
        non_ambig = ["A", "C", "G", "T"]
        snp_df = snp_df.loc[(snp_df["ref"].isin(non_ambig))
                            & (snp_df["alt"].isin(non_ambig)), :]
    
    chrom_list = list(snp_df["chrom"].unique())
    
    hap1_dict = {}
    hap2_dict = {}
    for chrom, chrom_df in snp_df.groupby(by=["chrom"], sort=False):

        hap1_matrix, hap2_matrix = get_chrom_hap(onehot_dict[chrom].copy(),
                                                 chrom_df, chrom=chrom,
                                                 allele_dict=encode_spec)
        
        hap1_dict[chrom] = hap1_matrix
        hap2_dict[chrom] = hap2_matrix

    print(f"Processed haplotype data in {timeit.default_timer() - start_time:.2f} seconds!")
    return hap1_dict, hap2_dict
