from pathlib import Path

import timeit

import h5py
import numpy as np
import pandas as pd

from pysam import VariantFile


def vcf_dataframe(in_vcf, chrom=None, sample=None):
    """Create dataframe with variant data

    :param in_vcf: _description_
    :type in_vcf: _type_
    :param chrom: _description_, defaults to None
    :type chrom: _type_, optional
    :param sample: _description_, defaults to None
    :type sample: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    if sample:
        with VariantFile(in_vcf, "r") as vcf:
            vcf.subset_samples([sample])
            vcf_data = vcf.fetch(chrom)
            snp_list = [(record.contig, record.start, record.stop, record.ref, record.alts[0],
                         record.samples[sample]['GT'][0], record.samples[sample]['GT'][1]) 
                        for record in vcf_data if (
                            (len(record.ref) == 1) and (len(record.alts) == 1) and (len(record.alts[0]) == 1) 
                            and (((record.samples[sample]['GT'][0] == 0) and (record.samples[sample]['GT'][1] == 1)) 
                                 or ((record.samples[sample]['GT'][0] == 1) and (record.samples[sample]['GT'][1] == 0))
                                )
                        )]
            
            snp_df = pd.DataFrame(snp_list, columns=["chrom", "start", "stop", "ref", "alt", "phase1", "phase2"], dtype=object)
            snp_df = snp_df.astype({"start": np.uint32, "stop": np.uint32, "phase1": np.uint8, "phase2": np.uint8})
        
    else:
        with VariantFile(in_vcf, "r", drop_samples=True) as vcf:
            vcf_data = vcf.fetch(chrom)
            snp_list = [(record.contig, record.start, record.stop, record.ref, record.alts[0])
                        for record in vcf_data if (
                            (len(record.ref) == 1) and (len(record.alts) == 1) and (len(record.alts[0]) == 1)
                        )]

            snp_df = pd.DataFrame(snp_list, columns=["chrom", "start", "stop", "ref", "alt"], dtype=object)
            snp_df = snp_df.astype({"start": np.uint32, "stop": np.uint32})

    return snp_df


def chrom_hap(seq_matrix, snp_list, chrom, out_h5):
    """Function called by write_hap_h5()

    :param seq_matrix: _description_
    :type seq_matrix: _type_
    :param snp_list: _description_
    :type snp_list: _type_
    :param chrom: _description_
    :type chrom: _type_
    :param out_h5: _description_
    :type out_h5: _type_
    """
    start_hap = timeit.default_timer()
    
    allele_dict = {"A": np.array([1, 0 , 0, 0], dtype=np.uint8),
                   "C": np.array([0, 1 , 0, 0], dtype=np.uint8),
                   "G": np.array([0, 0 , 1, 0], dtype=np.uint8),
                   "T": np.array([0, 0 , 0, 1], dtype=np.uint8)}
    
    p1_matrix = seq_matrix.copy()
    p2_matrix = seq_matrix.copy()
    
    for p1_a, p2_a, pos, *alleles in snp_list:
        p1_matrix[pos] = allele_dict[alleles[p1_a]]
        p2_matrix[pos] = allele_dict[alleles[p2_a]]
    
    
    with h5py.File(out_h5, "a") as file:
        chrom_group = file.require_group(chrom)
        chrom_group.create_dataset("haplotype1", data=p1_matrix, compression="gzip")
        chrom_group.create_dataset("haplotype2", data=p2_matrix, compression="gzip")
    
    print(f"Created {chrom} data in {timeit.default_timer() - start_hap} seconds!")


def write_hap_h5(in_genome, in_vcf, sample, out_dir, h5_name=None, chrom_list=None):
    """Retrieve onehot genome from h5 and write onehot encoded
    haplotypes for a given sample

    :param in_genome: _description_
    :type in_genome: _type_
    :param in_vcf: _description_
    :type in_vcf: _type_
    :param sample: _description_
    :type sample: _type_
    :param out_dir: _description_
    :type out_dir: _type_
    :param h5_name: _description_, defaults to None
    :type h5_name: _type_, optional
    :param chrom_list: _description_, defaults to None
    :type chrom_list: _type_, optional
    """
    
    if h5_name:
        out_h5 = str(Path(out_dir) / h5_name) # add check later to make sure not dir
    else:
        out_h5 = str(Path(out_dir) / f"{sample}_haplotypes.h5")

    start_time = timeit.default_timer()

    if not chrom_list:
        with VariantFile(in_vcf, "r") as vcf:
            chrom_list = list(vcf.header.contigs)
        # Add case to get all at once?
        # snp_df = vcf_dataframe(in_vcf, chrom=None, sample=sample)

    for chrom in chrom_list:
        snp_df = vcf_dataframe(in_vcf, chrom=chrom, sample=sample)
        variant_list = snp_df[["phase1", "phase2", "start", "ref", "alt"]
                             ].itertuples(index=False, name=None)

        with h5py.File(in_genome, "r") as in_h5:
            onehot_matrix = in_h5[chrom]["sequence"][:]
        
        chrom_hap(onehot_matrix, variant_list, chrom, out_h5)

    print(f"Finished in {timeit.default_timer() - start_time} seconds!")
    print(f"One-Hot encoded haplotypes written to {out_h5}")

