import numpy as np
import pandas as pd
import polars as pl
from pysam import VariantFile


def vcf_snp_gen(vcf, chrom_list=None, sample=None):
    
    for chrom in chrom_list:
        for record in vcf.fetch(chrom):
            
            if ((len(record.ref) == 1) and (len(record.alts) == 1) and (len(record.alts[0]) == 1)):
                if sample:
                    yield (record.contig, record.start, record.stop,
                           record.ref, record.alts[0], record.samples[sample]["GT"][0],
                           record.samples[sample]["GT"][1])
                else:
                    yield (record.contig, record.start, record.stop, record.ref, record.alts[0])


def load_vcf(in_vcf, chrom_list=None, sample=None):
    """Retrieve SNP's from VCF as dataframe with columns: 'chrom', 'start', 'stop', 'ref', 'alt'
    (If sample is given with phased VCF also includes columns 'phase1' and 'phase2')

    :param in_vcf: VCF file to retrieve SNP's
    :type in_vcf: str
    :param chrom_list: Chromosomes to parse, defaults to ALL chroms
    :type chrom: list[str] or str, optional
    :param sample: name in VCF to retrieve genotypes, defaults to None
    :type sample: str, optional
    :return: Dataframe containing SNP or SNP+GT data
    :rtype: pd.DataFrame
    """
    
    if chrom_list is None or isinstance(chrom_list, str):
        chrom_list = [chrom_list]

    if sample:
        with VariantFile(in_vcf, "r") as vcf:
            vcf.subset_samples([sample])
            
            snp_df = pd.DataFrame([record for record in
                                   vcf_snp_gen(vcf, chrom_list=chrom_list, sample=sample)],
                                  columns=["chrom", "start", "stop",
                                           "ref", "alt", "phase1", "phase2"],
                                  dtype=object)
            
            snp_df = snp_df.astype(
                {
                    "start": np.uint32,
                    "stop": np.uint32,
                    "phase1": np.uint8,
                    "phase2": np.uint8,
                }
            )

    else:
        with VariantFile(in_vcf, "r", drop_samples=True) as vcf:
            
            snp_df = pd.DataFrame([record for record in vcf_snp_gen(vcf, chrom_list=chrom_list)],
                                  columns=["chrom", "start", "stop", "ref", "alt"],
                                  dtype=object)
            
            snp_df = snp_df.astype({"start": np.uint32, "stop": np.uint32})

    return snp_df


def load_vcf_polars(in_vcf, chrom_list=None, sample=None):
    """Retrieve SNP's from VCF as dataframe with columns: 'chrom', 'start', 'stop', 'ref', 'alt'
    (If sample is given with phased VCF also includes columns 'phase1' and 'phase2')

    :param in_vcf: VCF file to retrieve SNP's
    :type in_vcf: str
    :param chrom_list: Chromosomes to parse, defaults to ALL chroms
    :type chrom: list[str] or str, optional
    :param sample: name in VCF to retrieve genotypes, defaults to None
    :type sample: str, optional
    :return: Dataframe containing SNP or SNP+GT data
    :rtype: pd.DataFrame
    """
    if chrom_list is None or isinstance(chrom_list, str):
        chrom_list = [chrom_list]

    if sample:
        with VariantFile(in_vcf, "r") as vcf:
            vcf.subset_samples([sample])


            snp_df = pl.DataFrame([record for record in vcf_snp_gen(vcf, chrom_list=chrom_list, sample=sample)],
                                  columns=["chrom", "start", "stop",
                                           "ref", "alt", "phase1", "phase2"]
                                  ).lazy().with_columns([pl.col("start").cast(pl.UInt32),
                                                         pl.col("stop").cast(
                                                             pl.UInt32),
                                                         pl.col("phase1").cast(
                                                             pl.UInt8),
                                                         pl.col("phase2").cast(pl.UInt8)]
                                                        ).collect()

    else:
        with VariantFile(in_vcf, "r", drop_samples=True) as vcf:

            snp_df = pl.DataFrame([record for record in vcf_snp_gen(vcf, chrom_list=chrom_list)],
             columns=["chrom", "start", "stop", "ref", "alt"]
                                  ).lazy().with_columns([pl.col("start").cast(pl.UInt32),
                                                         pl.col("stop").cast(pl.UInt32)]
                                                        ).collect()

    return snp_df
