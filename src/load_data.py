from pathlib import Path

import numpy as np
import pandas as pd

from pysam import VariantFile


def load_vcf(in_vcf, chrom=None, sample=None):

    if sample:
        with VariantFile(in_vcf, "r") as vcf:
            vcf.subset_samples([sample])
            vcf_data = vcf.fetch(chrom)
            snp_list = [(record.contig, record.start, record.stop, record.ref, record.alts[0],
                         record.samples[sample]['GT'][0], record.samples[sample]['GT'][1])
                        for record in vcf_data if ((len(record.ref) == 1) and (len(record.alts) == 1)
                                                   and (len(record.alts[0]) == 1))]

            snp_df = pd.DataFrame(snp_list, columns=[
                                  "chrom", "start", "stop", "ref", "alt", "phase1", "phase2"], dtype=object)
            snp_df = snp_df.astype(
                {"start": np.uint32, "stop": np.uint32, "phase1": np.uint8, "phase2": np.uint8})

    else:
        with VariantFile(in_vcf, "r", drop_samples=True) as vcf:
            vcf_data = vcf.fetch(chrom)
            snp_list = [(record.contig, record.start, record.stop, record.ref, record.alts[0])
                        for record in vcf_data if (
                            (len(record.ref) == 1) and (len(record.alts)
                                                        == 1) and (len(record.alts[0]) == 1)
            )]

            snp_df = pd.DataFrame(
                snp_list, columns=["chrom", "start", "stop", "ref", "alt"], dtype=object)
            snp_df = snp_df.astype({"start": np.uint32, "stop": np.uint32})

    return snp_df
