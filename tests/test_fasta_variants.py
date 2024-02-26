from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
import pysam
from pytest_cases import parametrize_with_cases

import genvarloader as gvl


def varseq_fasta_pgen():
    fasta = gvl.Fasta(
        "seq",
        Path(__file__).parent
        / "data"
        / "fasta"
        / "Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
        pad="N",
    )
    pgen = gvl.Pgen(Path(__file__).parent / "data" / "pgen" / "sample.pgen")
    return gvl.FastaVariants("varseq", fasta, pgen, jitter_long=False)


def varseq_fasta_zarr():
    fasta = gvl.Fasta(
        "seq",
        Path(__file__).parent
        / "data"
        / "fasta"
        / "Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
        pad="N",
    )
    var = gvl.variants.Variants.from_vcf(
        Path(__file__).parent / "data" / "vcf" / "filtered_sample.vcf.gz"
    )
    return gvl.FastaVariants("varseq", fasta, var, jitter_long=False)


@parametrize_with_cases("varseq", cases=".", prefix="varseq_")
def test_fasta_variants(varseq: gvl.FastaVariants):
    regions = (
        pl.read_csv(
            Path(__file__).parent / "data" / "vcf" / "sample.bed",
            separator="\t",
            has_header=False,
            new_columns=["contig", "start", "end"],
            dtypes={"contig": pl.Utf8},
        )
        .with_row_count()
        .select("contig", "start", "end", "row_nr")
    )
    samples = ["NA00001", "NA00002", "NA00003"]
    for sample, hap, region in product(samples, range(1, 3), regions.iter_rows()):
        contig: str
        start: int
        end: int
        contig, start, end, row_nr = region
        seq_path = (
            Path(__file__).parent
            / "data"
            / "vcf"
            / "consensus"
            / f"sample_{sample}_nr{row_nr}_h{hap}.fa"
        )

        try:
            gvl_seq = varseq.read(
                contig,
                np.array([start], dtype=np.int64),
                np.array([end], dtype=np.int64),
                sample=[sample],
                ploid=[hap - 1],
            ).squeeze()
        except SystemError as e:
            print(f"Failed {sample} hap{hap-1} {contig}:{start}-{end} row {row_nr}")
            raise e

        with pysam.FastaFile(str(seq_path)) as f:
            bcftools_seq = f.fetch(f.references[0])
        bcftools_seq = np.frombuffer(bcftools_seq.encode(), "S1")[: end - start]
        length = min(len(bcftools_seq), len(gvl_seq))

        try:
            assert len(gvl_seq) >= len(bcftools_seq)
            np.testing.assert_equal(gvl_seq[:length], bcftools_seq[:length])
        except AssertionError as e:
            print(f"Failed {sample} hap{hap-1} {contig}:{start}-{end} row {row_nr}")
            raise e


if __name__ == "__main__":
    test_fasta_variants(varseq_fasta_pgen())
