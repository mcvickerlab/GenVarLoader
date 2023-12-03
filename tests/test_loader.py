from pathlib import Path

import numpy as np
import polars as pl

import genvarloader as gvl
from genvarloader.loader import BatchDict


def varseq_fasta_pgen():
    fasta = gvl.Fasta(
        "seq",
        Path.cwd() / "data" / "fasta" / "Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
        pad="N",
    )
    pgen = gvl.Pgen(Path.cwd() / "data" / "pgen" / "sample.pgen")
    return gvl.FastaVariants("varseq", fasta, pgen, jitter_long=False)


def test_concat_batches():
    varseq = varseq_fasta_pgen()
    bed = pl.read_csv(
        Path.cwd() / "data" / "vcf" / "sample.bed",
        separator="\t",
        has_header=False,
        new_columns=["chrom", "chromStart", "chromEnd"],
        dtypes={"chrom": pl.Utf8},
    )

    gvloader = gvl.GVL(
        varseq,
        bed=bed,
        batch_dims=["sample", "ploid"],
        batch_size=4,
        fixed_length=4,
        max_memory_gb=4,
    )

    batch1: BatchDict = {
        "varseq": (["batch", "length"], np.array([["A", "T"], ["C", "G"]], dtype="S1"))
    }
    batch2: BatchDict = {
        "varseq": (["batch", "length"], np.array([["G", "C"], ["T", "A"]], dtype="S1"))
    }
    partial_batches = [batch1, batch2]
    concatenated_batches = {
        "varseq": (
            ["batch", "length"],
            np.array([["A", "T"], ["C", "G"], ["G", "C"], ["T", "A"]], dtype="S1"),
        )
    }
    out = gvloader.concat_batches(partial_batches)

    for name, (dim, arr) in concatenated_batches.items():
        assert out[name][0] == dim
        np.testing.assert_equal(out[name][1], arr)
