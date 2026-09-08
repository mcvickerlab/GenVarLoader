"""Regression test for #336: SVAR2 read path must normalize contig naming.

A dataset's contigs come from its BED (``_write._prep_bed``: ``natsorted(bed["chrom"]
.unique())``), while a ``.svar2`` store carries whatever spelling its source VCF used.
The two are allowed to differ — a UCSC-named (``chr1``) BED over an Ensembl-named
(``1``) store is a supported arrangement, and ``gvl.write`` reconciles them.

Before the fix, only the *write* path normalized: ``Svar2Haps`` passed
``ds_contigs[ci]`` straight into the read-bound Rust kernels, which look contigs up
in the store's own keyspace. So such a dataset built successfully and then raised
``ValueError: contig chr1 not in store`` on the first read.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

import genvarloader as gvl

# Same 40 bp reference and variants as the other svar2 fixtures, but named "1"
# (Ensembl) rather than "chr1" (UCSC), so the store's keyspace differs from the BED's.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@pytest.fixture(scope="module")
def ensembl_svar2_store(tmp_path_factory) -> Path:
    """A ``.svar2`` store whose only contig is spelled ``1``."""
    from genoray import _core

    d = tmp_path_factory.mktemp("svar2_contig_naming")
    ref = d / "ref.fa"
    ref.write_text(f">1\n{_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / "store.svar2"
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["1"],
        str(out),
        ["S0", "S1"],
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


@pytest.fixture(scope="module")
def chr_named_dataset(ensembl_svar2_store: Path, tmp_path_factory) -> Path:
    """A dataset whose regions are ``chr1`` over the ``1``-named store."""
    from genoray import SparseVar2

    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 5], "chromEnd": [20, 15]}
    )
    out = tmp_path_factory.mktemp("svar2_contig_naming_ds") / "ds.gvl"
    # The write path already reconciles chr1 <-> 1; this must succeed.
    gvl.write(
        out, bed, variants=SparseVar2(ensembl_svar2_store), samples=None, overwrite=True
    )
    return out


def test_svar2_read_normalizes_contig_naming(chr_named_dataset: Path):
    """A chr-named dataset over an Ensembl-named .svar2 store must be readable."""
    ds = gvl.Dataset.open(chr_named_dataset).with_seqs("variants")

    # The premise: the dataset and its store really do disagree on spelling.
    assert ds._seqs.ds_contigs == ["chr1"]
    assert ds._seqs.store_contigs == ["1"]

    # Raised `ValueError: contig chr1 not in store` before the fix.
    assert ds[0] is not None


def test_svar2_unmapped_contig_names_both_spellings(chr_named_dataset: Path):
    """A dataset contig with no store equivalent raises, naming both spellings.

    Resolution is lazy: an unmappable contig is only an error when that contig is
    actually queried, so building the resolution table at open must not itself raise.
    """
    ds = gvl.Dataset.open(chr_named_dataset).with_seqs("variants")

    # replace() re-runs __post_init__, so the resolution table is rebuilt.
    haps = replace(ds._seqs, ds_contigs=["chrZ"])
    assert haps._ds_to_store == [None]  # built without raising

    with pytest.raises(ValueError, match=r"chrZ.*no equivalent"):
        haps._store_contig(0)
