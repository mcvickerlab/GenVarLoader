import subprocess
from pathlib import Path

import genvarloader as gvl
import pytest

data_dir = Path(__file__).resolve().parents[1] / "data"
ref = data_dir / "fasta" / "hg38.fa.bgz"
issue_vcf_raw = data_dir / "issue_153.vcf"
issue_bed = data_dir / "issue_153.bed"


@pytest.fixture(scope="module")
def issue_vcf(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("issue_153_vcf")
    gz = tmp / "issue_153.vcf.gz"
    subprocess.run(["bgzip", "-c", str(issue_vcf_raw)], stdout=gz.open("wb"), check=True)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


@pytest.fixture(scope="module")
def issue_ds(issue_vcf, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("issue_153_ds")
    ds_path = tmp / "issue_153.gvl"
    gvl.write(path=ds_path, bed=issue_bed, variants=issue_vcf)
    return gvl.Dataset.open(ds_path, ref).with_len("ragged").with_seqs("haplotypes")


def test_issue_153_hap_lengths(issue_ds):
    """Ragged haplotype lengths must match expected indel net diff.

    Regression for GH #153: * (spanning deletion) alleles were incorrectly
    counted as negative ilen in get_diffs_sparse, undersizing the output buffer.

    Expected:
      hap1: 42645 + 4(G->GAGGA) + 1(G->GT) - 9(GGCAGCGCCA->G) = 42641
      hap2: 42645 - 4(GAGGA->G) + 5(C->CCATCT) + 1(G->GT)     = 42647
    """
    haps = issue_ds[0, "SAMPLE1"]
    assert len(haps[0]) == 42641, f"hap1 len={len(haps[0])}, expected 42641"
    assert len(haps[1]) == 42647, f"hap2 len={len(haps[1])}, expected 42647"
