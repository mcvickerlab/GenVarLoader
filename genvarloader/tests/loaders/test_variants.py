import re
from pathlib import Path

import hypothesis.extra.pandas as st_pd
import hypothesis.strategies as st
import numpy as np
from cyvcf2 import VCF
from hypothesis import given
from pytest_cases import fixture

import genvarloader
import genvarloader.loaders as gvl


def strategy_variants_queries(variants: gvl.Variants):
    longest_contig = int(250e6)
    contigs = [str(i) for i in range(1, 23)] + ["X", "Y"]
    contig = st_pd.column(name="contig", elements=st.sampled_from(contigs))
    start = st_pd.column(name="start", elements=st.integers(0, longest_contig + 1))
    sample = st_pd.column(
        name="sample", elements=st.sampled_from(list(variants.samples))
    )
    ploid_idx = st_pd.column(name="ploid_idx", elements=st.integers(0, 1))
    df = st_pd.data_frames(columns=[contig, start, sample, ploid_idx])
    return df.map(gvl.Queries)


def strategy_length():
    return st.integers(600, 1200).filter(lambda x: x % 2 == 0)


@fixture
def wdir():
    return Path(genvarloader.__file__).parent / "tests"


@fixture
def variants():
    return gvl.Variants.create(
        {
            "OCI-AML5": "/cellar/users/dlaub/repos/genome-loader/sbox/CDS-OJhAUD_cnn_filtered.zarr",
            "NCI-H660": "/cellar/users/dlaub/repos/genome-loader/sbox/CDS-oZPNvc_cnn_filtered.zarr",
        }
    )


@given(queries=strategy_variants_queries(variants()), length=strategy_length())
def test_variants(
    variants: gvl.Variants, queries: gvl.Queries, length: int, wdir: Path
):
    res = variants.sel(queries, length)
    vcf = VCF(str(wdir / "data" / "ccle_snp_wes.reduced.bcf"))
    if res["alleles"].size == 0:
        for query in queries.itertuples():
            region_string = f"{query.contig}:{query.start+1}-{query.start + length}"
            sample_idx = vcf.samples.index(query.sample)
            known_allele_count = 0
            for record in vcf(region_string):
                geno_string = record.gt_bases[sample_idx]
                geno = re.split(r"/|\|", geno_string)[query.ploid_idx].encode("ascii")
                if geno == b".":
                    continue
                known_allele_count += 1
            assert known_allele_count == 0
    else:
        gvl_alleles = np.split(res["alleles"], res["offsets"][1:])
        gvl_positions = np.split(res["positions"], res["offsets"][1:])

        for query, gvl_var, gvl_pos in zip(
            queries.itertuples(), gvl_alleles, gvl_positions
        ):
            sample_idx = vcf.samples.index(query.sample)
            region_string = f"{query.contig}:{query.start+1}-{query.start + length}"
            _vcf_var = []
            _vcf_pos = []
            for record in vcf(region_string):
                geno_string = record.gt_bases[sample_idx]
                geno = re.split(r"/|\|", geno_string)[query.ploid_idx].encode("ascii")
                if geno == b".":
                    continue
                pos = record.POS - 1  # we use 0-indexed positions
                _vcf_var.append(geno)
                _vcf_pos.append(pos)
            vcf_var = np.array(_vcf_var, dtype="|S1")
            vcf_pos = np.array(_vcf_pos)
            np.testing.assert_equal(gvl_var, vcf_var)
            np.testing.assert_equal(gvl_pos, vcf_pos)
