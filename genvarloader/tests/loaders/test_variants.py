import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest_cases as pt
from cyvcf2 import VCF
from pytest_cases import fixture, parametrize, parametrize_with_cases

import genvarloader
import genvarloader.loaders as gvl


@pt.fixture
def data_dir():
    return Path(genvarloader.__file__).parent / "tests" / "data"


@pt.fixture
def variants(data_dir: Path):
    zarrs = [
        data_dir / "CDS-RL1iWJ.zarr",
        data_dir / "CDS-69IkMA.zarr",
    ]
    return gvl.Variants.create(zarrs=zarrs)


def queries_hom_alt():
    # 20:276086 = T->A
    # 21:10592359 = A->G
    queries = gvl.Queries(
        {
            "contig": ["20", "20", "20", "20", "21", "21", "21", "21"],
            "start": [
                276085,
                276085,
                276085,
                276085,
                10592358,
                10592358,
                10592358,
                10592358,
            ],  # note this is 0-indexed, VCF is 1-indexed
            "sample": [
                "NCI-H660",
                "NCI-H660",
                "OCI-AML5",
                "OCI-AML5",
                "NCI-H660",
                "NCI-H660",
                "OCI-AML5",
                "OCI-AML5",
            ],
            "ploid_idx": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    return queries


def queries_hom_ref():
    hom_alt = queries_hom_alt()
    hom_ref = hom_alt.assign(
        start=hom_alt["start"] - 1
    )  # confirmed that there are no SNPs here
    return hom_ref


def queries_hom_alt_hom_ref():
    both = pd.concat([queries_hom_alt(), queries_hom_ref()], ignore_index=True)
    return both


@pt.case(tags="xfail")
def queries_nonexistent_sample():
    nonexistent_sample = queries_hom_alt().assign(sample="definitely_not_a_sample")
    return nonexistent_sample


def queries_negative_start():
    negative_start = queries_hom_alt().assign(start=-1)
    return negative_start


def queries_out_of_bounds_end():
    out_of_bounds_end = queries_hom_alt().assign(start=int(250e6))
    return out_of_bounds_end


@pt.parametrize_with_cases("queries", prefix="queries_")
@pt.parametrize_with_cases("length", [1])
def test_variants(
    variants: gvl.Variants,
    queries: pd.DataFrame,
    length: int,
    data_dir: Path,
    current_cases,
):
    queries_id, queries_fn, queries_params = current_cases["queries"]
    xfail = pt.matches_tag_query(queries_fn, has_tag="xfail")
    res = variants.sel(queries, length)
    vcf = VCF(str(data_dir / "ccle_snp_wes.reduced.bcf"))
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
