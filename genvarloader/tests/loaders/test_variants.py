import re
from pathlib import Path
from typing import Dict

import numpy as np
from cyvcf2 import VCF
from pytest_cases import fixture, parametrize_with_cases

import genvarloader
from genvarloader.loaders.types import Queries
from genvarloader.loaders.variants import Variants


def sel_args_1_region_1_samp():
    queries = Queries(
        {"contig": ["20"], "start": [96319], "sample": ["OCI-AML5"], "ploid_idx": [1]}
    )
    length = 5
    sorted = False
    missing_value = "reference"
    return dict(
        queries=queries, length=length, sorted=sorted, missing_value=missing_value
    )


def sel_args_1_chrom_1_samp():
    queries = Queries(
        {
            "contig": ["20", "20"],
            "start": [96319, 279175],
            "sample": ["OCI-AML5", "OCI-AML5"],
            "ploid_idx": [1, 0],
        }
    )
    length = 5
    sorted = False
    missing_value = "reference"
    return dict(
        queries=queries, length=length, sorted=sorted, missing_value=missing_value
    )


def sel_args_1_chrom_2_samp():
    queries = Queries(
        {
            "contig": ["20", "20"],
            "start": [96319, 279175],
            "sample": ["OCI-AML5", "NCI-H660"],
            "ploid_idx": [1, 0],
        }
    )
    length = 5
    sorted = False
    missing_value = "reference"
    return dict(
        queries=queries, length=length, sorted=sorted, missing_value=missing_value
    )


def sel_args_2_chrom_2_samp():
    queries = Queries(
        {
            "contig": ["21", "20", "20"],
            "start": [10414881, 96319, 279175],
            "sample": ["OCI-AML5", "NCI-H660", "NCI-H660"],
            "ploid_idx": [1, 0, 1],
        }
    )
    length = 5
    sorted = False
    missing_value = "reference"
    return dict(
        queries=queries, length=length, sorted=sorted, missing_value=missing_value
    )


@fixture
def wdir():
    return Path(genvarloader.__file__).parent / "tests"


@fixture
def var_loader():
    return Variants(
        {
            "OCI-AML5": "/cellar/users/dlaub/repos/genome-loader/sbox/CDS-OJhAUD_cnn_filtered.zarr",
            "NCI-H660": "/cellar/users/dlaub/repos/genome-loader/sbox/CDS-oZPNvc_cnn_filtered.zarr",
        }
    )


@parametrize_with_cases("sel_args", cases=".", prefix="sel_args_")
def test_ss_var_loader(var_loader: Variants, sel_args: Dict, wdir: Path):
    res = var_loader.sel(**sel_args)
    vcf = VCF(str(wdir / "data" / "ccle_snp_wes.reduced.bcf"))
    if res is None:
        for query in sel_args["queries"].itertuples():
            region_string = (
                f"{query.contig}:{query.start+1}-{query.start + sel_args['length']}"
            )
            sample_idx = vcf.samples.index(query.sample)
            known_allele_count = 0
            for variant in vcf(region_string):
                geno_string = variant.gt_bases[sample_idx]
                geno = re.split(r"/|\|", geno_string)[query.ploid_idx].encode("ascii")
                if geno == b".":
                    continue
                known_allele_count += 1
            assert known_allele_count == 0
    else:
        variants, positions, offsets = res
        gvl_vars = np.split(variants, offsets[1:])
        gvl_poss = np.split(positions, offsets[1:])

        for query, gvl_var, gvl_pos in zip(
            sel_args["queries"].itertuples(), gvl_vars, gvl_poss
        ):
            sample_idx = vcf.samples.index(query.sample)
            region_string = (
                f"{query.contig}:{query.start+1}-{query.start + sel_args['length']}"
            )
            _vcf_var = []
            _vcf_pos = []
            for variant in vcf(region_string):
                geno_string = variant.gt_bases[sample_idx]
                geno = re.split(r"/|\|", geno_string)[query.ploid_idx].encode("ascii")
                if geno == b".":
                    continue
                pos = variant.POS - 1  # we use 0-indexed positions
                _vcf_var.append(geno)
                _vcf_pos.append(pos)
            vcf_var = np.array(_vcf_var, dtype="|S1")
            vcf_pos = np.array(_vcf_pos)
            np.testing.assert_equal(gvl_var, vcf_var)
            np.testing.assert_equal(gvl_pos, vcf_pos)
