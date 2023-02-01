from pathlib import Path
from typing import Dict

from pysam import FastaFile
from pytest_cases import fixture, parametrize_with_cases

import genvarloader
from genvarloader.loaders.sequence import ZarrSequence
from genvarloader.loaders.types import Queries


def sel_args_1_region_1_samp():
    queries = Queries({"contig": ["20"], "start": [96319], "sample": ["OCI-AML5"]})
    length = 5
    sorted = False
    encoding = "bytes"
    return dict(queries=queries, length=length, sorted=sorted, encoding=encoding)


def sel_args_1_chrom_1_samp():
    queries = Queries(
        {
            "contig": ["20", "20"],
            "start": [96319, 279175],
            "sample": ["OCI-AML5", "OCI-AML5"],
        }
    )
    length = 5
    sorted = False
    encoding = "bytes"
    return dict(queries=queries, length=length, sorted=sorted, encoding=encoding)


def sel_args_1_chrom_2_samp():
    queries = Queries(
        {
            "contig": ["20", "20"],
            "start": [96319, 279175],
            "sample": ["OCI-AML5", "NCI-H660"],
        }
    )
    length = 5
    sorted = False
    encoding = "bytes"
    return dict(queries=queries, length=length, sorted=sorted, encoding=encoding)


def sel_args_2_chrom_2_samp():
    queries = Queries(
        {
            "contig": ["21", "20", "20"],
            "start": [10414881, 96319, 279175],
            "sample": ["OCI-AML5", "NCI-H660", "NCI-H660"],
        }
    )
    length = 5
    sorted = False
    encoding = "bytes"
    return dict(queries=queries, length=length, sorted=sorted, encoding=encoding)


@fixture
def wdir():
    return Path(genvarloader.__file__).parent / "tests"


@fixture
def zarr_sequence():
    return ZarrSequence(
        "/cellar/users/dlaub/repos/genome-loader/genvarloader/tests/data/grch38.20.21.zarr"
    )


@parametrize_with_cases("sel_args", cases=".", prefix="sel_args_")
def test_zarrsequence(zarr_sequence: ZarrSequence, sel_args: Dict, wdir: Path):
    seqs = zarr_sequence.sel(**sel_args).astype("U")
    ref_fasta = wdir / "data" / "fasta" / "grch38.20.21.fa.gz"

    for seq, query in zip(seqs, sel_args["queries"].itertuples()):
        contig = query.contig
        start = query.start
        length = sel_args["length"]
        with FastaFile(str(ref_fasta)) as f:
            ref_seq = f.fetch(contig, start, start + length)
        assert ref_seq == "".join(seq)
