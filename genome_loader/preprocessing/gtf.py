from typing import Optional

import polars as pl

from genome_loader.utils import PathType

# TODO: use gffutils to turn a GFFv3 (and/or GTF??) into a nice regions file (bed)
# and expose a few simple/common filters that people might use to select features of
# interest. Examples:
# TSS's of genes in a list
# span of genes in a list
# exonic regions, intronic regions


def gff_to_query_regions(
    gtf: PathType,
    out_path: PathType,
    genes: Optional[list[str]],
):
    # GFF v3 file specification:
    # https://uswest.ensembl.org/info/website/upload/gff.html
    # Note that starts and ends are 1-indexed and end-inclusive
    gtf_col_dtype = {
        "seqid": pl.Utf8,
        "type": pl.Utf8,
        "start": pl.Int32,
        "end": pl.Int32,
        "score": pl.Float32,
        "strand": pl.Utf8,
        "frame": pl.Int32,
        "attribute": pl.Utf8,
    }
    queries = pl.read_csv(
        gtf,
        columns=[0, 2, 3, 4],
        new_columns=list(gtf_col_dtype.keys()),
        dtypes=gtf_col_dtype,
    )

    queries.write_csv(out_path)
