from pathlib import Path
from typing import Dict, List, Optional

import typer

from genvarloader.types import Tn5CountMethod

app = typer.Typer()


def read_in_bams(in_bams: Path):
    import pandas as pd
    import pandera as pa
    import pandera.typing as pat

    class InBams(pa.SchemaModel):
        name: pat.Series[str]
        path: pat.Series[str]

    if in_bams.suffix == ".csv":
        bams_df = pd.read_csv(in_bams)
    elif in_bams.suffix in {".tsv", ".txt"}:
        bams_df = pd.read_csv(in_bams, sep="\t")
    else:
        raise ValueError("Need a CSV or TSV for `in_bams`.")

    bams_df = InBams.to_schema()(bams_df)

    _in_bams: Dict[str, Path] = dict(
        zip(bams_df.iloc[:, 0], map(Path, bams_df.iloc[:, 1]))
    )

    return _in_bams


@app.command()
def depth_only(
    in_bams: Path = typer.Argument(
        ...,
        help="A CSV/TSV where the first column is the sample name and the second column is the path to that sample's BAM.",
    ),
    out_zarr: Path = typer.Argument(...),
    contigs: Optional[List[str]] = typer.Argument(
        None, help="If None, write all contigs."
    ),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing samples in the output Zarr."
    ),
    n_jobs: Optional[int] = typer.Option(None, min=1),
):
    from genvarloader.writers.coverage import write_coverages

    if n_jobs is None:
        n_jobs = -2

    _in_bams = read_in_bams(in_bams)

    write_coverages(_in_bams, out_zarr, contigs, n_jobs, overwrite)


@app.command()
def tn5(
    in_bams: Path = typer.Argument(
        ...,
        help="A CSV/TSV where the first column is the sample name and the second column is the path to that sample's BAM.",
    ),
    out_zarr: Path = typer.Argument(...),
    contigs: Optional[List[str]] = typer.Argument(
        None, help="Contigs to write, defaults to all contigs."
    ),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing samples in the output Zarr."
    ),
    n_jobs: Optional[int] = typer.Option(None, min=1),
    offset_tn5: bool = typer.Option(
        True, help="Whether to offset read lengths for Tn5."
    ),
    count_method: Tn5CountMethod = typer.Option(
        "cutsite", help="What to count for coverage."
    ),
):
    """Write Tn5 coverage from BAM to Zarr"""
    from genvarloader.writers.coverage import write_tn5_coverages

    if n_jobs is None:
        n_jobs = -2

    _in_bams = read_in_bams(in_bams)

    write_tn5_coverages(
        _in_bams, out_zarr, contigs, n_jobs, overwrite, offset_tn5, count_method
    )
