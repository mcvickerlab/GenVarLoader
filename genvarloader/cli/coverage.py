from pathlib import Path
from typing import Optional

import typer

from genvarloader.types import Tn5CountMethod
from genvarloader.writers.coverage import coverage, tn5_coverage

app = typer.Typer()

coverage = app.command("depth-only")(coverage)


@app.command()
def tn5(
    in_bam: Path,
    out_zarr: Path,
    contigs: Optional[str] = typer.Argument(
        None, help="Comma separated list of contigs to write, defaults to all contigs."
    ),
    offset_tn5: bool = typer.Option(
        True, help="Whether to offset read lengths for Tn5."
    ),
    count_method: Tn5CountMethod = typer.Option(
        "cutsite", help="What to count for coverage."
    ),
):
    """Write Tn5 coverage from BAM to Zarr"""
    if contigs is not None:
        _contigs = contigs.split(",")
    else:
        _contigs = contigs
    tn5_coverage(in_bam, out_zarr, _contigs, None, offset_tn5, count_method)
