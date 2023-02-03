from pathlib import Path
from typing import List, Optional

import typer

from genvarloader.types import Tn5CountMethod
from genvarloader.writers import coverage

app = typer.Typer()


@app.command()
def depth_only(
    in_bam: Path,
    out_zarr: Path,
    contigs: Optional[List[str]] = typer.Argument(
        None, help="If None, write all contigs."
    ),
):
    coverage.coverage(in_bam, out_zarr, contigs)


@app.command()
def tn5(
    in_bam: Path,
    out_zarr: Path,
    contigs: Optional[List[str]] = typer.Argument(
        None, help="Contigs to write, defaults to all contigs."
    ),
    offset_tn5: bool = typer.Option(
        True, help="Whether to offset read lengths for Tn5."
    ),
    count_method: Tn5CountMethod = typer.Option(
        "cutsite", help="What to count for coverage."
    ),
):
    """Write Tn5 coverage from BAM to Zarr"""
    coverage.tn5_coverage(in_bam, out_zarr, contigs, None, offset_tn5, count_method)
