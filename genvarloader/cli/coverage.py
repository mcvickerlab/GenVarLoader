import logging
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import typer

from genvarloader.types import Tn5CountMethod

app = typer.Typer(help="Write BAM files to Zarr using various read counting methods.")


def read_in_bams(in_bams: Path):
    import pandas as pd

    if in_bams.suffix == ".csv":
        bams_df = pd.read_csv(in_bams, header=None)
    elif in_bams.suffix in {".tsv", ".txt"}:
        bams_df = pd.read_csv(in_bams, header=None, sep="\t")
    else:
        raise ValueError("Need a CSV or TSV for `in_bams`.")

    _in_bams: Dict[str, Path] = dict(
        zip(bams_df.iloc[:, 0].astype(str), map(Path, bams_df.iloc[:, 1].astype(str)))
    )

    return _in_bams


@app.command()
def depth_only(
    in_bams: Path = typer.Argument(
        ...,
        resolve_path=True,
        help="A CSV/TSV where the first column is the sample name and the second column is the path to that sample's BAM. Should not have a header.",
    ),
    out_zarr: Path = typer.Argument(..., resolve_path=True),
    contigs: Optional[str] = typer.Option(
        None, help="Comma separated list of contigs to write, defaults to all contigs."
    ),
    contig_file: Optional[Path] = typer.Option(
        None,
        help="File with a comma separated list of contigs to write. Supersedes --contigs.",
    ),
    overwrite_samples: bool = typer.Option(
        False, help="Whether to overwrite existing samples in the output Zarr."
    ),
    n_jobs: Optional[int] = typer.Option(None, min=1),
):
    """Write plain depth from BAM to Zarr. If the Zarr already exists, this will add samples, possibly overwriting.

    NOTE: this is relatively memory intensive and requires at least 80x bytes of memory than the longest contig processed PER JOB.
    For example, for the human genome the longest contig is chromosome 1 at ~250 mb -> each job needs upwards of 10 GB of RAM."""

    # This is memory intensive because of how pysam.AlignmentFile.count_coverage is implemented.

    from genvarloader.writers.coverage import write_coverages

    if n_jobs is None:
        n_jobs = -2

    _in_bams = read_in_bams(in_bams)

    if contig_file is not None:
        with open(contig_file) as f:
            _contigs = f.read().strip().split(",")
    elif contigs is not None:
        _contigs = contigs.split(",")
    else:
        _contigs = contigs

    t1 = perf_counter()

    write_coverages(_in_bams, out_zarr, _contigs, overwrite_samples, n_jobs)

    logging.info(f"Wrote coverages in {timedelta(seconds=perf_counter() - t1)}")


@app.command()
def tn5(
    in_bams: Path = typer.Argument(
        ...,
        resolve_path=True,
        help="A CSV/TSV where the first column is the sample name and the second column is the path to that sample's BAM.",
    ),
    out_zarr: Path = typer.Argument(..., resolve_path=True),
    contigs: Optional[str] = typer.Option(
        None, help="Comma separated list of contigs to write, defaults to all contigs."
    ),
    contig_file: Optional[Path] = typer.Option(
        None,
        help="File with a comma separated list of contigs to write. Supersedes --contigs.",
    ),
    overwrite_samples: bool = typer.Option(
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
    """Write Tn5 depth from BAM to Zarr. If the Zarr already exists, this will add samples, possibly overwriting."""
    from genvarloader.writers.coverage import write_tn5_coverages

    _in_bams = read_in_bams(in_bams)

    if contig_file is not None:
        with open(contig_file) as f:
            _contigs = f.read().strip().split(",")
    elif contigs is not None:
        _contigs = contigs.split(",")
    else:
        _contigs = contigs

    t1 = perf_counter()

    write_tn5_coverages(
        _in_bams,
        out_zarr,
        _contigs,
        n_jobs,
        overwrite_samples,
        offset_tn5,
        count_method,
    )

    logging.info(f"Wrote coverages in {timedelta(seconds=perf_counter() - t1)}")
