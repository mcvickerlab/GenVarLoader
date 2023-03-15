#! /usr/bin/env python

import logging
import sys

import typer

from genvarloader.cli import LoggingLevel, coverage
from genvarloader.cli.sequence import fasta_to_zarr_cli
from genvarloader.cli.variants import write_zarrs

app = typer.Typer(
    name="GenVarLoader", help="""Write files to "GenVarLoader-ready" data structures."""
)


@app.callback()
def main(logging_level: LoggingLevel = typer.Option("INFO")):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging_level.value,
        format="%(levelname)s:%(name)s:%(asctime)s:%(message)s",
    )
    logging.captureWarnings(True)


fasta_to_zarr_cli = app.command("fasta")(fasta_to_zarr_cli)
write_zarr = app.command("vcf")(write_zarrs)
app.add_typer(coverage.app, name="coverage")

if __name__ == "__main__":
    app()
