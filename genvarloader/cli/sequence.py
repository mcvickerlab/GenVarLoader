import warnings
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Optional

import typer

from genvarloader.cli import LoggingLevel
from genvarloader.types import ALPHABETS, SequenceEncoding

Alphabet = Enum("Alphabet", dict(zip(ALPHABETS.keys(), ALPHABETS.keys())))


def fasta_to_zarr_cli(
    fasta_path: Path,
    out_path: Path,
    alphabet: Alphabet,  # type: ignore | dynamic Enum
    encodings: str = typer.Argument(
        ...,
        help=dedent(
            f"""
            Comma separated list of encodings to write. Must be unique. 
            Available encodings: {[e.value for e in SequenceEncoding]}.
            """
        ).strip(),
    ),
    contigs: Optional[str] = typer.Argument(
        None, help="Comma separated list of contig names to write. Must be unique."
    ),
    ignore_case: bool = typer.Option(
        False,
        "--ignore-case",
        help="Ignore the case of FASTA sequences and treat everything as uppercase.",
    ),
    compression_level: int = typer.Option(5, min=0, max=9),
):
    from genvarloader.writers.sequence import fasta_to_zarr

    if not fasta_path.exists():
        raise ValueError("FASTA not found.")
    if out_path.exists():
        raise ValueError("Output file already exists.")

    _alphabet = ALPHABETS[alphabet.value]
    _encodings = {SequenceEncoding(e) for e in encodings.split(",")}
    if len(_encodings) != len(encodings.split(",")):
        raise ValueError("Got duplicate encodings.")
    if contigs is None:
        _contigs = contigs
    else:
        _contigs = {c for c in contigs.split(",")}
        if len(_contigs) != len(contigs.split(",")):
            warnings.warn("Duplicate contigs were ignored.")

    fasta_to_zarr(
        fasta_path,
        out_path,
        _alphabet,
        _encodings,
        _contigs,
        ignore_case,
        compression_level,
    )


if __name__ == "__main__":
    typer.run(fasta_to_zarr_cli)
