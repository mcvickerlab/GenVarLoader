#!/usr/bin/env python

from pathlib import Path
from typing import List, Optional
import typer
import sys

app = typer.Typer()


@app.command()
def writefasta(
    output: Path = typer.Option(..., "-o", "--output", help="Output h5 file and path"),
    input: Path = typer.Argument(..., help="Fasta file to write to H5"),
    spec: Optional[str] = typer.Option(
        ...,
        "-s",
        "--spec",
        "--order",
        help="Ordered string of non-repeating chars. Denotes encoded bases and order (ie: ACGT, Default: ACGTN)",
    ),
    chroms: Optional[List[str]] = typer.Option(
        None,
        "-c",
        "--chroms",
        "--contigs",
        help="Chromosomes to write. Writes all by default",
    ),
    encode: bool = typer.Option(
        False, "--encode/ ", "-e/ ", "--onehot/ ", help="Write onehot-encoded genome"
    ),
):
    if output.suffix not in {".h5", ".hdf5", ".hdf", ".he5"}:
        output.suffix = ".h5"
    directory = output.parent
    name = output.name

    if spec:
        if not encode:
            raise ValueError(f"Encoding specification given without --encode flag!")

        if not spec.isalpha():
            raise ValueError(f"Spec: '{spec}' contains non-characters!")

        if len(set(spec)) != len(spec):
            raise ValueError(f"Spec: '{spec}' can't contain duplicate characters!")

    from genome_loader.write_h5 import write_genome_seq, write_encoded_genome

    if encode:
        write_encoded_genome(
            str(input),
            str(directory),
            h5_name=name,
            chrom_list=chroms,
            encode_spec=spec,
        )
    else:
        write_genome_seq(str(input), str(directory), h5_name=name, chrom_list=chroms)


@app.command()
def writedepth(
    output: Path = typer.Option(..., "-o", "--output", help="Output h5 file and path"),
    input: Path = typer.Argument(..., help="BAM file to write to H5"),
    chroms: Optional[List[str]] = typer.Option(
        None,
        "-c",
        "--chroms",
        "--contigs",
        help="Chromosomes to write. Writes all by default",
    ),
    lens: Optional[List[int]] = typer.Option(
        None,
        "-l",
        "--lens",
        "--lengths",
        "--chromlens",
        help="Lengths of provided chroms (Auto retrieved if not provided)",
    ),
):
    if output.suffix not in {".h5", ".hdf5", ".hdf", ".he5"}:
        output.suffix = ".h5"
    directory = output.parent
    name = output.name

    if lens:
        if not chroms:
            print("WARNING: Lengths ignored, provided w/o chroms")
        elif len(chroms) != len(lens):
            return (
                f"Number of chroms({len(chroms)}) and lengths don't match({len(lens)})"
            )

    from genome_loader.write_h5 import write_read_depth

    write_read_depth(
        str(input), str(directory), h5_name=name, chrom_list=chroms, chrom_lens=lens
    )


@app.command()
def writecoverage(
    output: Path = typer.Option(..., "-o", "--output", help="Output h5 file and path"),
    input: Path = typer.Argument(..., help="BAM file to write to H5"),
    chroms: Optional[List[str]] = typer.Option(
        None,
        "-c",
        "--chroms",
        "--contigs",
        help="Chromosomes to write. Writes all by default",
    ),
):
    if output.suffix not in {".h5", ".hdf5", ".hdf", ".he5"}:
        output.suffix = ".h5"
    directory = output.parent
    name = output.name

    from genome_loader.write_h5 import write_allele_coverage

    write_allele_coverage(str(input), str(directory), h5_name=name, chrom_list=chroms)


if __name__ == "__main__":
    root_dir = Path(__file__).parent
    sys.path.append(str(root_dir.joinpath("genome_loader")))
    app()
