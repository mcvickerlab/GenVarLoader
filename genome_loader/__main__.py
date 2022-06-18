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
def writefrag(
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
    ignore_offset: bool = typer.Option(False, "--ignore_offset", help="Don't offset tn5 cutsites"),
    count_method: str = typer.Option("cutsite", "--method", help="Counting method, Choice of 'cutsite', 'midpoint, 'fragment'")
    
):
    if output.suffix not in {".h5", ".hdf5", ".hdf", ".he5"}:
        output.suffix = ".h5"
    directory = output.parent
    name = output.name

    if lens:
        if not chroms:
            typer.echo("WARNING: Lengths ignored, provided w/o chroms")
        elif len(chroms) != len(lens):
            typer.echo(f"Number of chroms({len(chroms)}) and lengths don't match({len(lens)})")
            raise typer.Exit(code=1)
    
    if ignore_offset:
        offset_tn5=False
    else:
        offset_tn5=True

    # Check for valid input
    if count_method not in {"cutsite", "midpoint", "fragment"}:
        typer.echo("Please input valid count method! ('cutsite', 'midpoint, 'fragment')")
        raise typer.Exit(code=1)

    from genome_loader.write_h5 import write_frag_depth

    write_frag_depth(
        str(input), str(directory), h5_name=name,
        chrom_list=chroms, chrom_lens=lens,
        offset_tn5=offset_tn5, count_method=count_method
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
