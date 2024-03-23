from pathlib import Path
from typing import Annotated, Optional

from typer import Argument, Option, Typer

app = Typer(rich_markup_mode="rich")


@app.command()
def main(
    path: Annotated[
        Path,
        Argument(
            ...,
            help="Path to the output directory, using the .gvl extension for clarity is encouraged.",
        ),
    ],
    bed: Annotated[Path, Argument(..., help="Path to the BED file.")],
    vcf: Annotated[Optional[Path], Option(help="Path to the VCF file.")] = None,
    bigwig_table: Annotated[
        Optional[Path],
        Option(
            help='CSV or TSV with columns "sample" and "path" mapping each sample to its BigWig.'
        ),
    ] = None,
    samples: Annotated[
        Optional[str],
        Option(help="Comma-separated list of samples to include/subset to."),
    ] = None,
    length: Annotated[
        Optional[int],
        Option(
            help="Length of the sequences. If not provided, will default to maximum length seen in the BED file.",
        ),
    ] = None,
    max_jitter: Annotated[
        int,
        Option(
            help="Maximum jitter to allow. Permitted by expanding the region length by 2 * max jitter.",
        ),
    ] = 0,
):
    """Write a GenVarLoader dataset from a BED3+ file and a VCF file and/or BigWig files.

    If a VCF is included, the dataset will support yielding haplotypes.
    If BigWigs are included, the dataset will support yielding base-pair resolution tracks.
    One of either a VCF or BigWigs must be provided.

    [b]Sample subsetting behavior:[/b]
    If a VCF and BigWigs are provided, the samples will be subset to the intersection of the samples in the VCF and BigWigs.
    If a list of specific samples are provided via --samples, that subset will take precedence.
    """

    from genvarloader.bigwig import BigWigs
    from genvarloader.write import write

    if bigwig_table is not None:
        bigwigs = BigWigs.from_table("bws", bigwig_table)
    else:
        bigwigs = None

    if samples is not None:
        _samples = samples.split(",")
    else:
        _samples = None

    write(path, bed, vcf, bigwigs, _samples, length, max_jitter)


if __name__ == "__main__":
    app()
