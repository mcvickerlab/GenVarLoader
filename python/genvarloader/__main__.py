from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

from typer import Argument, Option, Typer

__all__ = []


app = Typer(rich_markup_mode="rich")


class LOG_LEVEL(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@app.command(no_args_is_help=True)
def main(
    path: Annotated[
        Path,
        Argument(
            help="Path to the output directory, using the .gvl extension for clarity is encouraged."
        ),
    ],
    bed: Annotated[Path, Argument(help="Path to the BED file.")],
    variants: Annotated[
        Optional[Path], Option(help="Path to variants file, either VCF or PGEN.")
    ] = None,
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
            help="Length of the sequences. If not provided, will default to maximum length seen in the BED file."
        ),
    ] = None,
    max_jitter: Annotated[
        int,
        Option(
            help="Maximum jitter to allow. Permitted by expanding the region length by 2 * max jitter."
        ),
    ] = 0,
    overwrite: Annotated[
        bool,
        Option(help="Overwrite the output directory if it exists."),
    ] = False,
    max_memory: Annotated[
        str,
        Option(
            help="Hint for maximum memory to use. Can be a number or use suffixes M and G to specify units. Actual usage will be marginally higher than this. Default is 4 GiB."
        ),
    ] = "4G",
    log_level: Annotated[
        LOG_LEVEL,
        Option(
            help="Log level to use. One of DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO."
        ),
    ] = LOG_LEVEL.INFO,
):
    """Write a GenVarLoader dataset from a BED3+ file and a VCF file and/or BigWig files.

    If a VCF is included, the dataset will support yielding haplotypes.
    If BigWigs are included, the dataset will support yielding base-pair resolution tracks.
    One of either a VCF or BigWigs must be provided.

    [b]Sample subsetting behavior:[/b]
    If a VCF and BigWigs are provided, the samples will be subset to the intersection of the samples in the VCF and BigWigs.
    If a list of specific samples are provided via --samples, that subset will take precedence.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory {path} already exists. Use --overwrite to overwrite."
        )

    import sys

    from loguru import logger

    from genvarloader._bigwig import BigWigs
    from genvarloader._dataset._write import write

    if bigwig_table is not None:
        bigwigs = BigWigs.from_table("bws", bigwig_table)
    else:
        bigwigs = None

    if samples is not None:
        _samples = samples.split(",")
    else:
        _samples = None

    if max_memory[-1] == "M":
        _max_memory = int(max_memory[:-1]) * 1024**2
    elif max_memory[-1] == "G":
        _max_memory = int(max_memory[:-1]) * 1024**3
    else:
        _max_memory = int(max_memory)

    logger.remove()
    logger.add(sys.stderr, level=log_level.value)
    logger.enable("genvarloader")

    write(
        path=path,
        bed=bed,
        variants=variants,
        bigwigs=bigwigs,
        samples=_samples,
        length=length,
        max_jitter=max_jitter,
        overwrite=overwrite,
        max_mem=_max_memory,
    )


if __name__ == "__main__":
    app()
