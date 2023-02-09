from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def write_zarr(
    vcf: Path,
    out_zarr: Path,
    n_threads: int = typer.Option(1, min=1),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing output files."
    ),
    filter: bool = typer.Option(
        False, help="Whether to filter the input VCF for SNPs before writing to Zarr."
    ),
    out_vcf: Optional[Path] = typer.Option(
        None, help="Required if filtering. Output VCF."
    ),
    reference: Optional[Path] = typer.Option(None, help="Ignored if not filtering."),
    rename_contigs: Optional[Path] = typer.Option(
        None,
        help="Ignored if not filtering. File for renaming contigs, e.g. if the VCF has different contig names than the reference genome. See `bcftools annotate --rename-chrs` (https://samtools.github.io/bcftools/bcftools.html#annotate).",
    ),
):
    import logging

    from genvarloader.writers.variants import filt, write_zarr

    if filter:
        logging.info("Filtering VCF.")
        filt(vcf, out_vcf, reference, rename_contigs, n_threads, overwrite)

        logging.info("Converting VCF to Zarr.")
        write_zarr(out_vcf, out_zarr, n_threads, overwrite)
    else:
        logging.info("Converting VCF to Zarr.")
        write_zarr(vcf, out_zarr, n_threads, overwrite)


if __name__ == "__main__":
    app()
