from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def write_vcf_to_zarr(
    vcf: Path = typer.Argument(..., resolve_path=True),
    zarr: Path = typer.Argument(
        ...,
        resolve_path=True,
        help="If a directory, will keep name and change extension to .zarr",
    ),
    n_jobs: int = typer.Option(1, min=1),
    overwrite: bool = typer.Option(False, help="Whether to overwrite existing output."),
    filter: bool = typer.Option(
        False, help="Whether to filter the input VCF for SNPs before writing to Zarr."
    ),
    out_vcf: Optional[Path] = typer.Option(
        None,
        resolve_path=True,
        help="Required if filtering. Ouput path for filtered VCFs.",
    ),
    reference: Optional[Path] = typer.Option(
        None, resolve_path=True, help="Ignored if not filtering."
    ),
    rename_contigs: Optional[Path] = typer.Option(
        None,
        resolve_path=True,
        help="Ignored if not filtering. File for renaming contigs, e.g. if the VCF has different contig names than the reference genome. See `bcftools annotate --rename-chrs` (https://samtools.github.io/bcftools/bcftools.html#annotate).",
    ),
):
    """Write a VCF to Zarr.

    NOTE: if dask raises warnings about unmanaged memory, this is most likely an out-of-memory error.
    """
    import logging

    from genvarloader.writers.variants import filt, write_zarr

    if filter and out_vcf is None:
        raise ValueError("Need an output VCF directory if filtering.")
    elif filter and out_vcf is not None:
        logging.info("Filtering VCFs.")
        filt(vcf, out_vcf, reference, rename_contigs, n_jobs, overwrite)

        logging.info("Converting VCFs to Zarr.")
        write_zarr(out_vcf, zarr, n_jobs, overwrite)
    else:
        logging.info("Converting VCFs to Zarr.")
        write_zarr(vcf, zarr, n_jobs, overwrite)


if __name__ == "__main__":
    app()
