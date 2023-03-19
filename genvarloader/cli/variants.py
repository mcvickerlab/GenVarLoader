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
        help="If a directory, will use the VCF name with the file extension changed to .zarr",
    ),
    n_jobs: int = typer.Option(1, min=1),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing output files."
    ),
    filter: bool = typer.Option(
        False, help="Whether to filter the input VCF for SNPs before writing to Zarr."
    ),
    out_vcf: Optional[Path] = typer.Option(
        None, resolve_path=True, help="Required if filtering. Output VCF."
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
    """Write a single-sample VCF to a Zarr."""
    import logging
    import re

    from genvarloader.writers.variants import filt, write_zarr

    vcf_name_regex = re.compile(r"(.*)\.(?:vcf|vcf\.gz|bcf)")
    vcf_name = vcf_name_regex.match(vcf.name)
    if vcf_name is None:
        raise ValueError("VCF file extension is not one of: .vcf, .vcf.gz, or .bcf")

    if zarr.is_dir():
        zarr = zarr / f"{vcf_name.group(1)}.zarr"

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
