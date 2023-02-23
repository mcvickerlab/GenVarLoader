from pathlib import Path
from typing import Optional

import typer

app = typer.Typer()


@app.command()
def write_zarrs(
    vcf_dir: Path,
    zarr_dir: Path,
    n_jobs: int = typer.Option(1, min=1),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing output files."
    ),
    filter: bool = typer.Option(
        False, help="Whether to filter the input VCF for SNPs before writing to Zarr."
    ),
    out_vcf_dir: Optional[Path] = typer.Option(
        None, help="Required if filtering. Output VCF."
    ),
    reference: Optional[Path] = typer.Option(None, help="Ignored if not filtering."),
    rename_contigs: Optional[Path] = typer.Option(
        None,
        help="Ignored if not filtering. File for renaming contigs, e.g. if the VCF has different contig names than the reference genome. See `bcftools annotate --rename-chrs` (https://samtools.github.io/bcftools/bcftools.html#annotate).",
    ),
):
    import logging

    from genvarloader.writers.variants import filt_vcfs, write_zarrs

    if filter and out_vcf_dir is None:
        raise ValueError("Need an output VCF directory if filtering.")
    elif filter and out_vcf_dir is not None:
        logging.info("Filtering VCF.")
        filt_vcfs(vcf_dir, out_vcf_dir, reference, rename_contigs, n_jobs, overwrite)

        logging.info("Converting VCF to Zarr.")
        write_zarrs(out_vcf_dir, zarr_dir, n_jobs, overwrite)
    else:
        logging.info("Converting VCF to Zarr.")
        write_zarrs(vcf_dir, zarr_dir, n_jobs, overwrite)


if __name__ == "__main__":
    app()
