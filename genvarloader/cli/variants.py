from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import assert_never

app = typer.Typer()


class Steps(str, Enum):
    WRITE = "write"
    FILTER = "filter"
    MERGE = "merge"
    MERGE_FILTER = "merge-filter"


@app.command()
def write_zarr(
    steps: Steps,
    out_zarr: Path,
    out_vcf: Optional[Path] = typer.Option(
        None, help="Output VCF. Required if using merge, filter, or merge-filter"
    ),
    reference: Optional[Path] = typer.Option(None),
    rename_contigs: Optional[Path] = typer.Option(
        None,
        help="File for renaming contigs, e.g. if the VCF has different contig names than the reference genome. See `bcftools annotate --rename-chrs` (https://samtools.github.io/bcftools/bcftools.html#annotate).",
    ),
    n_threads: int = typer.Option(1, min=1),
    single_sample: bool = typer.Option(
        False,
        "--single-sample",
        help="Whether the output Zarr contains VCF data from a single sample or not.",
    ),
    overwrite: bool = typer.Option(
        False, help="Whether to overwrite existing output files."
    ),
    vcfs: List[Path] = typer.Argument(...),
):
    import logging
    import tempfile
    import warnings

    from genvarloader.writers.variants import filt, merge, write_zarr

    if steps is Steps.MERGE_FILTER:
        if len(vcfs) < 2:
            raise ValueError
        if single_sample == True:
            warnings.warn(
                "Setting 'single_sample' to False since VCFs are being merged."
            )

        temp_merged_vcf = tempfile.NamedTemporaryFile()

        logging.info(f"Writing merged VCFs to a tempfile {temp_merged_vcf.name}")
        merge(vcfs, Path(temp_merged_vcf.name), n_threads, overwrite)

        logging.info("Filtering VCF.")
        filt(
            Path(temp_merged_vcf.name),
            out_vcf,
            reference,
            rename_contigs,
            n_threads,
            overwrite,
        )

        temp_merged_vcf.close()

        logging.info("Converting VCF to Zarr.")
        write_zarr(
            out_vcf, out_zarr, n_threads, single_sample=False, overwrite=overwrite
        )
    elif steps is Steps.MERGE:
        if len(vcfs) is None:
            raise ValueError
        if single_sample == True:
            warnings.warn(
                "Setting 'single_sample' to False since VCFs are being merged."
            )

        logging.info(f"Writing merged VCFs.")
        merge(vcfs, out_vcf, n_threads, overwrite)

        logging.info("Converting VCF to Zarr.")
        write_zarr(
            out_vcf, out_zarr, n_threads, single_sample=False, overwrite=overwrite
        )
    elif steps is Steps.FILTER:
        logging.info("Filtering VCF.")
        filt(vcfs[0], out_vcf, reference, rename_contigs, n_threads, overwrite)

        logging.info("Converting VCF to Zarr.")
        write_zarr(out_vcf, out_zarr, n_threads, single_sample, overwrite)
    elif steps is Steps.WRITE:
        logging.info("Converting VCF to Zarr.")
        write_zarr(vcfs[0], out_zarr, n_threads, single_sample, overwrite)
    else:
        assert_never(steps)


if __name__ == "__main__":
    app()
