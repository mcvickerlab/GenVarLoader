from pathlib import Path
from typing import List


def vcfs_merge_filter_to_zarr(
    out_merged: Path,
    out_zarr: Path,
    reference: Path,
    renamer: Path,
    n_threads: int,
    vcfs: List[Path],
):
    import logging
    import sys

    from dask.distributed import Client, LocalCluster
    from sgkit.io.vcf import vcf_to_zarr

    from genome_loader.utils import run_shell

    logging.basicConfig(stream=sys.stdout, level="INFO")

    if out_merged.exists():
        raise ValueError("Merged VCF already exists.")
    if out_zarr.exists():
        raise ValueError("Zarr already exists.")

    out_merged.parent.mkdir(0o744, parents=True, exist_ok=True)
    out_zarr.parent.mkdir(0o744, parents=True, exist_ok=True)

    logging.info("Merging, filtering, and normalizing VCFs.")
    status = run_shell(
        f"""
        bcftools merge -O b --threads {n_threads} {" ".join(map(str, vcfs))} \\
        | bcftools filter -i 'TYPE="snp" | TYPE="mnp"' -O b --threads {n_threads} \\
        | bcftools filter -e 'ALT="*"' -O b --threads {n_threads} \\
        | bcftools norm -a -O b --threads {n_threads} \\
        | bcftools norm -d none -O b --threads {n_threads} \\
        | bcftools annotate --rename-chr {renamer} -O b --threads {n_threads} \\
        | bcftools norm -f {reference} -O b --threads {n_threads} \\
        > {out_merged}
        """
    )

    logging.info("Indexing merged VCF.")
    status = run_shell(f"bcftools index --threads {n_threads} {out_merged}")

    logging.info("Converting merged VCF to Zarr.")

    cluster = LocalCluster(n_workers=n_threads // 2, threads_per_worker=1)
    client = Client(cluster)

    vcf_to_zarr(out_merged, out_zarr, chunk_width=100)
