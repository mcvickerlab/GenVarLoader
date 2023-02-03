from pathlib import Path
from typing import List, Optional, cast

import numpy as np
import zarr
from dask.distributed import Client, LocalCluster
from natsort import natsorted
from numpy.typing import NDArray
from sgkit.io.vcf import vcf_to_zarr

from genvarloader.utils import run_shell


def merge(
    vcfs: List[Path], out_vcf: Optional[Path], n_threads: int, overwrite: bool = False
):
    if out_vcf is None:
        raise ValueError("Need an output VCF.")
    if not overwrite and out_vcf.exists:
        raise ValueError("Output VCF already exists.")

    status = run_shell(
        f"""
        bcftools merge -O b --threads {n_threads} {" ".join(map(str, vcfs))} \\
        >{'|' if overwrite else ''} {out_vcf}
        """
    )


def filt(
    vcf: Path,
    out_vcf: Optional[Path],
    reference: Optional[Path],
    rename_contigs: Optional[Path],
    n_threads: int,
    overwrite: bool = False,
):
    if out_vcf is None:
        raise ValueError("Need an output VCF.")
    if not overwrite and out_vcf.exists:
        raise ValueError("Output VCF already exists.")

    cmd = (
        f"""bcftools filter -i 'TYPE="snp" | TYPE="mnp"' -O b --threads {n_threads} {vcf} \\\n"""
        f"""| bcftools filter -e 'ALT="*"' -O b --threads {n_threads} \\\n"""
        f"""| bcftools norm -a -O b --threads {n_threads} \\\n"""
        f"""| bcftools norm -d none -O b --threads {n_threads} \\\n"""
    )
    if rename_contigs is not None:
        cmd += f"| bcftools annotate --rename-chr {rename_contigs} -O b --threads {n_threads} \\\n"
    if reference is not None:
        cmd += f"| bcftools norm -f {reference} -O b --threads {n_threads} \\\n"
    cmd += f">{'|' if overwrite else ''} {out_vcf}"

    status = run_shell(cmd)


def write_zarr(
    vcf: Optional[Path],
    out_zarr: Path,
    n_threads: int,
    single_sample: bool,
    overwrite: bool = False,
):
    """Write a VCF to a Zarr file. Note that this currently breaks compatibility with sgkit because TensorStore
    does not support filters nor object/byte arrays."""
    if vcf is None:
        raise ValueError("Need an input VCF.")
    if not overwrite and out_zarr.exists():
        raise ValueError("Zarr already exists.")

    cluster = LocalCluster(n_workers=n_threads // 2, threads_per_worker=1)
    client = Client(cluster)

    vcf_to_zarr(vcf, out_zarr)

    z = zarr.open_group(out_zarr)

    ## TensorStore doesn't support filters so remove them from any group we need to access

    # convert alleles from object to uint8 (ok because everything is SNPs)
    z.create_dataset(
        "variant_allele",
        data=z["variant_allele"].astype("S1")[:].view("u1"),  # type: ignore
        chunks=z["variant_allele"].chunks,
        compressor=z["variant_allele"].compressor,
        filters=None,
        overwrite=True,
    )

    if z["variant_position"].filters is not None:
        z.create_dataset(
            "variant_position",
            data=z["variant_position"][:],
            chunks=z["variant_position"].chunks,
            compressor=z["variant_position"].compressor,
            filters=None,
            overwrite=True,
        )

    # don't need this group for single sample Zarrs
    if not single_sample:
        z.create_dataset(
            "call_genotype_mask",
            data=z["call_genotype_mask"].astype(bool)[:],  # type: ignore
            chunks=z["call_genotype_mask"].chunks,
            compressor=z["call_genotype_mask"].compressor,
            filters=None,
            overwrite=True,
        )

    # squeeze sample dimension for single sample Zarr
    if single_sample:
        chunks = [
            c
            for c, d in zip(z["call_genotype"].chunks, z["call_genotype"].shape)
            if d != 1
        ]
        z.create_dataset(
            "call_genotype",
            data=z["call_genotype"][:].squeeze(),  # type: ignore
            chunks=chunks,
            compressor=z["call_genotype"].compressor,
            filters=None,
            overwrite=True,
        )
        chunks = [
            c
            for c, d in zip(
                z["call_genotype_mask"].chunks, z["call_genotype_mask"].shape
            )
            if d != 1
        ]
        z.create_dataset(
            "call_genotype_mask",
            data=z["call_genotype_mask"].astype(bool)[:].squeeze(),  # type: ignore
            chunks=chunks,
            compressor=z["call_genotype_mask"].compressor,
            filters=None,
            overwrite=True,
        )

    # add contig offsets to reduce initialization time
    v_contig = cast(NDArray, z["variant_contig"][:])
    contig_offsets = np.searchsorted(v_contig, np.arange(len(z.attrs["contigs"])))
    z.create_dataset("contig_offsets", data=contig_offsets, overwrite=True)

    if z["variant_contig"].filters is not None:
        z.create_dataset(
            "variant_contig",
            data=v_contig,
            chunks=z["variant_contig"].chunks,
            compressor=z["variant_contig"].compressor,
            filters=None,
            overwrite=True,
        )
