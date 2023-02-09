import shutil
from pathlib import Path
from subprocess import CalledProcessError
from typing import Optional, cast

import numpy as np
import zarr
from dask.distributed import Client, LocalCluster
from numpy.typing import NDArray
from sgkit.io.vcf import vcf_to_zarr

from genvarloader.utils import run_shell


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

    # check that bcftools is installed
    try:
        status = run_shell("type bcftools")
    except CalledProcessError:
        raise RuntimeError("Filtering requires bcftools to be installed.")

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
    overwrite: bool = False,
    variants_per_chunk: int = int(1e4),
):
    """Write a VCF to a Zarr file. Note that this currently breaks compatibility with sgkit because TensorStore
    does not support filters nor object/byte arrays."""
    if vcf is None:
        raise ValueError("Need an input VCF.")
    if not overwrite and out_zarr.exists():
        raise ValueError("Zarr already exists.")

    cluster = LocalCluster(n_workers=n_threads // 2, threads_per_worker=1)
    client = Client(cluster)

    vcf_to_zarr(vcf, out_zarr, chunk_length=variants_per_chunk)

    z = zarr.open_group(out_zarr)

    # add contig offsets to reduce initialization time
    v_contig = cast(NDArray, z["variant_contig"][:])
    contig_offsets = np.searchsorted(v_contig, np.arange(len(z.attrs["contigs"])))
    z.create_dataset(
        "contig_offsets", data=contig_offsets, compressor=None, chunks=1, overwrite=True
    )

    gvl_groups = {
        "variant_allele",
        "variant_contig",
        "variant_position",
        "call_genotype" "contig_offsets",
    }

    def edit(name, val):
        # TensorStore doesn't support filters so remove them from any group we need to access
        # convert alleles from object to uint8 for TensorStore (ok because everything is SNPs)
        if name == "variant_allele":
            z.create_dataset(
                name,
                data=val.astype("S1")[:].view("u1"),  # type: ignore
                chunks=val.chunks,
                compressor=val.compressor,
                filters=None,
                overwrite=True,
            )
        elif name == "call_genotype":
            chunks = [c for c, d in zip(val.chunks, val.shape) if d != 1]
            z.create_dataset(
                name,
                data=val[:].squeeze(),  # type: ignore
                chunks=chunks,
                compressor=val.compressor,
                filters=None,
                overwrite=True,
            )
        elif name == "variant_position":
            positions = val[:]
            c_pos = np.split(positions, contig_offsets[1:])
            g = z.create_group(name, overwrite=True)
            for i, pos in enumerate(c_pos):
                if len(pos) > 0:
                    g.create_dataset(
                        str(i),
                        data=pos,
                        chunks=False,
                        compressor=val.compressor,
                        filters=None,
                        overwrite=True,
                    )
        elif name in gvl_groups and val.filters is not None:
            z.create_dataset(
                name,
                data=val[:],
                chunks=val.chunks,
                compressor=val.compressor,
                filters=None,
                overwrite=True,
            )

    z.visititems(edit)
