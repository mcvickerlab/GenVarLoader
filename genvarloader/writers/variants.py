import re
from itertools import chain
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, cast

import dask
import joblib
import numpy as np
import zarr
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster
from numpy.typing import NDArray
from sgkit.io.vcf import vcf_to_zarr

from genvarloader.utils import run_shell

# def filt_vcfs(
#     vcf_dir: Path,
#     out_dir: Path,
#     reference: Optional[Path],
#     rename_contigs: Optional[Path],
#     n_jobs: int,
#     overwrite: bool = False,
# ):
#     if not vcf_dir.is_dir():
#         raise ValueError
#     if not out_dir.is_dir():
#         raise ValueError

#     vcfs: List[Path] = (
#         list(vcf_dir.glob("*.vcf"))
#         + list(vcf_dir.glob("*.vcf.gz"))
#         + list(vcf_dir.glob("*.bcf"))
#     )
#     vcf_name = re.compile(r"(.*)\.(?:vcf|vcf\.gz|bcf)")
#     vcf_names: List[str] = [vcf_name.match(vcf.name).group(1) for vcf in vcfs]  # type: ignore
#     out_vcfs = [out_dir / f"{name}.bcf" for name in vcf_names]

#     with joblib.Parallel(n_jobs, prefer="threads") as exe:
#         tasks = [
#             joblib.delayed(filt(vcf, out_vcf, reference, rename_contigs, overwrite))
#             for vcf, out_vcf in zip(vcfs, out_vcfs)
#         ]
#         _ = exe(tasks)


def filt(
    vcf: Path,
    out_vcf: Path,
    reference: Optional[Path],
    rename_contigs: Optional[Path],
    n_threads: int,
    overwrite: bool = False,
):
    if out_vcf is None:
        raise ValueError("Need an output VCF.")
    if not overwrite and out_vcf.exists():
        raise ValueError("Output VCF already exists.")

    # check that bcftools is installed
    try:
        status = run_shell("type bcftools")
    except CalledProcessError:
        raise RuntimeError("Filtering requires bcftools to be installed.")

    cmd = ""
    if rename_contigs is not None:
        cmd += f"| bcftools annotate --rename-chr {rename_contigs} -O u --threads {n_threads} \\\n"
    if reference is not None:
        cmd += f"| bcftools norm -f {reference} -O u --threads {n_threads} \\\n"
    cmd += (
        f"""bcftools filter -i 'TYPE="snp" | TYPE="mnp"' -O u --threads {n_threads} {vcf} \\\n"""
        f"""| bcftools filter -e 'ALT="*"' -O u --threads {n_threads} \\\n"""
        f"""| bcftools norm -a -O u --threads {n_threads} \\\n"""
        f"""| bcftools norm -d none -O b --threads {n_threads} \\\n"""
    )
    cmd += f">{'|' if overwrite else ''} {out_vcf}"

    status = run_shell(cmd)


# When writing WGS this hangs due to unmanaged memory issues.
# def write_zarrs(
#     vcf_dir: Path,
#     zarr_dir: Path,
#     n_jobs: int,
#     overwrite: bool = False,
#     variants_per_chunk: int = int(1e4),
# ):
#     if not vcf_dir.is_dir():
#         raise ValueError('Path to VCF directory is a file or does not exist.')
#     if not zarr_dir.is_dir():
#         raise ValueError('Path to Zarr directory is a file or does not exist.')

#     cluster = LocalCluster(n_workers=n_jobs // 2, threads_per_worker=1)
#     client = Client(cluster)

#     vcfs: List[Path] = list(chain(
#             vcf_dir.glob("*.vcf"),
#             vcf_dir.glob("*.vcf.gz"),
#             vcf_dir.glob("*.bcf")
#         ))
#     vcf_name = re.compile(r"(.*)\.(?:vcf|vcf\.gz|bcf)")
#     vcf_names: List[str] = [vcf_name.match(vcf.name).group(1) for vcf in vcfs]  # type: ignore
#     zarrs = [zarr_dir / f"{name}.zarr" for name in vcf_names]

#     tasks = [
#         delayed(write_zarr)(vcf, zarr, overwrite, variants_per_chunk)
#         for vcf, zarr in zip(vcfs, zarrs)
#     ]
#     dask.compute(tasks)  # type: ignore


def write_zarr(
    vcf: Path,
    out_zarr: Path,
    n_jobs: int,
    overwrite: bool = False,
    variants_per_chunk: int = int(1e4),
) -> None:
    """Write a VCF to a Zarr file. Note that this currently breaks compatibility with sgkit because TensorStore
    does not support filters nor object/byte arrays."""
    if vcf is None:
        raise ValueError("Need an input VCF.")
    if not overwrite and out_zarr.exists():
        raise ValueError("Zarr already exists.")

    cluster = LocalCluster(n_workers=n_jobs // 2, threads_per_worker=1)
    client = Client(cluster)

    vcf_to_zarr(vcf, out_zarr, chunk_length=variants_per_chunk)

    z = zarr.open_group(out_zarr)

    # add contig offsets to reduce initialization time
    v_contig = cast(NDArray, z["variant_contig"][:])

    contigs_with_variants, contig_offsets = np.unique(v_contig, return_index=True)
    z.create_dataset(
        "contig_offsets",
        data=contig_offsets,
        compressor=False,
        chunks=1,
        overwrite=overwrite,
    )
    contigs = cast(List[str], z.attrs["contigs"])
    contig_idx = dict(zip(contigs, range(len(contigs))))
    z.attrs["contig_idx"] = contig_idx
    z.attrs["contig_offset_idx"] = dict(
        zip(contigs_with_variants, range(len(contigs_with_variants)))
    )

    gvl_keys = {
        "variant_allele",
        "variant_contig",
        "variant_position",
        "call_genotype",
        "contig_offsets",
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
                overwrite=overwrite,
            )
        elif name == "call_genotype":
            chunks = [c for c, d in zip(val.chunks, val.shape) if d != 1]
            z.create_dataset(
                name,
                data=val[:].squeeze(),  # type: ignore
                chunks=chunks,
                compressor=val.compressor,
                filters=None,
                overwrite=overwrite,
            )
        elif name == "variant_position":
            positions = val[:]
            c_pos = np.split(positions, contig_offsets[1:])
            g = z.create_group(name, overwrite=overwrite)
            for i, pos in enumerate(c_pos):
                if len(pos) > 0:
                    g.create_dataset(
                        str(i),
                        data=pos,
                        chunks=False,
                        compressor=val.compressor,
                        filters=None,
                        overwrite=overwrite,
                    )
        elif name in gvl_keys and val.filters is not None:
            z.create_dataset(
                name,
                data=val[:],
                chunks=val.chunks,
                compressor=val.compressor,
                filters=None,
                overwrite=overwrite,
            )

    z.visititems(edit)

    to_del = [g for g in z.keys() if g not in gvl_keys]
    for g in to_del:
        del z[g]
