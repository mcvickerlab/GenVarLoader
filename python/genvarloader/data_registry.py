"""Data registry for GVL tutorial datasets."""

from pathlib import Path
from typing import Literal

import pooch

N_RETRIES = 3


def fetch(name: Literal["geuvadis_ebi", "1kgp"]) -> dict[str, Path]:
    """Download and cache data for constructing/opening a GVL dataset. Files are cached in the
    user's home directory under :code:`~/.cache/genvarloader`.

    Parameters
    ----------
    name
        The name of the dataset to fetch. Can be one of:

        - "geuvadis_ebi": Geuvadis data for the original analyses by Lappalainen et al. 2013. Phased, normalized, and split into biallelic variants.
        - "1kgp": 1000 Genomes Project, all 3,202 individuals. Phased, normalized, and split into biallelic variants.

    Returns
    -------
        A dictionary of paths to the fetched data.
    """
    if name == "geuvadis_ebi":
        return _geuvadis_ebi()
    elif name == "1kgp":
        return _1kgp()
    raise ValueError(f"Unknown dataset: {name}")  # type: ignore


def _geuvadis_ebi():
    huang = pooch.Pooch(
        pooch.os_cache("genvarloader/geuvadis_ebi"),
        "https://github.com/ni-lab/personalized-expression-benchmark/raw/refs/heads/main/",
        registry={
            "data/gene_list.csv": "md5:ed1b207b14313dbddef520413cbf9e40",
            "consensus/samples.txt": "md5:0ba048e6dbb39dc6a6a9835e4528621f",
        },
        retry_if_failed=N_RETRIES,
    )
    basenji = pooch.Pooch(
        pooch.os_cache("genvarloader/geuvadis_ebi"),
        "https://github.com/calico/basenji/raw/refs/heads/master/manuscripts/cross2020/",
        registry={"targets_human.txt": "md5:61c7fe3aa9da2f309124830ddb282ce3"},
        retry_if_failed=N_RETRIES,
    )
    zenodo = pooch.Pooch(
        pooch.os_cache("genvarloader/geuvadis_ebi"),
        "https://zenodo.org/records/15596289/files/",
        registry={
            "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz": "md5:500bffeed8e0f770c157e0189e9e50ae",
            "geuvadis.pgen": "md5:2815f8e20f8a0acbdead2b0c0d6a00dc",
            "geuvadis.psam": "md5:3c33d631dceccb43222341485b8658f8",
            "geuvadis.pvar.zst": "md5:51aad81267254695191d6284352b0523",
        },
        retry_if_failed=N_RETRIES,
    )
    genes = Path(huang.fetch("data/gene_list.csv", progressbar=True))
    samples = Path(huang.fetch("consensus/samples.txt", progressbar=True))
    basenji2_targets = Path(basenji.fetch("targets_human.txt", progressbar=True))
    expr = Path(
        zenodo.fetch(
            "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz", progressbar=True
        )
    )
    pgen = Path(zenodo.fetch("geuvadis.pgen", progressbar=True))
    _ = zenodo.fetch("geuvadis.psam", progressbar=True)
    _ = zenodo.fetch("geuvadis.pvar.zst", progressbar=True)
    return {
        "genes": genes,
        "expr": expr,
        "pgen": pgen,
        "basenji2_targets": basenji2_targets,
        "samples": samples,
    }


def _1kgp():
    zenodo = pooch.Pooch(
        pooch.os_cache("genvarloader/1kgp"),
        "https://zenodo.org/records/15596480/files/",
        registry={
            "1kGP.snp_indel.split_multiallelics.pgen.xz": "md5:3e07b63b48e5a205a2b2ae5022e01871",
            "1kGP.snp_indel.split_multiallelics.psam": "md5:cb8ea444a41cceb27483df26bb76ad1b",
            "1kGP.snp_indel.split_multiallelics.pvar.zst": "md5:cb00b271504713dbbdbbf17ee0b1825e",
        },
        retry_if_failed=N_RETRIES,
    )
    pgen = Path(
        zenodo.fetch(
            "1kGP.snp_indel.split_multiallelics.pgen.xz",
            pooch.Decompress(),
            progressbar=True,
        )
    )
    _ = zenodo.fetch("1kGP.snp_indel.split_multiallelics.psam", progressbar=True)
    _ = zenodo.fetch("1kGP.snp_indel.split_multiallelics.pvar.zst", progressbar=True)
    return {"pgen": pgen}
