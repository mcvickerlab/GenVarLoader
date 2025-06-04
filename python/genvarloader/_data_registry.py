from pathlib import Path
from typing import cast

import pooch
from attrs import define


def fetch(name: str) -> Path | list[Path]:
    url = registry[name].url
    processor = registry[name].processor
    path = cast(str | list[str], retriever.fetch(url, processor, progressbar=True))

    if isinstance(path, list):
        path = Path(path[0]).parent
    elif isinstance(path, str):
        path = Path(path)

    if name == "geuvadis_rna_ebi":
        files = [
            path / "EUR373.gene.cis.FDR5.best.rs137.txt.gz",
            path / "GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz",
        ]
        return files
    if name == "geuvadis_genos_ebi":
        files = [
            path / "geuvadis.pgen",
            path / "geuvadis.psam",
            path / "geuvadis.pvar",
            path / "geuvadis.pvar.gvi",
        ]
        return files
    else:
        return path


@define
class RegistryItem:
    url: str
    known_hash: str
    processor: pooch.processors.ExtractorProcessor | None = None


registry = {
    "geuvadis_rna_ebi": RegistryItem(
        url="https://ftp.ebi.ac.uk/biostudies/fire/E-GEUV-/003/E-GEUV-3/Files/analysis_results.zip",
        known_hash="md5:1ab7c60c9b9d7c567c09c9ab9c4fca34",
        processor=pooch.Unzip(),
    ),
    "geuvadis_genos_ebi": RegistryItem(url="", known_hash=""),
    "1kgp_genos_pgen": RegistryItem(url="", known_hash="", processor=pooch.Untar()),
    "geuvadis_rna_bw": RegistryItem(url="", known_hash="", processor=pooch.Untar()),
}

retriever = pooch.create(
    path=pooch.os_cache("genvarloader"),
    base_url="",
    version="v0.1.0",
    registry={v.url: v.known_hash for v in registry.values()},
    retry_if_failed=5,
)
