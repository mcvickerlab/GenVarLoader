"""Shared fixtures for the GenVarLoader test suite.

Centralizes paths to on-disk artifacts under ``tests/data/``. Test files
should import these via fixture injection rather than constructing paths
or opening datasets at module scope.

Fixtures here intentionally yield *paths*, not opened Datasets. Opening
a Dataset costs real time; pushing that decision to the test (or to a
small, locally scoped fixture in the test file) keeps fixture cost
predictable. Where a session-scoped opened Dataset is genuinely useful,
prefer adding it inside the integration test file that needs it.
"""

from pathlib import Path

import pytest


# --- root paths --------------------------------------------------------------

@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data_dir(tests_dir: Path) -> Path:
    return tests_dir / "data"


# --- reference -------------------------------------------------------------

@pytest.fixture(scope="session")
def ref_fasta(data_dir: Path) -> Path:
    """bgzipped hg38 reference used by the default toy datasets."""
    return data_dir / "fasta" / "hg38.fa.bgz"


# --- toy phased datasets (one per variant source) ----------------------------

@pytest.fixture(scope="session")
def phased_vcf_gvl(data_dir: Path) -> Path:
    return data_dir / "phased_dataset.vcf.gvl"


@pytest.fixture(scope="session")
def phased_pgen_gvl(data_dir: Path) -> Path:
    return data_dir / "phased_dataset.pgen.gvl"


@pytest.fixture(scope="session")
def phased_svar_gvl(data_dir: Path) -> Path:
    return data_dir / "phased_dataset.svar.gvl"


# --- 1kg datasets (slow tier) ------------------------------------------------

@pytest.fixture(scope="session")
def kg_dir(data_dir: Path) -> Path:
    return data_dir / "1kg"


@pytest.fixture(scope="session")
def kg_bcf_gvl(kg_dir: Path) -> Path:
    return kg_dir / "phased_1kg.bcf.gvl"


@pytest.fixture(scope="session")
def kg_pgen_gvl(kg_dir: Path) -> Path:
    return kg_dir / "phased_1kg.pgen.gvl"


@pytest.fixture(scope="session")
def kg_svar_gvl(kg_dir: Path) -> Path:
    return kg_dir / "phased_1kg.svar.gvl"


# --- raw inputs / regression artifacts --------------------------------------

@pytest.fixture(scope="session")
def source_bed(data_dir: Path) -> Path:
    return data_dir / "source.bed"


@pytest.fixture(scope="session")
def source_vcf(data_dir: Path) -> Path:
    return data_dir / "source.vcf"


@pytest.fixture(scope="session")
def issue_153_bed(data_dir: Path) -> Path:
    return data_dir / "issue_153.bed"


@pytest.fixture(scope="session")
def issue_153_vcf(data_dir: Path) -> Path:
    return data_dir / "issue_153.vcf"


# --- raw variant source subdirectories --------------------------------------

@pytest.fixture(scope="session")
def vcf_dir(data_dir: Path) -> Path:
    """Directory containing filtered VCF files used as variant sources."""
    return data_dir / "vcf"


@pytest.fixture(scope="session")
def pgen_dir(data_dir: Path) -> Path:
    """Directory containing filtered PGEN files used as variant sources."""
    return data_dir / "pgen"


@pytest.fixture(scope="session")
def filtered_svar(data_dir: Path) -> Path:
    """Pre-built SparseVar dataset used for svar-link and var-filter tests."""
    return data_dir / "filtered.svar"


# --- bigwig tracks -----------------------------------------------------------

@pytest.fixture(scope="session")
def bigwig_dir(data_dir: Path) -> Path:
    """Directory containing sample BigWig files."""
    return data_dir / "bigwig"


# --- ground-truth / consensus directories -----------------------------------

@pytest.fixture(scope="session")
def consensus_dir(data_dir: Path) -> Path:
    """Directory containing per-haplotype consensus FASTA ground-truth files."""
    return data_dir / "consensus"
