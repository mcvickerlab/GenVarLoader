from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import typer
from loguru import logger

WDIR = Path(__file__).resolve().parent
ONE_KG_DIR = WDIR / "1kg"
CONS_DIR = WDIR / "1kg_consensus"
REF = WDIR / "fasta" / "hg38.fa.bgz"

N_REGIONS = 100
REGION_LEN = 10_000
SEED = 0

ZENODO_BCF_URL = "https://zenodo.org/records/20132907/files/1kg.chr21_chr22.5samples.bcf"
ZENODO_CSI_URL = "https://zenodo.org/records/20132907/files/1kg.chr21_chr22.5samples.bcf.csi"
# Fill these in on first successful run; the script prints observed hashes
# and exits when they are None.
ZENODO_BCF_HASH: str | None = None
ZENODO_CSI_HASH: str | None = None


def run_shell(cmd: list[str], input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, input=input)
    except subprocess.CalledProcessError as e:
        print("Command:", " ".join(e.cmd))
        print("Stdout:", e.stdout.decode(errors="replace"))
        print("Stderr:", e.stderr.decode(errors="replace"))
        raise


def main() -> None:
    """Generate 1000 Genomes ground-truth haplotypes via bcftools consensus."""
    log_file = WDIR / "generate_1kg_ground_truth.log"
    if log_file.exists():
        log_file.unlink()
    _ = logger.add(log_file, level="DEBUG")

    t0 = perf_counter()

    if not REF.exists():
        raise SystemExit(
            f"Reference {REF} not found. Run `pixi run -e dev gen` first to "
            "fetch hg38."
        )

    ONE_KG_DIR.mkdir(0o777, parents=True, exist_ok=True)
    if CONS_DIR.exists():
        shutil.rmtree(CONS_DIR)
    CONS_DIR.mkdir(0o777, parents=True, exist_ok=True)

    logger.info("Pipeline scaffold OK")
    logger.info(f"Finished in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    typer.run(main)
