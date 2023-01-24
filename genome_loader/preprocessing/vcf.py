import logging
from pathlib import Path
from subprocess import CalledProcessError, run
from textwrap import dedent
from typing import List

import typer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_shell(args: str):
    try:
        status = run(dedent(args).strip(), check=True, shell=True)
    except CalledProcessError as e:
        logging.info(e.stdout)
        logging.error(e.stderr)
        raise e
    return status


def filter_and_norm(
    vcfs: list[Path], out_dir: Path, renamer: Path, ref: Path, n_threads: int
) -> List[Path]:
    """Create filtered BCFs from VCFs."""
    logging.info(
        "Select SNPs, atomize, remove duplicates, remove indirect deletions (i.e. '*')."
    )
    out_dir.mkdir(0o744, parents=True, exist_ok=True)
    snp_vcfs: List[Path] = []
    pbar = tqdm(vcfs)
    for vcf in pbar:
        valid_filetypes = {".vcf", ".vcf.gz", ".bcf"}
        filetype = next(filter(lambda ft: vcf.name.endswith(ft), valid_filetypes))
        out_file = out_dir / vcf.name.replace(filetype, ".bcf")
        snp_vcfs.append(out_file)
        pbar.set_description(f"Filtering {vcf} to {out_file}\n")
        if out_file.exists():
            logging.info(f"Out file {out_file} already exists.")
            continue
        status = run_shell(
            f"""
            bcftools view -i 'TYPE="snp" | TYPE="mnp"' -O b --threads {n_threads} {vcf} |\
            bcftools view -e 'ALT="*"' -O b --threads {n_threads} {vcf} |\
            bcftools norm -a -O b --threads {n_threads} |\
            bcftools norm -d none -O b --threads {n_threads} |\
            bcftools annotate --rename-chr {renamer} -O b --threads {n_threads} {vcf} |\
            bcftools norm -f {ref} -O b --threads {n_threads} >|\
            {out_file}
            """
        )
        status = run_shell(f"bcftools index --threads {n_threads} {out_file}")
    return snp_vcfs


def merge_samples(vcfs: list[Path], out_file: Path, n_threads: int):
    pass


def main():
    pass


if __name__ == "__main__":
    typer.run(main)
