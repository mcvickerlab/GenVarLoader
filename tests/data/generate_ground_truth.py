import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Annotated, List

import typer

WDIR = Path(__file__).resolve().parent
SEQ_LEN = 20


def run_shell(cmd: List[str], input=None):
    try:
        prc = subprocess.run(cmd, check=True, capture_output=True, input=input)
    except subprocess.CalledProcessError as e:
        print("Command:", " ".join(e.cmd))
        print("Stdout:", e.stdout.decode())
        print("Error message:", e.stderr.decode())
        raise e
    return prc


def main(
    name: Annotated[
        str,
        typer.Argument(
            help=dedent(
                """
            Prefix for files. There should be a file named <name>.vcf and then this
            script will generate <name>.bed and <name>_<sample>_nr<row_nr>_h<hap_nr>.fa
            files.
            """
            ),
        ),
    ] = "sample",
    reference: Annotated[
        Path,
        typer.Argument(
            help="Path to reference genome.",
        ),
    ] = WDIR / "fasta" / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.bgz",
    out_dir: Path = WDIR / "consensus",
    indels: Annotated[bool, typer.Option(help="Whether to include indels.")] = True,
    structural_variants: Annotated[
        bool, typer.Option(help="Whether to include structural variants.")
    ] = False,
    multiallelic: Annotated[
        bool, typer.Option(help="Whether to allow multiallelic variants.")
    ] = False,
):
    """Generate ground truth variant sequences using `bcftools consensus`."""
    import shutil
    import sys
    from time import perf_counter

    import genvarloader as gvl
    import polars as pl
    import polars.selectors as cs
    from loguru import logger
    from tqdm.auto import tqdm

    log_file = Path(__file__).parent / "generate_ground_truth.log"
    log_file.unlink()
    logger.enable("genvarloader")
    logger.remove(0)
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, level="DEBUG")

    logger.info(
        f"""Running command:
        generate_ground_truth.py {name} {reference} {out_dir} --indels {indels} --structural-variants {structural_variants} --multiallelic {multiallelic}
        """
    )

    t0 = perf_counter()

    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(0o777, parents=True, exist_ok=True)
    vcf_path = WDIR / "vcf" / f"{name}.vcf"
    filtered_vcf = WDIR / "vcf" / f"filtered_{name}.vcf"

    with open(vcf_path, "r") as f:
        vcf = f.read().encode()
    # left-align
    norm_cmd = [
        "bcftools",
        "norm",
        "-f",
        str(reference),
    ]
    result = run_shell(norm_cmd, input=vcf)
    logger.info(f"Left-alignment: {result.stderr.decode()}")
    vcf = result.stdout

    if not multiallelic:
        logger.info("Splitting multiallelic variants.")
        split_multiallelics_cmd = [
            "bcftools",
            "norm",
            "-a",
            "--atom-overlaps",
            ".",
            "-f",
            str(reference),
            "-m",
            "-",
        ]
        result = run_shell(split_multiallelics_cmd, input=vcf)
        logger.info(
            f"Atomizing variants and splitting multiallelics: {result.stderr.decode()}"
        )
        vcf = result.stdout
    if not indels:
        logger.info("Ignoring indels.")
        remove_indel_cmd = [
            "bcftools",
            "view",
            "-e",
            'TYPE="indel"',
        ]
        vcf = run_shell(remove_indel_cmd, input=vcf).stdout
    if not structural_variants:
        logger.info("Ignoring structural variants.")
        remove_sv_cmd = [
            "bcftools",
            "view",
            "-e",
            'TYPE="OTHER"',
        ]
        vcf = run_shell(remove_sv_cmd, input=vcf).stdout
    with open(filtered_vcf, "w+t") as f:
        f.write(vcf.decode())

    # BGZIP the VCF or bcftools consensus errors out
    run_shell(
        [
            "bcftools",
            "view",
            "-O",
            "z",
            "-o",
            str(filtered_vcf.with_suffix(".vcf.gz")),
            str(filtered_vcf),
        ]
    )
    filtered_vcf = filtered_vcf.with_suffix(".vcf.gz")
    run_shell(["bcftools", "index", str(filtered_vcf)])

    logger.info("Generating PGEN file.")
    run_shell(
        [
            "plink2",
            "--vcf",
            str(filtered_vcf),
            "--make-pgen",
            "--vcf-half-call",
            "r",
            "--out",
            str(WDIR / "pgen" / f"filtered_{name}"),
        ]
    )

    bed = pl.read_csv(
        filtered_vcf,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        new_columns=[
            "chrom",
            "pos",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO",
            "FORMAT",
            "NA00001",
            "NA00002",
            "NA00003",
        ],
    )
    samples = bed.select(cs.matches(r"^NA\d{5}$")).columns

    #! when writing the source VCF, make sure that the most distant variants within a group are not more than SEQ_LEN // 2 apart
    #! also ensure that groups of variants are more than SEQ_LEN apart
    #! otherwise, the logic below won't work
    bed = (
        bed.group_by("chrom", maintain_order=True)
        .agg(
            "pos",
            (pl.col("pos").diff().fill_null(0) > SEQ_LEN).cum_sum().alias("group"),
        )
        .explode("pos", "group")
        .group_by("chrom", "group", maintain_order=True)
        .agg(
            start=pl.col("pos").min() - SEQ_LEN // 2,
            end=pl.col("pos").min() + SEQ_LEN // 2,
        )
        .drop("group")
        .sample(fraction=1, shuffle=True, seed=0)
        .with_row_index()
    )

    logger.info("Generating BED file.")
    (
        bed.select("chrom", "start", "end").write_csv(
            WDIR / "vcf" / f"{name}.bed", include_header=False, separator="\t"
        )
    )

    logger.info("Generating consensus sequences.")
    pbar = tqdm(total=bed.height * len(samples) * 2)
    for row in bed.select("index", "chrom", "start", "end").iter_rows():
        row_nr, chrom, start, end = row
        subseq_cmd = [
            "samtools",
            "faidx",
            str(reference),
            f"{chrom}:{start + 1}-{end}",
        ]
        for sample in samples:
            for hap in range(2):
                out_fasta = out_dir / f"{name}_{sample}_nr{row_nr}_h{hap}.fa"
                consensus_cmd = [
                    "bcftools",
                    "consensus",
                    "-H",
                    str(hap + 1),
                    "-s",
                    sample,
                    "-o",
                    str(out_fasta),
                    str(filtered_vcf),
                ]
                seq = run_shell(subseq_cmd)
                run_shell(consensus_cmd, input=seq.stdout)
                index_cmd = ["samtools", "faidx", str(out_fasta)]
                run_shell(index_cmd)
                pbar.update()
    pbar.close()

    bed = WDIR / "vcf" / f"{name}.bed"

    logger.info("Generating phased dataset.")
    gvl.write(
        path=WDIR / "phased_dataset.gvl",
        bed=bed,
        variants=filtered_vcf,
        overwrite=True,
    )

    logger.info("Generating unphased dataset.")
    gvl.write(
        path=WDIR / "unphased_dataset.gvl",
        bed=bed,
        variants=filtered_vcf,
        overwrite=True,
        phased=False,
        dosage_field="VAF",
    )

    logger.info(f"Finished in {perf_counter() - t0} seconds.")


if __name__ == "__main__":
    typer.run(main)
