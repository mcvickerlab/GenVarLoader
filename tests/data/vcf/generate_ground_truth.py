import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Annotated, List

import typer


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
            )
        ),
    ],
    reference: Annotated[
        Path,
        typer.Argument(
            help="""
        Path to reference genome.
        """
        ),
    ],
    out_dir: Path,
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
    from time import perf_counter

    import polars as pl
    import polars.selectors as cs
    from loguru import logger
    from tqdm.auto import tqdm

    logger.add(Path(__file__).parent / "generate_ground_truth.log", level="INFO")

    logger.info(
        f"""Running command:
        generate_ground_truth.py {name} {reference} {out_dir} --indels {indels} --structural-variants {structural_variants} --multiallelic {multiallelic}
        """
    )

    t0 = perf_counter()

    SEQ_LEN = 10

    out_dir = out_dir.resolve()
    shutil.rmtree(out_dir)
    out_dir.mkdir(0o777, parents=True, exist_ok=True)
    prefix = Path(__file__).parent / name
    vcf_path = str(prefix.with_suffix(".vcf"))
    filtered_vcf = Path.cwd() / f"filtered_{name}.vcf"

    with open(vcf_path, "r") as f:
        vcf = f.read().encode()
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
        vcf = run_shell(split_multiallelics_cmd, input=vcf).stdout
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
            str(Path.cwd().parent / "pgen" / "sample"),
        ]
    )

    bed = (
        pl.read_csv(
            filtered_vcf,
            separator="\t",
            comment_prefix="#",
            has_header=False,
            new_columns=[
                "#CHROM",
                "POS",
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
        .rename({"#CHROM": "CHROM"})
        .with_columns(
            start=pl.col("POS") - SEQ_LEN // 2, end=pl.col("POS") + -(-SEQ_LEN // 2)
        )
        .with_row_count()
    )

    samples = bed.select(cs.matches(r"^NA\d{5}$")).columns
    logger.info("Generating BED file.")
    (
        bed.select("CHROM", "start", "end").write_csv(
            prefix.with_suffix(".bed"), include_header=False, separator="\t"
        )
    )

    logger.info("Generating consensus sequences.")
    pbar = tqdm(total=bed.height * len(samples) * 2)
    for row in bed.select("row_nr", "CHROM", "start", "end").iter_rows():
        row_nr, chrom, start, end = row
        subseq_cmd = [
            "samtools",
            "faidx",
            str(reference),
            f"{chrom}:{start+1}-{end}",
        ]
        for sample in samples:
            for hap in range(1, 3):
                out_fasta = out_dir / f"{name}_{sample}_nr{row_nr}_h{hap}.fa"
                consensus_cmd = [
                    "bcftools",
                    "consensus",
                    "-H",
                    str(hap),
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

    logger.info(f"Finished in {perf_counter() - t0} seconds.")

    # filtered_vcf.with_suffix('').unlink() # remove .vcf
    # filtered_vcf.unlink() # remove .vcf.gz
    # filtered_vcf.with_suffix('.gz.csi').unlink() # remove .vcf.gz.csi


if __name__ == "__main__":
    typer.run(main)
