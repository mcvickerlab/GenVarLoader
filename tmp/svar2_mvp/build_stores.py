"""Build .svar (SVAR1) and .svar2 (SVAR2) stores from a normalized biallelic BCF."""
import sys
from pathlib import Path

from genoray import VCF, SparseVar, _core

def build(bcf: str, chrom: str, samples: list[str], out_prefix: str, ploidy: int):
    bcf = str(bcf)
    # SVAR 1.0
    SparseVar.from_vcf(f"{out_prefix}.svar", VCF(bcf), "8g", overwrite=True)
    # SVAR 2.0
    _core.run_conversion_pipeline(
        bcf, "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa",
        [chrom], f"{out_prefix}.svar2", samples,
        25_000, ploidy, 8, 8 * 1024 * 1024,
    )
    print(f"built {out_prefix}.svar and {out_prefix}.svar2")

if __name__ == "__main__":
    # argv: <norm.bcf> <chrom> <out_prefix>
    bcf, chrom, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    import subprocess
    samples = subprocess.run(
        ["bcftools", "query", "-l", bcf], capture_output=True, text=True, check=True
    ).stdout.split()
    build(bcf, chrom, samples, out_prefix, ploidy=2)
