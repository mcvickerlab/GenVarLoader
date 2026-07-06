"""E1 single-path profiling driver. Exercises ONE code path in a warm loop so
py-spy/perf attribute time to that path only.

  python prof_driver.py <svar1|svar2> <germline|somatic> <K>

Prints: per_call_s=<median-ish mean over K warm calls>
For svar1, the 3-region .gvl is written ONCE before the loop (we profile the
query, not gvl.write)."""

import sys
import time

import numpy as np

W = "/carter/users/dlaub/repos/for_loukik/svar2_mvp"
REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
CHROM = "chr21"
REGIONS = [(20_000_000, 20_001_000), (30_000_000, 30_000_500), (40_000_000, 40_001_000)]


def _ref():
    import pysam

    rb = pysam.FastaFile(REF).fetch(CHROM).encode()
    return np.frombuffer(rb, np.uint8), np.array([0, len(rb)], np.int64)


def make_svar2(cohort):
    from genoray import SparseVar2
    from genvarloader._dataset._svar2_source import SparseVar2Source

    src = SparseVar2Source(SparseVar2(f"{W}/{cohort}.svar2"))
    ru, ro = _ref()

    def call():
        src.reconstruct(
            CHROM, REGIONS, ru, ro, pad_char=ord("N"), shifts=None, output_length=-1
        )

    return call


def make_svar1(cohort):
    import polars as pl
    import genvarloader as gvl
    from genoray import SparseVar2

    n_s = SparseVar2(f"{W}/{cohort}.svar2").n_samples
    bed = pl.DataFrame(
        {
            "chrom": [CHROM] * len(REGIONS),
            "chromStart": [s for s, _ in REGIONS],
            "chromEnd": [e for _, e in REGIONS],
        }
    )
    ds_path = f"{W}/{cohort}.gvl"
    gvl.write(ds_path, bed, variants=f"{W}/{cohort}.svar", overwrite=True)  # ONCE
    ds_hap = gvl.Dataset.open(ds_path, reference=REF).with_seqs("haplotypes")

    def call():
        ds_hap[: len(REGIONS), :n_s]

    return call


def main(backend, cohort, K):
    call = {"svar1": make_svar1, "svar2": make_svar2}[backend](cohort)
    call()  # warm
    t0 = time.perf_counter()
    for _ in range(K):
        call()
    dt = time.perf_counter() - t0
    print(f"per_call_s={dt / K:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
