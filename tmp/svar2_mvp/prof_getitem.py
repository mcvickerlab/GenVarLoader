"""Profile the LIVE SVAR2 read-bound Dataset.__getitem__ path (not the union
oracle) for the in-scope modes. One (mode, cohort) per process so cProfile/perf
attribute cleanly.

  python tmp/svar2_mvp/prof_getitem.py <haplotypes|variants> <germline|somatic> <K>

gvl.write + Dataset.open run ONCE (we profile the READ, not the write). Prints
per_call_s over K warm calls. Tracks mode is out of scope; variant-windows is
guarded NotImplementedError in Svar2Haps and cannot be profiled yet."""
import sys
import time
from pathlib import Path

import polars as pl

STORE_DIR = Path("/carter/users/dlaub/projects/svar2_mvp")
REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
CHROM = "chr21"
REGIONS = [(20_000_000, 20_001_000), (30_000_000, 30_000_500), (40_000_000, 40_001_000)]
WORK = Path("tmp/svar2_mvp/prof_out/readbound")


def _bed():
    return pl.DataFrame({
        "chrom": [CHROM] * len(REGIONS),
        "chromStart": [s for s, _ in REGIONS],
        "chromEnd": [e for _, e in REGIONS],
    })


def make_call(mode, cohort):
    import genvarloader as gvl
    from genoray import SparseVar2

    prefix = STORE_DIR / cohort
    sv2 = SparseVar2(f"{prefix}.svar2")
    n_s = sv2.n_samples
    ds_path = WORK / f"{cohort}_{mode}.gvl"
    WORK.mkdir(parents=True, exist_ok=True)

    gvl.write(ds_path, _bed(), variants=SparseVar2(f"{prefix}.svar2"),
              samples=None, max_jitter=0, overwrite=True)
    ds = gvl.Dataset.open(ds_path, reference=REF)
    view = ds.with_seqs(mode)   # "haplotypes" or "variants"

    R = len(REGIONS)

    def call():
        view[:R, :n_s]

    return call


def main(mode, cohort, K):
    call = make_call(mode, cohort)
    call()  # warm
    t0 = time.perf_counter()
    for _ in range(K):
        call()
    print(f"per_call_s={(time.perf_counter() - t0) / K:.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
