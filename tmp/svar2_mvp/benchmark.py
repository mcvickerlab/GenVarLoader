"""Benchmark SVAR1 (gvl Dataset over .svar) vs SVAR2 (SparseVar2Source over
.svar2): hap latency, variant latency, store size, for one source prefix.
Fair workload: ALL samples for a fixed region set. Warm caches, median of N."""

import sys
import time
import subprocess
from statistics import median

import numpy as np
import genvarloader as gvl
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source

REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
N = 5  # repeats


def _contig_ref(fasta, chrom):
    import pysam

    return pysam.FastaFile(fasta).fetch(chrom).encode()


def _timed(fn, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return median(ts)


def main(prefix, chrom):
    regions = [
        (20_000_000, 20_001_000),
        (30_000_000, 30_000_500),
        (40_000_000, 40_001_000),
    ]
    ref_bytes = _contig_ref(REF, chrom)
    ref_u8 = np.frombuffer(ref_bytes, np.uint8)
    ref_off = np.array([0, len(ref_bytes)], np.int64)

    # SVAR2 backend
    sv2 = SparseVar2(f"{prefix}.svar2")
    src = SparseVar2Source(sv2)
    svar2_hap = _timed(
        lambda: src.reconstruct(
            chrom,
            regions,
            ref_u8,
            ref_off,
            pad_char=ord("N"),
            shifts=None,
            output_length=-1,
        )
    )
    svar2_var = _timed(lambda: sv2.decode(chrom, regions))

    # SVAR1 backend (all samples, same regions)
    import polars as pl

    bed = pl.DataFrame(
        {
            "chrom": [chrom] * len(regions),
            "chromStart": [s for s, _ in regions],
            "chromEnd": [e for _, e in regions],
        }
    )
    ds_path = f"{prefix}.gvl"
    # Write the Dataset over the SAME region set the SVAR2 path benchmarks, so both
    # backends measure an identical workload (fairness rule). validate.py may have left
    # a .gvl with a different region set, so always (re)write here.
    gvl.write(ds_path, bed, variants=f"{prefix}.svar", overwrite=True)
    ds = gvl.Dataset.open(ds_path, reference=REF)
    ds_hap = ds.with_seqs("haplotypes")
    ds_var = ds.with_seqs("variants")
    n_s = sv2.n_samples
    svar1_hap = _timed(lambda: ds_hap[: len(regions), :n_s])
    svar1_var = _timed(lambda: ds_var[: len(regions), :n_s])

    def du(path):
        return subprocess.run(
            ["du", "-sb", path], capture_output=True, text=True
        ).stdout.split()[0]

    print(
        f"source={prefix.split('/')[-1]} chrom={chrom} n_samples={n_s} "
        f"regions={len(regions)} N={N}"
    )
    print(f"  hap_latency_s   svar1={svar1_hap:.4f}  svar2={svar2_hap:.4f}")
    print(f"  var_latency_s   svar1={svar1_var:.4f}  svar2={svar2_var:.4f}")
    print(
        f"  store_bytes     svar1={du(prefix + '.svar')}  svar2={du(prefix + '.svar2')}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])  # argv: <prefix> <chrom>
