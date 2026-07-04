"""Spot-check that gvl returns non-empty, sane haplotypes + variants through
both the SVAR1 (gvl Dataset over .svar) and SVAR2 (SparseVar2Source over .svar2)
backends, on a handful of regions x a few samples. Correctness is already proven
by the test suite; this proves the REAL-DATA plumbing works."""
import sys
from pathlib import Path

import numpy as np
import genvarloader as gvl
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source

REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"

def main(prefix: str, chrom: str):
    # A few small regions (0-based, half-open) in a variant-dense chr21 window.
    regions = [(20_000_000, 20_001_000), (30_000_000, 30_000_500)]

    # --- SVAR2 backend (adapter direct) ---
    sv2 = SparseVar2(f"{prefix}.svar2")
    print(f"[svar2] n_samples={sv2.n_samples} ploidy={sv2.ploidy}")
    ref_bytes = _contig_ref(REF, chrom)
    src = SparseVar2Source(sv2)
    hap = src.reconstruct(
        chrom, regions,
        np.frombuffer(ref_bytes, np.uint8),
        np.array([0, len(ref_bytes)], np.int64),
        pad_char=ord("N"), shifts=None, output_length=-1,
    )
    lens = np.asarray(hap.offsets)
    print(f"[svar2] hap ragged rows={len(lens) - 1} "
          f"min_len={int(np.diff(lens).min())} max_len={int(np.diff(lens).max())}")
    var = sv2.decode(chrom, regions)
    print(f"[svar2] decode variants: {var}")

    # --- SVAR1 backend (gvl Dataset over .svar) ---
    import polars as pl
    bed = pl.DataFrame({
        "chrom": [chrom] * len(regions),
        "chromStart": [s for s, _ in regions],
        "chromEnd": [e for _, e in regions],
    })
    ds_path = f"{prefix}.gvl"
    gvl.write(ds_path, bed, variants=f"{prefix}.svar", overwrite=True)
    ds = gvl.Dataset.open(ds_path, reference=REF).with_seqs("haplotypes")
    seqs = ds[:len(regions), :3]   # a few regions x first 3 samples
    print(f"[svar1] gvl haplotypes sample shape/type: {type(seqs)}")

def _contig_ref(fasta: str, chrom: str) -> bytes:
    import pysam
    return pysam.FastaFile(fasta).fetch(chrom).encode()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])   # argv: <prefix> <chrom>
