"""SVAR2 streaming: synchronous-path cold-cache baseline + IO-vs-CPU bound split.

MEASURES ONLY. Uses the vcfixture-rs bulk builder for cohort-scale stores; skips
cleanly if vcfixture/bcftools/samtools are absent. Reports best-of-N on a shared,
noisy node -- treat ratios as color, not a gate (project perf-gate convention).

Run:
    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev \
        python benchmarking/streaming/svar2_cold_cache.py --samples 500 2000 --records 20000
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import genvarloader as gvl


def _sweep(samples_list, records, repeats):
    import sys

    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from tests.benchmarks.data.build_svar2_stream_bulk import build
    import pyinstrument

    for n in samples_list:
        best = float("inf")
        tmp = None
        try:
            for rep in range(repeats):
                tmp = Path(tempfile.mkdtemp())  # fresh inode -> never-faulted
                fx = build(tmp, n_samples=n, records=records, seed=1000 + rep)
                sds = gvl.StreamingDataset(
                    fx.bed, reference=fx.reference, variants=fx.svar2_path
                ).with_seqs("haplotypes")
                t0 = time.perf_counter()
                for _ in sds.to_iter(batch_size=32):
                    pass
                best = min(best, time.perf_counter() - t0)
                # Keep the LAST rep's store alive for the pyinstrument sweep below;
                # earlier reps' stores are done being used, so clean them up now.
                if rep < repeats - 1:
                    shutil.rmtree(tmp, ignore_errors=True)
            print(f"n_samples={n:>6}: synchronous best-of-{repeats} = {best:.3f}s")

            # IO-vs-CPU split on the LAST rep's store, one sweep under pyinstrument.
            prof = pyinstrument.Profiler()
            prof.start()
            for _ in sds.to_iter(batch_size=32):
                pass
            prof.stop()
            print(prof.output_text(unicode=True, color=False))
            print(
                "  -> read `_find_ranges` (search) vs gather+kernel (fill) share above."
            )
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--records", type=int, default=20_000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    try:
        _sweep(args.samples, args.records, args.repeats)
    except FileNotFoundError as e:
        print(f"SKIP: {e}")


if __name__ == "__main__":
    main()
