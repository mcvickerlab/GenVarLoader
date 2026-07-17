"""pyinstrument profiler for the SVAR2 spliced-variant decode (PR #286).

Builds one bulk fixture and profiles repeated spliced ``with_seqs("variants")``
getitems. pyinstrument is a Python statistical profiler: the Python-side regroup in
``_query._fetch_spliced_variants`` shows as call frames, while time inside the Rust
SVAR2 decode aggregates into the extension call -- so the split between "Python
regroup" and "Rust kernel" is directly readable, which is the Phase-4 hypothesis
test (memory: svar2-spliced-gather-bottleneck says the Python gather dominates).

    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev python \
        tests/benchmarks/profiling/profile_svar2_splice_variants.py \
        --work /scratch/svar2bench --samples 25000 --transcripts 256 \
        --threads 8 --iters 20 --html out.html
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--samples", type=int, default=25_000)
    p.add_argument("--transcripts", type=int, default=256)
    p.add_argument("--records", type=int, default=20_000)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--html", type=Path, default=None)
    args = p.parse_args()

    os.environ["GVL_FORCE_PARALLEL"] = "1"
    os.environ["GVL_NUM_THREADS"] = str(args.threads)

    import genvarloader as gvl
    from pyinstrument import Profiler

    from tests.benchmarks.data.build_svar2_splice_bulk import build

    fx = build(
        args.work / "fixtures",
        n_samples=args.samples,
        n_transcripts=args.transcripts,
        records=args.records,
    )
    ds = (
        gvl.Dataset.open(fx.gvl_path, reference=fx.reference)
        .with_settings(
            splice_info=("transcript_id", "exon_number"), var_filter="exonic"
        )
        .with_seqs("variants")
    )
    idx = (slice(0, args.transcripts), slice(0, args.samples))
    for _ in range(args.warmup):
        ds[idx]

    prof = Profiler(interval=0.001)
    prof.start()
    for _ in range(args.iters):
        ds[idx]
    prof.stop()
    print(prof.output_text(unicode=True, color=False, show_all=True), flush=True)
    if args.html is not None:
        args.html.write_text(prof.output_html())
        print(f"wrote {args.html}", flush=True)


if __name__ == "__main__":
    main()
