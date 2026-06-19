"""Time + measure gvl.write() for a bigWig track (legacy vs rust).

  pixi run -e dev python tests/benchmarks/profiling/profile_bigwig_write.py --impl rust
  pixi run -e dev python tests/benchmarks/profiling/profile_bigwig_write.py --impl legacy

Set GVL_RUST_BIGWIG_WRITE via --impl. Reports wall-clock; run under memray for RSS.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so `tests` is importable when run standalone
# (pytest adds "." via pythonpath config, but pixi run doesn't).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CORPUS = Path(__file__).resolve().parents[1] / "data" / "bigwig_corpus"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["legacy", "rust"], required=True)
    p.add_argument("--n-regions", type=int, default=2000)
    args = p.parse_args()
    os.environ["GVL_RUST_BIGWIG_WRITE"] = "1" if args.impl == "rust" else "0"

    if not CORPUS.exists():
        raise SystemExit(
            "Corpus missing. Run "
            "`pixi run -e dev python tests/benchmarks/data/build_bigwig_corpus.py`."
        )

    import tempfile

    import genvarloader as gvl
    from genvarloader._dataset._write import _write_track
    from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions

    paths = sorted(CORPUS.glob("sample_*.bw"))
    samples = [p.stem for p in paths]
    track = gvl.BigWigs("signal", {s: str(p) for s, p in zip(samples, paths)})
    per_contig = max(1, args.n_regions // len(DEFAULT_CONTIGS))
    bed = make_regions(DEFAULT_CONTIGS, n_per_contig=per_contig, width=5000, seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "signal"
        out.mkdir()
        t0 = time.perf_counter()
        _write_track(out, bed, track, samples, 4 << 30)
        dt = time.perf_counter() - t0
    print(f"impl={args.impl} regions={bed.height} samples={len(samples)} wall={dt:.3f}s")


if __name__ == "__main__":
    main()
