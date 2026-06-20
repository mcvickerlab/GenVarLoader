"""Build a large reproducible bigWig corpus for write/update benchmarking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `tests` is importable when run standalone
# (pytest adds "." via pythonpath config, but pixi run doesn't).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._bigwig_corpus import DEFAULT_CONTIGS, make_synthetic_bigwigs  # noqa: E402

OUT = Path(__file__).resolve().parent / "bigwig_corpus"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--density", type=float, default=0.05)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = make_synthetic_bigwigs(
        OUT, n_samples=args.n_samples, contigs=DEFAULT_CONTIGS, density=args.density
    )
    print(f"wrote {len(paths)} bigWigs to {OUT}")


if __name__ == "__main__":
    main()
