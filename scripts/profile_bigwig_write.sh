#!/usr/bin/env bash
# py-spy needs sudo on macOS. Run this yourself; do not let the agent invoke py-spy.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=tests/benchmarks/profiling
pixi run -e dev python tests/benchmarks/data/build_bigwig_corpus.py
for impl in legacy rust; do
  sudo py-spy record -o "$OUT/bigwig_write_$impl.speedscope.json" -f speedscope -- \
    pixi run -e dev python "$OUT/profile_bigwig_write.py" --impl "$impl"
done
echo "wrote $OUT/bigwig_write_{legacy,rust}.speedscope.json"
