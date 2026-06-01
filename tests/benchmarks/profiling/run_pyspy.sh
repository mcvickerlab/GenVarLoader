#!/usr/bin/env bash
# py-spy requires root on macOS. Run:  sudo bash tests/benchmarks/profiling/run_pyspy.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="$(pixi run -e dev which python)"
PYSPY="$(pixi run -e dev which py-spy)"
OUT=tests/benchmarks/profiling
for mode in tracks haplotypes variants; do
  echo "=== py-spy $mode ==="
  "$PYSPY" record -o "$OUT/${mode}.speedscope.json" -f speedscope -- \
    "$PY" "$OUT/profile.py" --mode "$mode"
done
echo "Wrote $OUT/{tracks,haplotypes,variants}.speedscope.json"
