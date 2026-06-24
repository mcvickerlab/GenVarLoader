#!/usr/bin/env bash
# py-spy requires sudo on macOS. Hand-off script for David.
#
# Prerequisites:
#   pixi run -e dev python tests/benchmarks/data/build_realistic.py
#
# Usage:
#   sudo bash tests/benchmarks/profiling/baseline_getitem.sh
#
# Writes per-mode speedscope JSON to tests/benchmarks/profiling/:
#   haplotypes.speedscope.json
#   tracks.speedscope.json
#   variants.speedscope.json
#
# After running, open each .speedscope.json at https://speedscope.app and
# note the total sample count shown at bottom-left. Together with the wall
# clock printed by profile.py, that gives the baseline getitem throughput.
set -euo pipefail
cd "$(dirname "$0")/../../.."

PY="$(pixi run -e dev which python)"
PYSPY="$(pixi run -e dev which py-spy)"
OUT="tests/benchmarks/profiling"
DRIVER="$OUT/profile.py"

DS="tests/benchmarks/data/chr22_geuv.gvl"
if [ ! -d "$DS" ]; then
    echo "ERROR: Dataset $DS not found."
    echo "Run: pixi run -e dev python tests/benchmarks/data/build_realistic.py"
    exit 1
fi

for mode in haplotypes tracks variants; do
    echo "=== py-spy $mode ==="
    "$PYSPY" record \
        -o "$OUT/${mode}.speedscope.json" \
        -f speedscope \
        -- "$PY" "$DRIVER" --mode "$mode"
    echo "Wrote $OUT/${mode}.speedscope.json"
    echo "  -> open in https://speedscope.app; note total samples (bottom-left)"
    echo "  -> wall-clock and throughput are printed above by profile.py"
done

echo ""
echo "All done. Speedscope files:"
for mode in haplotypes tracks variants; do
    echo "  $OUT/${mode}.speedscope.json"
done
