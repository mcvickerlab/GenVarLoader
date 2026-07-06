#!/usr/bin/env bash
# Native-layer attribution for the live read-bound path via perf (py-spy is
# unusable: ptrace_scope=2; Python 3.10 has no perf trampoline so Python frames
# are opaque -> DSO-level + Rust-symbol self-time is the split).
set -eu
cd "$(git rev-parse --show-toplevel)"
OUT=tmp/svar2_mvp/prof_out/readbound
mkdir -p "$OUT"
PERF=/carter/users/dlaub/.pixi/bin/perf
PY=.pixi/envs/dev/bin/python
FREQ=299
REPORT="$OUT/native_baseline.md"
echo "# SVAR2 read-bound native baseline ($(date -I))" > "$REPORT"

probe_K () {  # mode cohort -> K sized to ~40s
  local per
  per=$("$PY" tmp/svar2_mvp/prof_getitem.py "$1" "$2" 5 | sed 's/per_call_s=//')
  "$PY" -c "import math;print(max(20,math.ceil(40/max(float('$per'),1e-4))))"
}

for c in germline somatic; do for m in haplotypes variants; do
  tag="${m}_${c}"; K=$(probe_K "$m" "$c")
  echo "## $tag (K=$K)" | tee -a "$REPORT"
  # instruction-count reference (the Phase-B gate baseline)
  echo '### perf stat' >> "$REPORT"
  { "$PERF" stat -e instructions,cycles -- "$PY" tmp/svar2_mvp/prof_getitem.py "$m" "$c" "$K" ; } \
    2>> "$REPORT" 1>/dev/null || echo "(perf stat HW counters unavailable)" >> "$REPORT"
  # sampling profile
  "$PERF" record -g --call-graph fp -F $FREQ -o "$OUT/$tag.data" -- \
    "$PY" tmp/svar2_mvp/prof_getitem.py "$m" "$c" "$K" >/dev/null 2>&1
  echo '### DSO split' >> "$REPORT"; echo '```' >> "$REPORT"
  "$PERF" report --stdio --sort=dso --no-children -g none -i "$OUT/$tag.data" 2>/dev/null \
    | grep -vE '^\s*#|^\s*$' | head -12 >> "$REPORT"; echo '```' >> "$REPORT"
  echo '### top Rust/native self-time symbols' >> "$REPORT"; echo '```' >> "$REPORT"
  "$PERF" report --stdio --sort=symbol --no-children -g none -i "$OUT/$tag.data" 2>/dev/null \
    | grep -vE '^\s*#|^\s*$' | head -20 >> "$REPORT"; echo '```' >> "$REPORT"
  echo '### call graph (top)' >> "$REPORT"; echo '```' >> "$REPORT"
  "$PERF" report --stdio --sort=overhead,symbol -i "$OUT/$tag.data" 2>/dev/null \
    | grep -vE '^\s*#|^\s*$' | head -45 >> "$REPORT"; echo '```' >> "$REPORT"
done; done
echo "NATIVE_BASELINE_DONE -> $REPORT"
