#!/usr/bin/env bash
# E1: per-(backend x cohort) query-latency attribution via perf ONLY.
# py-spy is unusable here (ptrace_scope=2, no sudo). perf works (paranoid=2, uses perf_event).
#
# Profile the env's python DIRECTLY (.pixi/envs/default/bin/python) — NOT via `pixi run`:
# the pixi launcher otherwise eats ~60% of samples, and the extensions import fine standalone
# (RPATH handles deps). Large K so the steady-state reconstruct loop drowns import/startup
# (genvarloader pulls in torch — a heavy one-time import).
#
# Python frames are opaque on 3.10, so the split is at the DSO level (self-time):
#   gvl genvarloader.abi3.so + genoray _core.so = native Rust hot path
#   numpy _multiarray_umath*.so                  = conversion/ascontiguousarray overhead
#   python3.10 / libpython                        = interpreter/orchestration
# Frame-pointer call graph (built with -C force-frame-pointers=yes); DWARF overloads the node.
# Runs DIRECTLY on the current 2-cpu carter-cn-04 allocation (no srun).
# NOTE: no `pipefail` — the `| head` truncations send SIGPIPE (141) upstream to perf report,
# which under pipefail+set-e would abort the whole sweep after the first combo.
set -eu
cd "$(git rev-parse --show-toplevel)"
OUT=tmp/svar2_mvp/prof_out/e1
mkdir -p "$OUT"
PERF=/carter/users/dlaub/.pixi/bin/perf
PY=.pixi/envs/default/bin/python
TARGET_S=60           # ~60s hot loop per capture so startup is negligible
FREQ=199

probe_K () {          # backend cohort -> K sized to ~TARGET_S
  local per
  per=$("$PY" tmp/svar2_mvp/prof_driver.py "$1" "$2" 5 | sed 's/per_call_s=//')
  "$PY" -c "import math; print(max(20, math.ceil($TARGET_S/max(float('$per'),1e-4))))"
}

: > "$OUT/dso_split.txt"; : > "$OUT/perf_top.txt"; : > "$OUT/callgraph.txt"; : > "$OUT/K_used.txt"

for b in svar2 svar1; do for c in germline somatic; do
  tag="${b}_${c}"
  K=$(probe_K "$b" "$c"); echo "$tag K=$K" | tee -a "$OUT/K_used.txt"
  $PERF record -g --call-graph fp -F $FREQ -o "$OUT/${tag}.perf.data" -- \
    "$PY" tmp/svar2_mvp/prof_driver.py "$b" "$c" "$K" >/dev/null 2>&1
  echo "== DSO split $tag ==" | tee -a "$OUT/dso_split.txt"
  $PERF report --stdio --sort=dso --no-children -g none -i "$OUT/${tag}.perf.data" 2>/dev/null \
    | "$PY" tmp/svar2_mvp/e1_bucket_dso.py | tee -a "$OUT/dso_split.txt"
  echo "== top self-time symbols $tag ==" | tee -a "$OUT/perf_top.txt"
  $PERF report --stdio --sort=symbol --no-children -g none -i "$OUT/${tag}.perf.data" 2>/dev/null \
    | grep -vE '^\s*#|^\s*$' | head -15 | tee -a "$OUT/perf_top.txt"
  # call-graph for the svar2 paths (shows e.g. SearchTree::build <- overlap_batch)
  if [ "$b" = svar2 ]; then
    echo "== call graph $tag (top) ==" >> "$OUT/callgraph.txt"
    $PERF report --stdio --sort=overhead,symbol -i "$OUT/${tag}.perf.data" 2>/dev/null \
      | grep -vE '^\s*#|^\s*$' | head -40 >> "$OUT/callgraph.txt"
  fi
done; done
echo "E1_PROFILE_DONE  dso -> $OUT/dso_split.txt  symbols -> $OUT/perf_top.txt"
