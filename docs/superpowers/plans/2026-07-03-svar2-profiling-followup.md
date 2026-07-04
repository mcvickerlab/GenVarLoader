# SVAR2 Profiling Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute the SVAR2 adapter's query-latency gap to Python overhead vs Rust hot paths vs memory layout, and find the conversion thread-split policy for read-bound (few-contig) builds — measurement only, no optimization.

**Architecture:** Add small parametrized *profiling driver* scripts under `tmp/svar2_mvp/` that exercise a single code path (SVAR1 `Dataset` hap query, SVAR2 adapter `reconstruct`, or genoray `run_conversion_pipeline`) in a warm loop. Profile them with py-spy (Python-vs-native split, any Python version) and perf (Rust symbol detail, needs debug-symbolized extensions). Accumulate every result table into one notes file. All timed runs go on a dedicated `carter-compute` node.

**Tech Stack:** Python 3.10.20 via `pixi run -e default`; genoray 2.15.0 (Rust/PyO3, source at `/carter/users/dlaub/projects/genoray`); gvl Rust extension in this repo's `src/`; py-spy 0.x (sampling profiler); Linux `perf`; SLURM (`carter-compute`); bcftools.

## Global Constraints

Copied verbatim from the spec — every task's requirements implicitly include these.

- **This is measurement only.** No optimization, no Dataset wiring (Task B), no dense-layout change, no conversion rebalance. The output is *numbers that decide whether/how to do those*.
- **Env:** `pixi run -e default` (the only installed env). **Python 3.10.20.**
- **perf cannot symbolize Python frames on Python < 3.12.** This env is 3.10, so `perf` shows resolved **Rust/native** symbols but **opaque** Python frames (`_PyEval_EvalFrameDefault`). Use **py-spy** for the Python-vs-native split and Python hotspots; use **perf** only for Rust symbol detail once py-spy says native dominates.
- **Profilers (confirmed present):** py-spy at `.pixi/envs/default/bin/py-spy`; perf at `/carter/users/dlaub/.pixi/bin/perf`.
- **Compute execution reality (discovered at run start):** this session already runs **inside** a small interactive SLURM allocation on `carter-cn-04` (**2 CPUs, 8 GB RAM**, 14-day walltime, governor=performance, paranoid=2). Therefore: **run light single-threaded profiling directly on this node** — do **NOT** wrap it in `srun` (an in-allocation `srun` makes a constrained 2-CPU *step* and fails with "More processors requested than permitted" at >2 cpus). Route **heavy multi-core / large-RAM work through fresh `sbatch` jobs** (independent of this allocation) sized to the big partition nodes (carter-cn-02/03/04 = 96–128 CPU, 476–953 GB). Everywhere a task below says `srun … <cmd>`, execute `<cmd>` **directly on the node** instead (the 3-region E1/E2-bench/E3 workloads are single-threaded and fit in 8 GB). Keep `sbatch` for E2 store builds and the E4 thread sweep, adding explicit `--cpus-per-task` and `--mem`.
- **Noise control:** compute node via `sbatch -p carter-compute` for heavy jobs; do **NOT** use `--exclusive` — Carter has only 3 compute nodes, always shared, so `--exclusive` won't schedule. The node is shared and noisy, so **absolute wall-clock is not comparable across allocations** — only *relative* comparisons **within one short allocation on the same hardware** are valid (matches the prior perf-gate lesson: gate on same-session before/after, not absolute time). Therefore: **run both backends of any comparison inside a single `srun`** (back-to-back on the same node), warm caches, **median of N≥5**, record CPU governor / turbo. CPU governor on the target nodes is `performance` — record the actual value observed on the node you land on. Keep the MVP workload (same 3 chr21 regions, all samples) for continuity.
- **MVP workload regions** (0-based half-open, chr21): `(20_000_000, 20_001_000)`, `(30_000_000, 30_000_500)`, `(40_000_000, 40_001_000)`.
- **Stores (already built), `$W`:** `/carter/users/dlaub/repos/for_loukik/svar2_mvp` — `{germline,somatic}.{svar,svar2,gvl}` + `{chr21,gdc.chr21}.norm.filt.bcf`. germline = 3202 samples (1000G, high-AF/dense); somatic = 16007 samples (GDC, rare/sparse).
- **Reference FASTA:** `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`.
- **Contig:** both cohorts are chr21-only. germline chrom label and somatic chrom label are both `chr21`.
- **Working dir:** the gvl worktree `/carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel`. Scripts live in its `tmp/svar2_mvp/`.
- **Deliverable notes file:** `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md`, holding the E1–E4 tables + two recommendations.
- **RTK:** prefix git/build shell commands with `rtk` per project CLAUDE.md.
- **Commit style:** conventional commits (this is a `chore`/`docs` profiling branch — driver scripts are `chore`, notes are `docs`).

---

## File Structure

All new files land under `tmp/svar2_mvp/` (profiling drivers, throwaway — not gold-plated) except the notes deliverable.

- `tmp/svar2_mvp/prof_driver.py` — **E1.** Parametrized single-path driver: `python prof_driver.py <backend> <cohort> <K>`. Warms once, then loops the chosen path K times. For SVAR1 it writes the 3-region `.gvl` once *outside* the loop (so we profile the query, not `gvl.write`). Prints `per_call_s=<x>`.
- `tmp/svar2_mvp/split_folded.py` — **E1/E4.** Post-processes a py-spy folded (`--format raw`) stack file into a **leaf-attributed Python% vs native%** split, plus top-N leaf frames per class.
- `tmp/svar2_mvp/e1_profile.sh` — **E1.** Orchestrates py-spy (flamegraph SVG + speedscope + folded raw) and perf (dwarf call-graph) captures for each `(backend × cohort)` on the compute node; drops artifacts in `tmp/svar2_mvp/prof_out/e1/`.
- `tmp/svar2_mvp/e2_subsample.sh` — **E2.** Builds per-S sample lists and subsampled BCFs (`bcftools view -S`).
- `tmp/svar2_mvp/e2_build.sbatch` — **E2.** SLURM array: for each S, build `.svar` + `.svar2` (reuses `build_stores.py`).
- `tmp/svar2_mvp/e2_bench.py` — **E2.** Hap-latency-vs-S sweep for both backends (warm, median N=5), emits a TSV.
- `tmp/svar2_mvp/e3_probe.py` — **E3 (conditional).** Sweeps SVAR2 hap latency vs (a) region width and (b) `n_dense_variants`.
- `tmp/svar2_mvp/e4_convert_driver.py` — **E4.** Runs one single-contig `run_conversion_pipeline` with a given `max_threads`, under optional `GENORAY_SAMPLE_INTERVAL`. Prints build wall-clock.
- `tmp/svar2_mvp/e4_sweep.sbatch` — **E4.** SLURM: sweep `max_threads` on a fixed single-contig input; capture wall-clock + genoray's reported thread split.
- `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` — **the deliverable.** E1–E4 tables + two recommendations.

---

## Task 0: Symbolized-build setup + notes skeleton + node baseline

**Files:**
- Create: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md`
- Create: `tmp/svar2_mvp/env_baseline.txt` (recorded node/env state)
- No source edits — we inject debug info via Cargo env vars, not Cargo.toml edits.

**Interfaces:**
- Consumes: nothing.
- Produces: a gvl extension and a genoray wheel both built `--release` **with** `debug = line-tables-only` + frame pointers, installed into the `default` env; a notes file with empty E1–E4 section headers; a recorded env baseline that later tasks cite. The symbol `reconstruct_haplotypes_from_svar2` and genoray's `overlap_batch`/query symbols must resolve in perf after this task.

- [ ] **Step 1: Grab an interactive compute node and record the baseline**

Run (from the worktree root):
```bash
srun -p carter-compute --cpus-per-task=16 --pty bash
# then, on the node:
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
{ echo "host=$(hostname)"; echo "date=$(date -Iseconds)";
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)";
  echo "turbo_no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)";
  echo "nproc=$(nproc)";
  echo "paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"; } | tee tmp/svar2_mvp/env_baseline.txt
```
Expected: `governor=performance`, `paranoid` ≤ 2 (memory: paranoid=2 works, no sudo). If `paranoid` > 2, perf call-graph capture will be restricted — note it; py-spy still works.

> If interactive `srun` isn't practical for the agent, wrap each later timed command as `srun -p carter-compute --cpus-per-task=16 <cmd>` instead. Every timed/profiled run in this plan MUST be on a compute node, never the login node.

- [ ] **Step 2: Confirm profilers and store layout on the node**

Run:
```bash
.pixi/envs/default/bin/py-spy --version
/carter/users/dlaub/.pixi/bin/perf --version
ls /carter/users/dlaub/repos/for_loukik/svar2_mvp/{germline,somatic}.svar2
```
Expected: py-spy prints a version; perf prints a version; both `.svar2` dirs list.

- [ ] **Step 3: Rebuild the gvl extension with release + line-table debug + frame pointers**

Run:
```bash
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
RUSTFLAGS="-C force-frame-pointers=yes" \
rtk pixi run -e default maturin develop --release
```
Expected: `maturin` finishes; the compiled `.so` under `.pixi/envs/default/.../genvarloader/` is rebuilt. (This does NOT edit `Cargo.toml` — the `CARGO_PROFILE_RELEASE_DEBUG` env var injects debug info into the release profile for this build only.)

- [ ] **Step 4: Rebuild + swap the genoray wheel with the same debug flags**

genoray source is at `/carter/users/dlaub/projects/genoray` (pyproject version 2.15.0 — matches the installed wheel). Build a debug-symbolized release wheel and force-install it into gvl's `default` env:
```bash
cd /carter/users/dlaub/projects/genoray
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
RUSTFLAGS="-C force-frame-pointers=yes" \
rtk pixi run -e py310 maturin build --release
# install the freshly built wheel into gvl's default env (py310 ABI == default's 3.10.20):
WHEEL=$(ls -t target/wheels/genoray-2.15.0-*.whl | head -1)
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
rtk pixi run -e default pip install --force-reinstall --no-deps "/carter/users/dlaub/projects/genoray/$WHEEL"
```
Expected: a wheel is produced under `genoray/target/wheels/`; `pip install` reports genoray 2.15.0 reinstalled. Verify version unchanged:
```bash
rtk pixi run -e default python -c "import genoray; print(genoray.__version__)"
```
Expected: `2.15.0`.

> **Risk / degrade-gracefully:** if the genoray wheel rebuild fails (toolchain/htslib build issue) or its version drifts from 2.15.0, do NOT block the whole plan. gvl's own kernel (`reconstruct_haplotypes_from_svar2`, in *this* repo's `src/`) was rebuilt in Step 3 and will symbolize regardless. Record in the notes that genoray-internal frames (`overlap_batch`, `query.rs`, `svar2-codec`) may appear as raw addresses in perf, and lean on py-spy `--native` (which still attributes them to the genoray `.so` module even without line tables) for those. Keep going.

- [ ] **Step 5: Verify Rust symbolization end-to-end**

Run a 3-second perf capture over the SVAR2 driver (driver written in Task 1 — for now smoke with a one-liner) and grep for a known gvl symbol:
```bash
/carter/users/dlaub/.pixi/bin/perf record -g --call-graph dwarf -o /tmp/sym_check.data -- \
  rtk pixi run -e default python -c "
import numpy as np
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source
import pysam
REF='/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa'
rb=pysam.FastaFile(REF).fetch('chr21').encode()
ru=np.frombuffer(rb,np.uint8); ro=np.array([0,len(rb)],np.int64)
regs=[(20_000_000,20_001_000),(30_000_000,30_000_500),(40_000_000,40_001_000)]
src=SparseVar2Source(SparseVar2('/carter/users/dlaub/repos/for_loukik/svar2_mvp/germline.svar2'))
for _ in range(20): src.reconstruct('chr21',regs,ru,ro,pad_char=ord('N'),shifts=None,output_length=-1)
"
/carter/users/dlaub/.pixi/bin/perf report --stdio -i /tmp/sym_check.data 2>/dev/null | grep -iE "reconstruct_haplotypes_from_svar2|svar2|overlap|query" | head
```
Expected: at least `reconstruct_haplotypes_from_svar2` (and ideally genoray `overlap`/`query` symbols) appear as named frames, not bare `0x...` addresses. If only gvl symbols resolve, that matches the Step 4 degrade note — acceptable.

- [ ] **Step 6: Create the results-notes skeleton**

Write `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md`:
```markdown
# SVAR2 Profiling Results (E1–E4)

**Date:** 2026-07-03 · **Spec:** `docs/superpowers/specs/2026-07-03-svar2-profiling-followup.md`
**Node/env baseline:** see `tmp/svar2_mvp/env_baseline.txt`.
**Symbolization:** gvl + genoray rebuilt `--release` with `debug=line-tables-only` + frame pointers.

## E1 — Query-latency attribution
_(filled by Task 3)_

## E2 — Same-cohort sample sweep
_(filled by Task 5)_

## E3 — Dense-access layout probe
_(filled by Task 6; conditional on E1)_

## E4 — Conversion thread-allocation
_(filled by Task 8)_

## Recommendations
_(filled by Task 9)_
```

- [ ] **Step 7: Commit**

```bash
rtk git add tmp/svar2_mvp/env_baseline.txt docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
rtk git commit -m "chore: SVAR2 profiling env baseline + symbolized builds + notes skeleton"
```

---

## Task 1: E1 profiling driver (`prof_driver.py`)

**Files:**
- Create: `tmp/svar2_mvp/prof_driver.py`

**Interfaces:**
- Consumes: `$W` stores; `SparseVar2Source.reconstruct(contig, regions, ref_, ref_offsets, pad_char, shifts, output_length)` (signature in `python/genvarloader/_dataset/_svar2_source.py:51`); `gvl.Dataset.open(path, reference).with_seqs("haplotypes")[:R, :n_s]`.
- Produces: CLI `python prof_driver.py <backend> <cohort> <K>` where `backend ∈ {svar1, svar2}`, `cohort ∈ {germline, somatic}`, `K` = loop count. Prints exactly one line `per_call_s=<float>`. For `svar1` it writes the 3-region `.gvl` **once before the loop** so profiling captures the query, not `gvl.write`.

- [ ] **Step 1: Write the driver**

Create `tmp/svar2_mvp/prof_driver.py`:
```python
"""E1 single-path profiling driver. Exercises ONE code path in a warm loop so
py-spy/perf attribute time to that path only.

  python prof_driver.py <svar1|svar2> <germline|somatic> <K>

Prints: per_call_s=<median-ish mean over K warm calls>
For svar1, the 3-region .gvl is written ONCE before the loop (we profile the
query, not gvl.write)."""
import sys
import time

import numpy as np

W = "/carter/users/dlaub/repos/for_loukik/svar2_mvp"
REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
CHROM = "chr21"
REGIONS = [(20_000_000, 20_001_000), (30_000_000, 30_000_500), (40_000_000, 40_001_000)]


def _ref():
    import pysam
    rb = pysam.FastaFile(REF).fetch(CHROM).encode()
    return np.frombuffer(rb, np.uint8), np.array([0, len(rb)], np.int64)


def make_svar2(cohort):
    from genoray import SparseVar2
    from genvarloader._dataset._svar2_source import SparseVar2Source
    src = SparseVar2Source(SparseVar2(f"{W}/{cohort}.svar2"))
    ru, ro = _ref()

    def call():
        src.reconstruct(CHROM, REGIONS, ru, ro, pad_char=ord("N"),
                        shifts=None, output_length=-1)
    return call


def make_svar1(cohort):
    import polars as pl
    import genvarloader as gvl
    from genoray import SparseVar2
    n_s = SparseVar2(f"{W}/{cohort}.svar2").n_samples
    bed = pl.DataFrame({"chrom": [CHROM] * len(REGIONS),
                        "chromStart": [s for s, _ in REGIONS],
                        "chromEnd": [e for _, e in REGIONS]})
    ds_path = f"{W}/{cohort}.gvl"
    gvl.write(ds_path, bed, variants=f"{W}/{cohort}.svar", overwrite=True)  # ONCE
    ds_hap = gvl.Dataset.open(ds_path, reference=REF).with_seqs("haplotypes")

    def call():
        ds_hap[:len(REGIONS), :n_s]
    return call


def main(backend, cohort, K):
    call = {"svar1": make_svar1, "svar2": make_svar2}[backend](cohort)
    call()  # warm
    t0 = time.perf_counter()
    for _ in range(K):
        call()
    dt = time.perf_counter() - t0
    print(f"per_call_s={dt / K:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 2: Smoke-test all four (backend × cohort) combos on the node**

Run (short K to keep it fast):
```bash
for b in svar2 svar1; do for c in germline somatic; do
  echo -n "$b $c -> "; rtk pixi run -e default python tmp/svar2_mvp/prof_driver.py $b $c 5
done; done
```
Expected: four `per_call_s=<x>` lines, all `>0`, roughly consistent with the MVP table (svar2 germline ~0.28, svar1 germline ~0.06, both somatic ~0.38). If svar1 numbers are ~10× higher than the MVP, the `.gvl` write leaked into timing — verify `gvl.write` is outside the loop.

- [ ] **Step 3: Commit**

```bash
rtk git add tmp/svar2_mvp/prof_driver.py
rtk git commit -m "chore: E1 single-path profiling driver for SVAR1/SVAR2 hap query"
```

---

## Task 2: Python-vs-native split post-processor (`split_folded.py`)

**Files:**
- Create: `tmp/svar2_mvp/split_folded.py`

**Interfaces:**
- Consumes: a py-spy `--format raw` folded-stack file (one `frame1;frame2;...;leaf <count>` line per unique stack). py-spy marks Python frames as `path.py:func:line` and native frames (with `--native`) as demangled symbols or `<module.so>` without a `.py:` token.
- Produces: CLI `python split_folded.py <folded.txt>` printing `python_pct=<x> native_pct=<x> total_samples=<n>` and the top-15 **leaf** frames with per-frame sample counts and class. Leaf-attribution = self-time. A frame is Python iff its text contains `.py:`.

- [ ] **Step 1: Write the post-processor**

Create `tmp/svar2_mvp/split_folded.py`:
```python
"""Split a py-spy --format raw (folded) stack file into Python vs native
self-time by LEAF frame. A leaf frame is Python iff it contains '.py:'.

  python split_folded.py <folded.txt>
"""
import sys
from collections import Counter


def is_python(frame: str) -> bool:
    return ".py:" in frame or frame.endswith(".py")


def main(path):
    py = nat = 0
    leaves = Counter()
    classed = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, cnt = line.rpartition(" ")
            try:
                n = int(cnt)
            except ValueError:
                continue
            leaf = stack.split(";")[-1]
            leaves[leaf] += n
            classed[leaf] = "python" if is_python(leaf) else "native"
            if is_python(leaf):
                py += n
            else:
                nat += n
    tot = py + nat
    if tot == 0:
        print("no samples parsed"); return
    print(f"python_pct={100 * py / tot:.1f} native_pct={100 * nat / tot:.1f} total_samples={tot}")
    print("top-15 leaf frames (self-time):")
    for leaf, n in leaves.most_common(15):
        print(f"  {100 * n / tot:5.1f}%  [{classed[leaf]:6s}]  {leaf}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Unit-smoke it on a synthetic folded file**

Run:
```bash
printf 'a.py:f:1;b.py:g:2 10\nc.py:h:3;native_symbol 30\n' > /tmp/folded_smoke.txt
rtk pixi run -e default python tmp/svar2_mvp/split_folded.py /tmp/folded_smoke.txt
```
Expected: `python_pct=25.0 native_pct=75.0 total_samples=40`, then two leaf lines (`b.py:g:2` python 25%, `native_symbol` native 75%).

- [ ] **Step 3: Commit**

```bash
rtk git add tmp/svar2_mvp/split_folded.py
rtk git commit -m "chore: py-spy folded-stack Python/native self-time splitter"
```

---

## Task 3: E1 — capture profiles + fill the attribution table

**Files:**
- Create: `tmp/svar2_mvp/e1_profile.sh`
- Modify: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` (E1 section)
- Artifacts: `tmp/svar2_mvp/prof_out/e1/` (flamegraphs, speedscope, folded, perf data) — gitignored via not committing binaries.

**Interfaces:**
- Consumes: `prof_driver.py` (Task 1), `split_folded.py` (Task 2), the symbolized builds (Task 0).
- Produces: E1 table `{backend, cohort, python_pct, native_pct, per_call_s}` for all four combos + top-~10 Rust symbols for the two SVAR2 paths, and a written classification **A / B / C** (Python-adapter overhead / Rust hot-path / mix).

- [ ] **Step 1: Write the capture orchestrator**

Create `tmp/svar2_mvp/e1_profile.sh`:
```bash
#!/usr/bin/env bash
# E1: py-spy split (all 4) + perf Rust detail (svar2 only). Run ON a compute node.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=tmp/svar2_mvp/prof_out/e1
mkdir -p "$OUT"
PYSPY=.pixi/envs/default/bin/py-spy
PERF=/carter/users/dlaub/.pixi/bin/perf
K=200   # warm loops per capture; adjust so each capture runs >=15s of wall time

run_pyspy () {  # backend cohort
  local b=$1 c=$2 tag="${1}_${2}"
  # native flamegraph (visual) + folded raw (for the split) + speedscope
  $PYSPY record --native --rate 500 --format flamegraph -o "$OUT/${tag}.svg" -- \
    pixi run -e default python tmp/svar2_mvp/prof_driver.py "$b" "$c" "$K"
  $PYSPY record --native --rate 500 --format raw       -o "$OUT/${tag}.folded" -- \
    pixi run -e default python tmp/svar2_mvp/prof_driver.py "$b" "$c" "$K"
  $PYSPY record --native --rate 500 --format speedscope -o "$OUT/${tag}.speedscope.json" -- \
    pixi run -e default python tmp/svar2_mvp/prof_driver.py "$b" "$c" "$K"
  echo "== split $tag ==" | tee -a "$OUT/splits.txt"
  pixi run -e default python tmp/svar2_mvp/split_folded.py "$OUT/${tag}.folded" | tee -a "$OUT/splits.txt"
}

run_perf () {   # backend cohort  (svar2 only)
  local b=$1 c=$2 tag="${1}_${2}"
  # Use frame-pointer call graph (extensions were built with -C force-frame-pointers=yes).
  # DWARF call-graph on this workload produced a 1GB perf.data and overloaded the shared node;
  # fp gives clean source-level Rust symbols at ~1MB. Lower -F to keep it light on the shared node.
  $PERF record -g --call-graph fp -F 199 -o "$OUT/${tag}.perf.data" -- \
    pixi run -e default python tmp/svar2_mvp/prof_driver.py "$b" "$c" "$K"
  echo "== perf top symbols $tag ==" | tee -a "$OUT/perf_top.txt"
  $PERF report --stdio -i "$OUT/${tag}.perf.data" --sort=overhead,symbol -g none 2>/dev/null \
    | grep -vE '^\s*#' | head -25 | tee -a "$OUT/perf_top.txt"
}

for b in svar2 svar1; do for c in germline somatic; do run_pyspy "$b" "$c"; done; done
for c in germline somatic; do run_perf svar2 "$c"; done
echo "DONE. splits -> $OUT/splits.txt ; perf -> $OUT/perf_top.txt"
```

- [ ] **Step 2: Run it on the compute node**

Run:
```bash
chmod +x tmp/svar2_mvp/e1_profile.sh
srun -p carter-compute --cpus-per-task=16 bash tmp/svar2_mvp/e1_profile.sh
```
Expected: prints four `python_pct=… native_pct=…` blocks and two perf top-symbol blocks; `tmp/svar2_mvp/prof_out/e1/splits.txt` and `perf_top.txt` populated. Sanity: the SVAR1 path (known Rust-bound from prior gvl profiling) should show high `native_pct`; if SVAR1 shows high `python_pct`, py-spy failed to unwind native — check that `--native` is active and the build has frame pointers (Task 0 Step 3).

- [ ] **Step 3: Extract per-call wall-clock for the table**

Run (clean timing, no profiler attached — profilers inflate wall time). **One allocation for all four** so the numbers are comparable (shared node — see Global Constraints):
```bash
srun -p carter-compute --cpus-per-task=16 bash -c '
for b in svar1 svar2; do for c in germline somatic; do
  echo -n "$b $c "; pixi run -e default python tmp/svar2_mvp/prof_driver.py $b $c 50
done; done'
```
Expected: four `per_call_s=` values; record them as the wall-clock column. (Absolute values are node-dependent; only their *ratios within this run* are load-bearing.)

- [ ] **Step 4: Read the top Rust symbols for the SVAR2 paths**

Run:
```bash
cat tmp/svar2_mvp/prof_out/e1/perf_top.txt
```
Read the top ~10 named Rust symbols (expect candidates: `reconstruct_haplotypes_from_svar2`, genoray `overlap`/`query` decode, dense presence `get_bit` gather, `svar2_codec` decode, memcpy/alloc). Note any that are bare addresses (genoray un-symbolized per Task 0 degrade note).

- [ ] **Step 5: Fill the E1 section of the notes file**

Replace the `## E1` placeholder in `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` with a table of the real captured numbers, in this exact shape (fill `<...>` from Steps 2–4):
```markdown
## E1 — Query-latency attribution

Workload: 3 chr21 regions × all samples, warm, K-loop. py-spy `--native` rate 500;
leaf-attributed Python/native self-time via `split_folded.py`. perf dwarf call-graph.

| backend | cohort   | per_call_s | python_pct | native_pct |
| ------- | -------- | ---------- | ---------- | ---------- |
| svar1   | germline | <...>      | <...>      | <...>      |
| svar1   | somatic  | <...>      | <...>      | <...>      |
| svar2   | germline | <...>      | <...>      | <...>      |
| svar2   | somatic  | <...>      | <...>      | <...>      |

**Top SVAR2 Rust symbols (perf, self-overhead):**
1. <symbol> — <overhead%>
   ... (~10)

**Top SVAR2 Python leaf frames (py-spy):**
- <frame> — <self%>   (e.g. `np.ascontiguousarray` conversions, `overlap_batch` marshalling, FFI boundary)

**Classification:** <A|B|C> — <one paragraph>:
- **A** = Python-adapter overhead dominates the SVAR2 gap → Task B Dataset wiring likely erases it.
- **B** = Rust hot-path / dense layout dominates → Task B must carry a kernel/layout fix.
- **C** = mix → quantify each share.
```

> Decision hint for classifying: the SVAR2 adapter does a long chain of `np.ascontiguousarray` copies + `overlap_batch` marshalling in `_svar2_source.py:_query`/`reconstruct` (lines 36–88). If those Python leaf frames dominate → A. If the dense-presence gather / codec decode dominates native → B. Report the split, don't force a single bucket.

- [ ] **Step 6: Commit (scripts + notes; not the binary artifacts)**

```bash
rtk git add tmp/svar2_mvp/e1_profile.sh docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
rtk git commit -m "chore: E1 query-latency attribution — py-spy split + perf Rust detail + table"
```

> Do NOT commit `tmp/svar2_mvp/prof_out/` (flamegraph SVGs, speedscope JSON, perf.data are large binaries). If `git status` shows them, add `tmp/svar2_mvp/prof_out/` to `.gitignore` in this commit.

---

## Task 4: E2 — subsample somatic + build per-S stores

**Files:**
- Create: `tmp/svar2_mvp/e2_subsample.sh`
- Create: `tmp/svar2_mvp/e2_build.sbatch`
- Artifacts: `$W/somatic_s{1000,2000,4000,8000,16007}.{svar,svar2}` + sample lists + per-S variant counts.

**Interfaces:**
- Consumes: `$W/gdc.chr21.norm.filt.bcf` (the somatic filtered BCF); `build_stores.py build(bcf, chrom, samples, out_prefix, ploidy)` (existing, `tmp/svar2_mvp/build_stores.py`).
- Produces: for each `S ∈ {1000, 2000, 4000, 8000, 16007}`, a subsampled BCF `$W/gdc.chr21.s${S}.bcf`, its `.svar`+`.svar2` at prefix `$W/somatic_s${S}`, and a recorded per-S variant count (subsampling drops sites monomorphic in the subset — record counts so curves are interpretable). S=16007 = the full cohort (reuse existing `somatic.*` — do not rebuild).

- [ ] **Step 1: Write the subsample script**

Create `tmp/svar2_mvp/e2_subsample.sh`:
```bash
#!/usr/bin/env bash
# E2: fixed-cohort (somatic) sample-count sweep. Build per-S subsampled BCFs.
set -euo pipefail
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
SRC=$W/gdc.chr21.norm.filt.bcf
CHROM=chr21
bcftools query -l "$SRC" > "$W/somatic.samples.txt"
TOTAL=$(wc -l < "$W/somatic.samples.txt")
echo "total somatic samples=$TOTAL"   # expect 16007
for S in 1000 2000 4000 8000; do
  head -n "$S" "$W/somatic.samples.txt" > "$W/somatic.s${S}.list"
  bcftools view -S "$W/somatic.s${S}.list" --force-samples -Ob \
    -o "$W/gdc.chr21.s${S}.bcf" "$SRC"
  bcftools index -f "$W/gdc.chr21.s${S}.bcf"
  N=$(bcftools view -H "$W/gdc.chr21.s${S}.bcf" | wc -l)
  echo "S=$S variants=$N" | tee -a "$W/e2_variant_counts.txt"
done
# S=16007 is the full cohort: reuse existing somatic.* store; just record its count.
N=$(bcftools view -H "$SRC" | wc -l)
echo "S=16007 variants=$N" | tee -a "$W/e2_variant_counts.txt"
```

- [ ] **Step 2: Run subsampling on the node**

Run:
```bash
chmod +x tmp/svar2_mvp/e2_subsample.sh
srun -p carter-compute --cpus-per-task=8 bash tmp/svar2_mvp/e2_subsample.sh
cat /carter/users/dlaub/repos/for_loukik/svar2_mvp/e2_variant_counts.txt
```
Expected: `total somatic samples=16007`; five `S=... variants=...` lines with variant counts monotonically non-decreasing in S (mild — see spec caveat). Sub-BCFs and `.csi` indexes present.

- [ ] **Step 3: Write the build array job**

Create `tmp/svar2_mvp/e2_build.sbatch`:
```bash
#!/usr/bin/env bash
#SBATCH -p carter-compute
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3
#SBATCH -J e2build
#SBATCH -o /carter/users/dlaub/repos/for_loukik/svar2_mvp/e2_build_%a.log
set -euo pipefail
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
SIZES=(1000 2000 4000 8000)
S=${SIZES[$SLURM_ARRAY_TASK_ID]}
pixi run -e default python tmp/svar2_mvp/build_stores.py \
  "$W/gdc.chr21.s${S}.bcf" chr21 "$W/somatic_s${S}"
```
> `build_stores.py` reads its own sample list via `bcftools query -l <bcf>`, so the subsampled BCF fully determines the cohort. S=16007 is NOT in the array — it reuses the existing `$W/somatic.*` store.

- [ ] **Step 4: Submit and wait for builds**

Run:
```bash
rtk pixi run -e default sbatch tmp/svar2_mvp/e2_build.sbatch   # note: sbatch itself needs no env, but rtk-wrapping is harmless
# poll:
squeue -u "$USER" -n e2build
```
Expected: four array tasks queue; on completion each `e2_build_<i>.log` ends with `built .../somatic_s<S>.svar and .../somatic_s<S>.svar2`. Verify:
```bash
ls -d /carter/users/dlaub/repos/for_loukik/svar2_mvp/somatic_s{1000,2000,4000,8000}.{svar,svar2}
```
Expected: eight store dirs exist.

- [ ] **Step 5: Commit the scripts + variant counts**

```bash
cp /carter/users/dlaub/repos/for_loukik/svar2_mvp/e2_variant_counts.txt tmp/svar2_mvp/e2_variant_counts.txt
rtk git add tmp/svar2_mvp/e2_subsample.sh tmp/svar2_mvp/e2_build.sbatch tmp/svar2_mvp/e2_variant_counts.txt
rtk git commit -m "chore: E2 somatic sample-count sweep — subsample + per-S store builds"
```

---

## Task 5: E2 — hap-latency-vs-S benchmark + curves

**Files:**
- Create: `tmp/svar2_mvp/e2_bench.py`
- Modify: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` (E2 section)

**Interfaces:**
- Consumes: the per-S stores from Task 4 (prefix `$W/somatic_s${S}` for S<16007, `$W/somatic` for S=16007); the same warm-median timing shape as `prof_driver.py`.
- Produces: a TSV `tmp/svar2_mvp/prof_out/e2_curve.tsv` with columns `S  variants  svar1_hap_s  svar2_hap_s`, and slopes (Δlatency/ΔS) for both backends written into the notes.

- [ ] **Step 1: Write the sweep benchmark**

Create `tmp/svar2_mvp/e2_bench.py`:
```python
"""E2: hap-latency vs sample-count S for both backends on the SOMATIC cohort.
Fixed dataset family (subsampled somatic), same 3 regions, warm, median N=5.

  python e2_bench.py > tmp/svar2_mvp/prof_out/e2_curve.tsv
"""
import time
from statistics import median

import numpy as np
import polars as pl
import genvarloader as gvl
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source

W = "/carter/users/dlaub/repos/for_loukik/svar2_mvp"
REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
CHROM = "chr21"
REGIONS = [(20_000_000, 20_001_000), (30_000_000, 30_000_500), (40_000_000, 40_001_000)]
N = 5
SIZES = [1000, 2000, 4000, 8000, 16007]


def prefix(S):
    return f"{W}/somatic" if S == 16007 else f"{W}/somatic_s{S}"


def timed(fn, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return median(ts)


def variants_at(S):
    # read the recorded count file written by e2_subsample.sh
    for line in open(f"{W}/e2_variant_counts.txt"):
        if line.startswith(f"S={S} "):
            return int(line.split("variants=")[1])
    return -1


def main():
    import pysam
    rb = pysam.FastaFile(REF).fetch(CHROM).encode()
    ru, ro = np.frombuffer(rb, np.uint8), np.array([0, len(rb)], np.int64)
    print("S\tvariants\tsvar1_hap_s\tsvar2_hap_s")
    for S in SIZES:
        p = prefix(S)
        sv2 = SparseVar2(f"{p}.svar2")
        src = SparseVar2Source(sv2)
        n_s = sv2.n_samples
        svar2 = timed(lambda: src.reconstruct(CHROM, REGIONS, ru, ro,
                                               pad_char=ord("N"), shifts=None, output_length=-1))
        bed = pl.DataFrame({"chrom": [CHROM] * len(REGIONS),
                            "chromStart": [s for s, _ in REGIONS],
                            "chromEnd": [e for _, e in REGIONS]})
        gvl.write(f"{p}.gvl", bed, variants=f"{p}.svar", overwrite=True)
        ds_hap = gvl.Dataset.open(f"{p}.gvl", reference=REF).with_seqs("haplotypes")
        svar1 = timed(lambda: ds_hap[:len(REGIONS), :n_s])
        print(f"{S}\t{variants_at(S)}\t{svar1:.4f}\t{svar2:.4f}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sweep on the node**

Run:
```bash
mkdir -p tmp/svar2_mvp/prof_out
srun -p carter-compute --cpus-per-task=16 \
  pixi run -e default python tmp/svar2_mvp/e2_bench.py | tee tmp/svar2_mvp/prof_out/e2_curve.tsv
```
Expected: a 5-row TSV; `svar2_hap_s` should track the full-cohort somatic value (~0.38) at S=16007 and both columns rise with S. Sanity: at S=16007 the two backends should be near the MVP parity point (~0.38 each).

- [ ] **Step 3: Compute slopes and fill the E2 notes section**

Compute least-squares slope per backend:
```bash
rtk pixi run -e default python -c "
import numpy as np
rows=[l.split() for l in open('tmp/svar2_mvp/prof_out/e2_curve.tsv').read().splitlines()[1:]]
S=np.array([int(r[0]) for r in rows]); a=np.array([float(r[2]) for r in rows]); b=np.array([float(r[3]) for r in rows])
print('svar1 slope s/sample=%.3e'%np.polyfit(S,a,1)[0])
print('svar2 slope s/sample=%.3e'%np.polyfit(S,b,1)[0])
"
```
Replace the `## E2` placeholder with the TSV rendered as a markdown table plus both slopes, and a one-line **decision output**: does SVAR2 latency rise *less steeply* with S than SVAR1 (confirming the MVP's declined hypothesis) or not? Cite the per-S variant counts as the interpretability caveat (subsampling drops monomorphic sites).

- [ ] **Step 4: Commit**

```bash
rtk git add tmp/svar2_mvp/e2_bench.py tmp/svar2_mvp/prof_out/e2_curve.tsv docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
rtk git commit -m "chore: E2 same-cohort hap-latency-vs-S sweep + slopes"
```

---

## Task 6: E3 — dense-access layout probe (conditional)

**Files:**
- Create: `tmp/svar2_mvp/e3_probe.py`
- Modify: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` (E3 section)

**Interfaces:**
- Consumes: E1's classification (Task 3). **Gate:** run this task ONLY if E1 attributed significant native time to the dense-presence gather (classification B or C with the dense gather in the top Rust symbols). Otherwise, write "E3 skipped — E1 showed the SVAR2 gap is <A: Python-adapter overhead / native but not dense-gather>" and commit that.
- Consumes: `SparseVar2` (`n_samples`, `overlap_batch`), `SparseVar2Source.reconstruct`.
- Produces: two curves — SVAR2 hap latency vs (a) **region width** (fixed all-samples) and (b) proxy for **`n_dense_variants`** (dense count touched by widening the region toward the whole contig) — into `tmp/svar2_mvp/prof_out/e3.tsv`, and an estimated size of the "contig-wide stride + bit-by-bit read" cost.

- [ ] **Step 1: Decide the gate**

Read the E1 classification from the notes file. If it is **A** (Python-adapter overhead) or **native-but-not-dense-gather**, skip to Step 4 (write the skip note). If **B/C with dense gather hot**, proceed to Step 2.

- [ ] **Step 2: Write the layout probe**

Create `tmp/svar2_mvp/e3_probe.py`:
```python
"""E3: SVAR2 hap-latency sensitivity to region width and dense count.
Germline (high-AF -> large n_dense_variants) is the stress cohort.

  python e3_probe.py > tmp/svar2_mvp/prof_out/e3.tsv
"""
import time
from statistics import median

import numpy as np
import pysam
from genoray import SparseVar2
from genvarloader._dataset._svar2_source import SparseVar2Source

W = "/carter/users/dlaub/repos/for_loukik/svar2_mvp"
REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
CHROM = "chr21"
N = 5
START = 20_000_000
WIDTHS = [200, 1_000, 5_000, 25_000, 100_000, 500_000]


def timed(fn):
    fn()
    ts = []
    for _ in range(N):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return median(ts)


def main():
    rb = pysam.FastaFile(REF).fetch(CHROM).encode()
    ru, ro = np.frombuffer(rb, np.uint8), np.array([0, len(rb)], np.int64)
    sv2 = SparseVar2(f"{W}/germline.svar2")
    src = SparseVar2Source(sv2)
    print("width\tn_dense_in_region\thap_s")
    for w in WIDTHS:
        regs = [(START, START + w)]
        d = sv2.overlap_batch(CHROM, [(START, START + w)])
        # dense variants actually spanned by this region (dense_range gives [lo,hi) per region)
        dr = np.asarray(d["dense_range"]).reshape(-1, 2)
        n_dense = int((dr[:, 1] - dr[:, 0]).sum())
        t = timed(lambda: src.reconstruct(CHROM, regs, ru, ro,
                                          pad_char=ord("N"), shifts=None, output_length=-1))
        print(f"{w}\t{n_dense}\t{t:.4f}", flush=True)


if __name__ == "__main__":
    main()
```
> The dense-presence *stride* is contig-wide (`n_dense_variants`), but the *count read* per query scales with the dense variants spanned by the region — widening the region increases both the gather work and cache-line scatter. This probe measures latency vs that count. A full Rust microbench of `get_bit`-per-column vs word-parallel is out of scope for measurement; note it as the follow-up if the curve is steep.

- [ ] **Step 3: Run the probe**

Run:
```bash
srun -p carter-compute --cpus-per-task=16 \
  pixi run -e default python tmp/svar2_mvp/e3_probe.py | tee tmp/svar2_mvp/prof_out/e3.tsv
```
Expected: latency rising with width and `n_dense_in_region`. A super-linear rise vs `n_dense_in_region` is evidence for the scatter/bit-by-bit cost.

- [ ] **Step 4: Fill the E3 notes section (probe result OR skip note)**

If run: replace `## E3` with the TSV as a table + a sentence on whether latency scales with `n_dense_in_region` (evidence for/against the contig-wide-stride + bit-by-bit cost) and a rough magnitude of the potential win from a region-local / word-parallel layout — input to how much layout work Task B carries.
If skipped: replace `## E3` with the one-line skip note citing E1's classification.

- [ ] **Step 5: Commit**

```bash
rtk git add tmp/svar2_mvp/e3_probe.py docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
# include the tsv only if the probe ran:
[ -f tmp/svar2_mvp/prof_out/e3.tsv ] && rtk git add tmp/svar2_mvp/prof_out/e3.tsv
rtk git commit -m "chore: E3 dense-access layout probe (or documented skip per E1)"
```

---

## Task 7: E4 — conversion phase breakdown (build side)

**Files:**
- Create: `tmp/svar2_mvp/e4_convert_driver.py`
- Modify: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` (E4 section, part 1)

**Interfaces:**
- Consumes: `genoray._core.run_conversion_pipeline(vcf_path, reference_path, chroms, output_dir, samples, chunk_size=25000, ploidy=2, max_threads=None, long_allele_capacity=8388608)` (confirmed signature); `GENORAY_SAMPLE_INTERVAL` env (genoray's built-in channel-fill/per-thread sampler — the only genoray env knob present); `split_folded.py` (Task 2).
- Produces: CLI `python e4_convert_driver.py <bcf> <chrom> <out_prefix> <max_threads>` that runs one conversion and prints `build_wall_s=<x>`; plus a py-spy `--native` phase breakdown (htslib read/decompress vs encode vs Phase-2 merge) for a single build, written to the E4 notes section.

- [ ] **Step 1: Write the conversion driver**

Create `tmp/svar2_mvp/e4_convert_driver.py`:
```python
"""E4: run ONE single-contig svar2 conversion with a chosen max_threads.
  python e4_convert_driver.py <bcf> <chrom> <out_prefix> <max_threads>
Prints: build_wall_s=<x>
Set GENORAY_SAMPLE_INTERVAL in the environment to enable genoray's sampler."""
import sys
import time
import subprocess

from genoray import _core

REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"


def main(bcf, chrom, out_prefix, max_threads):
    samples = subprocess.run(["bcftools", "query", "-l", bcf],
                             capture_output=True, text=True, check=True).stdout.split()
    t0 = time.perf_counter()
    _core.run_conversion_pipeline(
        bcf, REF, [chrom], f"{out_prefix}.svar2", samples,
        25_000, 2, int(max_threads), 8 * 1024 * 1024,
    )
    print(f"build_wall_s={time.perf_counter() - t0:.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
```

- [ ] **Step 2: Capture the phase breakdown with py-spy --native + genoray sampler**

Use the germline single-contig BCF (~11 min full; use a subsampled BCF from Task 4 if you want a faster capture — e.g. `gdc.chr21.s1000.bcf` builds in a few minutes and exercises the same phases). Run on the node:
```bash
mkdir -p tmp/svar2_mvp/prof_out/e4
GENORAY_SAMPLE_INTERVAL=1000 \
srun -p carter-compute --cpus-per-task=16 \
  .pixi/envs/default/bin/py-spy record --native --rate 250 --format raw \
    -o tmp/svar2_mvp/prof_out/e4/convert.folded -- \
    pixi run -e default python tmp/svar2_mvp/e4_convert_driver.py \
      /carter/users/dlaub/repos/for_loukik/svar2_mvp/chr21.norm.filt.bcf chr21 \
      /tmp/e4_probe 8
```
Expected: a `.folded` file; py-spy sampled all Rust worker threads (htslib/encode/merge). genoray's sampler prints channel-fill / per-thread CPU% to stderr — capture it too (append `2> tmp/svar2_mvp/prof_out/e4/genoray_sampler.txt` if desired).

- [ ] **Step 3: Bucket native frames into phases**

Run the split, then bucket by symbol substring (approximate — regex over demangled names):
```bash
rtk pixi run -e default python tmp/svar2_mvp/split_folded.py tmp/svar2_mvp/prof_out/e4/convert.folded | tee tmp/svar2_mvp/prof_out/e4/split.txt
# rough phase buckets from the folded leaves:
rtk pixi run -e default python -c "
from collections import Counter
buckets=Counter()
for line in open('tmp/svar2_mvp/prof_out/e4/convert.folded'):
    stack,_,cnt=line.rstrip().rpartition(' ')
    try: n=int(cnt)
    except ValueError: continue
    leaf=stack.split(';')[-1].lower()
    if any(k in leaf for k in ('bgzf','inflate','zlib','htslib','decompress','read')): b='read/decompress'
    elif any(k in leaf for k in ('encode','bitgrid','codec','pack')): b='encode'
    elif any(k in leaf for k in ('merge','dense_merge','transpose')): b='phase2-merge'
    else: b='other'
    buckets[b]+=n
tot=sum(buckets.values())
for b,n in buckets.most_common(): print(f'{100*n/tot:5.1f}%  {b}')
" | tee -a tmp/svar2_mvp/prof_out/e4/split.txt
```
Expected: a percentage per phase; the MVP log's claim ("VCF read/decompress dominating") predicts `read/decompress` is the largest bucket. The `other` bucket catches un-symbolized/misc frames — if it dominates, note that symbol names didn't match the regex (genoray un-symbolized per Task 0) and lean on the genoray sampler output instead.

- [ ] **Step 4: Write the E4 part-1 notes**

Replace the `## E4` placeholder's first half with: the phase-bucket table, the genoray sampler's reported thread split (e.g. `1 concurrent chromosome | N HTSlib decompression threads`), and a one-line confirmation/refutation of "conversion is read-bound."

- [ ] **Step 5: Commit**

```bash
rtk git add tmp/svar2_mvp/e4_convert_driver.py docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
rtk git commit -m "chore: E4.1 conversion phase breakdown (read/decompress vs encode vs merge)"
```

---

## Task 8: E4 — thread-split sweep + recommended policy

**Files:**
- Create: `tmp/svar2_mvp/e4_sweep.sbatch`
- Modify: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` (E4 section, part 2)

**Interfaces:**
- Consumes: `e4_convert_driver.py` (Task 7). The Python-exposed knob is `max_threads` (total pipeline threads); genoray internally derives the decompression-vs-executor/writer split from it and reports it in its log line. There is **no** finer env knob exposed (only `GENORAY_SAMPLE_INTERVAL`) — so this sweep varies `max_threads` (and total cores) and records genoray's *reported* internal split at each point.
- Produces: a table `max_threads → build_wall_s` (+ genoray's reported decompress/executor split) for a fixed single-contig input, and a recommended few-contig thread policy.

- [ ] **Step 1: Pick the sweep input**

Use a single-contig input that builds in a few minutes so the sweep is affordable: the somatic `S=1000` subsample (`$W/gdc.chr21.s1000.bcf`, from Task 4) OR full germline (`$W/chr21.norm.filt.bcf`, ~11 min each). Prefer the smaller input for a denser sweep; note the chosen input in the notes. Sweep `max_threads ∈ {2, 4, 8, 16}` (cap at node `nproc`).

- [ ] **Step 2: Write the sweep job**

Create `tmp/svar2_mvp/e4_sweep.sbatch`:
```bash
#!/usr/bin/env bash
#SBATCH -p carter-compute
#SBATCH --cpus-per-task=16
#SBATCH -J e4sweep
#SBATCH -o /carter/users/dlaub/repos/for_loukik/svar2_mvp/e4_sweep.log
set -euo pipefail
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
W=/carter/users/dlaub/repos/for_loukik/svar2_mvp
BCF=$W/gdc.chr21.s1000.bcf      # single-contig, fast; swap to chr21.norm.filt.bcf for germline
for T in 2 4 8 16; do
  echo "=== max_threads=$T ==="
  GENORAY_SAMPLE_INTERVAL=1000 pixi run -e default python tmp/svar2_mvp/e4_convert_driver.py \
    "$BCF" chr21 "/tmp/e4_sweep_t${T}" "$T" 2>&1 | grep -E "build_wall_s|concurrent chromosome|decompression"
  rm -rf "/tmp/e4_sweep_t${T}.svar2"
done
```

- [ ] **Step 3: Submit and collect**

Run:
```bash
rtk pixi run -e default sbatch tmp/svar2_mvp/e4_sweep.sbatch
squeue -u "$USER" -n e4sweep     # wait for completion
cat /carter/users/dlaub/repos/for_loukik/svar2_mvp/e4_sweep.log
```
Expected: four `max_threads=T` blocks, each with a `build_wall_s=` and genoray's reported thread-split line. Wall-clock should fall then plateau (or regress) as threads rise — the minimum identifies the read-bound sweet spot. The sweep runs sequentially in **one** allocation so all points share the same node, but a shared node still adds noise — if a point looks anomalous (non-monotonic in a way threads don't explain), re-run the whole `.sbatch` and prefer the run whose curve is smooth; note contention in the E4 section.

- [ ] **Step 4: Fill the E4 part-2 notes + recommended policy**

Replace the `## E4` placeholder's second half with the `max_threads → build_wall_s (+ reported split)` table and a concrete recommendation for **few-contig** jobs (e.g. "for single-contig input, max_threads=N minimizes wall-clock; the read-bound phase wants most threads on htslib decompress"). Add the cross-reference line: this feeds `genoray:docs/roadmap/architecture.md` → Open questions → *read-bound conversion / thread allocation*.

- [ ] **Step 5: Commit**

```bash
rtk git add tmp/svar2_mvp/e4_sweep.sbatch docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
rtk git commit -m "chore: E4.2 conversion thread-split sweep + recommended few-contig policy"
```

---

## Task 9: Synthesis — two recommendations + cross-links

**Files:**
- Modify: `docs/superpowers/notes/2026-07-03-svar2-profiling-results.md` (Recommendations section)

**Interfaces:**
- Consumes: the filled E1–E4 sections.
- Produces: the two concrete recommendations the spec requires, plus a decision on E3's layout-work sizing, with pointers back to the spec's Task B and the genoray architecture open question.

- [ ] **Step 1: Write the two recommendations**

Replace the `## Recommendations` placeholder with exactly two headed recommendations, each grounded in the tables above:

1. **Where SVAR2 query latency actually goes → what Task B must include.** State the E1 classification (A/B/C) with the Python% vs native% numbers. If A: "Task B Dataset wiring likely erases the gap — no kernel/layout work needed." If B/C: name the hot Rust symbol(s) and cite E3 for the layout-win size, i.e. "Task B must carry a <region-local / word-parallel presence> layout change worth ~<X>." Reference the E2 slopes for the scales-with-S question (confirmed/refuted).
2. **Conversion thread-split policy.** State the E4 read-bound confirmation and the recommended `max_threads` / split for few-contig jobs, with the wall-clock delta between default and recommended.

- [ ] **Step 2: Add cross-links**

Add a short "Feeds" list: (a) the spec's Task B (Dataset wiring) — this notes file is the gate that decides its scope; (b) `genoray:docs/roadmap/architecture.md` → Open questions → read-bound conversion / thread allocation (E4). Note that split-by-contig layout remains unassessed (single-contig only) per the MVP notes' open question 3.

- [ ] **Step 3: Verify the notes file is complete and self-consistent**

Run (each alternative is a literal leftover-placeholder marker):
```bash
rtk grep -nF -e "filled by" -e "<...>" -e "TBD" docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
```
Expected: **no matches** — every placeholder replaced with real numbers. If any remain, the corresponding experiment task's Step didn't write its result; go back and fill it.

- [ ] **Step 4: Commit**

```bash
rtk git add docs/superpowers/notes/2026-07-03-svar2-profiling-results.md
rtk git commit -m "docs: SVAR2 profiling results synthesis — latency attribution + thread-split recommendations"
```

---

## Notes on scope & honesty (carry into every task)

- **No optimization.** If a hot path is obvious, record it — do not fix it. Task B and the thread rebalance are explicitly out of scope (spec "Out of scope").
- **Profilers inflate wall-clock.** Always take the reported latency numbers from *unprofiled* runs (`prof_driver.py`/`e2_bench.py` timing), and use py-spy/perf only for *attribution* (the %-split and symbol ranks).
- **Degrade gracefully on genoray symbols.** If the genoray wheel rebuild (Task 0 Step 4) didn't land, say so in the notes and rely on py-spy `--native` module attribution + the genoray sampler for genoray-internal phases; gvl's own kernel symbolizes regardless.
- **Record dispersion honestly.** Median of N=5 with no CI — call parity "indistinguishable at this resolution," not "equal," matching the MVP notes' framing.
