# SVAR2 read-bound `Dataset.__getitem__` profiling & optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute and then remove the hottest per-read costs in the live SVAR2 read-bound `Dataset.__getitem__` path for the two supported in-scope modes — **haplotypes** and **variants** — Python by inspection and Rust by `cargo asm`, with byte-identical parity preserved.

**Architecture:** Phase A stands up a deterministic profiling harness on the *real* `Dataset.open(svar2).with_seqs(mode)[...]` path (not the retired union oracle) and captures a committed baseline: cProfile + pyinstrument for Python functions, `perf` DSO-split + Rust symbol self-time for native, and `perf stat -e instructions` for a noise-free gate. Phase B applies optimizations one at a time — each confirmed against the Phase A profile before it is written, then gated on byte-identical parity plus a same-session instruction-count delta. gvl-side changes land on `svar2-m6b-kernel` (PR #266); genoray-side changes land on genoray `svar-2`.

**Tech Stack:** Python 3.10 (`-e dev` pixi env), Rust (maturin/PyO3), numpy, genoray_core (Rust crate path-dep) + genoray wheel, cProfile/pyinstrument, `perf`, `cargo asm`.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-05-svar2-readbound-getitem-perf-design.md` (read it first).
- **In-scope modes:** **haplotypes** and **variants** only. **Tracks is out of scope** (not profiled) but its parity tests MUST stay green — no change may break the tracks path. **variant-windows** is a consumer target but is currently guarded `NotImplementedError` in `Svar2Haps.__call__` (`_FlatVariantWindows`), so it cannot be profiled or optimized here — note it as deferred until implemented.
- **Two repos / branches:** gvl = this worktree, branch `svar2-m6b-kernel` (PR #266). genoray = `/carter/users/dlaub/projects/genoray`, branch `svar-2` (currently @ `aaf44fd`).
- **Rebuild before testing/profiling any Rust change:** `pixi run -e dev maturin develop --release` — pytest/perf import the stale `.so` otherwise. genoray crate edits are picked up by this same gvl rebuild (path-dep); a genoray **wheel** rebuild is only needed if a genoray **Python-API** surface changes (it will not here).
- **Profiling build must keep frame pointers:** build the profiled `.so` with `RUSTFLAGS="-C force-frame-pointers=yes" pixi run -e dev maturin develop --release` (perf fp call-graph needs them).
- **Parity is a hard gate.** After every change: `pixi run -e dev pytest tests -q` (full tree — the svar2 suite + parity oracle live across `tests/dataset` and `tests/unit`; this includes the tracks parity tests), plus `cargo test` on the repo(s) touched. The read-bound kernels are byte-identical to the union oracle; any divergence blocks the change. Two documented intentional non-identities (pure-DEL ALT `b""`; SVAR1 `max_ends` tie under-extension) are pre-existing — do not "fix" them.
- **Measurement discipline.** No cross-session absolute wall-clock claims (shared Carter node). Gate on **same-session before/after** + **`perf stat -e instructions,cycles`** deltas. If HW counters are unavailable (`perf stat` errors with "not supported"), fall back to cProfile total primitive-call counts + same-session median wall-clock, and say so in the recorded result.
- **Fixed inputs (verified present):**
  - Stores: `/carter/users/dlaub/projects/svar2_mvp/{germline,somatic}.{svar,svar2}`.
  - Reference: `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa`, contig `chr21`.
  - Env python (call directly for perf; never via `pixi run` — the launcher eats ~60% of samples): `.pixi/envs/dev/bin/python`.
  - perf binary: `/carter/users/dlaub/.pixi/bin/perf`.
- **py-spy is unusable here** (`ptrace_scope=2`, no sudo) and Python 3.10 has no perf trampoline — do not attempt py-spy or `perf` Python-frame resolution. Python attribution comes only from cProfile/pyinstrument.
- **Commit hygiene:** docs-only commits may need `--no-verify` (the `pyrefly` pre-commit hook fails spuriously on zero-Python-file commits and collides with the unstaged `pixi.lock`). Code commits run hooks normally; ensure prek hooks are installed first (`prek install` if needed).
- **Live read path recipe** (how every driver builds the real path):
  ```python
  import genvarloader as gvl
  from genoray import SparseVar2
  gvl.write(ds_path, bed, variants=SparseVar2(f"{prefix}.svar2"), samples=None,
            max_jitter=0, overwrite=True)          # ONCE, before the profiled loop
  ds = gvl.Dataset.open(ds_path, reference=REF)
  ds.with_seqs("haplotypes")[:, :]                  # or with_seqs("variants")
  ```

---

## Phase A — Profiling harness & committed baseline

### Task A1: Add pyinstrument + write the live-path profiling driver

**Files:**
- Modify: `pixi.toml` (add `pyinstrument` to the dev feature deps)
- Create: `tmp/svar2_mvp/prof_getitem.py`

**Interfaces:**
- Produces: a CLI `python tmp/svar2_mvp/prof_getitem.py <mode> <cohort> <K>` where `mode ∈ {haplotypes, variants}`, `cohort ∈ {germline, somatic}`; runs `gvl.write` + `Dataset.open` ONCE, then calls the selected read `K` times in a warm loop; prints `per_call_s=<float>`. Exposes a module function `make_call(mode, cohort) -> Callable[[], object]` reused by the cProfile/perf drivers in A2/A3.

- [ ] **Step 1: Add pyinstrument to the dev env**

Find the dev feature's `pypi-dependencies` (or `dependencies`) table in `pixi.toml` and add `pyinstrument`:

```toml
# in the [feature.dev.pypi-dependencies] (or nearest dev deps) table
pyinstrument = "*"
```

- [ ] **Step 2: Install it**

Run: `pixi install -e dev`
Expected: resolves and installs `pyinstrument`; no other deps change materially.

- [ ] **Step 3: Verify import**

Run: `.pixi/envs/dev/bin/python -c "import pyinstrument; print(pyinstrument.__version__)"`
Expected: prints a version (e.g. `4.x`).

- [ ] **Step 4: Write the driver**

Create `tmp/svar2_mvp/prof_getitem.py`:

```python
"""Profile the LIVE SVAR2 read-bound Dataset.__getitem__ path (not the union
oracle) for the in-scope modes. One (mode, cohort) per process so cProfile/perf
attribute cleanly.

  python tmp/svar2_mvp/prof_getitem.py <haplotypes|variants> <germline|somatic> <K>

gvl.write + Dataset.open run ONCE (we profile the READ, not the write). Prints
per_call_s over K warm calls. Tracks mode is out of scope; variant-windows is
guarded NotImplementedError in Svar2Haps and cannot be profiled yet."""
import sys
import time
from pathlib import Path

import polars as pl

STORE_DIR = Path("/carter/users/dlaub/projects/svar2_mvp")
REF = "/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa"
CHROM = "chr21"
REGIONS = [(20_000_000, 20_001_000), (30_000_000, 30_000_500), (40_000_000, 40_001_000)]
WORK = Path("tmp/svar2_mvp/prof_out/readbound")


def _bed():
    return pl.DataFrame({
        "chrom": [CHROM] * len(REGIONS),
        "chromStart": [s for s, _ in REGIONS],
        "chromEnd": [e for _, e in REGIONS],
    })


def make_call(mode, cohort):
    import genvarloader as gvl
    from genoray import SparseVar2

    prefix = STORE_DIR / cohort
    sv2 = SparseVar2(f"{prefix}.svar2")
    n_s = sv2.n_samples
    ds_path = WORK / f"{cohort}_{mode}.gvl"
    WORK.mkdir(parents=True, exist_ok=True)

    gvl.write(ds_path, _bed(), variants=SparseVar2(f"{prefix}.svar2"),
              samples=None, max_jitter=0, overwrite=True)
    ds = gvl.Dataset.open(ds_path, reference=REF)
    view = ds.with_seqs(mode)   # "haplotypes" or "variants"

    R = len(REGIONS)

    def call():
        view[:R, :n_s]

    return call


def main(mode, cohort, K):
    call = make_call(mode, cohort)
    call()  # warm
    t0 = time.perf_counter()
    for _ in range(K):
        call()
    print(f"per_call_s={(time.perf_counter() - t0) / K:.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 5: Smoke-test both modes (germline, small K)**

Run:
```bash
cd "$(git rev-parse --show-toplevel)"
for m in haplotypes variants; do
  echo "== $m =="; .pixi/envs/dev/bin/python tmp/svar2_mvp/prof_getitem.py $m germline 3
done
```
Expected: each prints `per_call_s=<float>` with no exception. (haplotypes needs the reference — it is set; no BigWig is required for either mode.)

- [ ] **Step 6: Commit**

```bash
git add pixi.toml pixi.lock tmp/svar2_mvp/prof_getitem.py
git commit -m "perf(svar2): live read-bound Dataset.__getitem__ profiling driver + pyinstrument"
```

---

### Task A2: Capture the Python-layer baseline (cProfile + pyinstrument)

**Files:**
- Create: `tmp/svar2_mvp/prof_python.py`
- Create (output): `tmp/svar2_mvp/prof_out/readbound/python_baseline.md`

**Interfaces:**
- Consumes: `prof_getitem.make_call` (A1).
- Produces: committed per-(mode×cohort) cProfile top-cumulative tables + pyinstrument trees identifying the hottest **Python** functions on the read path.

- [ ] **Step 1: Write the Python-layer profiler**

Create `tmp/svar2_mvp/prof_python.py`:

```python
"""cProfile + pyinstrument over the live read (Python-layer attribution).

  python tmp/svar2_mvp/prof_python.py <mode> <cohort> <K>

cProfile ranks Python functions by cumulative time; pyinstrument gives a
low-overhead statistical wall-clock call tree as a cross-check (cProfile's own
per-call overhead can distort tiny hot loops)."""
import cProfile
import io
import pstats
import sys

from prof_getitem import make_call


def main(mode, cohort, K):
    call = make_call(mode, cohort)
    call()  # warm

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(K):
        call()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(f"### cProfile {mode} {cohort} (K={K}), sort=cumulative\n")
    print("```\n" + s.getvalue() + "```\n")

    from pyinstrument import Profiler
    p = Profiler(interval=0.0005)
    p.start()
    for _ in range(K):
        call()
    p.stop()
    print(f"### pyinstrument {mode} {cohort} (K={K})\n")
    print("```\n" + p.output_text(unicode=False, color=False, show_all=False) + "```\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 2: Run all four combos into the baseline report**

Run:
```bash
cd tmp/svar2_mvp
OUT=prof_out/readbound/python_baseline.md
echo "# SVAR2 read-bound Python-layer baseline ($(date -I))" > "$OUT"
for c in germline somatic; do for m in haplotypes variants; do
  echo "== $m $c =="
  ../../.pixi/envs/dev/bin/python prof_python.py $m $c 200 >> "$OUT" 2>&1
done; done
cd ../..
```
Expected: `python_baseline.md` populated with 4 cProfile tables + 4 pyinstrument trees, no tracebacks. (Lower K to ~50 for `somatic` if a combo is slow.)

- [ ] **Step 3: Record the top Python functions**

Read `tmp/svar2_mvp/prof_out/readbound/python_baseline.md`. Append a short `## Top Python functions (ranked)` section listing, per mode, the 3–5 highest-cumulative gvl Python functions. Expect for **variants**: `_reconstruct_variants`, `_gather_inputs`, `_ragged_arange_gather` / `_ragged_arange_gather_2level`, `_contig_groups`; for **haplotypes**: `get_haps_and_shifts`, `_gather_inputs`, `_assemble_haps`. Note which are pure-Python overhead vs. thin FFI wrappers.

- [ ] **Step 4: Commit**

```bash
git add tmp/svar2_mvp/prof_python.py tmp/svar2_mvp/prof_out/readbound/python_baseline.md
git commit --no-verify -m "perf(svar2): Python-layer baseline profile (cProfile + pyinstrument)"
```

---

### Task A3: Capture the native/Rust baseline (perf) + instruction-count reference

**Files:**
- Create: `tmp/svar2_mvp/prof_perf.sh`
- Create (output): `tmp/svar2_mvp/prof_out/readbound/native_baseline.md`

**Interfaces:**
- Consumes: `prof_getitem.py` (A1).
- Produces: committed per-(mode×cohort) DSO split, Rust symbol self-time, fp call-graph, and a `perf stat -e instructions,cycles` reference used as the Phase-B gate baseline.

- [ ] **Step 1: Rebuild the `.so` with frame pointers**

Run: `RUSTFLAGS="-C force-frame-pointers=yes" pixi run -e dev maturin develop --release`
Expected: build succeeds; this is the binary all A3/Phase-B perf captures use.

- [ ] **Step 2: Write the perf capture script**

Create `tmp/svar2_mvp/prof_perf.sh`:

```bash
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
```

- [ ] **Step 3: Run it**

Run: `bash tmp/svar2_mvp/prof_perf.sh`
Expected: prints `NATIVE_BASELINE_DONE`; `native_baseline.md` has DSO split + symbol + call-graph blocks for all 4 combos. Confirm the split shows **no** `SearchTree::build` / `dense_union` / `overlap_batch` samples (that would mean the union path is being hit — a driver bug). Expect `gather_haps_readbound`, `merge_keys`, `split_to_flat`, `decode_variants_from_split` (variants), `reconstruct_haplotypes_from_svar2` (haplotypes), and numpy cache-slice symbols (`PyArray_Repeat`, `mapiter_get`) instead.

- [ ] **Step 4: Record the ranked native targets**

Append a `## Ranked native targets` section to `native_baseline.md`: per mode, the top 3 native self-time symbols and their DSO (gvl vs genoray_core vs numpy). For **haplotypes**, note whether `gather_haps_readbound` + `split_to_flat` self-time is roughly **2× the diffs-only need** — that confirms the redundant double gather (Task B1). For **variants**, note `split_to_flat` / `decode_variants_from_split` allocation churn (`_int_malloc`/`SpecFromIter` under them → Task B2).

- [ ] **Step 5: Commit**

```bash
git add tmp/svar2_mvp/prof_perf.sh tmp/svar2_mvp/prof_out/readbound/native_baseline.md
git commit --no-verify -m "perf(svar2): native-layer baseline profile (perf DSO/symbol/callgraph + instr reference)"
```

---

## Phase B — Optimizations (each confirmed against Phase A, then parity + instruction-count gated)

> **Before each Task below:** re-read the Phase A baseline and confirm the target actually ranks in the top few. If a candidate does NOT rank (e.g. B1 removed it, or the profile disagrees), skip that Task and instead apply the **same fix→gate template** to whichever function tops the profile, using `cargo asm`/`perf annotate` for native fns. Do not implement an optimization the profile does not justify.

### Task B1: Eliminate the redundant pre-reconstruct gather (haplotypes)

**Rationale (confirm against A3):** `Svar2Haps.get_haps_and_shifts` calls `hap_diffs_from_svar2_readbound` (full `gather_haps_readbound` + `split_to_flat` + `hap_diffs_svar2`) **and then** `reconstruct_haplotypes_from_svar2_readbound` (which repeats gather + split + diffs internally for output sizing). For a haplotypes read the pre-reconstruct diffs are needed **only** when jitter randomizes shifts; the warm `ds[:, :]` read is deterministic/ragged (`shifts = 0`), so the entire first gather is redundant there. `hap_lengths` (which uses `diffs`) is discarded on the haplotypes path (`__call__` keeps only `haps`); the tracks path (out of scope, but must stay green) reuses `get_haps_and_shifts` and *does* need `diffs`/`hap_lengths`, so the guard must preserve them there via an explicit flag.

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`get_haps_and_shifts`, ~lines 319-424; and its tracks caller)
- Test: `tests/dataset/test_svar2_readbound_haps.py`, `tests/dataset/test_svar2_dataset.py` (existing parity oracles — must stay green, including tracks)

**Interfaces:**
- Consumes: `hap_diffs_from_svar2_readbound`, `reconstruct_haplotypes_from_svar2_readbound` (unchanged FFI).
- Produces: `get_haps_and_shifts(..., need_hap_lengths: bool = False)` computes the diffs loop only when its result is consumed (randomized shifts, OR a caller that needs `hap_lengths`/`diffs` — i.e. tracks passes `need_hap_lengths=True`). Same return tuple shape.

- [ ] **Step 1: Add a failing micro-test asserting the diffs kernel is not called for a deterministic haps read**

Add to `tests/dataset/test_svar2_readbound_haps.py`:

```python
def test_deterministic_haps_read_skips_pre_reconstruct_diffs(monkeypatch):
    """A deterministic (shifts=0) haplotypes read must NOT call the separate
    hap_diffs readbound kernel — reconstruct sizes itself internally. Guards the
    double-gather regression."""
    import genvarloader._dataset._svar2_haps as m

    calls = {"diffs": 0}
    real = m.hap_diffs_from_svar2_readbound
    def counting(*a, **k):
        calls["diffs"] += 1
        return real(*a, **k)
    monkeypatch.setattr(m, "hap_diffs_from_svar2_readbound", counting)

    # Build the same small live svar2 dataset the module parity tests use, then:
    #   ds.with_seqs("haplotypes")[:, :]
    # (reuse this file's existing fixture that yields a ds2 Svar2Haps-backed view;
    #  if none is exposed, lift the _open_pair helper from test_svar2_dataset.py.)
    ds2 = _svar2_haps_dataset()          # existing/lifted fixture -> haplotypes view
    ds2[:, :]
    assert calls["diffs"] == 0
```

If no fixture yields a live Svar2Haps haplotypes view in this file, lift `_open_pair` from `tests/dataset/test_svar2_dataset.py` into a local helper `_svar2_haps_dataset()` returning `ds2.with_seqs("haplotypes")`.

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_readbound_haps.py::test_deterministic_haps_read_skips_pre_reconstruct_diffs -v`
Expected: FAIL with `assert 1 == 0` (the diffs kernel is currently called unconditionally).

- [ ] **Step 3: Guard the diffs computation in `get_haps_and_shifts`**

In `python/genvarloader/_dataset/_svar2_haps.py`, add `need_hap_lengths: bool = False` to `get_haps_and_shifts`'s signature, and replace the unconditional diffs block + shifts block (currently ~lines 352-384) with:

```python
        groups = self._contig_groups(contig_ids)

        # diffs are needed pre-reconstruct ONLY to (a) bound randomized jitter
        # shifts, or (b) return hap_lengths/diffs to a caller that uses them
        # (the tracks path). A deterministic/ragged haplotypes read needs
        # neither: reconstruct sizes itself internally. Avoid the redundant
        # gather+split+diffs in that (common warm-read) case.
        randomized = not (deterministic or isinstance(output_length, str))
        need_diffs = randomized or need_hap_lengths

        if need_diffs:
            diffs = np.empty((b, P), np.int32)
            for ci, qsel in groups:
                gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)
                d = hap_diffs_from_svar2_readbound(
                    self.store, self.ds_contigs[ci],
                    gi[0], gi[1], gi[2], gi[3], gi[4], gi[5], gi[6], P,
                )
                diffs[qsel] = np.asarray(d, np.int32).reshape(len(qsel), P)
            hap_lengths = (lengths[:, None] + diffs).astype(np.int32)
        else:
            diffs = np.zeros((b, P), np.int32)      # placeholder (unused downstream)
            hap_lengths = np.broadcast_to(
                lengths[:, None].astype(np.int32), (b, P)
            ).copy()

        if randomized:
            max_shift = diffs.clip(min=0)
            max_shift = max_shift + (lengths - output_length).clip(min=0)[:, None]
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)
        else:
            shifts = np.zeros((b, P), np.int32)
```

Then have the tracks caller pass `need_hap_lengths=True`. Find the caller: `grep -n "get_haps_and_shifts" python/genvarloader/_dataset/*.py` — it is invoked from `HapsTracks` dispatch (the tracks path) and from `Svar2Haps.__call__` (haplotypes). Update the tracks call site to `get_haps_and_shifts(..., need_hap_lengths=True)`; leave the haplotypes call site at the `False` default.

> NOTE: when `randomized` is False but `output_length` is a fixed `int`, `hap_lengths` in the placeholder branch is `lengths` broadcast — but `reconstruct_haplotypes_from_svar2_readbound` with a fixed `output_length >= 0` recomputes per-hap lengths internally and ignores `hap_lengths`, so this is safe. Confirm no downstream consumer reads `hap_lengths` for a fixed-int deterministic haps read; if one does, set `need_hap_lengths=True` for that path too.

- [ ] **Step 4: Python-only change — run the new test + full svar2 parity (incl. tracks)**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_svar2_readbound_haps.py::test_deterministic_haps_read_skips_pre_reconstruct_diffs -v
pixi run -e dev pytest tests/dataset -k svar2 -q
```
Expected: the new test PASSES (`calls["diffs"] == 0`); all svar2 parity tests stay green (haplotypes/variants **and tracks** byte-identical to oracle — the tracks path still calls the diffs kernel via `need_hap_lengths=True`).

- [ ] **Step 5: Full-tree parity + instruction-count delta**

Run:
```bash
pixi run -e dev pytest tests -q
P=.pixi/envs/dev/bin/python
/carter/users/dlaub/.pixi/bin/perf stat -e instructions,cycles -- \
  $P tmp/svar2_mvp/prof_getitem.py haplotypes germline 500 2>&1 | grep -E 'instructions|cycles'
```
Expected: full tree green; instructions/read materially lower than the A3 baseline for `haplotypes` (record the % drop). If HW counters are unavailable, record cProfile primitive-call-count drop instead.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_haps.py tests/dataset/test_svar2_readbound_haps.py
git commit -m "perf(svar2): skip redundant pre-reconstruct gather for deterministic haplotype reads"
```

---

### Task B2: Pre-size `split_to_flat` + `decode_variants_from_split` allocations (gvl Rust)

**Rationale (confirm against A3):** both functions build output `Vec`s with `Vec::new()` (no capacity) and grow them in hot loops — `split_to_flat` (`src/svar2/mod.rs:159`) for `dense_pos`/`dense_key`/`dense_present` (used by BOTH modes — variants decode and haplotype diffs/reconstruct call it), and `decode_variants_from_split` (`src/svar2/mod.rs:262`) for `pos`/`ilen`/`alt_bytes`/`str_off` (variants). `_int_malloc`/`SpecFromIter` were hot even in the old profile. Pre-sizing is byte-identical. **Only implement the parts that rank in the A3 native top symbols after B1.**

**Files:**
- Modify: `src/svar2/mod.rs` (`split_to_flat` 159-237; `decode_variants_from_split` 262-320)
- Test: `cargo test` in gvl (`test_split_to_flat_*`, `test_decode_variants_from_split_*` in `src/svar2/mod.rs`) + the pytest parity suite

**Interfaces:**
- Consumes/Produces: `split_to_flat(&BatchResultSplit) -> FlatChannels` and `decode_variants_from_split(&BatchResultSplit, &[u8], &[i64]) -> VariantsSoa` — signatures and output bytes unchanged; only allocation strategy changes.

- [ ] **Step 1: Confirm the existing unit tests cover both**

Run: `grep -n "fn test_split_to_flat\|fn test_decode_variants_from_split" src/svar2/mod.rs`
Expected: `test_split_to_flat_marshals_readbound_split`, `test_split_to_flat_trailing_zero_byte_is_allocated`, `test_decode_variants_from_split_merges_and_decodes` exist — the byte-identity guard.

- [ ] **Step 2: Pre-size `split_to_flat`**

In `split_to_flat`, compute the total dense entry count once and reserve; pre-size the presence bitstream and drop the on-set `resize` growth. Replace the `dense_pos`/`dense_key` init:

```rust
    let dense_total: usize = (0..n_q)
        .map(|q| {
            let (ss, se) = br.dense_snp_range[q];
            let (is_, ie) = br.dense_indel_range[q];
            (se - ss) + (ie - is_)
        })
        .sum();
    let mut dense_pos: Vec<i32> = Vec::with_capacity(dense_total);
    let mut dense_key: Vec<i32> = Vec::with_capacity(dense_total);
```

and the presence bitstream:

```rust
    let total_bits: usize = (0..h_count)
        .map(|h| {
            let q = h / ploidy;
            let (ss, se) = br.dense_snp_range[q];
            let (is_, ie) = br.dense_indel_range[q];
            (se - ss) + (ie - is_)
        })
        .sum();
    let mut dense_present: Vec<u8> = vec![0u8; total_bits.div_ceil(8)];
```

Inside the hap loop, drop the `if dense_present.len() <= byte { resize }` guards and set bits directly (`dense_present[bit_acc / 8] |= 1 << (bit_acc % 8);`). Keep the final `dense_present.resize(bit_acc.div_ceil(8), 0);` as a no-op safety.

- [ ] **Step 3: Pre-size `decode_variants_from_split`**

Replace the `Vec::new()` inits (`src/svar2/mod.rs:272-275`) with capacity-reserved vectors (an over-estimate upper bound is fine for `with_capacity`):

```rust
    let flat = split_to_flat(br);
    let ploidy = br.ploidy;
    let n_q = br.n_regions;
    let h_count = n_q * ploidy;

    // Upper bound on total merged variants across all haps: every vk entry plus
    // every dense window entry (present or not). Over-reserving is harmless.
    let vk_total = flat.vk_off[h_count] as usize;
    let dense_bits = flat.dense_present_off[h_count] as usize;
    let cap = vk_total + dense_bits;
    let mut pos: Vec<i32> = Vec::with_capacity(cap);
    let mut ilen: Vec<i32> = Vec::with_capacity(cap);
    let mut alt_bytes: Vec<u8> = Vec::with_capacity(cap);
    let mut str_off: Vec<i64> = Vec::with_capacity(cap + 1);
    str_off.push(0);
    let mut var_off: Vec<i64> = Vec::with_capacity(h_count + 1);
    var_off.push(0);
```

(Leave the per-hap loop body unchanged.)

- [ ] **Step 4: Rebuild + Rust unit tests**

Run:
```bash
RUSTFLAGS="-C force-frame-pointers=yes" pixi run -e dev maturin develop --release
cargo test split_to_flat decode_variants_from_split 2>&1 | tail -20   # (set LD_LIBRARY_PATH to .pixi/envs/dev/lib if libpython load fails)
```
Expected: the named unit tests PASS (byte-identical output).

- [ ] **Step 5: Full-tree parity + instruction delta (both modes)**

Run:
```bash
pixi run -e dev pytest tests -q
P=.pixi/envs/dev/bin/python
for m in variants haplotypes; do
  echo "== $m =="
  /carter/users/dlaub/.pixi/bin/perf stat -e instructions,cycles -- \
    $P tmp/svar2_mvp/prof_getitem.py $m somatic 300 2>&1 | grep -E 'instructions|cycles'
done
```
Expected: full tree green; instructions/read down vs A3 baseline for both modes (record %).

- [ ] **Step 6: Commit**

```bash
git add src/svar2/mod.rs
git commit -m "perf(svar2): pre-size split_to_flat + decode_variants_from_split allocations (byte-identical)"
```

---

### Task B3: De-duplicate the twin ragged gather in `_reconstruct_variants` (gvl Python)

**Rationale (confirm against A2):** in `_reconstruct_variants` (`python/genvarloader/_dataset/_svar2_haps.py:585-586`), `pos` and `ilen` are permuted back to global order by two separate `_ragged_arange_gather(pos_c, grouped_var_off, perm)` / `(ilen_c, grouped_var_off, perm)` calls with **identical** `offsets` and `perm` — so the `lens`/`new_off`/`within`/`src` index arrays are computed twice. Compute the source-index permutation once and apply it to both. **Only implement if `_ragged_arange_gather` / `_reconstruct_variants` ranks in the A2 variants top functions.**

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`_ragged_arange_gather` region + `_reconstruct_variants`)
- Test: `tests/dataset/test_svar2_readbound_variants.py`, `tests/dataset/test_svar2_dataset.py` (variants parity — must stay green)

**Interfaces:**
- Produces: a helper `_ragged_arange_src(offsets, perm) -> (src, new_off)` returning the 1-level reorder source index + new offsets; `_ragged_arange_gather` is re-expressed on it; `_reconstruct_variants` computes `src`/`var_off_g` once and does `pos_g = pos_c[src]`, `ilen_g = ilen_c[src]`.

- [ ] **Step 1: Add the shared-index helper and re-express `_ragged_arange_gather`**

In `python/genvarloader/_dataset/_svar2_haps.py`, add above `_ragged_arange_gather`:

```python
def _ragged_arange_src(
    offsets: NDArray[np.integer], perm: NDArray[np.integer]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Source-row index + new offsets for a 1-level ragged reorder by ``perm``.

    ``new_data == data[src]``; ``src`` and ``new_off`` depend only on
    ``(offsets, perm)`` — so callers reordering several parallel data arrays by
    the same key compute this ONCE and index each array.
    """
    offsets = np.asarray(offsets, np.int64)
    lens = np.diff(offsets)
    new_lens = lens[perm]
    new_off = lengths_to_offsets(new_lens, np.int64)
    n = int(new_off[-1])
    if n == 0:
        return np.zeros(0, np.int64), new_off
    within = np.arange(n, dtype=np.int64) - np.repeat(new_off[:-1], new_lens)
    src = np.repeat(offsets[perm], new_lens) + within
    return src, new_off
```

Then rewrite `_ragged_arange_gather` to use it:

```python
def _ragged_arange_gather(
    data: NDArray, offsets: NDArray[np.integer], perm: NDArray[np.integer]
) -> tuple[NDArray, NDArray[np.int64]]:
    """Reorder the rows of a 1-level ragged array ``(data, offsets)`` by ``perm``."""
    src, new_off = _ragged_arange_src(offsets, perm)
    if src.size == 0:
        return data[:0].copy(), new_off
    return data[src], new_off
```

- [ ] **Step 2: Use the shared index for pos + ilen in `_reconstruct_variants`**

Replace (`python/genvarloader/_dataset/_svar2_haps.py:585-586`):

```python
        pos_g, var_off_g = _ragged_arange_gather(pos_c, grouped_var_off, perm)
        ilen_g, _ = _ragged_arange_gather(ilen_c, grouped_var_off, perm)
```

with:

```python
        src, var_off_g = _ragged_arange_src(grouped_var_off, perm)
        if src.size == 0:
            pos_g = pos_c[:0].copy()
            ilen_g = ilen_c[:0].copy()
        else:
            pos_g = pos_c[src]
            ilen_g = ilen_c[src]
```

(The 2-level `alt` gather is left as-is; it is a different offset structure.)

- [ ] **Step 3: Run variants parity (Python-only change, no rebuild)**

Run:
```bash
pixi run -e dev pytest tests/dataset -k svar2 -q
```
Expected: all svar2 variants parity tests green (byte-identical `pos`/`ilen`/`alt` reorder).

- [ ] **Step 4: Full-tree parity + instruction delta**

Run:
```bash
pixi run -e dev pytest tests -q
P=.pixi/envs/dev/bin/python
/carter/users/dlaub/.pixi/bin/perf stat -e instructions,cycles -- \
  $P tmp/svar2_mvp/prof_getitem.py variants germline 500 2>&1 | grep -E 'instructions|cycles'
```
Expected: full tree green; variants instructions/read down vs baseline (record %; if small, keep anyway — it removes a duplicated O(total_variants) pass and is a clarity win).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_haps.py
git commit -m "perf(svar2): compute the pos/ilen ragged reorder index once in variants decode"
```

---

### Task B4: `cargo asm` pass on ALL hot native functions — parallel subagent fan-out

**Rationale:** after B1-B3, re-profile and collect the **full set** of native functions that still carry meaningful self-time (gvl and genoray), not just the #1. Because these functions are independent, optimize them **in parallel — one subagent per function**, each inspecting that function's assembly, finding a bounds-check / vectorization / allocation defect, and applying a byte-identical fix guarded by its **own per-function parity test**. The specific instruction change is *discovered from the asm*, not pre-guessed; each subagent's deliverable is a measured instruction-count reduction (or a documented "no safe win"). The branch owner then merges the fixes sequentially behind the full-tree parity gate.

**Execution model:** REQUIRED SUB-SKILLs — superpowers:dispatching-parallel-agents (fan-out) + superpowers:using-git-worktrees (isolation). Per project rules: subagent implementers run on **Sonnet or weaker**; worktrees live under `.claude/worktrees/` of the owning repo. Worktree isolation is REQUIRED because several candidates share a file (`src/svar2/mod.rs` holds `split_to_flat`/`merge_hap`/`decode_alt`; genoray `src/spine.rs` holds `merge_keys`) — parallel in-place edits to the same file would clobber each other.

**Files:**
- Modify (per subagent, in its own worktree): the file owning that function — gvl `src/reconstruct.rs` / `src/svar2/mod.rs`, **or** genoray `/carter/users/dlaub/projects/genoray/src/{spine.rs,query.rs}` (branch `svar-2`)
- Test (per subagent): a focused `cargo test` in the owning repo asserting byte-identical output for that function

**Interfaces:**
- Consumes: A3/post-B3 native profile ranking (the hot-function set).
- Produces: per function — same signature, byte-identical output, fewer hot-loop instructions, and a committed per-function parity test.

- [ ] **Step 1: Re-profile and enumerate the hot-function set**

Run: `cp tmp/svar2_mvp/prof_out/readbound/native_baseline.md tmp/svar2_mvp/prof_out/readbound/native_after_b1b3.md` then `bash tmp/svar2_mvp/prof_perf.sh`. From the per-mode symbol tables, list every gvl/genoray self-time symbol above a cutoff (e.g. ≥1.5% self across any mode). **Exclude** libc (`_int_malloc`/`memmove`/`memset`) and numpy symbols — those are structural, not asm-fixable. Write the resulting list (function → owning repo/file → which mode exercises it) into `tmp/svar2_mvp/prof_out/readbound/asm_targets.md`. This list is the fan-out work-list.

- [ ] **Step 2: Create one worktree per target function**

For each function `F` in the work-list, create an isolated worktree off the correct branch. gvl functions:
```bash
cd "$(git rev-parse --show-toplevel)"
git worktree add ".claude/worktrees/asm-<F>" svar2-m6b-kernel
```
genoray functions:
```bash
cd /carter/users/dlaub/projects/genoray
git worktree add ".claude/worktrees/asm-<F>" svar-2
```
Record the worktree path per function in `asm_targets.md`.

- [ ] **Step 3: Dispatch one Sonnet subagent per function (in parallel)**

Dispatch all subagents in a single batch (superpowers:dispatching-parallel-agents). Give each subagent this exact brief, filled in for its function `F`, file, worktree path, and the mode/cohort that exercises it:

> You are optimizing exactly ONE Rust function, `F`, in worktree `<path>` (do not touch any other file). Model: Sonnet.
> 1. Dump its assembly: `cargo asm --release --lib "<fully::qualified::F>" 2>/dev/null | head -200` (if the path is ambiguous, run `cargo asm --release --lib` and grep the candidate list). Cross-reference `perf annotate -i tmp/svar2_mvp/prof_out/readbound/<tag>.data --stdio` for the hot source lines.
> 2. Identify a byte-identical defect: per-iteration `panic`/bounds-check in the hot loop (→ iterators or `get_unchecked` behind a proven invariant), scalar copy/merge loops (→ `extend_from_slice`/`copy_from_slice`), or in-loop allocation (→ `with_capacity`). Comment the specific instruction pattern on the fix.
> 3. Apply the minimal fix. Behavior MUST stay byte-identical.
> 4. Write a **per-function parity test** in the owning repo: a `#[test]` that runs `F` on representative inputs and asserts the exact output (compare against a hand-checked expected value, or against a slow reference implementation of `F` inline in the test). Name it `test_<F>_byte_identical`.
> 5. `cargo test test_<F>_byte_identical` (set `LD_LIBRARY_PATH` to `.pixi/envs/dev/lib` if libpython fails to load in gvl). It MUST pass.
> 6. Measure: `RUSTFLAGS="-C force-frame-pointers=yes" pixi run -e dev maturin develop --release` (from the gvl worktree; for genoray targets, build gvl against the genoray worktree path-dep), then same-session `perf stat -e instructions,cycles -- .pixi/envs/dev/bin/python tmp/svar2_mvp/prof_getitem.py <mode> <cohort> <K>` before vs after. Report the instruction delta.
> 7. Commit in your worktree: `perf(...): <F> asm fix — <what> (byte-identical, -N% instr)`. If you found NO safe byte-identical win, revert your edits, commit nothing, and report "no safe win for F" with the asm reason.

- [ ] **Step 4: Collect results**

Gather each subagent's report (fix applied? instruction delta? or no-safe-win) into `asm_targets.md`. Discard worktrees for no-win functions: `git worktree remove .claude/worktrees/asm-<F>`.

- [ ] **Step 5: Merge the winning fixes sequentially behind the parity gate**

For each winning function, in turn (gvl fixes onto `svar2-m6b-kernel`, genoray fixes onto `svar-2`):
```bash
# gvl example (from the main worktree):
cd "$(git rev-parse --show-toplevel)"
git cherry-pick <worktree-commit-sha>     # or merge the worktree branch
RUSTFLAGS="-C force-frame-pointers=yes" pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q           # full-tree parity after EACH merge
```
If a merge conflicts (two fixes in the same file) or the full tree goes red, resolve the conflict / drop the offending fix, re-run, and record the decision. genoray fixes: cherry-pick onto `svar-2` in the genoray checkout, then rebuild gvl (crate path-dep) and re-run the gvl full tree. After all merges, remove the worktrees.

- [ ] **Step 6: Record**

Append the final per-function outcomes (merged / dropped / no-win) and cumulative instruction deltas per mode to `tmp/svar2_mvp/prof_out/readbound/native_after_b1b3.md`.

---

### Task B5: Consolidate results & update the profiling report

**Files:**
- Create: `tmp/svar2_mvp/prof_out/readbound/RESULTS.md`
- Modify: genoray `docs/roadmap/svar-2.md` **only if** a kernel signature changed (record it under the relevant milestone note); gvl design spec §9 open-question resolution

- [ ] **Step 1: Write the results summary**

Create `tmp/svar2_mvp/prof_out/readbound/RESULTS.md` with a table: per mode (haplotypes, variants) × cohort, baseline vs final instructions/read (and cProfile fallback where HW counters were absent), the optimizations applied (B1/B2/B3/B4), and any candidate that was profiled-but-skipped (with the reason). State the parity result (full tree + svar2 suite green, incl. tracks) explicitly. Note the **deferred** items: tracks (out of scope this round; its `get_haps_and_shifts` double-gather is unaddressed and still runs the diffs kernel via `need_hap_lengths=True`), and variant-windows (guarded `NotImplementedError` — profile once implemented).

- [ ] **Step 2: Reconcile docs if signatures changed**

If any FFI/kernel signature changed, update the genoray roadmap note and the gvl design spec's §9 open-question resolution. If nothing signature-level changed, state "read-path internals only; no API/format/doc-surface change" in RESULTS.md. (The `get_haps_and_shifts` `need_hap_lengths` param is an internal signature, not public API — no skill/api.md/docs update needed.)

- [ ] **Step 3: Commit**

```bash
git add tmp/svar2_mvp/prof_out/readbound/RESULTS.md
git commit --no-verify -m "perf(svar2): read-bound getitem optimization results summary"
# + any genoray docs commit on svar-2 if applicable
```

---

## Self-Review

**Spec coverage:**
- Profile the real read-bound path (not union oracle) → A1–A3 (driver forces `Dataset.open(svar2)`; A3 Step 3 asserts no `SearchTree`/`overlap_batch` samples). ✓
- Tooling substitution (cProfile+pyinstrument / perf) → A2, A3; pyinstrument dep → A1. ✓
- In-scope modes haplotypes + variants; tracks dropped from profiling but parity-protected; variant-windows deferred (guarded) → Global Constraints + A1 driver + B5 deferred note. ✓
- Two-repo landing (gvl #266 / genoray svar-2) → Global Constraints + B4/B5 commit steps. ✓
- Measure→confirm→fix→re-measure + parity + instr-delta → per-Task gates; Phase B preamble enforces "confirm before implementing." ✓
- Haplotypes double-gather → B1 (concrete, tracks-safe via `need_hap_lengths`). Variants+shared alloc churn → B2 (concrete). Variants Python twin-gather → B3 (concrete). cargo asm on ALL hot Rust fns → B4 (parallel subagent fan-out, per-function parity tests, worktree-isolated). ✓
- No format/API change → B5 Step 2 reconciliation. ✓

**Placeholder scan:** B4 is investigation-then-fix by nature (asm-discovered) and now a parallel fan-out, but specifies the exact hot-set enumeration rule + cutoff, worktree-per-function isolation, the verbatim per-subagent brief (with exact `cargo asm`/`perf annotate`/`perf stat` commands, the defect classes, the per-function `test_<F>_byte_identical` requirement), and a sequential merge-behind-parity gate — not a hand-wave. B1/B2/B3 carry full code. No "TBD"/"similar to Task N".

**Type consistency:** `make_call(mode, cohort)` defined in A1, reused verbatim in A2. `get_haps_and_shifts` gains `need_hap_lengths: bool = False` (B1) — threaded to the tracks caller in the same task; haplotypes caller uses the default. `_ragged_arange_src(offsets, perm) -> (src, new_off)` defined and consumed within B3; `_ragged_arange_gather` re-expressed on it (same return contract). `split_to_flat`/`decode_variants_from_split` signatures unchanged (B2). Store sample count via `SparseVar2.n_samples` (verified attribute).
