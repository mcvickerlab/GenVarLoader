# Rust Migration Phase 5 Wrap-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 5's finalization threads (thin-shim audit, cargo-standalone verification, seqpro-core released-dep verification, W6 perf re-baseline) and land them as one PR into `rust-migration`, leaving the `rust-migration → master` merge to the maintainer.

**Architecture:** Four mostly-independent units. Three are verification + roadmap documentation (no production code); one (Unit B) may carry a small build/config fix if `cargo test` does not run standalone. Unit D is a measurement pass on Carter. A final task sets the Phase 5 status marker and runs the full gate.

**Tech Stack:** Rust (PyO3 0.28 abi3, ndarray, rayon, seqpro-core 0.1), Python 3.10–3.13, maturin, pixi (`-e dev`), pytest + pytest-benchmark, cargo test, ruff/pyrefly/clippy.

**Spec:** `docs/superpowers/specs/2026-06-27-rust-migration-phase-5-wrapup-design.md`

## Global Constraints

- **Branch:** `phase-5-w6-wrapup` (already created off `rust-migration`). All commits land here.
- **PR target:** `rust-migration` (NOT master). Do not merge to master — the maintainer triggers `rust-migration → master` separately, no-squash.
- **Out of scope:** Phase 6 (absorb genoray); the "single big `__getitem__` kernel" architectural collapse (Unit A *audits* it, does not build it).
- **Rebuild before testing Rust:** `pixi run -e dev maturin develop --release` BEFORE any pytest run that imports the extension. pytest does NOT rebuild Rust.
- **No numba A/B:** numba was deleted in W5. There is no live numba backend; all perf comparison is rust serial-vs-rayon (same session) + the W4-recorded numba figures. Do NOT re-checkout a numba commit.
- **Carter perf caveat:** shared HPC node; absolute wall-clock drifts ≥2× across sessions. Durable signals = byte-identical parity (already gated) + same-session improve-or-hold + deterministic counts. See `[[gvl-rust-perf-gate-shared-node-noise]]`.
- **Corpus:** `chr22_geuv.gvl` (format 2.0, 165 regions × 5 samples). Assumed present from W4/W5; Task 4 Step 1 verifies and rebuilds if absent.
- **Roadmap is source of truth:** `docs/roadmaps/rust-migration.md` — tick items, set the Phase 5 marker, add a notes-log entry, record measurements under the checkpoint.

---

### Task 1: Thin-shim audit (Unit A)

Investigation + documentation only. **No production code changes.** Produce a precise "what's left to collapse the PyO3 surface" verdict and write it into the roadmap.

**Files:**
- Create: `docs/roadmaps/phase-5-w6-thin-shim-audit.md` (the detailed audit)
- Modify: `docs/roadmaps/rust-migration.md` (Phase 5 section + a notes-log entry referencing the audit)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the audit verdict (bucket-2 "remaining collapsible glue" list) that Task 5 reads to set the Phase 5 status marker.

- [ ] **Step 1: Inventory the read-path call chain**

Trace `Dataset.__getitem__` to its FFI calls and list every Python function on the hot path between the public API and the `from ..genvarloader import ...` call. Use:

```bash
rtk grep -n "def __getitem__\|_reconstruct\|reconstruct_haplotypes_fused\|intervals_and_realign_track_fused\|assemble_variant_buffers" \
  python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_reconstruct.py \
  python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_query.py
```

Read `_dataset/_reconstruct.py`, `_dataset/_haps.py`, `_dataset/_query.py` in full to see the per-batch work each does before/after the FFI crossing.

- [ ] **Step 2: Inventory the FFI surface**

List the registered pyfunctions and which are fused `__getitem__` kernels:

```bash
rtk grep -n "wrap_pyfunction!\|add_class" src/lib.rs
```

Expected: ~28 entries incl. the five fused kernels (`reconstruct_haplotypes_fused`, `reconstruct_annotated_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused`, `reconstruct_annotated_haplotypes_spliced_fused`, `intervals_and_realign_track_fused`) and `assemble_variant_buffers_{u8,i32}`.

- [ ] **Step 3: Confirm the dispatch layer is fully gone**

```bash
ls python/genvarloader/_dispatch.py 2>&1                 # expect: No such file
rtk grep -rn "GVL_BACKEND\|_dispatch\|import numba\|from numba\|nb\.njit\|nb\.prange" python/genvarloader/ --include=*.py
```

Expected: zero matches (confirms W5 removed the rust/numba switch and Python calls Rust directly). Also delete the stale bytecode so it cannot mislead future greps:

```bash
rm -f python/genvarloader/__pycache__/_dispatch.cpython-*.pyc
```

- [ ] **Step 4: Classify each read-path Python step into the three buckets**

For every per-batch Python step found in Step 1, classify as: (1) **intentional shim** (indexing sugar / torch / validation / error messages — stays in Python), (2) **remaining collapsible glue** (per-batch coercion/alloc/object churn worth a future kernel), or (3) **already-collapsed** (one FFI crossing, no material Python work). Cross-reference the Phase 3 optimization-targets section of the roadmap (zero-copy `_ffi_array`, `_HapsFfiStatic` caching, uninit buffers) — those already eliminated the major bucket-2 items.

- [ ] **Step 5: Write the audit document**

Write `docs/roadmaps/phase-5-w6-thin-shim-audit.md` containing: the read/write-path call-chain inventory, the FFI surface list, the three-bucket classification table (one row per Python step with its bucket + justification), and a one-paragraph **verdict**: either "shim is already thin — bucket-2 list is empty/negligible, the single-big-kernel collapse is not warranted as Phase 5 work" OR "bucket-2 glue remains: <explicit list>". Include the `to_rc` / RC handling and any `np.ascontiguousarray` survivors (there should be none on per-sample-scale memmaps — that was the scale-guard fix; confirm via `rtk grep -rn "ascontiguousarray" python/genvarloader/_dataset/`).

- [ ] **Step 6: Update the roadmap Phase 5 section**

In `docs/roadmaps/rust-migration.md`, under Phase 5, annotate the "Collapse the PyO3 surface so Python is a true shim" checklist item with the audit verdict (link to the audit doc). Do NOT tick or mark the phase yet — Task 5 sets the final marker. Add a notes-log entry dated 2026-06-27 (Phase 5 W6 — thin-shim audit) summarizing the verdict.

- [ ] **Step 7: Commit**

```bash
rtk git add docs/roadmaps/phase-5-w6-thin-shim-audit.md docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): Phase 5 W6 thin-shim audit — classify remaining PyO3 surface glue

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: cargo-testable standalone verification (Unit B)

Confirm `cargo test` builds and runs the Rust suite without the pixi/maturin/Python-extension layer. This is the only task that may carry a code/config fix.

**Files:**
- Modify (only if broken): `Cargo.toml` and/or `.cargo/config.toml` (whatever the minimal fix requires)
- Modify: `docs/roadmaps/rust-migration.md` (record the standalone result + the canonical invocation)

**Interfaces:**
- Consumes: nothing.
- Produces: the verified standalone-test invocation string recorded in the roadmap; Task 5's gate reuses it.

- [ ] **Step 1: Run the standalone cargo suite from a clean shell**

Run WITHOUT pixi, from the repo root:

```bash
cargo test --release 2>&1 | tail -30
```

Expected (pass case): all tests pass (W5 reported 114 cargo tests). If it links and passes, the crate is already standalone-testable — skip to Step 4.

- [ ] **Step 2: If it fails to link/build, diagnose**

The most likely failure is pyo3 needing a libpython at link time (the `extension-module` feature is non-default, so `cargo test` links a real interpreter). Capture the exact error:

```bash
cargo test --release 2>&1 | grep -iE "error|undefined|python|link" | head -20
```

If it is a libpython discovery issue, the minimal fix is to ensure a Python is discoverable (e.g. `PYO3_PYTHON=$(pixi run -e dev which python) cargo test --release`). Prefer documenting the invocation over adding config that could perturb the abi3 wheel build. Only edit `Cargo.toml`/`.cargo/config.toml` if there is no env-only path.

- [ ] **Step 3: Re-run to confirm the fix**

```bash
PYO3_PYTHON=$(pixi run -e dev which python) cargo test --release 2>&1 | tail -15   # or the plain command if no fix was needed
```

Expected: all tests pass.

- [ ] **Step 4: Record the result in the roadmap**

In `docs/roadmaps/rust-migration.md` Phase 5, annotate the "Confirm the crate is fully cargo-testable standalone" item with the verified invocation and the pass count (do NOT tick yet — Task 5 does the final marker). If a fix was needed, note it.

- [ ] **Step 5: Commit**

```bash
rtk git add Cargo.toml .cargo/config.toml docs/roadmaps/rust-migration.md 2>/dev/null; rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): verify crate is cargo-testable standalone (Phase 5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: seqpro-core released-dep verification (Unit C)

Confirm seqpro-core resolves from crates.io with no path/patch override, and correct the stale Phase 1 roadmap note.

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (correct the stale Phase 1 "editable path-dep" note)

**Interfaces:**
- Consumes: nothing.
- Produces: corrected roadmap text.

- [ ] **Step 1: Confirm the resolved source is the registry**

```bash
rtk grep -n -A3 'name = "seqpro-core"' Cargo.lock
rtk grep -rn "seqpro-core\|\[patch\|path =" Cargo.toml
```

Expected: `Cargo.lock` shows `version = "0.1.0"`, `source = "registry+https://github.com/rust-lang/crates.io-index"`, with a checksum; `Cargo.toml` shows `seqpro-core = "0.1"` and NO `[patch]` or `path =` override.

- [ ] **Step 2: Confirm a clean build resolves it without a local checkout**

```bash
cargo build --release 2>&1 | grep -iE "seqpro|error" | head; echo "exit: ${PIPESTATUS[0]}"
```

Expected: builds clean, seqpro-core pulled from registry (no "path" / local-edit lines).

- [ ] **Step 3: Correct the stale Phase 1 roadmap note**

In `docs/roadmaps/rust-migration.md`, find the Phase 1 bullet and notes-log lines that say seqpro-core is "editable; flip to git/crates.io before shipping" / "path dep (editable…)". Replace with text stating it is already a released crates.io dependency (`seqpro-core 0.1.0`, registry source, verified in `Cargo.lock`), so the shipping prerequisite is satisfied.

- [ ] **Step 4: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): seqpro-core is already a released crates.io dep (correct stale Phase 1 note)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: W6 perf re-baseline — serial vs rayon (Unit D)

Measure the rayon multi-thread speedup curve + peak-RSS deltas on Carter and record under the Phase 5 checkpoint. Long pole.

**Files:**
- Create: `docs/roadmaps/phase-5-w6-perf-rebaseline.md` (full tables + methodology)
- Modify: `docs/roadmaps/rust-migration.md` (summary under the Phase 5 checkpoint)

**Interfaces:**
- Consumes: the verified release build (rebuild in Step 2).
- Produces: the rayon speedup curve + RSS deltas referenced by Task 5's checkpoint update.

- [ ] **Step 1: Verify the corpus exists (rebuild if absent)**

```bash
ls -la tests/benchmarks/data/chr22_geuv.gvl 2>&1
```

If present, continue. If absent, rebuild (needs `/carter` or `GVL_BENCH_SOURCE`):

```bash
pixi run -e dev python tests/benchmarks/data/build_realistic.py
```

- [ ] **Step 2: Rebuild the extension release and identify the parallel toggle**

```bash
pixi run -e dev maturin develop --release
```

Find how the read kernels expose the W5 `parallel` gate and how to force serial vs parallel (the `should_parallelize(total_out_bytes)` threshold in `_threads.py` and `RAYON_NUM_THREADS`):

```bash
rtk grep -rn "should_parallelize\|RAYON_NUM_THREADS\|parallel" python/genvarloader/_threads.py
```

- [ ] **Step 3: Capture the serial baseline (1 thread)**

Run the de-noised e2e harness pinned to one rayon thread for the seq/track paths, and `profile.py` for the variants paths:

```bash
RAYON_NUM_THREADS=1 pixi run -e dev pytest tests/benchmarks/test_e2e.py -q 2>&1 | tail -30
RAYON_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 2000
RAYON_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variant-windows --n-batches 2000
```

Record ms/batch (pedantic min for e2e modes; wall avg for variants modes) per mode.

- [ ] **Step 4: Capture the thread sweep (2 / 4 / 8 / all cores)**

Repeat Step 3's commands with `RAYON_NUM_THREADS=2`, `=4`, `=8`, and unset (default = all cores). Capture ms/batch per mode per thread count. Also capture peak RSS for one representative parallel run vs the serial run via memray:

```bash
pixi run -e dev memray-tracks 2>&1 | tail; pixi run -e dev memray-haps 2>&1 | tail   # then: memray stats <output>
```

(If `should_parallelize`'s byte threshold suppresses parallelism on this small corpus for some modes, note which modes never crossed the threshold — that is itself a finding, not a failure.)

- [ ] **Step 5: Write the perf doc**

Write `docs/roadmaps/phase-5-w6-perf-rebaseline.md` with: methodology (corpus, harness, HEAD, machine, `maturin develop --release`), a per-mode serial-vs-thread-count table (ms/batch + speedup vs serial), the peak-RSS serial-vs-parallel deltas, a note that numba A/B is unavailable (W5 deletion) with a pointer to the W4 figures (`docs/roadmaps/phase-5-w4-final-ab.md`), and the node-noise caveat. State the gvl-attributable conclusion (rayon speedup achieved; modes below the parallelism threshold noted).

- [ ] **Step 6: Record the summary in the roadmap checkpoint**

In `docs/roadmaps/rust-migration.md` Phase 5 "Checkpoint" area, add the rayon speedup summary + RSS deltas (link to the perf doc). This satisfies "full perf re-baseline recorded here."

- [ ] **Step 7: Commit**

```bash
rtk git add docs/roadmaps/phase-5-w6-perf-rebaseline.md docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): Phase 5 W6 perf re-baseline — rayon serial-vs-multithread speedup + RSS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Phase 5 status disposition + full gate + PR

Set the Phase 5 marker from the audit verdict, run the full project gate, finalize the roadmap, and open the PR into `rust-migration`.

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (tick items, set Phase 5 marker, final notes-log entry)

**Interfaces:**
- Consumes: Task 1 audit verdict, Task 2 standalone result, Task 3 seqpro verification, Task 4 perf re-baseline.
- Produces: the PR.

- [ ] **Step 1: Rebuild and run the full pytest tree**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q 2>&1 | tail -20
```

Expected: green (single rust-only run; numba backend gone). Note pass/skip/xfail counts; the W5 baseline was parity+dataset+unit = 692 passed / 35 skipped / 2 xfailed and whole-tree green.

- [ ] **Step 2: Run cargo tests + lint + format + typecheck + clippy**

```bash
cargo test --release 2>&1 | tail -5
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
cargo clippy --release 2>&1 | tail -10
```

Expected: cargo 114 passed; ruff/format/typecheck/clippy all clean.

- [ ] **Step 3: Confirm the abi3 wheel builds**

```bash
pixi run -e dev maturin build --release 2>&1 | tail -5
```

Expected: wheel builds clean.

- [ ] **Step 4: Set the Phase 5 status marker**

Per the spec disposition, using Task 1's verdict:
- If the audit found the shim already thin AND checkpoint criteria are met (numba count = 0 ✓, perf re-baseline ✓, cargo-standalone ✓): tick the "Collapse PyO3 surface" item with the audit verdict, tick "cargo-testable standalone", set Phase 5 marker to **✅**, and re-file any residual collapse as a separate optimization track entry.
- If bucket-2 glue remains: keep Phase 5 **🚧**, tick only the completed items (cargo-standalone, perf recorded), and leave the collapse item open with the audited remainder list.

Add a final notes-log entry dated 2026-06-27 (Phase 5 W6 — wrap-up) summarizing: thin-shim verdict, cargo-standalone confirmation, seqpro-core released confirmation, perf re-baseline result, and the chosen Phase 5 marker. Note that the `rust-migration → master` merge is left to the maintainer.

- [ ] **Step 5: Commit the finalization**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): finalize Phase 5 W6 — set status marker + gate results

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Push and open the PR into rust-migration**

```bash
rtk git push -u origin phase-5-w6-wrapup
gh pr create --base rust-migration --head phase-5-w6-wrapup \
  --title "Phase 5 W6 wrap-up: thin-shim audit + cargo-standalone + seqpro verification + perf re-baseline" \
  --body "$(cat <<'EOF'
Wraps up Phase 5 finalization threads (sans genoray, sans the single-big-kernel collapse).

- **Thin-shim audit** (Unit A): classified remaining PyO3-surface Python glue; verdict in `docs/roadmaps/phase-5-w6-thin-shim-audit.md`.
- **cargo-testable standalone** (Unit B): verified `cargo test` runs without the pixi/Python layer.
- **seqpro-core released** (Unit C): confirmed `seqpro-core 0.1.0` resolves from crates.io; corrected the stale Phase 1 path-dep note.
- **W6 perf re-baseline** (Unit D): rayon serial-vs-multithread speedup curve + peak-RSS deltas in `docs/roadmaps/phase-5-w6-perf-rebaseline.md`.

Gate: full pytest tree green, cargo test green, ruff/format/pyrefly/clippy clean, abi3 wheel builds.

**Merge note:** targets `rust-migration` only. The `rust-migration → master` merge is left to the maintainer (no-squash).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the implementer

- This plan is audit/measure/document-heavy, not feature code. Only Task 2 may touch source/config, and only if `cargo test` does not already run standalone.
- Every roadmap edit is additive/corrective text — preserve the existing structure and the status-legend conventions (⬜/🚧/✅).
- Do NOT mark Phase 5 ✅ before Task 5; intermediate tasks annotate but do not set the phase marker.
- Do NOT merge to master under any circumstances.
