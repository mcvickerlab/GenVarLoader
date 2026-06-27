# Round-3 Instruction-Level Kernel Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the Rust read-path kernels to rust ≥ numba single-threaded on all four read paths (tracks-only, haplotypes, variants, variant-windows) by tuning their generated machine code, using perf to localize and cargo-show-asm (+llvm-mca) to inspect and verify.

**Architecture:** Profile-all-first to build one consolidated, aggregate-weighted target list, then run a fixed per-kernel tune loop (inspect asm → fix → confirm asm delta → confirm throughput → confirm parity → commit-or-revert) in descending target order. No format/API/semantic change; this round only changes the instruction sequences hot kernels compile to.

**Tech Stack:** Rust (ndarray, PyO3, rayon present but unused this round), `cargo-show-asm` v0.2.61 (`cargo asm`), `perf`, `maturin`, `pixi`, `pytest` + `pytest-benchmark`, `hypothesis` (parity).

**Spec:** `docs/superpowers/specs/2026-06-25-round3-instruction-level-kernel-tuning-design.md`

## Global Constraints

Every task implicitly includes these. Values copied verbatim from the spec.

- **Parity is sacrosanct:** rust output must stay **byte-identical** to numba on both backends. The two documented numba-bug exclusions (the #242-family `intervals_to_tracks` start<query clip; the reconstruct trailing-under-write overshoot) stay **unchanged** — do not touch them.
- **Gate = wall-clock throughput, not instruction count.** A change that drops instructions but does **not** improve (or at least hold) ms/batch is **reverted**. Instruction/llvm-mca deltas are recorded as evidence only.
- **`unsafe` budget:** safe idioms first (slice hoisting, iterators, `assert!` bound hints, codegen attrs). Targeted `unsafe` (`get_unchecked` / explicit SIMD) only where the bound is provably safe but the optimizer keeps the check; every `unsafe` carries a `// SAFETY:` comment and is gated by passing parity.
- **No scope creep:** no on-disk format change, no public API change, no new kernels, no rayon/batch parallelism (Phase 5), no numba deletion (Phase 5).
- **Measurement env (every throughput/asm number):** corpus `tests/benchmarks/data/chr22_geuv.gvl` (format 2.0, 165 regions × 5 samples, 82 neg / 83 pos strand), `with_len(16384)`, `BATCH=32`, `NUMBA_NUM_THREADS=1`, `maturin develop --release`, Carter HPC (AMD EPYC 7543, linux-64). **Report the rust ÷ numba ratio, not absolute batch/s** (shared-node load varies across sessions).
- **Per-path gate harness:** tracks-only & haplotypes → `tests/benchmarks/test_e2e.py` pytest-benchmark **pedantic min** (ms/batch). variants & variant-windows → `tests/benchmarks/profiling/profile.py` **wall-clock average** (2000 batches) — `test_e2e_variants` is xfailed (`_FlatVariants.to_fixed` gap) so no pedantic min exists for those two.
- **Gate numbers come only from the plain `--release` build.** The `[profile.profiling]` profile is for perf attribution only and is never the measured artifact.
- **HPC note:** dataset/parity tests need `--basetemp=$(pwd)/.pytest_tmp` (avoids `os.link` cross-device Errno 18).
- **Roadmap contract:** this work lands as "Optimization targets — round 3" under Phase 3 in `docs/roadmaps/rust-migration.md` (not a new phase); the roadmap must be updated as part of the work.

---

### Task 1: Worktree + fresh pixi env + release build smoke

**Files:**
- Create: new git worktree directory (outside the repo tree), branch `opt/round3-instruction-tuning` off `rust-migration`.

**Interfaces:**
- Consumes: nothing.
- Produces: an isolated worktree with its **own** pixi env and a working `--release` build; all later tasks run here.

- [ ] **Step 1: Create the worktree via the using-git-worktrees skill**

Use the `superpowers:using-git-worktrees` skill to create a worktree for branch `opt/round3-instruction-tuning` based on `rust-migration`. Do **not** symlink `.pixi` into it — `maturin develop` repoints the shared env's `.pth`/`.so` and would corrupt the parent workspace (per the `gvl-parallel-worktrees-fresh-pixi-env` note).

- [ ] **Step 2: Install a fresh dev pixi env in the worktree**

Run (from the worktree root): `pixi install -e dev`
Expected: a populated `.pixi/envs/dev` local to the worktree.

- [ ] **Step 3: Release build + smoke the four profile modes**

Run: `pixi run -e dev maturin develop --release`
Then smoke each mode at a tiny batch count to confirm the corpus + build work:
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode tracks --n-batches 20`
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode haplotypes --n-batches 20`
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 20`
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variant-windows --n-batches 20`
Expected: each prints a `done wall=... throughput=... batch/s` line, no exception. (If the corpus is missing, build it: `pixi run -e dev python tests/benchmarks/data/build_realistic.py`.)

- [ ] **Step 4: Confirm `cargo asm` resolves a symbol against this build**

Run: `cargo asm --simplify genvarloader::intervals::intervals_to_tracks 2>&1 | head -30`
Expected: x86-64 assembly for the function prints (confirms cargo-show-asm v0.2.61 sees the release artifact and resolves the symbol). If it lists candidates instead, copy the exact mangled path it offers — that is the canonical symbol name for later tasks.

- [ ] **Step 5: Commit (worktree marker)**

No code change yet; nothing to commit. Proceed.

---

### Task 2: Add the `[profile.profiling]` profile

**Files:**
- Modify: `Cargo.toml` (append a profile section).

**Interfaces:**
- Consumes: nothing.
- Produces: a `profiling` cargo profile for perf call-graph attribution (used in Task 3 only when flat self-time is ambiguous). Never the measured artifact.

- [ ] **Step 1: Append the profile to `Cargo.toml`**

Add at the end of `Cargo.toml`:

```toml
# Perf call-graph attribution only (`perf report --children`). Inherits release
# codegen and adds line tables + frame pointers. NEVER the gate artifact — all
# throughput/asm gate numbers come from the plain `--release` build.
[profile.profiling]
inherits = "release"
debug = "line-tables-only"
force-frame-pointers = true
```

- [ ] **Step 2: Verify it builds**

Run: `pixi run -e dev cargo build --profile profiling 2>&1 | tail -5`
Expected: `Finished` line, no error. (This validates the profile parses; the gate build remains `maturin develop --release`.)

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "build(rust): add [profile.profiling] for perf call-graph attribution"
```

---

### Task 3: Fresh baseline + ranked aggregate target list

**Files:**
- Create: `docs/roadmaps/round3-profile-baseline.md` (the consolidated table; the roadmap round-3 section links to it).

**Interfaces:**
- Consumes: the release build from Task 1.
- Produces: `round3-profile-baseline.md` containing (a) per-path rust ÷ numba starting ratios and (b) a consolidated flat-self-time table with an aggregate-weight column. **No tuning task starts until this file exists** — it determines target order and overrides the "expected targets" in the spec.

- [ ] **Step 1: Capture per-path throughput baselines (rust vs numba)**

tracks-only & haplotypes (pedantic min):
Run: `pixi run -e dev pytest tests/benchmarks/test_e2e.py::test_e2e_tracks_only tests/benchmarks/test_e2e.py::test_e2e_haplotypes --benchmark-only -q`
Run again with `GVL_BACKEND=numba` prefixed to get the numba min for the same two.

variants & variant-windows (profile.py wall-clock avg, 2000 batches):
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 2000`
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variant-windows --n-batches 2000`
Run each again with `GVL_BACKEND=numba` prefixed.

Record the four rust ÷ numba ratios.

- [ ] **Step 2: Capture flat self-time perf profiles for all four paths (rust)**

For each `MODE` in `tracks haplotypes variants variant-windows`:

```bash
NUMBA_NUM_THREADS=1 perf record -F 999 -o p_$MODE.data -- \
    .pixi/envs/dev/bin/python tests/benchmarks/profiling/profile.py --mode $MODE --n-batches 12000
perf report --stdio --no-children -i p_$MODE.data > report_$MODE.txt
```

Expected: each `report_*.txt` lists symbols by self-time with `genvarloader::...` Rust symbols resolved. (12k batches drowns one-time import/JIT.)

- [ ] **Step 3: Build the consolidated aggregate-weighted table**

In `docs/roadmaps/round3-profile-baseline.md`, write a table: rows = Rust kernel symbols that appear in any path's top self-time, columns = self-time % per path, plus an **Aggregate** column = sum of self-time % across the paths the kernel appears in. Shared kernels (e.g. `intervals_to_tracks`, `shift_and_realign_tracks_sparse` appear in both tracks and haplotypes) rank by total read-path cost. Include the four starting ratios from Step 1 above the table.

- [ ] **Step 4: Commit**

```bash
git add docs/roadmaps/round3-profile-baseline.md
git commit -m "docs(roadmap): round-3 profiling baseline + aggregate target list"
```

---

### Task 4: TUNE LOOP TEMPLATE — apply to each target in descending aggregate-weight order

> **This is the procedure every tuning task follows.** The exact code fix **cannot** be pre-written — it is determined by reading the kernel's assembly (an instruction-count pass is asm-driven by definition; fabricating a diff here would be a lie). What IS fixed and concrete: the inspect commands, the asm→fix decision tree with worked examples from this codebase, and the three gates (asm delta recorded, throughput non-regression, parity byte-identical). Instantiate this loop as a **separate commit per kernel**, taking targets from Task 3's table in order. Tasks 5–7 list the expected targets with their real source anchors; Task 3's profile reorders/prunes them.

For a target kernel `K` at `crate::module::K` in `src/<file>.rs`:

- [ ] **Step 1: Record the asm baseline (evidence)**

Run: `cargo asm --rust crate::module::K > asm_K_before.txt`
Run: `cargo asm --mca crate::module::K > mca_K_before.txt`
Note from `asm_K_before.txt`: total instruction count, and from `mca_K_before.txt`: llvm-mca "Total Cycles" / "Block RThroughput". Identify the dominant cost using the decision tree in Step 3.

- [ ] **Step 2: Record the throughput baseline for K's path (gate)**

Run K's path harness (see Global Constraints "Per-path gate harness") for **both** backends and record the rust ÷ numba ratio. This is the number the change must improve or hold.

- [ ] **Step 3: Diagnose from the asm, pick a fix class**

Map the asm symptom to a fix (worked examples are real transformations from this codebase / its history):

  - **Per-element bounds check** (`cmp`/`jae` to a panic block around an indexed write in the hot loop) → hoist the slice once before the loop and index the raw `&mut [T]`. *Worked example (already landed as T5, `src/intervals.rs:29,69`):* `out.as_slice_mut().unwrap()` hoisted before the interval loop, inner body `out_slice[a..b].fill(value)` on `&mut [f32]` — dropped per-interval `SliceInfo` + bounds check, no `unsafe`. If the compiler still cannot prove `a..b` in range, add `assert!(b <= out_slice.len())` before the loop (one check feeds the optimizer), or as a last resort `out_slice.get_unchecked_mut(a..b)` with `// SAFETY: a,b are clamped to [0,length] and out_s+length == out_e <= out_slice.len()`.
  - **Scalar byte loop that should vectorize** (e.g. `rc_flat_rows_inplace`'s `for b in row.iter_mut() { *b = COMP[*b as usize] }`, `src/reverse.rs:54-56`) → the gather through `COMP` blocks autovectorization. Try: process in fixed chunks, or split reverse+complement so the reverse is a `slice::reverse` (already SIMD) and the complement is a separate tight pass; inspect whether llvm vectorizes the complement after the split. Keep the COMP table semantics identical (parity).
  - **Redundant copy / materialization** in the loop → eliminate the intermediate, write directly into the output slice.
  - **Register spill** (stack `mov`s in the inner loop) → reduce live values, pull invariants out of the loop, or split the function so the hot loop monomorphizes tighter.
  - **Integer width churn** (`movsxd`/`cdqe` from `as i64`/`as usize` per element) → compute loop-invariant casts once outside the loop.

Apply the chosen fix to `src/<file>.rs`. Safe idiom first; `unsafe` only per the Global Constraints budget, always with a `// SAFETY:` comment.

- [ ] **Step 4: Rebuild and confirm the asm delta (evidence)**

Run: `pixi run -e dev maturin develop --release`
Run: `cargo asm --rust crate::module::K > asm_K_after.txt` and `cargo asm --mca crate::module::K > mca_K_after.txt`
Expected: lower instruction count and/or lower llvm-mca cycles vs the `*_before.txt`. Record the delta.

- [ ] **Step 5: Confirm throughput (gate) — REVERT if no win**

Re-run K's path harness for both backends; recompute the rust ÷ numba ratio.
- If ms/batch **improved or held** and parity (Step 6) passes → keep.
- If instructions dropped but ms/batch **did not improve** → **`git checkout -- src/<file>.rs`** and record in the roadmap that K is memory/branch-bound at this floor (honest non-result). Do not force it.

- [ ] **Step 6: Confirm parity (byte-identical, both backends)**

Run the kernel's parity suite (Task 5–7 name the exact file per kernel), e.g.:
Run: `pixi run -e dev pytest tests/parity/<test_file>.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS. Then the relevant cargo unit tests:
Run: `pixi run -e dev cargo test <module> 2>&1 | tail -5`
Expected: `test result: ok`.

- [ ] **Step 7: Commit (one kernel per commit)**

```bash
git add src/<file>.rs
git commit -m "perf(rust): tune <K> — <instr before>→<after> instrs, <ratio before>→<after>"
```

---

### Task 5: Tune the tracks/haplotypes shared kernels (expected highest aggregate weight)

> Instantiate the Task-4 loop for each, in the order Task 3's aggregate column gives. Real source anchors and parity files below. Skip any whose Task-3 self-time is already negligible.

**Files:**
- Modify (as the asm dictates): `src/intervals.rs`, `src/tracks/mod.rs`, `src/reverse.rs`.
- Test: `tests/parity/test_intervals_to_tracks_parity.py`, `tests/parity/test_fused_tracks_parity.py`, `tests/parity/test_shift_and_realign_tracks_parity.py`, `tests/parity/test_dataset_parity.py`.

**Interfaces:**
- Consumes: Task 3's ranked table.
- Produces: tuned kernels with recorded asm + ratio deltas; tracks-only and tracks-seqs paths at/above numba.

- [ ] **Step 1: `genvarloader::intervals::intervals_to_tracks`** (`src/intervals.rs:16`) — run the Task-4 loop. Hot inner loop already raw-slice (T5); look for residual per-interval `as i64`/`as usize` casts (`src/intervals.rs:52-53,67-68`) and the `out_slice.fill(0.0)` prelude. Parity: `test_intervals_to_tracks_parity.py` + `test_fused_tracks_parity.py`. Gate path: `test_e2e_tracks_only`.
- [ ] **Step 2: `genvarloader::tracks::shift_and_realign_tracks_sparse`** (`src/tracks/mod.rs`) — run the Task-4 loop. Parity: `test_shift_and_realign_tracks_parity.py` + `test_fused_tracks_parity.py`. Gate path: `test_e2e_tracks_only` and `test_e2e_tracks` (shared).
- [ ] **Step 3: `genvarloader::reverse::reverse_flat_rows_inplace`** (`src/reverse.rs:25`, the f32 track-reverse half) — run the Task-4 loop only if Task 3 shows it hot on the tracks path. Parity: `test_fused_tracks_parity.py`. Gate path: `test_e2e_tracks_only`.
- [ ] **Step 4: Re-confirm both gate paths after all kept changes**

Run: `pixi run -e dev pytest tests/benchmarks/test_e2e.py::test_e2e_tracks_only tests/benchmarks/test_e2e.py::test_e2e_tracks --benchmark-only -q` (rust, then `GVL_BACKEND=numba`).
Expected: recorded rust ÷ numba ratio ≥ the Task-3 starting ratio for both.

---

### Task 6: Tune the haplotype kernels

> Instantiate the Task-4 loop for each, in Task-3 aggregate order.

**Files:**
- Modify (as the asm dictates): `src/reconstruct/mod.rs`, `src/reverse.rs`.
- Test: `tests/parity/test_reconstruct_haplotypes_parity.py`, `tests/parity/test_fused_haps_parity.py`, `tests/parity/test_haplotypes_dataset_parity.py`.

**Interfaces:**
- Consumes: Task 3's ranked table.
- Produces: tuned haplotype kernels; haplotypes path at/above numba.

- [ ] **Step 1: `genvarloader::reconstruct::reconstruct_haplotypes_from_sparse`** (`src/reconstruct/mod.rs`) — run the Task-4 loop. Parity: `test_reconstruct_haplotypes_parity.py` + `test_fused_haps_parity.py`. Gate path: `test_e2e_haplotypes`.
- [ ] **Step 2: `genvarloader::reverse::rc_flat_rows_inplace`** (`src/reverse.rs:41`, the byte revcomp half) — run the Task-4 loop. Decision-tree hint: the `COMP[*b as usize]` gather (`src/reverse.rs:54-56`) blocks autovectorization; try splitting `row.reverse()` (already SIMD) from the complement pass and inspect whether the complement vectorizes. Parity: `test_fused_haps_parity.py` + `test_dataset_parity.py`. Gate path: `test_e2e_haplotypes`.
- [ ] **Step 3: Re-confirm the gate path after all kept changes**

Run: `pixi run -e dev pytest tests/benchmarks/test_e2e.py::test_e2e_haplotypes --benchmark-only -q` (rust, then `GVL_BACKEND=numba`).
Expected: recorded rust ÷ numba ratio ≥ the Task-3 starting ratio.

---

### Task 7: Tune the variant-windows kernels

> Instantiate the Task-4 loop for each, in Task-3 aggregate order. These are the T7 profile top.

**Files:**
- Modify (as the asm dictates): `src/variants/windows.rs`.
- Test: `tests/parity/test_assemble_variant_buffers_parity.py`, `tests/parity/test_flat_variants_parity.py`, `tests/parity/test_variants_dataset_parity.py`.

**Interfaces:**
- Consumes: Task 3's ranked table.
- Produces: tuned variant-window assembly kernels; variant-windows path further above numba.

- [ ] **Step 1: `genvarloader::variants::windows::tokenize`** (`src/variants/windows.rs`, T7 top leaf ~28%) — run the Task-4 loop. Gate path (profile.py wall-clock avg, 2000 batches): `--mode variant-windows`.
- [ ] **Step 2: `genvarloader::variants::windows::slice_flanks`** (`src/variants/windows.rs`, ~19%) — run the Task-4 loop.
- [ ] **Step 3: `genvarloader::variants::windows::assemble_alt_window`** (`src/variants/windows.rs`, ~13%) — run the Task-4 loop.
- [ ] **Step 4: Re-confirm the gate path after all kept changes**

Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variant-windows --n-batches 2000` (rust, then `GVL_BACKEND=numba`).
Expected: recorded rust ÷ numba ratio ≥ the Task-3 starting ratio (T7 baseline 1.83×).

Parity for all three: `tests/parity/test_assemble_variant_buffers_parity.py` + `tests/parity/test_flat_variants_parity.py`.

---

### Task 8: Full-tree gate + roadmap update + finish

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (add the round-3 section).

**Interfaces:**
- Consumes: all kept tuning commits + their recorded deltas.
- Produces: a landed, fully-verified round-3 pass with the roadmap updated per the migration contract.

- [ ] **Step 1: Full tree, rust backend**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: all pass except the known pre-existing xfails (`test_e2e_variants`, `test_haps_property` ×2, `test_indexing::test_parse_idx[missing]`, `test_ref_ds::test_getitem[no_regions]`). 0 unexpected failures.

- [ ] **Step 2: Full tree, numba backend**

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: same pass/xfail profile (byte-identical parity proven on both backends).

- [ ] **Step 3: cargo tests + lint + format + typecheck + wheel build**

Run: `pixi run -e dev cargo test 2>&1 | tail -5` → `test result: ok`
Run: `pixi run -e dev ruff check python/ tests/` → clean
Run: `pixi run -e dev ruff format --check python/ tests/` → clean
Run: `pixi run -e dev typecheck` → clean
Run: `pixi run -e dev maturin build 2>&1 | tail -3` → abi3 wheel builds

- [ ] **Step 4: Write the round-3 roadmap section**

In `docs/roadmaps/rust-migration.md`, under Phase 3's optimization-targets area, add an "Optimization targets — round 3 (instruction-level, profiled <date>)" subsection containing: the Task-3 starting ratios, the consolidated target table, a per-kernel row (symbol · instr before→after · llvm-mca cycles before→after · rust÷numba before→after · kept/reverted), and the final four-path ratio summary. Add a dated entry to the "Notes & decisions log" summarizing the round (tooling = cargo-show-asm; gate = throughput; unsafe = targeted/parity-gated; any honest non-results). Update the sequencing note to mark round-3 done and restate that rayon (Phase 5) is the next lever.

- [ ] **Step 5: Commit the roadmap**

```bash
git add docs/roadmaps/rust-migration.md docs/roadmaps/round3-profile-baseline.md
git commit -m "docs(roadmap): record round-3 instruction-level tuning results"
```

- [ ] **Step 6: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to choose how to integrate `opt/round3-instruction-tuning` into `rust-migration` (the roadmap uses per-target PRs into `rust-migration`, e.g. #248/#249/#250 — follow that precedent; **no squash merge**, per the `no-squash-merges` note).

---

## Notes for the implementer

- **Why no pre-written fix diffs:** an instruction-count pass is asm-driven — the fix is whatever the disassembly reveals, discovered at execution. Task 4 gives the real decision tree (asm symptom → fix class → worked codebase example) and the three concrete gates. A fabricated diff would be a placeholder; the gates are the real deliverable.
- **Always rebuild `--release` before any `cargo asm` / throughput measurement.** `cargo asm` reads the last build's artifact; a stale debug build gives misleading asm.
- **One kernel per commit** so any reverted non-result is a clean, isolated revert.
- **Ratios over absolutes:** the Carter node is shared; numba absolute times drift between sessions. Always re-measure numba in the same session as rust and report the ratio.
