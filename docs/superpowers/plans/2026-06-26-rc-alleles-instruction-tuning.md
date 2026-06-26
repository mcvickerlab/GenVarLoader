# rc_alleles_inplace Instruction-Level Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the instruction count of `variants::rc_alleles_inplace` (the only compute kernel from PR #251, never covered by the round-3 #252 pass) by fusing its row→allele mask expansion and delegation into a single pass, byte-identical to today.

**Architecture:** Extract the per-row reverse+complement body (already round-3-vectorized inside `rc_flat_rows_inplace`) into a shared `#[inline]` helper `reverse::rc_row`, then rewrite `rc_alleles_inplace` to walk masked rows → alleles and call `rc_row` directly — deleting a per-call `Vec<bool>` heap alloc+memset, an `Array1` wrap, and a redundant full-allele rescan.

**Tech Stack:** Rust (ndarray, PyO3), `cargo-show-asm` (`cargo asm`), `maturin`, `pixi` (`-e dev`), `pytest` + `hypothesis` (parity), `cargo test`.

**Spec:** `docs/superpowers/specs/2026-06-26-rc-alleles-instruction-tuning-design.md`

## Global Constraints

Every task implicitly includes these. Values copied verbatim from the spec.

- **Parity is sacrosanct:** `rc_alleles_inplace` output must stay **byte-identical** to the seqpro reference on both backends. The migration contract; a change only lands when parity holds.
- **Gate = parity + instruction-count drop + no throughput regression** (NOT round-3's strict "improve throughput or revert"). This path (`rc_alleles` fires only on negative-strand variants / `RaggedVariants` reads) is wall-clock noise-dominated per the roadmap. Keep iff: parity byte-identical both backends; `cargo asm` instruction count drops; `profile.py --mode variants` rust÷numba **holds** (same session, both backends); and `rc_flat_rows_inplace` asm stays equivalent after the extract.
- **Risk control on the shared kernel:** `rc_flat_rows_inplace` is on the round-3-tuned haplotype hot path. The `#[inline]` extract must leave its codegen equivalent. If extraction perturbs it, fall back to duplicating the ~6-line complement locally in `rc_alleles_inplace` and leave `rc_flat_rows_inplace` byte-for-byte untouched.
- **No scope creep:** no on-disk format change, no public API change, no new kernels, no rayon/batch parallelism (Phase 5), no numba/seqpro-reference deletion (Phase 5). No change to `flank_tokens` or `_FlatVariantWindows` (never RC'd).
- **Always rebuild `--release` before any `cargo asm` / throughput measurement.** `cargo asm` reads the last build's artifact; a stale build gives misleading asm.
- **Measurement env:** corpus `tests/benchmarks/data/chr22_geuv.gvl`, `NUMBA_NUM_THREADS=1`, `maturin develop --release`, Carter HPC. Report the **rust ÷ numba ratio** measured in the *same session* (shared-node load drifts across sessions).
- **HPC note:** dataset/parity tests need `--basetemp=$(pwd)/.pytest_tmp` (avoids `os.link` cross-device Errno 18).
- **Worktrees:** never symlink `.pixi` into the worktree — `maturin develop` repoints the shared env's `.pth`/`.so` and corrupts the parent. Each worktree gets its own fresh pixi env.
- **Roadmap contract:** this lands under Phase 3, Target-6 / round-3 area of `docs/roadmaps/rust-migration.md`; the roadmap must be updated as part of the work.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Worktree + fresh pixi env + baseline asm capture

**Files:**
- Create: new git worktree directory (outside the repo tree), branch `opt/rc-alleles-instruction-tuning` off `rust-migration`.

**Interfaces:**
- Consumes: nothing.
- Produces: an isolated worktree with its own pixi env, a working `--release` build, and the recorded `asm_*_before.txt` baselines all later tasks compare against.

- [ ] **Step 1: Create the worktree via the using-git-worktrees skill**

Use the `superpowers:using-git-worktrees` skill to create a worktree for branch `opt/rc-alleles-instruction-tuning` based on `rust-migration`. Do **not** symlink `.pixi` into it (per Global Constraints).

- [ ] **Step 2: Install a fresh dev pixi env in the worktree**

Run (from the worktree root): `pixi install -e dev`
Expected: a populated `.pixi/envs/dev` local to the worktree.

- [ ] **Step 3: Release build + variants-mode smoke**

Run: `pixi run -e dev maturin develop --release`
Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 20`
Expected: a `done wall=... throughput=... batch/s` line, no exception. (If the corpus is missing, build it: `pixi run -e dev python tests/benchmarks/data/build_realistic.py`.)

- [ ] **Step 4: Record the asm baselines (evidence)**

Run: `cargo asm --rust genvarloader::variants::rc_alleles_inplace > asm_rc_alleles_before.txt 2>&1`
Run: `cargo asm --rust genvarloader::reverse::rc_flat_rows_inplace > asm_rc_flat_before.txt 2>&1`
Expected: each prints x86-64 assembly for the function. Note the total instruction count of each (used as the before-numbers in Task 2 and Task 3). If `cargo asm` lists candidates instead of a body, copy the exact mangled path it offers and use that verbatim in later tasks.

- [ ] **Step 5: Record the throughput baseline (gate reference)**

Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 2000`
Run: `GVL_BACKEND=numba pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 2000`
Record both ms/batch and the rust ÷ numba ratio. This is the number the final change must hold (not regress).

No code change yet; nothing to commit.

---

### Task 2: Extract the shared `reverse::rc_row` helper

**Files:**
- Modify: `src/reverse.rs` (add `rc_row`; rewrite `rc_flat_rows_inplace`'s masked branch to call it)
- Test: `src/reverse.rs` `#[cfg(test)] mod tests` (existing reverse/rc tests are the regression lock)

**Interfaces:**
- Consumes: nothing new.
- Produces: `pub(crate) fn rc_row(row: &mut [u8])` — reverses `row` then applies the branchless-vectorized ACGT↔TGCA complement (identity for other bytes), byte-identical to the prior inline body. `rc_flat_rows_inplace` keeps its exact signature `(data: &mut [u8], offsets: ArrayView1<i64>, to_rc: ArrayView1<bool>)` and behavior.

- [ ] **Step 1: Confirm the existing reverse tests pass (regression baseline)**

Run: `pixi run -e dev cargo test --lib reverse 2>&1 | tail -5`
Expected: `test result: ok` (covers `rc_reverses_and_complements_masked_rows_only`, `rc_handles_odd_length_and_n`, `empty_row_and_all_false_are_noops`, `arith_complement_matches_comp_for_all_256_bytes`, the f32/i32 reverse tests). These are the byte-identity lock for the extract.

- [ ] **Step 2: Add `rc_row` and call it from `rc_flat_rows_inplace`**

In `src/reverse.rs`, add `rc_row` (the body is lifted verbatim from the current `rc_flat_rows_inplace` masked branch):

```rust
/// Reverse a single row of bytes then DNA-complement it in place via the
/// branchless ACGT↔TGCA arithmetic (identity for every other byte; A/T = XOR
/// 0x15, C/G = XOR 0x04). `#[inline]` so callers (rc_flat_rows_inplace,
/// rc_alleles_inplace) inline it back to the prior codegen.
#[inline]
pub(crate) fn rc_row(row: &mut [u8]) {
    row.reverse();
    for b in row.iter_mut() {
        let v = *b;
        let at = (((v == b'A') | (v == b'T')) as u8).wrapping_neg(); // 0xFF if A/T
        let cg = (((v == b'C') | (v == b'G')) as u8).wrapping_neg(); // 0xFF if C/G
        *b = v ^ (at & 21) ^ (cg & 4);
    }
}
```

Replace the body of `rc_flat_rows_inplace` with the helper call:

```rust
/// Reverse AND complement bytes within each masked row via `rc_row`.
pub fn rc_flat_rows_inplace(
    data: &mut [u8],
    offsets: ArrayView1<i64>,
    to_rc: ArrayView1<bool>,
) {
    for i in 0..to_rc.len() {
        if !to_rc[i] {
            continue;
        }
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        rc_row(&mut data[s..e]);
    }
}
```

- [ ] **Step 3: Rebuild and run the reverse tests — must still pass**

Run: `pixi run -e dev maturin develop --release`
Run: `pixi run -e dev cargo test --lib reverse 2>&1 | tail -5`
Expected: `test result: ok` (unchanged from Step 1 — proves the extract is byte-identical).

- [ ] **Step 4: Confirm `rc_flat_rows_inplace` asm is equivalent (risk gate)**

Run: `cargo asm --rust genvarloader::reverse::rc_flat_rows_inplace > asm_rc_flat_after.txt 2>&1`
Run: `diff asm_rc_flat_before.txt asm_rc_flat_after.txt; echo "exit=$?"`
Expected: identical or trivially-equivalent asm (same instruction count; only label/address churn). If the instruction count rose or the loop changed shape, the `#[inline]` extract perturbed the tuned kernel — **revert `rc_flat_rows_inplace` to its original inline body** (leave it byte-for-byte untouched) and instead duplicate the `rc_row` body locally inside `rc_alleles_inplace` in Task 3. Record which path was taken.

- [ ] **Step 5: Commit**

```bash
git add src/reverse.rs
git commit -m "refactor(rust): extract reverse::rc_row shared helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Fuse `rc_alleles_inplace`

**Files:**
- Modify: `src/variants/mod.rs` (rewrite `rc_alleles_inplace`, ~lines 88-118)
- Test: `src/variants/mod.rs` `#[cfg(test)] mod tests` (existing `rc_alleles_*` tests are the regression lock); `tests/parity/test_rc_alleles_parity.py`

**Interfaces:**
- Consumes: `crate::reverse::rc_row` (Task 2).
- Produces: `rc_alleles_inplace` keeps its exact signature `(byte_data: &mut [u8], seq_offsets: ArrayView1<i64>, var_offsets: ArrayView1<i64>, to_rc_row: ArrayView1<bool>)` and byte-identical output; no longer allocates a `Vec<bool>` / `Array1` or rescans all alleles.

- [ ] **Step 1: Confirm the existing rc_alleles cargo tests pass (regression baseline)**

Run: `pixi run -e dev cargo test --lib rc_alleles 2>&1 | tail -5`
Expected: `test result: ok` (`rc_alleles_rcs_only_masked_rows`, `rc_alleles_all_false_is_noop`, `rc_alleles_handles_empty_allele_and_n`). These pin byte-identity through the rewrite.

- [ ] **Step 2: Rewrite `rc_alleles_inplace` as a single fused pass**

In `src/variants/mod.rs`, replace the body of `rc_alleles_inplace` (keep the doc comment; update its last paragraph) with:

```rust
pub fn rc_alleles_inplace(
    byte_data: &mut [u8],
    seq_offsets: ndarray::ArrayView1<i64>,
    var_offsets: ndarray::ArrayView1<i64>,
    to_rc_row: ndarray::ArrayView1<bool>,
) {
    // Single fused pass: for each masked (b*p) row, reverse-complement each of
    // its alleles directly via `reverse::rc_row`. `var_offsets` partition the
    // alleles by row (contiguous, disjoint), so this RCs exactly the alleles the
    // old per-allele-mask delegation did, in the same order — byte-identical —
    // without the intermediate `Vec<bool>` alloc or the second full-allele scan.
    for g in 0..to_rc_row.len() {
        if !to_rc_row[g] {
            continue;
        }
        let a0 = var_offsets[g] as usize;
        let a1 = var_offsets[g + 1] as usize;
        for a in a0..a1 {
            let s = seq_offsets[a] as usize;
            let e = seq_offsets[a + 1] as usize;
            crate::reverse::rc_row(&mut byte_data[s..e]);
        }
    }
}
```

> If Task 2 Step 4 took the fallback path (kept `rc_flat_rows_inplace` untouched, no shared helper), inline the `rc_row` body here instead of calling `crate::reverse::rc_row` — i.e. `let row = &mut byte_data[s..e]; row.reverse(); for b in row.iter_mut() { ... }` with the same A/T XOR 21, C/G XOR 4 arithmetic.

- [ ] **Step 3: Rebuild and run the rc_alleles cargo tests — must still pass**

Run: `pixi run -e dev maturin develop --release`
Run: `pixi run -e dev cargo test --lib rc_alleles 2>&1 | tail -5`
Expected: `test result: ok` (unchanged from Step 1 — proves the fuse is byte-identical).

- [ ] **Step 4: Run the Python parity suite (byte-identical, both backends)**

Run: `pixi run -e dev pytest tests/parity/test_rc_alleles_parity.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (the hypothesis parity test + the `_FlatAlleles.reverse_masked` spy test). This compares the rust kernel against the seqpro reference across the allele-batch matrix.

- [ ] **Step 5: Record the asm delta (evidence)**

Run: `cargo asm --rust genvarloader::variants::rc_alleles_inplace > asm_rc_alleles_after.txt 2>&1`
Run: `diff asm_rc_alleles_before.txt asm_rc_alleles_after.txt; echo "exit=$?"`
Expected: lower total instruction count than `asm_rc_alleles_before.txt` (the `Vec<bool>` alloc, memset, `Array1::from_vec`, and second scan are gone). Record `<before>→<after>` instruction count.

- [ ] **Step 6: Confirm no throughput regression (gate)**

Run: `pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 2000`
Run: `GVL_BACKEND=numba pixi run -e dev python tests/benchmarks/profiling/profile.py --mode variants --n-batches 2000`
Expected: rust ÷ numba ratio **holds** vs the Task 1 Step 5 baseline (no regression; improvement is a bonus, not required). Record the ratio.

- [ ] **Step 7: Commit**

```bash
git add src/variants/mod.rs
git commit -m "perf(rust): fuse rc_alleles_inplace — <before>→<after> instrs, drop Vec<bool> alloc + rescan

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Full-tree gate + roadmap update + finish

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (Target-6 / round-3 area)

**Interfaces:**
- Consumes: the kept commits from Tasks 2-3 + their recorded asm/ratio deltas.
- Produces: a landed, fully-verified pass with the roadmap updated per the migration contract.

- [ ] **Step 1: Full pytest tree on BOTH backends**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Run: `GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: both green with the same passed/xfailed profile (byte-identical parity proven on both backends). Read the output; investigate any new failure before proceeding — do NOT claim success without it.

- [ ] **Step 2: cargo tests + lint + format + typecheck + wheel build**

Run: `pixi run -e dev cargo test 2>&1 | tail -5` → `test result: ok`
Run: `pixi run -e dev ruff check python/ tests/` → clean
Run: `pixi run -e dev ruff format --check python/ tests/` → clean
Run: `pixi run -e dev typecheck` → clean
Run: `pixi run -e dev maturin build 2>&1 | tail -3` → abi3 wheel builds

- [ ] **Step 3: Update the roadmap**

In `docs/roadmaps/rust-migration.md`, under the Target-6 "**✅ Variant-allele RC folded**" block (~lines 491-499), append a dated follow-up note recording the tuning:

```markdown
   **✅ rc_alleles_inplace instruction-tuned (follow-up, 2026-06-26).** The #251
   `variants::rc_alleles_inplace` kernel was not in the round-3 (#252) target list;
   this pass fused its row→allele mask expansion and `rc_flat_rows_inplace` delegation
   into a single pass via the shared `reverse::rc_row` helper, dropping a per-call
   `Vec<bool>` alloc+memset, an `Array1` wrap, and a redundant full-allele rescan.
   Instr <before>→<after> (`cargo asm`); variants-path rust÷numba held (noise-dominated
   path — gated on parity + instr drop + no regression, not throughput improvement);
   `rc_flat_rows_inplace` asm unchanged after the extract. Byte-identical parity on both
   backends. Spec/plan: `docs/superpowers/{specs/2026-06-26-rc-alleles-instruction-tuning-design,plans/2026-06-26-rc-alleles-instruction-tuning}.md`.
```

Fill `<before>→<after>` with the real numbers recorded in Task 3 Step 5.

- [ ] **Step 4: Commit the roadmap**

```bash
git add docs/roadmaps/rust-migration.md
git commit -m "docs(roadmap): record rc_alleles_inplace instruction tuning (Target 6 follow-up)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to integrate `opt/rc-alleles-instruction-tuning` into `rust-migration`. Follow the roadmap precedent of per-target PRs into `rust-migration` (e.g. #248/#249/#250); **no squash merge** (per the `no-squash-merges` note — preserve the real commit history).

---

## Notes for the implementer

- **Why no pre-written asm diffs:** the recorded instruction counts are discovered at execution by running `cargo asm` on this build — fabricating them here would be a placeholder. The transformation itself (fuse + shared helper) is fully specified above; the counts are evidence captured during Tasks 2-3.
- **One logical change per commit** (Task 2 extract, Task 3 fuse) so either is a clean isolated revert if its asm/throughput gate fails.
- **Ratios over absolutes:** the Carter node is shared; always re-measure numba in the same session as rust and report the ratio.
- **The reference IS the oracle:** there is no numba `rc_alleles` kernel; the seqpro path is the byte-identical reference. Parity tests compare rust vs that reference.
