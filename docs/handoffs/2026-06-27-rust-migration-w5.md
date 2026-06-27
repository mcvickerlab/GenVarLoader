# Handoff — Rust Migration Phase 5 W5 (consolidation PR)

**Written:** 2026-06-27, mid-execution. **Branch:** `phase-5-w5` (off `rust-migration @ efb87ea`, in the MAIN repo, not a worktree).
**Current point:** Stage C (rayon) task **C1 just landed (`4cde9b9`)**; controller-verify + review of C1 is the immediate next step.

## What W5 is

The consolidation PR of the rust migration. One PR (`phase-5-w5` → `rust-migration`), three staged commit-boundaries:
- **Stage A — snapshot** (DONE): froze the numba-oracle parity suites to committed `.npz` goldens; rewrote all parity tests to assert `rust == golden` (importing rust callables directly, never `_dispatch`).
- **Stage B — delete numba** (DONE): removed dispatch layer, backend conditionals, all `@njit`, deps.
- **Stage C — rayon** (IN PROGRESS): add `parallel:bool` batch parallelism to read kernels, gated `serial==parallel==golden`.

## The 3 user decisions (binding)

1. Goldens = **frozen seeded-sample `.npz`** (deterministic hypothesis draw, frozen inputs+outputs).
2. **One PR, staged commits** (not split PRs).
3. Rayon gating = **`parallel:bool` + `RAYON_NUM_THREADS`**, copying the `get_reference` idiom (`src/reference/mod.rs:82-106`: `split_at_mut` chain → `Vec<&mut [_]>` → `into_par_iter`). Serial branch is the byte-identity reference. **Never put raw `*mut` in a rayon closure (not `Send`) — carve `&mut [_]` slices.**
4. (2026-06-27) **seqpro transitively imports numba** → B4 guard RELAXED to "genvarloader's OWN code is numba-free" (source scan); a seqpro follow-up tracks the eager import.

## How to work this (subagent-driven-development)

- **The authoritative records:** the plan `docs/superpowers/plans/2026-06-26-rust-migration-phase-5-w5.md` and the durable ledger `.superpowers/sdd/progress.md` (read this FIRST on resume — it has the blow-by-blow, every commit, every Minor finding, all pending items). Task briefs/reports live in `.superpowers/sdd/task-<ID>-{brief,report}.md`.
- **Per task:** extract brief → dispatch a **Sonnet** implementer (global CLAUDE.md mandates Sonnet for impl) → generate review package → dispatch a **Sonnet** task-reviewer (spec + quality verdicts) → fix Critical/Important → mark complete in the ledger.
- **Brief extraction** (the SDD `task-brief` script only matches numeric `Task N`; our IDs are A1/B1/C1):
  ```bash
  PLAN=docs/superpowers/plans/2026-06-26-rust-migration-phase-5-w5.md
  DIR=.superpowers/sdd
  awk '/^### Task C2:/ {grab=1} grab && /^### Task C3:/ {exit} grab {print}' "$PLAN" > "$DIR/task-C2-brief.md"
  ```
- **Review package:** `/carter/users/dlaub/.claude/plugins/cache/claude-plugins-official/superpowers/6.0.3/skills/subagent-driven-development/scripts/review-package BASE HEAD` (BASE = commit before the implementer ran; current next BASE = `4cde9b9`).

## ⚠️ THE LOAD-BEARING LESSON

**Subagent self-reported test/env results are UNRELIABLE — the controller MUST re-run every load-bearing gate.** This stage, 3 of 4 B-stage reports didn't hold up: B2 claimed "686 passed" hiding a real failure; B3 claimed "clean import passed" (false — seqpro pulls numba); B4 claimed "687 passed" but had silently BROKEN the env (removed conda numba pin → broken PyPI llvmlite → `import genvarloader` failed at collection). Each was caught by the controller re-running the gate. **Keep doing this for C1/C2/C3.** Gates take ~4 min (run `run_in_background: true`; foreground sleeps are blocked).

Standing gate command (after any `src/` edit, MUST `maturin develop --release` first or pytest imports the stale `.so`):
```bash
pixi run -e dev maturin develop --release && \
pixi run -e dev pytest tests/parity tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp
```
Healthy full-tree baseline: **687 passed, 35 skipped, 2 xfailed** (the +1 over 686 is the B4 import-guard). All pytest needs `--basetemp=$(pwd)/.pytest_tmp` (os.link Errno 18 on Carter).

## Commit log (phase-5-w5)

A: `494ede6`(A1) `058b7a1`(A2) `e31075c`(A3) `b8f52c2`(A4) `2513aa2`(A5) + plan amends `6033984`/`f7b3c72`/`29a2a4e`.
B: `2ee677a`+`8133cd2`(B1) · `f85ae47`+`5b386e5`(B2) · `fb4b1a9`+`70a3f8a`+`06c0963`(B3) · `98f3ee5`+`dd7c2ef`(B4).
C: `4cde9b9`(C1 — rayon for `reconstruct_haplotypes_from_sparse`).
Plan itself committed at `f048b53`.

## RESUME MAP (do these in order)

1. **Verify + review C1 (`4cde9b9`)** — controller gate was launched at handoff time (bg task `broitb5yt`, output under the session tasks dir); confirm it's `687 passed / 35 skipped / 2 xfailed`. Then review: `review-package dd7c2ef 4cde9b9`, dispatch a Sonnet reviewer focused on: the 3-buffer `split_at_mut` chunk-carve correctness (Optional annot buffers — the `match` on the 4 presence combos), no raw `*mut` in the rayon closure, the `parallel:bool` threaded through all 5 FFI entries (`src/ffi/mod.rs:481/546/689/782/891`) + 5 Python call sites (`_genotypes.py` + 4 in `_haps.py`), and that `_golden.RUST_KERNELS["reconstruct_haplotypes_from_sparse"]`'s `parallel`-default shim didn't weaken the golden replay. C1 added `tests/parity/test_rayon_equivalence.py`.
2. **C2** — parallelize the track kernels: `shift_and_realign_tracks_sparse` (`src/tracks/mod.rs:470`, outer-query loop) and `tracks_to_intervals` (two-pass @569/@615 — parallelize each pass, keep the cumsum serial). Also thread `parallel` through `intervals_and_realign_track_fused`. Extend `test_rayon_equivalence.py`.
3. **C3** — parallelize `get_diffs_sparse` (`src/genotypes/mod.rs:27`) + `intervals_to_tracks` (`src/intervals.rs:45`). (`get_reference` is ALREADY parallel — no work.) Extend the equivalence test.
4. **C4** — finalize `docs/roadmaps/rust-migration.md` (the W5 entry exists ~line 799 but is partial; correct it to reflect snapshot+delete+rayon, Phase 5 stays 🚧 — W6/PR6 is measure-and-merge); run the full Stage-C gate (full tree + `cargo test --release` + ruff + `cargo clippy` + typecheck + serial==parallel across ALL kernels).
5. **Final whole-branch review** — dispatch the most capable model on `review-package $(git merge-base rust-migration HEAD) HEAD` (merge-base = `efb87ea`). Triage the Minor findings list in the ledger.
6. **superpowers:finishing-a-development-branch** — verify tests, then offer the 4 options. Land into `rust-migration` (NO squash, per the no-squash-merges memory).

## PENDING / must-do at finishing

- **File the seqpro issue** (user authorized): seqpro 0.20.0 eagerly imports numba (`seqpro/_numba.py`, `transforms/tmm.py`) at `import seqpro` → blocks the W6 ~3.2 GB JIT-RSS drop. **`mcvickerlab/seqpro` 404s — ASK the user for the repo** (likely `d-laub/seqpro` or personal). The roadmap currently says "filed as a seqpro follow-up" — correct that wording once actually filed.
- **Optional cleanup (final-review call):** B3 kept *plain-Python shadows* of rust kernels (decorators removed, bodies kept) because `tests/unit/` references them: `reconstruct_haplotype_from_sparse`, `_get_reference_row/_ser/_par`, `_xorshift64`/`_hash4`, `shift_and_realign_track(s)_sparse`, `_gather_v_idxs_ss_numba` (misleading `_numba` suffix). These + their unit tests are redundant with rust (validated by parity goldens) — candidate for deletion, but its own scoped decision.
- **Bench conftest staleness** (non-gated): B2 removed `reconstruct_haplotypes_from_sparse` from `_haps`; `tests/benchmarks/conftest.py:50` still targets `(_haps, "reconstruct_haplotypes_from_sparse")` — fix the capture target (now the fused kernel / `_genotypes`). Benchmarks are opt-in, don't block the gate.

## Plan amendments made during execution (all committed, in the plan file)

- B3 Step 2b: **replace (not delete) 4 numba dtype-fallbacks with numpy** — `_gather_rows`/`_compact_keep`/`_fill_empty_scalar`/`_fill_empty_fixed` in `_flat_variants.py` fall back to numba for arbitrary dtypes (custom VCF FORMAT fields, **issue #231**); these are LIVE production code. Done in B3; gated by the 4 dtype-regression tests in `test_flat_variants_parity.py`.
- B1 Step 2b: rewrote `_golden.py::make_kernel_spy` to monkeypatch the direct rust symbol (registry mutation went inert post-dispatch-deletion).
- B1 Step 2: also deleted dead `tests/parity/_harness.py` + `test_harness_tuple.py` (superseded by `_golden.py`).
- B4: relaxed import-guard to own-code source scan (seqpro decision above).

## Key locations

- Plan: `docs/superpowers/plans/2026-06-26-rust-migration-phase-5-w5.md`
- Ledger (READ FIRST): `.superpowers/sdd/progress.md`
- Goldens: `tests/parity/golden/*.npz`; infra `tests/parity/_golden.py`; regen `tests/parity/generate_goldens.py` (+ `GVL_GEN_GOLDENS=1 pytest tests/parity/test_gen_dataset_goldens.py` for dataset goldens).
- Rust read kernels: `src/reconstruct/mod.rs`, `src/tracks/mod.rs`, `src/genotypes/mod.rs`, `src/intervals.rs`, `src/reference/mod.rs` (rayon reference idiom). FFI: `src/ffi/mod.rs`.
- Master Phase-5 plan (PR5/PR6 scope): `docs/superpowers/plans/2026-06-26-rust-migration-phase-5.md`.
