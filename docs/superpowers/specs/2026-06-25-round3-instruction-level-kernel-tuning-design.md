# Round-3 instruction-level kernel tuning

**Date:** 2026-06-25
**Branch base:** `rust-migration` (Targets 5/6/7 merged: PRs #248/#249/#250)
**Roadmap home:** `docs/roadmaps/rust-migration.md` → Phase 3 "Optimization targets — round 3" (a new sub-section alongside rounds 1–2 and targets 5–7; **not** a new phase)

---

## Goal

Drive the now-Rust-dominated read-path kernels to **rust ≥ numba single-threaded** on all four
read paths — **tracks-only, haplotypes, variants, variant-windows** — by tuning the generated
machine code. Use `perf` to localize the hot Rust leaves and `cargo-show-asm` (+ llvm-mca via
`--mca`) to inspect and verify codegen at the instruction level.

This is a continuation of the established Phase-3 optimization rhythm (rounds 1–2, targets 5–7),
not a new architectural phase. It changes no on-disk format, no public API, and no kernel
semantics — only the instruction sequences the hot kernels compile to.

### Non-goals

- No rayon / batch parallelism (explicitly deferred to Phase 5; single-thread parity first).
- No on-disk format change, no public API change, no new kernels.
- No numba deletion (that is Phase 5).
- Not a correctness pass — byte-identical parity must hold unchanged throughout.

---

## Decisions (locked with the user, 2026-06-25)

1. **Gate = wall-clock throughput; asm instruction count is evidence, not the gate.**
   The round lands on the established **rust ÷ numba batch/s** metric. Per-kernel
   instruction-count / llvm-mca cycle deltas are recorded as supporting evidence in the roadmap,
   but a kernel that drops instructions without improving ms/batch is reverted. Instruction count
   is a proxy (kernels can be memory- or branch-bound); throughput is truth.

2. **Tooling = `cargo-show-asm`** (`cargo asm`, v0.2.61, installed). Gives `--mca` llvm-mca
   cycle/throughput estimates, `--rust` source interleave, and resolves modern monomorphized
   symbols. The 2019-era gnzlbg `cargo-asm` is not used.

3. **`unsafe` budget = targeted, parity-gated.** Prefer safe idioms first (slice hoisting,
   iterators, `assert!` bound hints, codegen attributes — the T5 playbook). Where the optimizer
   provably cannot elide a bound, allow `get_unchecked` / explicit SIMD, each with a `// SAFETY:`
   comment, contained by the byte-identical parity gate on both backends.

---

## Approach

**Profile-all-first ranked target list, driven by a per-kernel tune loop.** Reach for a Rust
criterion microbench only for a kernel where the in-process flat profile is ambiguous or where
llvm-mca on realistic inputs in isolation is needed — matching the roadmap's own guidance
("a Rust-only criterion harness is only worth building if we want to micro-optimize a kernel in
isolation from FFI/Python").

Rejected alternatives:
- *Per-path sequential* (tune kernels in path order): misses that several kernels are shared
  across paths, so path-order tuning fails to compound shared wins.
- *Criterion-first for every kernel*: more setup, and risks optimizing against unrealistic input
  shapes divorced from the real FFI call sites.

---

## Workspace

- **New git worktree** off `rust-migration` (via the `using-git-worktrees` skill).
- **Its own fresh pixi env** — do **not** symlink `.pixi`. `maturin develop` repoints the shared
  env's `.pth`/`.so`, so a shared env would corrupt the parent workspace's build
  (per the `gvl-parallel-worktrees-fresh-pixi-env` note).
- `cargo asm` (cargo-show-asm) already installed and on PATH (v0.2.61).
- Release builds via `maturin develop --release`.
- Add a `[profile.profiling]` to `Cargo.toml` that **inherits `release`** and adds
  `debug = "line-tables-only"` + `force-frame-pointers = true`, for perf call-graph attribution
  when flat self-time is ambiguous. Flat self-time on the plain release `.so` (symbols resolve
  from the symbol table) is the default; the profiling profile is only for `perf report --children`
  caller attribution. This profile must not change the codegen the gate measures — gate numbers
  always come from the plain `--release` build.

---

## Procedure

### Step 1 — Fresh baseline + ranked target list (no tuning until this exists)

The last perf profiles predate the T5/6/7 merges, so re-baseline at current HEAD.

For each of the four paths, run the established perf method (per `gvl-profiling-perf-not-pyspy-native`):

```bash
NUMBA_NUM_THREADS=1 perf record -F 999 -o p.data -- .pixi/envs/dev/bin/python \
    tests/benchmarks/profiling/profile.py --mode <mode> --n-batches 12000
perf report --stdio --no-children -i p.data        # flat self-time, Rust symbols resolved
```

Modes: `tracks`, `haplotypes`, `variants`, `variant-windows` (the four the user named;
`profile.py --mode` already supports all of `{haplotypes,annotated,tracks,tracks-seqs,variants,variant-windows}`).

Produce **one consolidated table**: rows = Rust kernel symbols, columns = per-path self-time %,
plus an **aggregate weight** (self-time % summed across the paths a kernel appears in, so shared
kernels like `intervals_to_tracks` and `shift_and_realign_tracks_sparse` rank by their total
read-path cost). Record current **rust ÷ numba ratios** per path as the round-3 starting line.

**Expected (to be confirmed, not assumed) targets:** `intervals_to_tracks` and
`shift_and_realign_tracks_sparse` (shared: tracks + haplotypes), `reconstruct_haplotypes_from_sparse`,
`rc_flat_rows_inplace`; and the variant-windows trio `tokenize` / `slice_flanks` /
`assemble_alt_window` (T7 left these as the profile top). Step 1's real profile overrides any
of these.

### Step 2 — Per-kernel tune loop (highest aggregate weight first)

For each target kernel, in descending aggregate-weight order:

1. **Inspect.** `cargo asm --rust --mca <crate>::<path>::<symbol>` → capture instruction count,
   llvm-mca cycle/throughput estimate, and the dominant cost (bounds check, redundant
   slice/copy, missed autovectorization, register spill, etc.).
2. **Fix.** Safe idioms first (hoist `as_slice_mut`, iterator forms, `assert!` to feed the
   bound checker, `#[inline]`/codegen hints). Targeted `unsafe` (`get_unchecked` / explicit
   SIMD) only where the bound is provably safe but the optimizer keeps the check; each `unsafe`
   carries a `// SAFETY:` comment.
3. **Confirm asm (evidence).** Re-run `cargo asm` → instruction/cycle drop recorded.
4. **Confirm throughput (gate).** Re-run the path's throughput harness → ms/batch improvement
   (or no regression). **If instructions dropped but ms/batch did not improve, revert** — it was
   a memory/branch-bound kernel and the change adds risk for no win.
5. **Confirm parity.** Run the kernel's `@pytest.mark.parity` suite → byte-identical on both
   backends.

### Step 3 — Gate + land

Before merge:
- Full tree on **both** backends: `pixi run -e dev pytest tests -q` under `GVL_BACKEND` rust and
  numba (use `--basetemp=$(pwd)/.pytest_tmp` per the HPC `os.link` note).
- `cargo test` green; lint (`ruff check python/ tests/`), format, `typecheck` clean; abi3 wheel
  builds.
- `docs/roadmaps/rust-migration.md` updated: round-3 target table, per-kernel asm deltas, final
  rust ÷ numba ratios, decisions log entry, and the optimization-targets sequencing note.

---

## Measurement harnesses (per-path, established — do not invent new ones)

| Path | Gate metric | Harness | Why |
|---|---|---|---|
| tracks-only | rust ÷ numba **pedantic min** (ms/batch) | `tests/benchmarks/test_e2e.py` (pytest-benchmark, `iterations=10, rounds=50, warmup=5`) | de-noised min is reproducible <1% |
| haplotypes | rust ÷ numba **pedantic min** (ms/batch) | same | same |
| variants | rust ÷ numba **wall-clock average** (ms/batch, 2000 batches) | `tests/benchmarks/profiling/profile.py` | `test_e2e_variants` is xfailed (`_FlatVariants.to_fixed` gap) → no pedantic min |
| variant-windows | rust ÷ numba **wall-clock average** (ms/batch, 2000 batches) | `profile.py` | same xfail; T7 used this harness |

All measurements: corpus `chr22_geuv.gvl` (format 2.0, 165 regions × 5 samples, 82 neg / 83 pos
strand), `with_len(16384)`, `BATCH=32`, `NUMBA_NUM_THREADS=1`, `maturin develop --release`,
Carter HPC (AMD EPYC 7543, linux-64). Report the **ratio**, not absolute batch/s (shared-node
load varies across sessions — the standing roadmap caveat).

---

## Parity contract (unchanged)

Byte-identical rust vs numba on both backends, via the existing `@pytest.mark.parity` hypothesis
suites + the spy-guarded dataset backstops. The two documented numba-bug sub-domains stay excluded
exactly as today (the #242-family `intervals_to_tracks` start<query clip and the reconstruct
trailing-under-write overshoot) — this round must not touch those exclusions. Any new `unsafe`
must produce output byte-identical to the safe path it replaces; the parity suite is the proof.

---

## Risks & stop rules

1. **Instruction count ≠ wall-clock.** Throughput is the gate precisely to catch this; revert
   instruction wins that don't move ms/batch (Step 2.4).
2. **Diminishing returns.** Stop tuning a kernel when a round yields < ~5% throughput on its path.
3. **Hard floors.** The cheapest path (tracks-only, ~1 ms/batch) is partly FFI fixed-cost- and
   memory-bound; there is a floor below which instruction tuning does nothing. Record honestly;
   do not force a win that isn't there.
4. **`unsafe` risk** is contained by the byte-identical parity gate on both backends; no `unsafe`
   lands without a `// SAFETY:` comment and a passing parity suite.
5. **Profiling-profile codegen drift.** Gate numbers come only from the plain `--release` build;
   the `[profile.profiling]` build is for perf attribution and is never the measured artifact.

---

## Deliverables

- New worktree on a `opt/round3-*` branch off `rust-migration`, fresh pixi env.
- `[profile.profiling]` added to `Cargo.toml`.
- Step-1 consolidated profile table (committed under `docs/roadmaps/` or the round-3 roadmap
  section).
- Per-kernel tuning commits, each with asm-delta + throughput + parity evidence in the message.
- Roadmap round-3 section with target table, asm deltas, final ratios, decisions-log entry.
- Full-tree-green on both backends, cargo test, lint/format/typecheck, abi3 build.
