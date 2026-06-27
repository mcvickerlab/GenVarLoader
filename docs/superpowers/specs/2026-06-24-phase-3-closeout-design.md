# Design: Phase 3 close-out — main merge, missing-kernel ports, seqpro 0.20

**Date:** 2026-06-24
**Branch:** `phase-3-reconstruction` (Phase 3 PR #245 → `rust-migration`)
**Status:** approved (design); pending implementation plan

## Context & motivation

Phase 3 of the Rust migration (reconstruction + track realignment) was marked `✅` in
`docs/roadmaps/rust-migration.md`, but the roadmap is internally inconsistent: the phase
header is `✅` while four sub-items (lines 282–285) are left unchecked, and the close-out
commits updated the file sloppily. Separately, two bug fixes that were surfaced *during*
Phase 3 landed on `origin/main` and are not yet on this branch. And seqpro shipped 0.20.0
with a faster `to_numpy(validate=False)` path that GVL should adopt at guaranteed-uniform
materialization sites.

This spec closes Phase 3 honestly: absorb the main fixes, port the one genuinely-missing
rust kernel, fuse the remaining unfused-but-rust read paths, bump seqpro, and reconcile the
roadmap with reality.

### Verified ground truth (the audit behind this plan)

- **`origin/main` is 9 commits ahead** of this branch with two real fixes:
  - **PR #244 / #242** — `fix(intervals): clip sub-query interval starts in both kernels`.
    Touches `python/genvarloader/_dataset/_intervals.py` (+13) and `src/intervals.rs` (+45).
  - **PR #243** — `fix(indexing): SpliceIndexer.parse_idx double-applies sample-subset map`.
    Touches `python/genvarloader/_dataset/_indexing.py`.
- **Merge interaction:** Phase 3 never modified `src/intervals.rs`, so main's clip fix merges
  clean on the Rust side. The Phase 3 fused tracks kernel
  `intervals_and_realign_track_fused` (`src/ffi/mod.rs:653`) **calls the shared
  `intervals::intervals_to_tracks` core**, so it inherits the #242 fix automatically — no
  manual Rust propagation. The only text conflict is `_intervals.py` (main +13 vs Phase 3 +45).
- **Backend reality on the default (no `GVL_BACKEND`) read path:**
  - Splice (`_haps.py:855`) and annotated (`_haps.py:903`) haps already run **rust** — they
    call the dispatch wrapper `reconstruct_haplotypes_from_sparse` (`default="rust"`), just
    **unfused** (2 FFI crossings instead of 1). They are *correct*, not broken.
  - `shift_and_realign_track_sparse` (singular) is **only** a numba parity reference — never
    on the default path. Nothing to port.
  - The one **genuinely-missing rust port** is `Reference.fetch` (`_fetch_impl_par`/
    `_fetch_impl_ser`, `_reference.py:164–183`): a thin per-row `padded_slice` loop with no
    rust impl, used by the spliced ref-only dataset path (`RefDataset._getitem_spliced`) and
    `_flat_flanks.py`.
- **seqpro 0.20.0** is the current PyPI release. Its skip-validation addition is
  `to_numpy(validate=False)` (skips the uniformity scan). The Rust `seqpro-core` is `0.1.0`
  from crates.io (independently versioned from the Python package).
- **~10 `#242` test exclusions** (`xfail(reason=_REASON_242)` + `assume(False)` guards) exist
  solely because #242 was unfixed; they become real passing tests once the fix is merged.

## Goals

1. Bring the branch to an honest, fully-rust-default state for Phase 3's banner
   (reconstruction + track realignment).
2. Absorb the bug fixes that landed on `main` during Phase 3.
3. Bump seqpro to 0.20.0 and adopt its skip-validation arg where safe.
4. Reconcile the roadmap with what is actually done.

## Non-goals (deferred, with honest roadmap notes)

- Deleting numba parity references — Phase 5.
- The broad "single big `__getitem__` kernel" beyond the specific fusions below — Phase 5.
- Write-path concerns / `Reference.fetch` callers beyond what parity requires — Phase 4.
- Any public-API change (this work is entirely internal).

## Work plan (dependency order)

### Step 1 — Merge `origin/main` into `phase-3-reconstruction`

- Merge commit (not squash; preserves history per maintainer preference).
- Brings #244 (#242) + #243 onto the branch. When this branch later merges to
  `rust-migration`, the fixes flow through.
- **Conflict resolution:** `python/genvarloader/_dataset/_intervals.py` — reconcile main's
  clip fix (+13) with Phase 3's edits (+45). `src/intervals.rs`, `_indexing.py` merge clean.
- **Acceptance:** branch builds (`cargo build`, `maturin develop`), no leftover conflict
  markers, `src/intervals.rs` carries the clip fix.

### Step 2 — Lift the now-obsolete #242 exclusions

- Remove `xfail(reason=_REASON_242)` markers and the `_REASON_242` constants from:
  - `tests/dataset/test_flat_intervals.py`
  - `tests/dataset/test_seqs_tracks.py`
  - `tests/dataset/test_realign_tracks.py`
  - `tests/unit/dataset/test_output_bytes_per_instance.py`
  - `tests/integration/dataset/test_dummy_dataset_insertion_fill.py`
- Remove the `assume(False)` #242-family guards in
  `tests/parity/test_reconstruct_haplotypes_parity.py` and
  `tests/parity/test_shift_and_realign_tracks_parity.py` **that correspond to the
  `itv.start < query_start` / `start>=clen` #242 domain only**.
- **Keep** the *reconstruct trailing-under-write* exclusion (overshoot pre-check +
  double-init guard) — that is a genuine numba-undefined domain, unrelated to #242.
- **Acceptance:** these tests now run (not xfail) and pass on `max_jitter>0` datasets under
  both `GVL_BACKEND=rust` and `GVL_BACKEND=numba`.

### Step 3 — Port `Reference.fetch` to rust

- Add a rust kernel (working name `fetch_reference`) in the `src/reference/` module that
  loops rows and calls the existing `padded_slice` core, mutating the caller's `out` buffer
  in place (mirrors `_fetch_impl_ser`/`_par`; serial is fine — disjoint per-row out-slices).
- Expose via `src/ffi/`; register in `python/genvarloader/_dataset/_reference.py` through
  `_dispatch.register(..., default="rust")`, keeping the numba `_fetch_impl_*` as the parity
  reference. Route `Reference.fetch` through the dispatcher.
- **Acceptance:** byte-identical parity (hypothesis suite, both impls) for `fetch_reference`;
  spliced ref-only dataset path (`RefDataset._getitem_spliced`) and `_flat_flanks.py`
  exercise the rust kernel by default. Closes the last 3 numba kernels of roadmap item 3.

### Step 4 — Fuse the annotated-haps and splice haps paths

Both currently run correct-but-unfused rust (2 FFI crossings via the dispatch wrapper).

- **Annotated haps:** add/extend a fused rust entry that fills `out`, `annot_v_idxs`, and
  `annot_ref_pos` in a single FFI crossing (currently `_haps.py:903` composes via the
  wrapper). Route `_reconstruct_annotated_haplotypes` (non-splice branch) through it when
  `GVL_BACKEND` is rust (default), mirroring the Task-13 `reconstruct_haplotypes_fused`
  pattern.
- **Splice haps:** add a fused rust entry that consumes the splice-permuted request
  (`flat_geno_idx`, `flat_shifts`, `permuted_regions`, permuted keep arrays,
  `splice_plan.permuted_out_offsets`) and reconstructs in one crossing (currently
  `_haps.py:855` composes via the wrapper). The Python-side splice permutation
  (`_permute_request_for_splice`) stays in Python; only the reconstruction crossing fuses.
- Annotated + splice combined (annotated path with a splice plan) may remain on the unfused
  dispatched rust path if fusing the combination is disproportionately complex — if so,
  document it as a Phase-5 residue rather than claiming 100%.
- **Acceptance:** byte-identical dataset parity vs the composed numba oracle for each fused
  path (same gate style as Tasks 13–14), across insertion-fill strategies where relevant.
  Closes roadmap items 1 and 4.

### Step 5 — Bump seqpro to 0.20.0 + adopt skip-validation

- `pixi.toml`: `seqpro = "==0.18.0"` → `"==0.20.0"`.
- `pyproject.toml`: `"seqpro>=0.18"` → `"seqpro>=0.20"`.
- Re-run `pixi install`/lock; confirm the env resolves and `import seqpro; __version__ == 0.20.0`.
- **Skip-validation adoption (propose-then-approve):** inventory read-path `.to_numpy()` /
  fixed-length materialization sites where row uniformity is *guaranteed by construction*
  (e.g. `with_len(L)` / `to_fixed` / `to_padded` outputs). Propose `validate=False` at those
  sites for maintainer approval before applying. Do **not** blanket-apply.
- **Rust compat check:** confirm `seqpro-core` 0.1.0's `Ragged` layout (offsets + data +
  itemsize) still matches what GVL's `src/ragged/mod.rs` bridge constructs against seqpro
  0.20.0. Low risk (core is pyo3-free and independently versioned), but verified via
  `cargo test` + the dataset parity backstop.
- **Acceptance:** full tree green on 0.20.0; any `validate=False` sites approved and parity
  unchanged.

### Step 6 — Roadmap + skill honesty pass

- `docs/roadmaps/rust-migration.md`:
  - Reconcile the `✅`-header / unchecked-boxes contradiction in Phase 3.
  - Check off items 1, 3, 4 (now truthfully done); reword item 2 to state tracks/intervals
    realign is rust-default + fused, with the remaining numba retained as Phase-5-deletion
    parity references.
  - Add a dated decisions-log entry recording: #242 fix merged + xfails lifted,
    `Reference.fetch` ported, annotated/splice fused, seqpro 0.20 bump.
- `skills/genvarloader/SKILL.md`: confirm no public-API change (expected no-op per CLAUDE.md
  maintenance rule). Update only if an exported symbol/signature changed (none expected).

## Verification gate (migration contract)

- `cargo test` green (incl. new `fetch_reference` + fused-kernel unit tests).
- Full pytest tree green: `pixi run -e dev pytest tests -q` (cover `tests/dataset` **and**
  `tests/unit` per CLAUDE.md), including the un-xfailed #242 tests, under **both**
  `GVL_BACKEND=rust` and `GVL_BACKEND=numba`.
  - Env note: dataset tests need `--basetemp=$(pwd)/.pytest_tmp` on Carter HPC (os.link
    cross-device Errno 18), same as Phases 2–3.
- Byte-identical parity for `fetch_reference` and the fused annotated/splice kernels.
- `ruff check python/ tests/`, `ruff format`, `typecheck` clean; abi3 wheel builds.
- Throughput recorded (not gated) for the newly-fused paths, appended to the Phase 3
  measurement block.

## Risks & mitigations

- **`_intervals.py` merge conflict** — small, mechanical; resolve by keeping both the clip
  fix and Phase 3's additions. Mitigation: re-run the intervals parity + #242 tests after.
- **Splice fusion complexity** — the permuted-request plumbing is the most involved piece.
  Mitigation: keep the Python permutation in Python; fuse only the reconstruction crossing;
  fall back to the documented unfused-rust path (with an honest roadmap note) if the
  annotated×splice combination proves disproportionate.
- **seqpro 0.20 Ragged layout drift** — could break the Rust bridge. Mitigation: `cargo test`
  + dataset parity backstop catch any layout mismatch immediately.
- **Lifting xfails exposes a latent failure** — if an un-xfailed test fails, that is a real
  signal (the clip fix didn't fully cover it). Mitigation: investigate rather than re-xfail;
  the #242 fix is the contract.

## Out-of-scope confirmations

No public API changes; no numba deletion; no write-path migration; no new perf gate (Phase 3
remains parity-gated, throughput recorded only, per the branch/gate strategy).
