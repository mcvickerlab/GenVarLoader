# SVAR2 M6b branch — final pre-merge pass (design)

**Date:** 2026-07-13
**Branch:** `svar2-m6b-kernel`
**Goal:** A finishing pass over the SVAR2 read-bound work — correctness/safety fixes,
shipping hygiene, and doc consistency — so the branch is mergeable modulo the genoray
release gate (documented, not resolved here).

## Context

The branch adds SVAR2 (`.svar2` sparse variant format) support to GenVarLoader: as a
`gvl.write` variant source, a live `Dataset` read-bound backend (all-Rust FFI kernels),
INFO/FORMAT field routing into `variants`/`variant-windows` outputs, and `unphased_union`.
It is ~23k insertions across 86 files. The full test suite is green at branch HEAD
(`pixi run -e dev pytest tests -x` exits 0; SVAR2 parity gate 31/31), and `ruff`/`ruff
format` are clean.

Three focused audits (docs / Python / Rust) plus manual verification produced the
work-list below. This pass does **not** add features or change the parity contract; every
change is either a correctness fix, a hygiene cleanup, or a doc correction. The SVAR1 path
must remain byte-unchanged (additive-only), and the SVAR2 parity gate must stay 31/31.

## Out of scope / the release gate (documented, not fixed)

The branch is dev-wired to build only on this machine, and that **cannot** be resolved in
this pass — it depends on a genoray release:

- `Cargo.toml`: `svar2-codec` and `genoray_core` are `path = "/carter/users/dlaub/projects/genoray..."`.
- `pixi.toml` (`feature.py310`): `genoray` installed from a local `dist/*.whl`.
- `pyproject.toml`: genoray version constraint dropped (`"genoray"`, unpinned).

PyPI's newest genoray is 2.15.0; the INFO/FORMAT field-read + read-bound gather API this
branch consumes lives on genoray `main`, unreleased. **Decision: leave the dev pins in
place** (they are required to build/test) and add a prominent **RELEASE-GATE checklist** so
nothing ships silently un-pinned. The checklist lives in two places:

1. The PR body (a "⛔ Do not merge until" section).
2. `docs/roadmaps/rust-migration.md` Phase 6a (a `Release gate` subsection).

Checklist contents (exact lines to flip at genoray release):
- `Cargo.toml`: `svar2-codec`/`genoray_core` path-deps → published crates.io versions.
- `pixi.toml` `feature.py310`: `genoray = { path = ".../dist/*.whl" }` → `genoray = "==<release>"`.
- `pyproject.toml`: `"genoray"` → `"genoray>=<release>,<next-major>"`.
- Re-run the full py3xx matrix once the wheel is on PyPI.

Also confirm/adjust the version pins already touched but not gated: `numpy 0.28→0.29`,
`pyo3 0.28.3→0.29`, `seqpro 0.20.0→0.21.1` — verify these are the intended floors at merge.

## Workstreams

### 1. Correctness & safety (Rust + Python) — must-fix

**1a. Unsafe serial path missing its guard.**
`src/reconstruct/mod.rs` — the raw-pointer slices at `:557,564,572` and `:857,864` carve
`out_e - out_s` from caller-supplied `out_offsets`. The `debug_assert!(out_e >= out_s)`
monotonicity guard exists only on the parallel path (`:442,:742`); the serial fallback has
none, so a non-monotonic offsets array underflows to a multi-GB slice (UB) in debug and
silently in release. Fix: hoist the same `debug_assert!` into the serial loops before each
`from_raw_parts_mut`. Confirm the SAFETY comment text matches what is actually asserted.

**1b. Doc comment describing a reverted optimization.**
`src/svar2/mod.rs:800` documents a `get_unchecked` read of `dense_present`. `get_unchecked`
appears nowhere in `src/` (verified by grep) — all three `present_bit` closures
(`svar2/mod.rs:88`, `reconstruct/mod.rs:677`, `tracks/mod.rs:761`) use checked indexing. The
optimization was reverted and the comment left stale. Fix: delete/rewrite the comment to
describe the checked read that is actually there. (A doc comment asserting a false unsafe
invariant is a correctness hazard for the next reader.)

**1c. Python-reachable panics → `PyErr`.**
Arrays arriving from Python are `.as_slice().unwrap()` / `.expect("must be contiguous")`'d in
the SVAR2 kernels (`ffi/mod.rs:821-829`, `reconstruct/mod.rs:639-645,692`,
`tracks/mod.rs:725-734,806-808,834-836`). A non-contiguous view (`a[::2]`) panics instead of
raising. Fix: validate contiguity once at the `#[pyfunction]` boundary and return
`PyValueError`. Also bounds-check the pure-DEL anchor index `contig_ref_s[pos..pos+1]`
(`reconstruct/mod.rs:703`) → `PyValueError` for a variant at/past contig end. Also
`svar2/mod.rs:358-365` `assert_eq!(vk_src.len(), ...)` fires in release from a
Python-reachable path — hoist the check to the FFI boundary (`ffi/mod.rs:1411`, where
`has_fields` is known) as a `PyValueError`.

**1d. `extend_to_length` silently ignored for `.svar2`.**
`_write.py:_write_from_svar2` accepts the flag and never reads it; `chromEnd` is always
extended. Passing `extend_to_length=False` yields a different dataset than requested, with no
signal. **Fix: raise `NotImplementedError`** when `extend_to_length is False` for a `.svar2`
source (consistent with the branch's Phase-1 guard-matrix policy of failing loudly on
unsupported combinations), and document the limitation. Do **not** silently honor `True`
only — make the unsupported case explicit.

### 2. Shipping hygiene — should-fix

**2a. Test-only oracle code out of the library.**
`python/genvarloader/_dataset/_svar2_source.py` (`SparseVar2Source`) and
`_svar2_store_py.py` (`build_readbound_*`) have zero importers under `python/` — only
`tests/` uses them (they are the parity oracle + FFI-input builders). Move both under
`tests/` (e.g. `tests/_oracles/`), update test imports, and confirm nothing in the shipped
package references them. Rename on the way: `_svar2_store_py.py` holds no store class (the
Rust `Svar2Store` is the store) — the `_py` suffix is meaningless; name it for what it does
(`svar2_readbound_inputs.py` or similar). `SparseVar2Source` may keep its name once it lives
in tests.

**2b. Drop dead FFI capability.**
`reconstruct_haplotypes_from_svar2_readbound`'s `annot_v_idxs`/`annot_ref_pos` params are
`None` at both call sites (`ffi/mod.rs:889-890,1045-1046`); 3 of the 4 match arms (~60 lines)
and the `:605-608` doc are unreachable. Remove the params and the dead arms (annotated-hap
output for `.svar2` is guarded `NotImplementedError` anyway). If there's a near-term plan to
wire them, leave a one-line note instead — but default to removal (YAGNI).

**2c. Vectorize `_svar2_region_max_ends`.**
`_write.py:1092-1138` is an `O(regions × samples × ploidy)` Python triple-loop over decoded
records at write time; its own docstring flags it as a scalability follow-up. The semantics
are a per-region max over haplotypes of `(pos, end)` with a `pos`-then-`end` tie-break. Fix:
vectorize with a scatter-reduce over the decoded ragged offsets — encode `(pos<<32) | end`
into an int64, `np.maximum.reduceat` (or `np.maximum.at` on a region-index scatter) to get
the per-region max, then unpack `end`. Must stay byte-identical to the current loop
(`test_write_svar2.py` locks the cache contents + same-POS-tie behavior).

**2d. Drop unused `region_starts`.**
`_write.py:1162,1205,1214` writes/memmaps `region_starts`; `_svar2_haps.py:86-88,95` loads it
and its own docstring says it is "kept for parity/debug, NOT fed to the FFI." Remove the
array, its `svar2_meta.json` entry, and the loader. (Confirm no test asserts its presence; if
one does, drop that assertion.)

**2e. Fix the `typecheck` pixi task (also helps every worktree).**
`pixi run -e dev typecheck` = `pyrefly check` with no paths → inside any `.claude/worktrees/`
checkout pyrefly matches **zero files** (root `.gitignore` ignores `.claude/`, pyrefly honors
ignore files) and exits 0. Typecheck has effectively never run on this branch. Fix: change the
task to `pyrefly check python/genvarloader tests` (explicit paths). Then clear the one real
finding it surfaces: unused `# pyrefly: ignore[no-matching-overload]` at `_ragged.py:325`.

**2f. Relocate `tmp/svar2_mvp/` into `tests/benchmarks/`.**
19 tracked files with hardcoded absolute paths (`/carter/shared/data/gdc/...`), and a
self-contradicting `.gitignore` entry (`tmp/svar2_mvp/prof_out/` ignored while its `.md`
files are tracked). `tests/benchmarks/` already has the right shape: `profiling/` for
`profile_*.py` + shell drivers, `data/build_*.py` for corpus builders, and session-scoped
path fixtures in `conftest.py`. Plan:
- Benchmark/profiling drivers (`benchmark.py`, `bench_gvl_svar1_vs_svar2.py`, `prof_*.py`,
  `prof_*.sh`, `build_stores.py`, `validate.py`, `split_folded.py`) → `tests/benchmarks/`
  (drivers) / `tests/benchmarks/profiling/` (perf shells), with hardcoded paths replaced by
  the existing `data_dir`/`kg_dir` fixtures or a module-level constant + CLI arg.
- `.sbatch` files and `env_baseline.txt` → drop (machine/cluster-specific scratch; the perf
  numbers they produced are already captured in `docs/superpowers/notes/`).
- `prof_out/*.md` reports → drop from git (superseded by the roadmap Phase-6a results +
  notes); keep on disk locally.
- Remove the `tmp/svar2_mvp/prof_out/` `.gitignore` line; add `tmp/` if we want scratch
  ignored going forward.
Nothing of value is lost — perf conclusions live in the roadmap and notes; only
machine-specific scratch is dropped.

### 3. Rust duplication extraction — approved, highest-risk

Guarded by the parity/oracle suite (31/31) + full-tree regression. Do this **last**, rebuild
and re-run parity after each extraction, and keep each extraction its own commit so a
regression bisects cleanly.

**3a. Chunk-carving + serial/parallel dispatch helper.**
`reconstruct/mod.rs:424-578` and `:724-880` are ~150 lines verbatim (bounds build,
`split_at_mut` carve ×3, 4-arm `(av, ap)` match, serial raw-ptr path). Extract a
`carve_chunks(&mut [T], &[(usize,usize)]) -> Vec<&mut [T]>` and one dispatcher generic over
the per-chunk work closure. This is the hot path — byte-identical output is mandatory.

**3b. Readbound FFI preamble helper.**
`ffi/mod.rs:934-998, 1086-1144, 1186-1250, 1358-1432` paste the same preamble 4× (reader
lookup → regions build → `arr2_to_ranges` ×4 → `HapRanges::new` → gather → `lut_arrays` →
`split_to_flat` → `dense_range` view). Extract one helper returning
`(FlatChannels, lut_bytes, lut_off, regions)`. The diffs→`out_offsets` prefix-sum loop
(`ffi/mod.rs:846-864, 1000-1019, 1252-1268`) is pasted 3× — extract
`offsets_from_diffs(...)`. Add a `type` alias for the three readbound return tuples to clear
the `clippy::type_complexity` warnings at `:930,1182,1344`.

**3c. Shared `present_bit`.**
`reconstruct/mod.rs:675-678` == `tracks/mod.rs:759-762` (identical closure + re-documented
LSB-first invariant). Move to a `svar2::present_bit` fn documented once.

### 4. Docs & comments — should-fix

**4a. Roadmap (`docs/roadmaps/rust-migration.md`).**
- Phase 6a guard-matrix bullet (~`:812-816`) still lists `unphased_union` and
  `"variant-windows"` as guarded `NotImplementedError`; both now ship. Move them to the
  supported list.
- Gate footnote (~`:822-826`) claiming variant-windows parity is untested is false — remove
  it (`test_svar2_readbound_variants.py`, `test_svar2_fields_read.py` cover it).
- The 2026-07-05 notes-log entry (~`:890`) repeats the stale exclusions — amend or add a
  2026-07-12/07-13 entry reflecting shipped scope.
- Add a ticked task line for the INFO/FORMAT field-routing work (plan
  `2026-07-12-svar2-info-format-field-routing.md`).
- Fill the `_PR: TBD_` link once the PR exists (project rule: a ✅ phase carries a PR link) —
  or leave 🚧 until the PR number is known, then update.

**4b. `skills/genvarloader/SKILL.md`.**
- `:193-195`: `var_fields` on `.svar2` does **not** accept `ref`/`dosage`; allowed set is
  `alt|ilen|start` + store INFO/FORMAT fields (`_svar2_haps.py:261` raises otherwise). Correct
  the statement.
- `:66,170,437`: "`min_af`/`max_af` requires SVAR-backed genotypes" → "`.svar` only (not
  `.svar2`)" — `.svar2` raises `NotImplementedError`.
- `:128,442`: note `extend_to_length` has no effect / is unsupported for a `.svar2` source
  (matches the 1d fix).
- `:91`: "byte-identical … all four output modes" is contradicted by the pure-deletion ALT
  paragraph below it — qualify with "except pure-deletion ALT bytes (see below)".

**4c. Prose docs (`docs/source/`).**
- `index.md:51`: "Currently supports VCF, PGEN, and BigWig" — mirror README's updated
  `.svar`/`.svar2` wording.
- `faq.md:81` + `write.md:98` point at `format.md` "for the full list" of unsupported
  `.svar2` combinations, which does not exist there → add the guard-matrix list to
  `format.md`, or repoint to the actual location.
- `write.md` §"Variants from a genoray sparse store": add a 2-line snippet showing how to
  *build* a `.svar`/`.svar2` (`genoray` `dense2sparse` / `SparseVar2.from_vcf`), since
  `faq.md:76` promises it.
- `dataset.md`: add a short "Variant fields (`var_fields`)" section — the branch's headline
  feature (`.svar2` store INFO/FORMAT fields on `variants`/`variant-windows`, e.g. `rv["AF"]`)
  is currently documented only in the skill.
- `format.md:145`: pin the `(unreleased)` changelog row to the target version at merge.

**4d. Strip internal plan/task numbering from shipped code.**
Comments/docstrings referencing planning artifacts (reader has no access to them):
`_svar2_haps.py:23` (stale — lists `unphased_union` as unsupported; delete that entry),
`:384,486,489` ("tracks follow-up (7c)"), `_reconstruct.py:143` ("Task 7c"), `:399` ("FIX 1
guard"), `_write.py:1196` ("Phase-1 wiring"); Rust `svar2/mod.rs:280` ("Task 1.3"),
`tracks/mod.rs:2467` ("Task 4 Part C"), `ffi/mod.rs:774` ("first cut minimal"). Rewrite each
in terms of behavior. The `_reconstruct.py:399` FlankSample fill-seed divergence that the
guard papers over needs a tracked GitHub issue (the comment is currently the only record) —
open one and reference it.

**4e. Missing docstrings.**
Add numpydoc-style docstrings to public/semi-public surfaces lacking them:
`_svar2_link.py:make_svar2_link`, `_svar2_haps.py:_reconstruct_variants` (sibling
`_reconstruct_variant_windows` has one), `_write.py:_write_from_svar2` (SVAR1's
`_write_from_svar` has one); Rust `svar2/store.rs:16,19,26,46` (`reader`, `store_path`,
`#[new]`, `contigs` — PyO3-exposed, become Python docstrings).

### 5. Clippy nits (`cargo clippy --all-targets`) — new-code only

Clear the new-code-attributable warnings; leave pre-existing ones (`bigwig.rs`,
`reference/mod.rs`, etc.) alone as out-of-scope:
- `reconstruct/mod.rs:248,251` `explicit_auto_deref` → drop `as_deref_mut()`.
- `reconstruct/mod.rs:19-37,278` `doc_overindented_list_items` (4→2 spaces).
- `svar2/mod.rs:593,600,772` + `reconstruct/mod.rs:905` `single_range_in_vec_init` /
  `redundant_closure` (tests).
- `type_complexity` aliases from 3b cover `:930,1182,1344`.

## Sequencing

1. Docs & comments (§4) + clippy nits (§5) — zero runtime risk, do first.
2. Correctness/safety (§1) — rebuild Rust (`maturin develop --release`), run SVAR2 parity.
3. Hygiene (§2) — oracle relocation, dead-capability removal, `max_ends` vectorization,
   `region_starts` drop, typecheck-task fix, `tmp/` relocation.
4. Rust duplication (§3) — last, one extraction per commit, parity re-run after each.

## Verification (gates — all must pass before PR)

- `pixi run -e dev maturin develop --release` (rebuild before any Python test touching Rust).
- `pixi run -e dev pytest tests -x` — full tree green (SVAR1 byte-unchanged, SVAR2 gate 31/31).
- `pixi run -e dev cargo test` (compiles from source; LD_LIBRARY_PATH per pixi activation.env).
- `pixi run -e dev ruff check python/ tests/` + `ruff format --check python/ tests/`.
- `pixi run -e dev pyrefly check python/genvarloader tests` (the fixed task) — clean.
- `pixi run -e dev cargo clippy --all-targets` — no **new** warnings.
- `api.md` ↔ `__all__` check (should remain "none"; no public symbol added).
- Manual: confirm no `python/` module imports the relocated oracle code; confirm the
  RELEASE-GATE checklist is present in both the PR body and the roadmap.

## Non-goals

- Resolving the genoray path-pins (release-gated; documented only).
- Any new SVAR2 feature, output mode, or change to the parity contract.
- Refactoring the unrelated genoray-3.0-API bump bundled into this branch (the
  `ContigNormalizer` module moves + `_vcf_region_chunks` rewrite) — it's clean; splitting it
  into its own commit is optional and noted for the reviewer, not required.
- Touching pre-existing clippy warnings outside the SVAR2 changes.
