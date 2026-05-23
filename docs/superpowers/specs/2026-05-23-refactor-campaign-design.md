# Refactor Campaign: Readability + Targeted Architecture

**Status:** Design — pending plan
**Date:** 2026-05-23
**Scope:** `python/genvarloader/` (no Rust changes)

## Goals

Make the GenVarLoader Python codebase easier to read and maintain, taking targeted
architectural redesign where the readability debt reveals genuine structural seams.
The work runs as a multi-PR campaign, each PR independently shippable.

## Non-goals

- No public API changes (the `gvl.write` / `Dataset.open` / `Dataset.with_*` /
  exported names in `python/genvarloader/__init__.py` `__all__` surface stays
  identical in behavior and signatures).
- No on-disk dataset format changes.
- No Rust changes.
- No performance work, except where a refactor incidentally enables it.
- No implementation of the future variant-major haplotype-reconstruction class
  (see "Forward compatibility" below). We only ensure architecture does not
  preclude it.

## Hard constraints

- The full test suite (`pixi run -e dev test`, both pytest and cargo) passes
  after each PR.
- Each PR is independently shippable, reviewable in one sitting, and uses
  conventional-commit messages (the project uses commitizen).
- Any PR that changes the public API surface must update
  `skills/genvarloader/SKILL.md` in the same PR (per `CLAUDE.md`).
- `__all__` in `python/genvarloader/__init__.py` stays unchanged. If a class is
  collapsed (e.g. `ArrayDataset`/`RaggedDataset`), the names remain importable
  (as type aliases if necessary) so user code does not break.

## Forward compatibility: future variant-major class

The user plans (post-campaign, not implemented here) to add a class for
reconstructing haplotypes directly from VCF/BCF/PGEN, optimized for inference
(arbitrary access pattern, variant-major + block-compressed over a subset of
variants and *all* samples). This is a different access pattern from the current
region/sample-major `Dataset`.

This campaign must leave the architecture able to accommodate it without
duplicating the haplotype-reconstruction kernel. Concretely:

- The sparse-genotype → sequence machinery (currently in `_genotypes.py` +
  `_reconstruct._get_haplotypes`) must be callable independent of `Dataset`'s
  region-major iteration. It should be a pure function (or method on a small
  pipeline object) that takes "which variants apply, which samples, which
  reference window" and returns sequences. The region-major iteration driver in
  `Dataset` is layered *on top* of this kernel, not woven into it.
- Reconstruction request objects (PR6 below) describe *what* to reconstruct,
  agnostic of *how* iteration is driven.
- Names: the write-side `VariantSource` protocol (PR2) is for writers; the
  future read-side variant-major API will be a separate protocol. To leave room,
  the writer protocol is named `VariantWriter` (or similar), not `VariantSource`.
- File layout: the package leaves room for a future `python/genvarloader/_online/`
  (or similarly named) sibling to `_dataset/`.

The future class is **not** implemented in this campaign.

## Debt audit (summary)

Heavyweight files:

- `_dataset/_impl.py` — 2253 lines
- `_dataset/_reconstruct.py` — 1525 lines
- `_dataset/_write.py` — 832 lines
- `_dataset/_reference.py` — 758 lines
- `_dataset/_genotypes.py` — 571 lines

Specific debt items the campaign targets:

1. `ArrayDataset` / `RaggedDataset` carry ~150 lines of overload boilerplate that
   only `super()`-calls — a type-system workaround.
2. Five reconstructor classes (`Haps`, `Ref`, `Tracks`, `RefTracks`, `HapsTracks`)
   express a combinatorial product (`Optional[Seqs] × Optional[Tracks]`).
   `RefTracks` and `HapsTracks` duplicate track-stitching logic.
3. `Dataset.with_settings()` is 154 lines validating ~8 different settings inline.
4. `Dataset.open()` is 206 lines mixing argument parsing, validation, splice
   resolution, reconstructor init, and indexer build.
5. `_write_from_vcf`, `_write_from_pgen`, `_write_from_svar` are a Strategy
   pattern done as if/elif with ~70% shared chunking logic.
6. `Haps._get_haplotypes` has 3 overloads in 167 lines (different return shapes
   for different caller combinations).
7. `Tracks.write_transformed_track` is 166 lines with deep branching over
   insertion-fill strategies.
8. Naming smells: `rsp_idx`, `perm` shadowing data vs loop var, plural/singular
   inconsistencies (`reconstruct_haplotype_from_sparse` vs
   `reconstruct_haplotypes_from_sparse`).
9. ~50 `# type: ignore` comments cluster around `Ragged` operations and
   `evolve()` calls — likely a type-modeling issue.

## PR sequence

Each PR is independently shippable and keeps the full test suite green.

### PR0 — Integrate pyrefly as a type-check gate

Add [pyrefly](https://github.com/facebook/pyrefly) as a type-checking gate for
the library code, configured for the `strict` preset as the baseline.

**Scope:**

- Add a `[tool.pyrefly]` section to `pyproject.toml` with `project_includes`
  scoped to both `python/genvarloader/` and `tests/`, with the `strict` preset
  enabled.
- Wire in the [`facebook/pyrefly-pre-commit`](https://github.com/facebook/pyrefly-pre-commit)
  hook in `.pre-commit-config.yaml` (creating the file if absent), pinned to a
  released revision.
- Add a CI job (matching the existing `pixi run -e dev` flow) that runs
  `pyrefly check` on the library and fails on any error.
- Make `pyrefly check` pass on `python/genvarloader/` AND `tests/` under
  `strict` as the baseline. Where strict is too noisy to fix in this PR,
  relax the rule in `pyproject.toml` (per-rule, with a short comment
  explaining why) rather than blanket-suppressing — the relaxed config
  becomes the de facto baseline that subsequent PRs must continue to satisfy.
- Remove `basedpyright`: delete the `[tool.basedpyright]` block from
  `pyproject.toml`, drop it from `pixi.toml`/dev tasks, and remove any
  references from `CLAUDE.md`. Pyrefly is the sole type checker.

**Touches:** `pyproject.toml`, new `.pre-commit-config.yaml` (or update if
present), CI config, possibly a small number of source files where strict
errors are cheaper to fix than to suppress.

**Risk:** medium. Strict may surface unexpected issues. The mitigation is
explicit: relax pyrefly config rather than expand the PR's blast radius.

**Win:** every subsequent refactor PR is type-checked under pyrefly-strict
automatically. This catches subtle regressions during architectural moves
(PRs 3, 5, 6 especially), and folds naturally into PR8's `type: ignore` audit
since by then pyrefly will be the source of truth for what suppressions are
actually needed.

### PR1 — Centralize reconstructor construction via a factory

**Revision note (post-PR0):** the original PR1 framing (`DatasetSettings` value
object) didn't fit the code well — only 4 of the 9 args to `with_settings` are
"settings" in a value-object sense; the others mutate the underlying `_seqs` /
`_recon` / `_sp_idxer` reconstructor state. The real source of the propagation
ugliness is that `_recon` duplicates state from `_seqs` (`HapsTracks.haps` is a
copy of `_seqs`), so changing `_seqs` requires `isinstance`-dispatched re-stitching
of `_recon` in every `with_*` method. We address that directly here.

**The fix:** make the 5 reconstructor classes (`Haps`, `Ref`, `Tracks`, `RefTracks`,
`HapsTracks`) a derived view of authoritative source state. Specifically:

- Authoritative state on `Dataset`: `_seqs: Haps | Ref | None` (where `Haps`
  already carries the output kind in its generic parameter) and `_tracks: Tracks | None`.
- Add `_build_reconstructor(seqs, tracks) -> Reconstructor` factory in
  `_reconstruct.py`. Single source of truth for which of the 5 classes to construct,
  given the (seqs, tracks) sources. The factory enforces the invariant that at
  least one source must be present.
- `_recon` stays as a stored field on `Dataset` (avoiding attrs-frozen property
  awkwardness), but is only ever assigned via the factory.
- Collapse the scattered construction sites:
  - `Dataset.open` (~15-line match) → factory call
  - `with_seqs` (~90-line match constructing different `RefTracks` / `HapsTracks`
    variants) → update `_seqs.kind` via `Haps.to_kind`, factory call
  - `with_tracks` → update `_tracks`, factory call
  - `with_settings` propagation block → update `_seqs`, factory call
- `sequence_type` property simplifies from a `match` on `_recon`'s class to a
  lookup on `_seqs.kind`.

**The 5 reconstructor classes are NOT collapsed.** They already form a valid ADT
(invalid `(None, None)` state isn't representable in their union), and the two
combined classes (`RefTracks`, `HapsTracks`) encode genuinely different semantics
— `RefTracks.__call__` is naive composition, `HapsTracks.__call__` does indel-aware
joint reconstruction (it calls `Haps.get_haps_and_shifts` and re-aligns tracks to
haplotype coordinates). Forcing both into one class would shove ~50 lines of
specialized joint-reconstruction logic alongside a 2-line naive composition.

**Implication for the campaign:** original PR5 ("Pipeline composition for
reconstructors") is no longer needed — the existing 5 classes are already the
ADT we want, just with their construction now centralized. PR5's other goal
(extracting the haplotype-reconstruction kernel for forward compat with the
future variant-major class) folds into PR5 (was PR6) below.

**Touches:** `_dataset/_reconstruct.py` (add factory), `_dataset/_impl.py`
(rewrite scattered construction sites).
**Risk:** medium. Touches `with_seqs`, `with_tracks`, `with_settings`,
`Dataset.open` — all heavily used. Full test suite is the parity check.
**Win:** propagation block collapses from ~10 lines of `isinstance` dispatch to
a one-liner factory call; the 90-line `with_seqs` match collapses; the
invariant ("either seqs or tracks must be present") moves into the type system
at the factory boundary.

### PR2 — `VariantWriter` protocol

Replace the `_write_from_vcf` / `_write_from_pgen` / `_write_from_svar` dispatch
with a small `VariantWriter` protocol (`iter_chunks() → Iterator[Chunk]`, or
similar). `write()` picks the writer; the shared chunking logic moves into a
single `_write_variants_chunked()` helper.

**Name choice:** "writer", not "source" — reserve "VariantSource" terminology
for the future variant-major read-side protocol.

**Touches:** `_dataset/_write.py`.
**Risk:** low. Behavior-preserving dedup.
**Win:** three near-duplicate writers collapse to one helper + three thin
protocol implementations.

### PR3 — Collapse `ArrayDataset` / `RaggedDataset`

Move from class-hierarchy expression of `Ragged|Padded` to generic typing on a
single `Dataset` class. The names `ArrayDataset` and `RaggedDataset` remain
importable as type aliases (or thin views) so `__all__` is unchanged and user
imports keep working. The ~32 overload stubs that just `super()` are deleted.

**Touches:** `_dataset/_impl.py`, `python/genvarloader/__init__.py` (if needed
for re-exports), `skills/genvarloader/SKILL.md`.
**Risk:** medium. Type-system change; downstream user type hints may need
adjustment. Public-API-adjacent.
**Win:** ~150 lines of boilerplate deleted; the actual generic shape becomes
visible.

### PR4 — `OpenRequest` + decompose `Dataset.open`

Extract an `OpenRequest` value object that holds the parsed/validated arguments
to `Dataset.open`, with a `.resolve() → Dataset` method. `Dataset.open` becomes
a thin classmethod that builds and resolves the request. This is possible *after*
PR1 because settings are already a value object.

**Touches:** `_dataset/_impl.py`, new `_dataset/_open.py`.
**Risk:** low–medium. Splits a 206-line method; needs careful parity testing.
**Win:** `open()` shrinks to ~30 lines; resolution stages testable in isolation.

### PR5 — `ReconstructionRequest` + extract haplotype kernel

Introduce a `ReconstructionRequest` value object describing *what* to reconstruct
(region, sample, splice plan if any, annotation flag). The three overloads of
`Haps._get_haplotypes` collapse to one method taking the request. Same treatment
for `Tracks.write_transformed_track` (166 lines) — split the insertion-fill
strategies into named helpers.

**Critical for forward compatibility:** the haplotype-reconstruction *kernel*
must remain callable independent of region-major iteration. The current
`Haps._get_haplotypes` does too much — it mixes the per-window reconstruction
kernel with iteration plumbing. The kernel is extracted as a pure function (or
method on `Haps`) that takes (variants, samples, reference window, optional
splice plan) and returns sequences. This was originally scoped to the old PR5
(Pipeline composition); since that PR is no longer needed, the kernel extraction
folds in here.

**Touches:** `_dataset/_reconstruct.py`, callers.
**Risk:** medium. Numerical parity verification.
**Win:** overload triplet disappears; 167-line and 166-line methods
decomposed; future variant-major class can construct requests without going
through region-major iteration.

### PR6 — File splits

With architectural boundaries now real, split `_impl.py` and `_reconstruct.py`
along the seams:

- `_dataset/_impl.py` keeps the `Dataset` class itself.
- `_dataset/_open.py` — already created in PR3.
- `_dataset/_query.py` — `__getitem__` and the spliced/unspliced query paths.
- `_dataset/_haps.py` / `_dataset/_ref.py` / `_dataset/_tracks.py` (latter
  already exists) — one source class per file.

**Touches:** mostly mechanical moves + import updates.
**Risk:** low. No behavior changes.
**Win:** `_impl.py` and `_reconstruct.py` drop from 2253 / 1525 lines to
something readable in one sitting.

### PR7 — Naming pass + `type: ignore` audit

Final sweep:

- Rename ambiguous abbreviations: `rsp_idx` (document what `rsp` means or
  rename), `perm` (split data-`permutation` from loop-`perm`),
  `geno_offset_idx` → `geno_idx` if equivalent.
- Make plural/singular consistent (e.g. `reconstruct_haplotype_from_sparse` vs
  `reconstruct_haplotypes_from_sparse` — pick one or split clearly).
- Audit the ~50 `# type: ignore` comments. Many likely become unnecessary once
  PRs 1, 4, 5 have improved type modeling. Where they remain, add a one-line
  reason comment so future-us knows whether they can be removed.

**Touches:** package-wide.
**Risk:** low.
**Win:** newcomer onboarding cost drops; LSP/type-checker noise goes down.

## Sequencing rationale

- PR0 lands first so every subsequent refactor is type-checked under pyrefly
  strict, surfacing regressions during the riskier moves (PRs 1, 4, 5).
- PR1 (reconstructor factory) goes second because it centralizes the
  construction logic that PRs 2, 4, 5 all rely on; with the factory in place,
  every `with_*` method collapses cleanly.
- PR2 / PR3 are low-risk warm-ups.
- PR4 is the highest-leverage single deletion (~150 lines) and is gated only by
  willingness to touch public-adjacent names; doing it before PR5 surfaces
  user feedback early.
- PR5 is the largest architectural change and carries numerical-parity risk.
  Sequenced after PRs 1–4 so the surrounding code is already tidied (clearer
  blast radius, easier review).
- PR6 is mechanical and only worth doing once the boundaries from PRs 1, 3, 5
  are real.
- PR7 is a finisher.

## Verification strategy

Each PR:

1. Full pytest run (`pixi run -e dev test`).
2. Cargo test run (covered by `pixi run -e dev test`).
3. Ruff clean.
4. Pyrefly check clean under the baseline config established in PR0.
5. For PRs 1, 4, 5: explicit numerical-parity tests against the pre-PR
   implementation, on the existing test fixtures.

For PR5 specifically, before merging: run the existing dataset tests AND add a
direct unit test that constructs a `ReconstructionRequest` from synthetic
inputs and verifies output against a tiny hand-computed expected.

## Out of scope (revisited)

Not in this campaign — captured here so future work can pick them up:

- The variant-major haplotype-reconstruction class for inference (post-campaign).
- BigWig backend or Rust extension changes.
- Splicing redesign (recently completed; do not re-touch).
- Reader protocol generalization beyond what PR2 does for writers.
- Test fixture overhaul.
