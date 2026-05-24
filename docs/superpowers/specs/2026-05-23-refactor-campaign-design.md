# Refactor Campaign: Readability + Targeted Architecture

**Status:** In progress — PR0, PR1, attrs migration, PR2, PR4, PR5, PR5c, PR6 shipped; PR3 dropped after investigation; PR7 pending
**Date:** 2026-05-23
**Scope:** `python/genvarloader/` (no Rust changes)

## Progress log

| Date | PR | What | Branch | Result |
|---|---|---|---|---|
| 2026-05-23 | #181 | PR0 — pyrefly strict baseline + drop basedpyright + CI lint workflow | `refactor/pr0-pyrefly` | Merged. 0 errors / 117 warnings under curated baseline. CI uses `j178/prek-action` via `prek` so `.pre-commit-config.yaml` is single source of truth. |
| 2026-05-23 | #182 | PR1 — promote reconstructor view-state to explicit `Dataset._seqs_kind` field; centralize construction in `_build_reconstructor(seqs, tracks, seqs_kind)` factory | `refactor/pr1-build-reconstructor` | Merged. ~150 lines net removed from `_impl.py`; `sequence_type` is now a one-line field lookup. Latent bug fixed in passing (`Tracks.active_tracks` is a dict, not optional list; was always-truthy so `with_insertion_fill` guard never fired after `with_tracks(False)`). Test suite 475 → 488 passing (+12 factory unit tests + 1 newly-active insertion-fill test). |
| 2026-05-23 | #183 | attrs → stdlib dataclasses migration (out-of-band, not numbered in the campaign) | `refactor/attrs-to-dataclass` | Merged. 10 source files + tests migrated to `@dataclass(slots=True)`; `attrs` dropped from `pyproject.toml` / `pixi.toml`. Zero behavioral changes. Motivated by pyrefly's weaker support for attrs. |
| 2026-05-23 | #185 | PR2 — VCF + PGEN writer dedup via shared `_write_phased_chunked` helper | `refactor/pr2-write-dedup` | Merged. `_write.py`: 832 → 817 lines (−15 net). `_write_from_vcf` / `_write_from_pgen` collapsed to small setup + per-region generators that feed a shared aggregation helper. SVAR untouched (per scope refinement). |
| 2026-05-23 | — | PR3 dropped after investigation (see "PR3 dropped" section below) | `refactor/pr3-collapse-datasets` (closed) | Dropped. Probed phantom-types + self-typed overloads; concluded the ~150 lines of overload stubs encode the Array/Ragged ADT in Python's type system and are not deletable boilerplate. Net realistic deletion would be ~30 lines of `super()` bodies — not worth the architectural churn. |
| 2026-05-23 | #187 | PR4 — `OpenRequest` + decompose `Dataset.open` | `refactor/pr4-open-request` | Merged. `_impl.py`: 2156 → 2015 (−141). New `_dataset/_open.py` (262 lines) houses `OpenRequest` with 10 small stage methods. `Dataset.open` becomes a ~20-line facade. Public API unchanged. |
| 2026-05-23 | #188 | PR5 — `ReconstructionRequest` + restructure `Haps._get_haplotypes` | `refactor/pr5-reconstruction-request` | Merged. `_reconstruct.py`: 1595 → 1682 lines (+87 net; reorganization adds dataclass + helper, deletes ~30 lines of overloads). `Haps._get_haplotypes` overload triplet replaced with `_reconstruct_haplotypes` / `_reconstruct_annotated_haplotypes`. Splice permutation prep extracted into shared `_permute_request_for_splice`. `get_haps_and_shifts` now a two-stage op (`_prepare_request` → dispatch). Spec revised inline to reflect audit findings; PR5c (dead-code in `write_transformed_track`) carved out as separate small PR. |
| 2026-05-23 | #189 | PR5c — delete dead body of `Tracks.write_transformed_track` + add `docs/superpowers/roadmap.md` | `refactor/pr5c-dead-code-roadmap` | Merged. `_reconstruct.py`: 1682 → 1539 (−143). 145 unreachable lines under unconditional `NotImplementedError` deleted. New internal roadmap doc captures the feature's intent (transformed track writing) for future revival; structured to grow. Public API behavior unchanged. |
| 2026-05-23 | (this PR) | PR6 — file splits + `QueryView` composition | `refactor/pr6-file-splits` | `_reconstruct.py` decomposed: `Ref` → new `_ref.py` (76 lines), `Haps` + `ReconstructionRequest` + `_Variants` → new `_haps.py` (870 lines), `Tracks` + `TrackType` merged into existing `_tracks.py` (379 → 736 lines). `Reconstructor` Protocol moved to new `_protocol.py` (34 lines) to break circular imports. `_reconstruct.py`: 1539 → 310 lines (now just `RefTracks` + `HapsTracks` + `_build_reconstructor` + re-exports). `Dataset.__getitem__` cluster (getitem + `_getitem_(un)spliced` + `_build_splice_plan` + `_rc` + `_pad` + module-level `_regroup`) extracted to new `_query.py` (403 lines) via a typed `QueryView` frozen dataclass; `Dataset.__getitem__` becomes a ~15-line facade. `_impl.py`: 2015 → 1712 lines. Audit-driven design folded in inline (see PR6 section below). |

**Current state of heavyweight files (post the merged PRs):**

- `_dataset/_impl.py` — 1712 lines (was 2253; −541 net, post-PR4+PR6)
- `_dataset/_reconstruct.py` — 310 lines (was 1525; −1215 net — now just compound classes + factory + re-exports)
- `_dataset/_haps.py` — 870 lines (new in PR6; houses `Haps`, `_Variants`, `ReconstructionRequest`)
- `_dataset/_tracks.py` — 736 lines (was 379; merged class + numba kernels in PR6)
- `_dataset/_query.py` — 403 lines (new in PR6; `QueryView` + free-function query path)
- `_dataset/_open.py` — 262 lines (new in PR4)
- `_dataset/_ref.py` — 76 lines (new in PR6)
- `_dataset/_protocol.py` — 34 lines (new in PR6; `Reconstructor` Protocol, lives alone to break import cycle)
- `_dataset/_write.py` — 817 lines (was 832; −15 after PR2)
- `_dataset/_reference.py` — 756 lines (unchanged-ish)
- `_dataset/_genotypes.py` — 571 lines (unchanged)

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
- `__all__` in `python/genvarloader/__init__.py` stays unchanged.

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
  `Dataset` is layered *on top* of this kernel, not woven into it. **PR5
  (`ReconstructionRequest` + haplotype kernel extraction) is the slot for this
  work.**
- Reconstruction request objects (PR5) describe *what* to reconstruct,
  agnostic of *how* iteration is driven.
- Names: any writer-side protocol introduced in PR2 should be named
  `VariantWriter`, not `VariantSource` — reserve "Source" terminology for the
  future read-side variant-major API.
- File layout: PR6 leaves room for a future `python/genvarloader/_online/`
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

1. ~~`ArrayDataset` / `RaggedDataset` carry ~150 lines of overload boilerplate that
   only `super()`-calls — a type-system workaround.~~ **Re-evaluated 2026-05-23
   (PR3 dropped):** these overloads encode the Array/Ragged ADT — they are not
   boilerplate but the only way to express the ADT in Python's type system. See
   the "PR3 — DROPPED" section below.
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

### PR0 — Integrate pyrefly as a type-check gate ✅ SHIPPED (#181)

Add [pyrefly](https://github.com/facebook/pyrefly) as a type-checking gate for
the library code, configured for the `strict` preset as the baseline.

**As shipped:** Pre-commit hook is a `local` hook invoking `pixi run -e dev pyrefly check` (rather than the `facebook/pyrefly-pre-commit` repo hook), because a `language: system` hook silently picks up a `~/.pixi/bin/pyrefly` outside the project env. CI runs `j178/prek-action@v2` after `setup-pixi activate-environment: true` so PATH carries the dev env's pyrefly. `pre-commit.ci` is configured to skip the pixi-invoking hooks (no pixi in that runner). debug-statements hook pinned to `python3.10` (the python3 default chokes on `match/case`). Baseline relaxations populated empirically in `[tool.pyrefly.errors]` with one-line reasons per category.

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

### PR1 — Centralize reconstructor construction via a factory ✅ SHIPPED (#182)

**As shipped (deviation from original spec):** The original "DatasetSettings value object" framing didn't fit the code — only 4 of 9 args to `with_settings` are settings; the rest mutate sources. Reading `with_seqs`/`with_tracks` revealed that `_recon`'s runtime class was doing double duty as (1) callable and (2) implicit view-state. Promoting view-state to an explicit `Dataset._seqs_kind: Literal[...] | None` field — combined with a single `_build_reconstructor(seqs, tracks, seqs_kind)` factory — collapsed the 90-line match in `with_seqs`, the isinstance ladders in `with_settings` propagation, the `with_tracks` match, and the `with_insertion_fill` guard. `sequence_type` is now `return self._seqs_kind`. Tracks-active state stays on `Tracks.active_tracks` (a dict — truthiness signals "active"; this is the latent bug found in passing). The 5 reconstructor classes (`Haps`, `Ref`, `Tracks`, `RefTracks`, `HapsTracks`) are preserved — they encode genuinely different combination strategies. The original PR5 ("Pipeline composition") was dropped as a result.

(The historical original-PR1 framing is below for reference but is superseded by the shipped work.)

---

**Original spec (superseded by what shipped):**

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

### PR2 — VCF + PGEN writer dedup (scope refined from original spec)

**Scope finding before implementation:** the three writers in `_write.py` are not
uniformly de-dupable. `_write_from_vcf` and `_write_from_pgen` share ~90% of
their structure (contig/chunk iteration, dense→sparse conversion,
`_write_phased_variants_chunk` calls). `_write_from_svar` is genuinely
different — it pre-allocates a memmap, uses `SparseVar._find_starts_ends` for
direct offset extraction, and returns a `(bed, SvarLink)` tuple. A
`VariantWriter` protocol that uniformly covers all three would paper over real
semantic differences.

**Revised scope:**

1. Extract a shared `_write_phased_chunked(out_dir, bed, source, ...)` helper
   used by VCF and PGEN writers only.
2. Leave SVAR's writer untouched.
3. If a small `VariantWriter` protocol falls out naturally during dedup (with
   SVAR as one of two implementations or as an opt-out), introduce it — but
   don't force the abstraction. Decide during implementation.
4. **Naming:** if a protocol is introduced, call it `VariantWriter`, not
   `VariantSource` — reserve "Source" terminology for the future variant-major
   read-side protocol.

**Touches:** `_dataset/_write.py`.
**Risk:** low. Behavior-preserving dedup.
**Win:** ~50–80 lines of duplication between VCF and PGEN writers removed.
Smaller architectural drama than the original spec implied.

### PR3 — Collapse `ArrayDataset` / `RaggedDataset` (DROPPED)

**Status:** Dropped after brainstorming + a concrete pyrefly probe.

**The misread:** the original spec framed `ArrayDataset` / `RaggedDataset` as
~150 lines of `super()`-only overload boilerplate. They aren't. The overloads
encode an algebraic data type — GenVarLoader does not support mixed Array and
Ragged return shapes, so a Dataset is *either* in the Array universe
(`NDArray[bytes_]`, `AnnotatedHaps`, `NDArray[float32]`, `RaggedIntervals`) *or*
in the Ragged universe (`RaggedSeqs`, `RaggedAnnotatedHaps`, `Ragged[float32]`,
`RaggedIntervals`). The `with_seqs` / `with_tracks` / `__getitem__` overloads
encode the per-universe associated-type mapping, and the cross-class `with_len`
overloads encode the state transition.

**Patterns considered:**

1. Pure type aliases (`ArrayDataset = Dataset`) — breaks `isinstance(ds, ArrayDataset)` in `_variants/_sitesonly.py`.
2. TYPE_CHECKING split — keeps narrowing, but the overloads still have to be written somewhere; LOC delta is essentially zero.
3. `.pyi` stub file — same line count, plus a second source of truth to keep in sync.
4. Decompose along typestate boundaries (separate shared core from leaf Array/Ragged classes) — cleaner architecture, but doesn't reduce overload count.
5. Higher-kinded types via `dry-python/returns` (`Kind1[T, X]`) — provides container polymorphism, not associated-type projection. Doesn't solve our problem.
6. Phantom types via `antonagestam/phantom-types` — see probe below.
7. Codegen — small declarative source generates the overloads. Adds build complexity.

**Phantom-types probe (concrete pyrefly check):**

A minimal synthetic Dataset with two `Phantom` subclasses (`ArrayDataset` /
`RaggedDataset`) using predicates on `output_length` was checked under pyrefly:

- ✅ Runtime `isinstance(ds, ArrayDataset)` works via Phantom's predicate-based `__instancecheck__`.
- ✅ Overloads on the phantom subclass narrow correctly (`ArrayDataset.with_seqs("haplotypes") → ArrayDataset[NDArrayBytes, ...]`).
- ❌ Moving overloads onto the base with self-typed dispatch returns `Unknown` for the SEQ/TRK params — the base's TypeVars can't reach into the subclass's universe-specific TypeVars.
- ❌ Dropping `return super().method(...)` impl bodies fails pyrefly: "Overloaded function must have an implementation."

**Honest LOC accounting if PR3 went ahead with phantom-types:** about +10 lines of phantom metaclass machinery and consolidation of constructor sites (`ArrayDataset(...)` / `RaggedDataset(...)` → `Dataset(...)`); the ~150 lines of overload stubs stay; the ~30 lines of `super()` bodies stay (required by pyrefly). Net deletion: roughly zero.

**Decision:** the ADT is the design intent — preserve it. The overload stubs
are the cost of expressing this ADT in Python's type system (which lacks
associated types). Documenting the finding here so future maintainers don't
re-litigate the same path.

**Future variant-major class:** if/when that class lands, it may benefit from a
shared `_DatasetCore` mixin housing the state-independent methods (settings,
subset_to, etc.), but that's a forward-compat concern, not deletion-driven.

### PR4 — `OpenRequest` + decompose `Dataset.open`

Extract an `OpenRequest` value object that holds the parsed/validated arguments
to `Dataset.open`, with a `.resolve() → Dataset` method. `Dataset.open` becomes
a thin classmethod that builds and resolves the request. This is possible *after*
PR1 because settings are already a value object.

**Touches:** `_dataset/_impl.py`, new `_dataset/_open.py`.
**Risk:** low–medium. Splits a 206-line method; needs careful parity testing.
**Win:** `open()` shrinks to ~30 lines; resolution stages testable in isolation.

### PR5 — `ReconstructionRequest` + restructure `Haps._get_haplotypes` (revised after audit)

**Audit findings (2026-05-23, before implementation):**

- ✅ Verified: `Haps._get_haplotypes` is 3 signatures (2 `@overload` stubs + impl) in ~167 lines, with two distinct internal paths (no-splice ~50 lines, splice-plan ~80 lines).
- ❌ Disproven: `Tracks.write_transformed_track` is 166 lines, but line 1189 unconditionally raises `NotImplementedError` — the 145 lines below are unreachable dead code. There is **zero insertion-fill branching** in it. The actual insertion-fill dispatch lives in `_dataset/_tracks.py:_apply_insertion_fill` (a small numba helper, ~30 lines, not 166). Moved to a separate small PR (see PR5c below).
- ❌ Already done: the "haplotype-reconstruction kernel must be callable independent of region-major iteration" is already true. `_dataset/_genotypes.py:reconstruct_haplotypes_from_sparse` is a numba-jit pure function taking raw arrays (variants, samples, ref window, optional keep mask, optional annotation buffers). A future variant-major class can call it directly. No further kernel extraction needed.
- ⚠️ Misframed: a `ReconstructionRequest` for `_get_haplotypes` alone (one caller chain) buys little. The *real* opportunity is to extract `_prepare_request()` out of `get_haps_and_shifts` (the ~60 lines that compute `geno_offset_idx`, `keep`, `keep_offsets`, `shifts`, `out_offsets`, `diffs`, `hap_lengths`) into a frozen dataclass `ReconstructionRequest` consumed by `_get_haplotypes` and `_get_variants`. That gives a clean prep → dispatch seam and a stable shape for the future variant-major class to construct directly.

**Revised scope (PR5):**

1. Introduce `ReconstructionRequest` (frozen dataclass with `slots=True`) holding the per-batch prep state.
2. Extract `Haps._prepare_request()` returning a `ReconstructionRequest`.
3. Replace `Haps._get_haplotypes` (2 overloads + 1 impl) with two named methods: `_reconstruct_haplotypes(req)` and `_reconstruct_annotated_haplotypes(req)`. Drop the overload triplet entirely.
4. Extract `_permute_request_for_splice(req)` as a shared helper for the splice-plan branch of both methods.
5. Rewire `get_haps_and_shifts` to: `req = _prepare_request(...); dispatch on self.kind`. Preserve the 7-tuple return shape so `HapsTracks.__call__` consumers stay unchanged.

**Out of scope (deferred to PR5c):**

- `Tracks.write_transformed_track` dead-code deletion (145 lines under unconditional `NotImplementedError`).

**Touches:** `_dataset/_reconstruct.py`.
**Risk:** low. The kernel calls are unchanged (same args, same library call); the change is purely a reorganization of the Python wrapper around the numba kernel. Numerical parity is verified by the existing test suite (488 pytest + cargo).
**Win:** overload triplet eliminated; `get_haps_and_shifts` is now a two-stage operation (prep → kernel); a future variant-major class can construct a `ReconstructionRequest` and call the kernel-facing methods directly without going through region-major preparation.

### PR5c — Delete dead code in `Tracks.write_transformed_track`

`Tracks.write_transformed_track` unconditionally raises `NotImplementedError` at line 1189 of `_dataset/_reconstruct.py`. The 145 lines of body below are unreachable. Replace the method body with `raise NotImplementedError(...)` and a one-line comment about why it's stubbed.

**Touches:** `_dataset/_reconstruct.py`.
**Risk:** None. Was unreachable.
**Win:** ~145 lines deleted; the file gets clearer.

### PR6 — File splits + `QueryView` composition (revised after audit)

**Audit findings (2026-05-23, before implementation):**

- ✅ Reconstructor classes in `_reconstruct.py` ARE cleanly splittable: each is
  self-contained with well-defined imports.
- ⚠️ Existing `_dataset/_tracks.py` is **all numba kernels** (~379 lines), not a
  home for the `Tracks` class (which lives in `_reconstruct.py`). The original
  spec wording elided that distinction. Chosen resolution: merge class + kernels
  into one file (semantically coherent — class is the public face of kernels).
- ❌ The `_dataset/_query.py` extraction is **not** a mechanical move. The
  query cluster (`__getitem__`, `_getitem_(un)spliced`, `_build_splice_plan`,
  `_rc`, `_pad`) touches ~9 fields of Dataset state. Spec called it "mostly
  mechanical, low risk" — wrong. Chosen resolution: composition via a typed
  `QueryView` frozen dataclass (matches the PR4 `OpenRequest` / PR5
  `ReconstructionRequest` idiom). Each query function is then a pure function
  of the view; testable in isolation; type-safe contract.
- ⚠️ Cycle hazard: `_ref.py` / `_haps.py` / `_tracks.py` all need
  `Reconstructor` Protocol, but the dispatcher (`_reconstruct.py`) also needs
  the leaves for `RefTracks`/`HapsTracks`/`_build_reconstructor`. Chosen
  resolution: extract `Reconstructor` Protocol into a tiny `_protocol.py`
  (~34 lines). Clean DAG, no cycles.

**Revised scope (as shipped):**

1. New `_dataset/_protocol.py` — `Reconstructor` Protocol (with `splice_plan`
   declared in the contract; the original Protocol omitted it even though all
   implementers used it).
2. New `_dataset/_ref.py` — `Ref`.
3. New `_dataset/_haps.py` — `Haps`, `_Variants`, `ReconstructionRequest`.
4. `_dataset/_tracks.py` merged — existing numba kernels + new `Tracks` +
   `TrackType` class block.
5. New `_dataset/_query.py` — `QueryView` dataclass + free functions
   (`getitem`, `_getitem_unspliced`, `_getitem_spliced`,
   `build_recon_splice_plan`, `reverse_complement_ragged` (was `Dataset._rc`),
   `pad` (was `Dataset._pad`), `_regroup`).
6. `_dataset/_reconstruct.py` slimmed — `RefTracks`, `HapsTracks`,
   `_build_reconstructor` factory, plus re-exports of all moved symbols so
   downstream imports (`_impl.py`, `_open.py`, `_dummy.py`) keep working.
7. `Dataset.__getitem__` becomes a ~15-line facade: build a `QueryView`, call
   `_query.getitem`. The `inner_ds = self.with_len("ragged")` trick from
   `_getitem_spliced` is replaced by a direct `output_length="ragged"`
   argument to the recon call — no inner Dataset construction needed.
8. `_regroup` moves to `_query.py` (only the query path uses it).
   `_annot_to_intervals` stays in `_impl.py` (used by `write_annot_tracks`,
   not the query path).

**Touches:** `_dataset/_reconstruct.py`, `_dataset/_impl.py`,
`_dataset/_tracks.py`; new `_dataset/{_protocol,_ref,_haps,_query}.py`.
**Risk:** Low for the splits (mechanical with re-exports preserving import
paths). Low for the QueryView extraction (pure composition; no MRO concerns;
the same kernel calls happen with the same arguments).
**Win:** `_reconstruct.py` 1539 → 310 (−1229); `_impl.py` 2015 → 1712 (−303).
Each leaf module is small and focused. The query path is testable as free
functions taking `QueryView`.

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

- ✅ PR0 landed first so every subsequent refactor is type-checked under pyrefly
  strict, surfacing regressions during the riskier moves.
- ✅ PR1 (reconstructor factory + view-state) collapsed the construction logic
  that PRs 2, 4, 5 rely on; with the factory in place, every `with_*` method
  now uses it.
- ✅ **Out-of-band: attrs → stdlib dataclasses migration** (PR #183). Reduced
  pyrefly false-positive surface area on `evolve()` chains and `field(...)`
  generic inference. Mechanical, behavior-preserving. Not numbered in the
  campaign because it's a tooling/quality concern, not architectural.
- ✅ **PR2 (shipped #185):** small, low-risk warm-up — VCF + PGEN writer dedup.
  Refined scope (SVAR stays as-is).
- ❌ **PR3 (dropped):** the "~150 lines of overload boilerplate" reading was
  wrong; the overloads encode the Array/Ragged ADT. See the PR3 section above.
- ✅ **PR4 (shipped #187):** `OpenRequest` + decompose `Dataset.open`.
  `_impl.py`: 2156 → 2015. New `_dataset/_open.py` with 10 small stage methods.
- ✅ **PR5 (shipped #188):** `ReconstructionRequest` + restructure
  `Haps._get_haplotypes`. Audit (before implementation) revealed two of the
  original spec's premises were wrong (kernel already extracted; insertion-fill
  branching not in `write_transformed_track`); spec was revised inline.
  Parity-risk turned out to be low (kernel call args unchanged); the existing
  test suite is the parity check.
- ✅ **PR5c (shipped #189):** delete the 145 dead lines under
  `Tracks.write_transformed_track`'s unconditional `NotImplementedError`;
  capture the feature intent in `docs/superpowers/roadmap.md`.
- ✅ **PR6 (this PR):** reconstructor file splits + `QueryView` composition.
  Audit before implementation surfaced two design wrinkles (existing
  `_tracks.py` name collision; the query extraction is not mechanical) that
  the spec underspecified; both resolved inline. `_reconstruct.py` now
  ~310 lines (was 1539); `_impl.py` 2015 → 1712.
- **PR6** is mechanical and only worth doing once the boundaries from PRs 1, 5
  are real.
- **PR7** is a finisher.

## Out-of-band tooling work

Work that doesn't fit the numbered campaign but lands alongside it:

| PR | Description | Status |
|---|---|---|
| #183 | Migrate `attrs` → stdlib `dataclasses(slots=True)`; drop the `attrs` dep. Motivated by pyrefly's weaker support for attrs. No NamedTuple/msgspec yet — no hot-loop construction sites identified. | Merged 2026-05-23 |

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
