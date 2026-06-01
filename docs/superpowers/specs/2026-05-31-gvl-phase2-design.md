# GenVarLoader test-migration Phase 2 — property-based coverage + fixture retirement (design)

**Date:** 2026-05-31 · **Branch:** `worktree-test-vcfixture` (worktree at
`.claude/worktrees/test-vcfixture`, based on `main`).

**Companion docs:**
- Phase 1 + overall migration design: `docs/superpowers/specs/2026-05-30-vcfixture-test-migration-design.md`
- Phase 1 handoff (as-built, gotchas): `docs/superpowers/plans/2026-05-30-vcfixture-HANDOFF.md`
- Upstream vcfixture (released **0.4.0**): reference-consistent strategy
  (`references()`, `documents(reference=, violations=, label_overrides=)`,
  `reference_and_documents()`), `ReferenceBuilder`/`ReferenceSpec`/`RepeatFeature`,
  per-variant `labels`/`GroundTruth.labels`.

---

## 1. Background

Phase 1 (done) replaced the 3 GB hg38 download and the hand-authored
`source.vcf` with a committed synthetic reference + a programmatic vcfixture
re-encode. It kept the **static** ground-truth pipeline: a single fixed
`source` VCF is run through `bcftools norm` → `bcftools consensus` (per
sample/hap) → committed `tests/data/consensus/*.fa`, and `gvl.write`s three
committed datasets `tests/data/phased_dataset.{vcf,pgen,svar}.gvl/`. The coupled
integration tests (notably `test_ds_haps.py`) compare gvl haplotypes against the
committed consensus FASTAs.

Phase 2 turns this into **property-based** coverage and retires the committed
binary fixtures.

### The dependency that unblocked this
`vcfixture` 0.4.0 (released to PyPI; gvl dev dep bumped to `>=0.4.0`) adds the
reference-consistent generation this plan consumes. Its `documents()` can emit
**deliberately non-canonical but reference-consistent** VCFs, opting into
violation classes that auto-tag provenance `labels`:

| `violations` class | emitted `labels` |
|---|---|
| `multiallelic` | `multiallelic` |
| `non_atomic` | `non_atomic` |
| `non_left_aligned` | `off_anchor` + `tandem_repeat` |

`reference_and_documents()` returns `(ReferenceSpec, VcfDocument, GroundTruth)`.

### The normalization tension (central design fact)
`gvl.write` documents that input must be **left-aligned, bi-allelic, and
atomized** (`_dataset/_write.py:73`). What it actually *enforces* today:

| Property | Enforced by gvl? | Evidence |
|---|---|---|
| **Bi-allelic** | **Yes** — raises `ValueError` | `_write.py:389-392` (VCF), `_write.py:455-458` (PGEN) |
| **Atomized** | **No** — silently corrupts | hardcoded `+1` REF/ALT overlap assumption in `_genotypes.py:69,313`, `_tracks.py:297` |
| **Left-aligned** | **No** — silently assumed | no checks in write / genotype paths |

`GroundTruth` describes the document **as authored** (pre-normalization); it does
not reconstruct haplotype *sequence*. So the haplotype oracle stays on
`bcftools consensus`, and `GroundTruth` is used only for genotype/AF assertions
where representation is preserved.

---

## 2. Goal & scope

Full remaining Phase 2, in one branch/PR:

1. Add a Hypothesis **property-test module** asserting gvl haplotypes ==
   `bcftools consensus` and gvl genotypes/AF == vcfixture `GroundTruth`, across
   VCF/PGEN/SVAR sources, plus a clean-rejection track for non-canonical input.
2. **Delete** committed `tests/data/consensus/` and
   `tests/data/phased_dataset.{vcf,pgen,svar}.gvl/`.
3. **Migrate** the ~6 coupled integration tests onto a session-scoped generated
   fixture; **standardize** on synthetic contigs (`chr1`/`chr2`) + samples
   `s0..s2`.

**Out of scope:** the actual gvl hardening fixes for atomization/left-alignment
(only the bugs are *filed* + `xfail`ed here); the 1kg slow tier; the bigwig/track
re-alignment oracle; `issue_153` regression fixtures.

---

## 3. Architecture / components

### 3.1 Shared case builder
A single builder is the source of truth for "turn a drawn `(spec, doc)` into
on-disk gvl inputs + oracle". Location: extend `tests/data/_synthetic.py` (or a
new `tests/_builders/case.py` — implementer's choice during planning, following
existing `tests/_builders/` conventions).

```
build_case(spec: ReferenceSpec,
           doc: VcfDocument,
           workdir: Path,
           *,
           sources: tuple[str, ...] = ("vcf", "pgen", "svar"),
           normalize: bool = True) -> Case
```

Steps:
1. `spec.write(workdir / "ref.fa.bgz")` — bgzipped + faidx reference.
2. `doc.render()` → raw VCF text → bgzip + index.
3. If `normalize`: `bcftools norm -f ref` (left-align) then
   `bcftools norm -a --atom-overlaps . -m -` (atomize + split multiallelic) →
   canonical VCF.
4. **Haplotype oracle:** per sample/hap, `samtools faidx` the region then
   `bcftools consensus -H {1,2} -s {sample}` → oracle FASTA. (Same shape as the
   current `generate_ground_truth.py` consensus loop.)
5. **PGEN:** `plink2 --vcf … --make-pgen --vcf-half-call r`.
6. **SVAR:** `genoray.SparseVar.from_vcf(...)` + `.cache_afs()`.
7. **BED:** derived deterministically from variant positions (group within
   `SEQ_LEN//2`, regions `SEQ_LEN` apart) — replacing nothing new; reuse the
   Phase 1 logic but keyed off `doc` positions rather than a re-read VCF.
8. `gvl.write(path, bed, variants=reader, max_jitter=2)` per requested source.

Returns a `Case` dataclass exposing: `ref_path`, per-source `gvl_path`,
`consensus_dir`, `bed`, the `GroundTruth`, sample names, and region table.

The slimmed `tests/data/generate_ground_truth.py` becomes a thin wrapper that
calls `build_case` with the fixed session document (so the `gen` pixi task and
the session fixture share one code path).

### 3.2 Property-test module
`tests/integration/dataset/test_haps_property.py`. Uses `vcfixture.strategies`.
Per-example isolation via a `tempfile.TemporaryDirectory()` created **inside the
test body** (not a function-scoped fixture — avoids Hypothesis's
function-scoped-fixture health check).

### 3.3 Session fixtures (conftest)
`synthetic_case` (session-scoped): builds one fixed-seed (seed 0), standardized
case (`chr1`/`chr2`, `s0..s2`) in `tmp_path_factory` once per session via
`build_case`. The existing `phased_{vcf,pgen,svar}_gvl` and `consensus_dir`
fixtures are **redefined to yield paths into `synthetic_case`** instead of
`tests/data/`, so coupled tests change only their data source, not their shape.

---

## 4. Data flow per example (the two tracks)

### Track 1a — haplotype correctness (norm-then-compare)
Sources ∈ {vcf, pgen, svar} (pytest param).
```
spec, doc, truth = draw(reference_and_documents(
    violations={"multiallelic", "non_atomic", "non_left_aligned"}))
case = build_case(spec, doc, tmp, sources=(src,), normalize=True)
ds   = gvl.Dataset.open(case.gvl_path[src], case.ref_path, rc_neg=False)
        .with_len("ragged").with_seqs("haplotypes").with_tracks(False)
for region, sample, hap:
    assert cast_seqs(ds[region, sample][hap]) == cast_seqs(consensus_oracle)
```
The violations exist precisely to make `bcftools norm` do real work (indels in
tandem repeats, atomized MNPs, split multiallelics), stressing gvl's
reconstruction on hard-but-valid normalized variants.

### Track 1b — genotypes / AF (canonical only)
Sources ∈ {vcf, pgen, svar}.
```
spec, doc, truth = draw(reference_and_documents(violations=frozenset()))
# canonical ⇒ norm is representation-preserving ⇒ GroundTruth still matches
case = build_case(spec, doc, tmp, normalize=True)
assert gvl_loaded_genotypes(ds) == truth.genotypes          # per region/sample
assert gvl_allele_frequencies(ds) ≈ af_from(truth.genotypes)
```
`af_from(...)` = ALT-allele count / total called alleles, computed from
`truth.genotypes`; compared to gvl's AF (e.g. `SparseVar` cached AFs / dataset
AF), with exact or tight-tolerance equality.

### Track 2 — clean rejection (raw, no norm), VCF source only
Per violation class, one parametrized test:
```
spec = draw(references())
doc  = draw(documents(reference=spec, violations={CLASS}))
case = build_case(spec, doc, tmp, sources=("vcf",), normalize=False)
multiallelic     → with pytest.raises(ValueError): gvl.write(... raw vcf ...)
non_atomic       → xfail(strict=True, reason="gvl #<atomize-validation>")
non_left_aligned → xfail(strict=True, reason="gvl #<leftalign-validation>")
```
PGEN/SVAR derive from a normalized VCF, so they cannot carry raw violations —
Track 2 is VCF-only by construction.

---

## 5. Oracles & assertions

- **Haplotype oracle:** `bcftools consensus` per sample/hap. gvl's deepest
  oracle; `GroundTruth` does not reconstruct sequence. Compared via
  `seqpro.cast_seqs` exact equality (matching `test_ds_haps.py` today).
- **Variant oracle:** vcfixture `GroundTruth.genotypes` + AF derived from it.
  Used **only on canonical examples**, where `bcftools norm` does not change the
  variant representation.

---

## 6. Hypothesis settings & cost

Each example shells out to bcftools/plink2/samtools, so the module configures:
- `deadline=None`,
  `suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]`.
- Per-example `tempfile.TemporaryDirectory()` inside the test body.
- Small draws: contigs ≤ 2000 bp, `max_samples=2`, `max_records=4`,
  `max_repeats=3`.
- Modest counts: `max_examples` ≈ 25 (Track 1a) / 15 (Track 1b) / 10 (per
  Track-2 class).

Not marked `@pytest.mark.slow` — this is core correctness coverage and must run
in the default suite. Counts are tuned to keep wall-clock to a few seconds; a
CI/env knob to scale `max_examples` may be added if needed (deferred — YAGNI
until the suite time is measured).

---

## 7. Error handling, xfail & hardening bugs

This work **surfaces** two latent gvl gaps as a deliverable:

1. File a gvl issue: `gvl.write` does not validate **atomization** (non-atomic
   input silently corrupts haplotype length math via the `+1` overlap
   assumption).
2. File a gvl issue: `gvl.write` does not validate **left-alignment**.

The two Track-2 `xfail(strict=True)` markers reference these issue numbers, so
they automatically flip to failures (alerting us) when hardening lands. The
`multiallelic` rejection is already enforced and is asserted green.

This matches the agreed Track-2 contract: gvl *should* cleanly reject
non-canonical input; gaps where it does not are hardening bugs, tracked via
`xfail`, fixed separately.

---

## 8. Deletions, migration & standardization

**Delete:**
- `tests/data/consensus/`
- `tests/data/phased_dataset.{vcf,pgen,svar}.gvl/`
- the monolithic generation body of `tests/data/generate_ground_truth.py`
  (reduced to a thin `build_case` wrapper) and any now-dead branches.

Migration splits into two disjoint mechanisms:

**(a) Transparent — fixture redefinition (no test edits).** Because
`phased_{vcf,pgen,svar}_gvl` and `consensus_dir` are redefined to yield from
`synthetic_case` (§3.3), every test that merely *injects* those fixtures and
iterates over them follows automatically. A grep audit found ~12 such consumers
(`test_ds_haps`, `test_ds_haps_modes`, `test_determinism`, `test_subset`,
`test_edge_shapes`, `test_cross_mode_equivalence`, `test_jitter`, `test_dataset`,
`test_issue_191_var_fields`, `test_annot_tracks`, `test_torch`,
`test_double_buffered_loader`) — none of these hardcode contig/sample literals,
so they need **no content changes** beyond the fixture redefinition itself.
`test_ds_haps` keeps comparing gvl haps to `consensus_dir` FASTAs, now generated.

**(b) Content migration (real edits) — tests asserting specific
contig/sample/position values** that change when the dataset is regenerated on
standardized `chr1`/`chr2` + `s0..s2`. Grep-identified candidates (literals
`chr19`/`chr20`/`NA0000`/`1010696`):
- `tests/dataset/test_with_methods.py`
- `tests/integration/dataset/test_write_edge_cases.py`
- `tests/integration/dataset/test_write_tracks_e2e.py`
- `tests/data/test_synthetic_inputs.py` and `tests/data/_synthetic.py` (the
  Phase-1 builder/test pair)
- `tests/unit/test_utils.py` (verify — its literal may be an unrelated string,
  not a dataset assertion)

Note the interaction with the **two-references** decision (§3.3, §8 "Keep"):
the committed `ref_fasta` and its Phase-1 builder (`_synthetic.py`'s
`write_synthetic_reference`/`build_source_vcf`) may legitimately retain
`chr19`/`chr20` for FASTA-only tests; only assertions coupled to the
**regenerated dataset** migrate to `chr1`/`chr2`/`s0..s2`. The exact per-file
edit set is resolved by the planning-time audit — a literal's contig may refer
to the unchanged committed reference rather than the dataset.

**Keep unchanged:**
- The committed synthetic `ref_fasta` (`tests/data/fasta/synthetic.fa.bgz`) for
  pure-FASTA/reference unit tests that involve no variants. The
  `synthetic_case` builds its own internally-consistent ref for dataset tests;
  having two references is intentional — they serve different test classes, and
  migrating the FASTA-only tests' contigs is low-value churn deferred out.
- `issue_153.{bed,vcf}`, bigwig fixtures, 1kg slow-tier — out of scope.

---

## 9. Acceptance gate

- `pixi run -e dev gen` runs offline (thin `build_case` wrapper), and
  `pixi run -e dev pytest tests -m "not slow"` is **green**, with
  `tests/data/consensus/` and `tests/data/phased_dataset.*.gvl/` **absent** from
  the tree.
- The new property module passes (Track 1a/1b green across VCF/PGEN/SVAR; Track 2
  `multiallelic` green, `non_atomic`/`non_left_aligned` `xfail`).
- Two gvl hardening issues filed and referenced by the `xfail`s.

---

## 10. File-by-file change map

| File | Change |
|---|---|
| `tests/data/_synthetic.py` (or `tests/_builders/case.py`) | add `build_case` + `Case` |
| `tests/data/generate_ground_truth.py` | reduce to thin `build_case` wrapper; drop dead branches |
| `tests/integration/dataset/test_haps_property.py` | **new** property module (Tracks 1a/1b/2) |
| `tests/conftest.py` | add `synthetic_case`; redefine `phased_*_gvl`/`consensus_dir` to yield from it (covers ~12 consumers transparently) |
| content-migration tests (§8b) | edit assertions hardcoding `chr19`/`chr20`/`NA0000x`/`1010696` → `chr1`/`chr2`/`s0..s2` (subset; per planning audit) |
| `tests/data/consensus/`, `tests/data/phased_dataset.*.gvl/` | **delete** |
| (gvl issue tracker) | file 2 hardening issues (atomize, left-align) |

---

## 11. Open items for planning (not blockers)

- Confirm `GroundTruth` exposes genotypes in a shape directly comparable to
  gvl's loaded genotypes, and how AF is best derived/compared (exact vs tight
  tolerance). Probe during the first plan task.
- Confirm the exact coupled-test set via the grep audit (§8).
- Decide `build_case` home (`_synthetic.py` vs `tests/_builders/case.py`) per
  existing conventions.
