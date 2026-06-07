# Reject Symbolic & Breakend Variants in `gvl.write` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gvl.write` reject any VCF/PGEN/SVAR input containing symbolic (`<DEL>`, `<INS>`, …) or breakend (`G[chr2:321[`, …) ALT alleles, which cannot be reconstructed into haplotypes.

**Architecture:** Bump genoray to 2.9.1 (adds `genoray.exprs.is_symbolic` / `is_breakend`). Add one consolidated validator `_reject_unsupported_variants(index, source)` that checks multi-allelic + symbolic + breakend over the full post-filter variant index, and call it from all three write paths (`_write_from_vcf`, `_write_from_pgen`, `_write_from_svar`). This replaces the two existing standalone multi-allelic guards and adds the missing guard to the SVAR path. No on-the-fly filtering.

**Tech Stack:** Python, polars, genoray, pixi, pytest, hypothesis, vcfixture.

**Spec:** `docs/superpowers/specs/2026-06-07-reject-symbolic-breakend-variants-design.md`

**Conventions:**
- All commands run through pixi: `pixi run -e dev <cmd>`.
- Prefix shell commands with `rtk` per repo `CLAUDE.md` (e.g. `rtk git commit`).
- The pre-commit `pyrefly` hook currently fails on PRE-EXISTING unrelated `seqpro.rag` import errors. For commits that don't touch those files this is noise; use `git commit --no-verify` ONLY when the failure is confirmed to be those pre-existing `seqpro.rag` errors and nothing you introduced. Otherwise fix what you broke.

---

### Task 1: Bump genoray dependency to 2.9.1 and verify the new exprs

**Files:**
- Modify: `pyproject.toml:14` (`"genoray>=2.7.3,<3"` → `"genoray>=2.9.1,<3"`)
- Modify: `pixi.toml:92` (`genoray = "==2.7.3"` → `genoray = "==2.9.1"`)
- Modify: `pixi.lock` (regenerated)

- [ ] **Step 1: Edit `pyproject.toml`**

Change line 14 from:
```toml
    "genoray>=2.7.3,<3",
```
to:
```toml
    "genoray>=2.9.1,<3",
```

- [ ] **Step 2: Edit `pixi.toml`**

Change line 92 (under `[feature.py310.pypi-dependencies]`) from:
```toml
genoray = "==2.7.3"
```
to:
```toml
genoray = "==2.9.1"
```
(This is the only genoray pin; py311/312/313 features inherit it.)

- [ ] **Step 3: Relock and install**

Run: `rtk pixi update genoray`
Expected: `pixi.lock` updates genoray to 2.9.1 for the dev env (and any solve groups) with no solver errors.

If `pixi update genoray` does not pick up the new constraint, run `pixi install` to force a resolve against the edited `pixi.toml`.

- [ ] **Step 4: Verify the new exprs exist and operate on an index frame**

Run:
```bash
pixi run -e dev python -c "
import genoray, polars as pl
print('genoray', genoray.__version__)
from genoray import exprs
idx = pl.DataFrame(
    {'CHROM': ['c','c','c'], 'POS': [1,2,3], 'REF': ['A','G','C'],
     'ALT': [['T'], ['<DEL>'], ['C[c:9[']]},
    schema_overrides={'ALT': pl.List(pl.Utf8)},
)
print(idx.select(
    sym=exprs.is_symbolic.cast(pl.Int64).sum(),
    bnd=exprs.is_breakend.cast(pl.Int64).sum(),
).row(0))
"
```
Expected: prints `genoray 2.9.1` then `(1, 1)` — one symbolic, one breakend.

If this errors with a missing `ILEN`/`CHROM`/other column, note which columns `is_symbolic`/`is_breakend` actually require; the validator in Task 2 and its unit-test fixtures must include those columns. (Per the genoray-api skill they need only `ALT`, but confirm here against the installed 2.9.1.)

- [ ] **Step 5: Commit**

```bash
rtk git add pyproject.toml pixi.toml pixi.lock
rtk git commit -m "build: bump genoray to 2.9.1 for is_symbolic/is_breakend exprs"
```

---

### Task 2: Add the consolidated `_reject_unsupported_variants` validator (TDD)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (add import + new function)
- Test: `tests/dataset/test_write_validation.py` (new)

The validator is pure (`pl.DataFrame` → raise or return), so it gets fast, direct unit tests independent of `gvl.write`.

- [ ] **Step 1: Write the failing test file**

Create `tests/dataset/test_write_validation.py`:
```python
import polars as pl
import pytest

from genvarloader._dataset._write import _reject_unsupported_variants


def _index(alts, refs=None):
    """Build a minimal genoray-style index frame from a list of ALT lists."""
    n = len(alts)
    refs = refs if refs is not None else ["A"] * n
    return pl.DataFrame(
        {
            "CHROM": ["chr1"] * n,
            "POS": list(range(1, n + 1)),
            "REF": refs,
            "ALT": alts,
            "ILEN": [[0]] * n,
        },
        schema_overrides={"ALT": pl.List(pl.Utf8), "ILEN": pl.List(pl.Int32)},
    )


def test_clean_index_passes():
    # all bi-allelic SNPs: no raise
    _reject_unsupported_variants(_index([["T"], ["C"], ["G"]]), "VCF")


def test_symbolic_is_rejected():
    idx = _index([["T"], ["<DEL>"]])
    with pytest.raises(ValueError, match="symbolic"):
        _reject_unsupported_variants(idx, "VCF")


def test_breakend_is_rejected():
    idx = _index([["T"], ["C[chr1:600["]])
    with pytest.raises(ValueError, match="breakend"):
        _reject_unsupported_variants(idx, "PGEN")


def test_multiallelic_is_rejected():
    idx = _index([["T", "C"]])
    with pytest.raises(ValueError, match="multi-allelic"):
        _reject_unsupported_variants(idx, "SVAR")


def test_error_reports_source_and_counts():
    idx = _index([["<DEL>"], ["<INS>"], ["C[chr1:600["]])
    with pytest.raises(ValueError) as exc:
        _reject_unsupported_variants(idx, "SVAR")
    msg = str(exc.value)
    assert "SVAR" in msg
    assert "2 symbolic" in msg
    assert "1 breakend" in msg
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_write_validation.py -v`
Expected: FAIL with `ImportError: cannot import name '_reject_unsupported_variants'`.

- [ ] **Step 3: Add the genoray exprs import**

In `python/genvarloader/_dataset/_write.py`, the existing genoray import is line 16:
```python
from genoray import PGEN, VCF, Reader, SparseVar
```
Add directly below it:
```python
from genoray import exprs as _gexprs
```

- [ ] **Step 4: Implement the validator**

Add this function to `python/genvarloader/_dataset/_write.py`, immediately above `def _write_from_vcf(` (around line 398):
```python
def _reject_unsupported_variants(index: pl.DataFrame, source: str) -> None:
    """Raise if the variant index contains alleles gvl cannot reconstruct.

    gvl expands each variant's ALT into literal haplotype sequence, so it
    requires bi-allelic, non-symbolic, non-breakend records. This runs over the
    FULL index (post any user-supplied filter), matching the "valid inputs only"
    contract. ``source`` names the input for the error message (e.g. "VCF").
    """
    n_multi, n_sym, n_bnd = index.select(
        n_multi=(pl.col("ALT").list.len() > 1).cast(pl.Int64).sum(),
        n_symbolic=_gexprs.is_symbolic.cast(pl.Int64).sum(),
        n_breakend=_gexprs.is_breakend.cast(pl.Int64).sum(),
    ).row(0)
    if n_multi or n_sym or n_bnd:
        raise ValueError(
            f"{source} contains unsupported variants: {n_multi} multi-allelic, "
            f"{n_sym} symbolic (e.g. <DEL>/<INS>), {n_bnd} breakend. gvl can only "
            f"reconstruct bi-allelic, non-symbolic, non-breakend variants. Remove "
            f"them upstream (bcftools/plink2 — split multi-allelics, drop SVs), or "
            f"construct the genoray reader with a filter such as "
            f"`filter=genoray.exprs.is_biallelic & ~genoray.exprs.is_symbolic & "
            f"~genoray.exprs.is_breakend`."
        )
```

Note: the message intentionally contains the substrings `multi-allelic`, `symbolic`, and `breakend` that the tests (and the existing `test_multiallelic_raw_is_rejected`) match on.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_write_validation.py -v`
Expected: all 5 PASS.

If `is_symbolic`/`is_breakend` require columns beyond `ALT` (discovered in Task 1 Step 4), add those columns to the `_index` helper in the test and re-run.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_validation.py
rtk git commit -m "feat(write): add consolidated unsupported-variant validator"
```

---

### Task 3: Wire the validator into the VCF write path

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:408-412` (`_write_from_vcf`)

- [ ] **Step 1: Replace the standalone multi-allelic guard**

In `_write_from_vcf`, replace:
```python
    if vcf._index.select((pl.col("ALT").list.len() > 1).any()).item():
        raise ValueError(
            "VCF with filtering applied still contains multi-allelic variants. Please filter or split them."
        )
```
with:
```python
    _reject_unsupported_variants(vcf._index, "VCF")
```
(The preceding `assert vcf._index is not None` stays.)

- [ ] **Step 2: Verify the existing multiallelic rejection test still passes**

Run: `pixi run -e dev pytest tests/integration/dataset/test_haps_property.py::test_multiallelic_raw_is_rejected -v`
Expected: PASS (new message still contains `multi-allelic`, which the test matches).

If the test data has not been generated in this environment, first run `pixi run -e dev gen`.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "feat(write): reject symbolic/breakend variants from VCF inputs"
```

---

### Task 4: Wire the validator into the PGEN write path

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:570-577` (`_write_from_pgen`)

- [ ] **Step 1: Replace the `_sei is None` multi-allelic guard**

In `_write_from_pgen`, replace:
```python
    if pgen._sei is None:
        raise ValueError(
            "PGEN with filtering has multi-allelic variants. Please filter or split them."
        )
    assert pgen._sei is not None
```
with:
```python
    assert pgen._index is not None, (
        "caller must init the PGEN index before _write_from_pgen"
    )
    _reject_unsupported_variants(pgen._index, "PGEN")
    # _sei is genoray's sparse-extraction index; it is only built for
    # bi-allelic data. The validator above already rejects multi-allelics, so a
    # None _sei here signals a distinct genoray-internal failure, not bad input.
    assert pgen._sei is not None, (
        "PGEN sparse-extraction index is None despite passing variant validation"
    )
```

- [ ] **Step 2: Verify the `_sei`/index relationship**

Run:
```bash
pixi run -e dev python -c "
import inspect, genoray
src = inspect.getsource(genoray._pgen._load_index)
print('_sei' in src, 'biallelic' in src.lower() or 'multi' in src.lower())
print(src[:1500])
"
```
Expected: confirms `_load_index` returns `_sei` and that it is `None` specifically when multi-allelic variants are present. If `_sei` can be `None` for a reason UNRELATED to multi-allelics (so that the new `_reject_unsupported_variants` would let such input through), keep a dedicated `if pgen._sei is None: raise ValueError(...)` guard AFTER the validator with a message explaining the genoray-internal condition, instead of the bare `assert`.

- [ ] **Step 3: Run the validator unit tests (regression)**

Run: `pixi run -e dev pytest tests/dataset/test_write_validation.py -v`
Expected: all PASS (no behavior change; sanity check that the import/edit didn't break the module).

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "feat(write): reject symbolic/breakend variants from PGEN inputs"
```

---

### Task 5: Wire the validator into the SVAR write path

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:735-744` (`_write_from_svar`, top of body)

This path currently has NO unsupported-variant guard; this adds one.

- [ ] **Step 1: Add the guard at the top of `_write_from_svar`**

In `_write_from_svar`, immediately after the docstring/signature and before `out_dir = path / "genotypes"`, insert:
```python
    _reject_unsupported_variants(svar.index, "SVAR")
```
So the function begins:
```python
def _write_from_svar(
    path: Path,
    bed: pl.DataFrame,
    svar: SparseVar,
    samples: list[str],
    extend_to_length: bool,
) -> tuple[pl.DataFrame, SvarLink]:
    _reject_unsupported_variants(svar.index, "SVAR")

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Run the validator unit tests (regression)**

Run: `pixi run -e dev pytest tests/dataset/test_write_validation.py -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "feat(write): reject symbolic/breakend variants from SVAR inputs"
```

---

### Task 6: Integration tests via `gvl.write` (hand-crafted VCF + SVAR; vcfixture if supported)

**Files:**
- Test: `tests/integration/dataset/test_haps_property.py` (add deterministic tests near the existing rejection tests, ~line 470)

A deterministic hand-crafted VCF is the primary, robust test. It reuses the module-level helpers already imported in this file (`_bgzip_index`, `_derive_bed` from `case`, and the `_raw_write_vcf` pattern).

- [ ] **Step 1: Add a shared raw-VCF text constant and helper**

Add near the top of `tests/integration/dataset/test_haps_property.py` (after the existing imports/helpers, e.g. just below `_raw_write_vcf`):
```python
# A minimal raw VCF containing one clean SNP, one symbolic <DEL>, and one
# breakend ALT. Used to assert gvl.write rejects symbolic/breakend inputs.
_SYM_BND_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=2000>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">
##ALT=<ID=DEL,Description="Deletion">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1
chr1\t100\t.\tA\tT\t.\t.\t.\tGT\t0|1\t1|0
chr1\t200\t.\tG\t<DEL>\t.\t.\tSVTYPE=DEL;END=300;SVLEN=-100\tGT\t0|1\t0|0
chr1\t400\t.\tC\tC[chr1:600[\t.\t.\tSVTYPE=BND\tGT\t0|1\t0|0
"""


def _write_vcf_text(text: str, tmp) -> tuple[Path, Path]:
    """bgzip+index the given VCF text and derive a BED. Returns (vcf_gz, bed_path)."""
    from case import _bgzip_index, _derive_bed

    tmp = Path(tmp)
    vcf_gz = _bgzip_index(text.encode(), tmp / "raw.vcf.gz")
    bed = _derive_bed(vcf_gz, None)
    bed_path = tmp / "source.bed"
    bed.select(
        "chrom",
        "start",
        "end",
        pl.lit(".").alias("name"),
        pl.lit(".").alias("score"),
        "strand",
    ).write_csv(bed_path, include_header=False, separator="\t")
    return vcf_gz, bed_path
```

- [ ] **Step 2: Write the failing VCF integration test**

Add:
```python
def test_symbolic_breakend_vcf_is_rejected():
    """gvl.write rejects a VCF containing symbolic and breakend ALTs."""
    import genvarloader as gvl
    from genoray import VCF

    with tempfile.TemporaryDirectory() as tmp:
        vcf_gz, bed_path = _write_vcf_text(_SYM_BND_VCF, tmp)
        reader = VCF(vcf_gz)
        if not reader._valid_index():
            reader._write_gvi_index()
        reader._load_index()
        with pytest.raises(ValueError, match="symbolic"):
            gvl.write(
                path=Path(tmp) / "ds.gvl",
                bed=bed_path,
                variants=reader,
                max_jitter=2,
            )
```

- [ ] **Step 3: Run it**

Run: `pixi run -e dev pytest "tests/integration/dataset/test_haps_property.py::test_symbolic_breakend_vcf_is_rejected" -v`
Expected: PASS. The VCF index built from `_SYM_BND_VCF` contains a `<DEL>` and a breakend; `_write_from_vcf` → `_reject_unsupported_variants` raises with a message containing `symbolic`.

If the test instead errors during `reader._write_gvi_index()`/`_load_index()` (genoray failing to index the symbolic/breakend ALT), that is itself acceptable rejection behavior, but adjust the test to wrap index construction in the same `pytest.raises` block and document why.

- [ ] **Step 4: Write the SVAR integration test**

Add:
```python
def test_symbolic_breakend_svar_is_rejected():
    """gvl.write rejects a .svar built (unfiltered) from symbolic/breakend input."""
    import genvarloader as gvl
    from genoray import VCF, SparseVar

    with tempfile.TemporaryDirectory() as tmp:
        vcf_gz, bed_path = _write_vcf_text(_SYM_BND_VCF, tmp)
        svar_path = Path(tmp) / "v.svar"
        # No filter: symbolic/breakend records are carried into the .svar index.
        SparseVar.from_vcf(svar_path, VCF(vcf_gz), max_mem="1g", overwrite=True)
        with pytest.raises(ValueError, match="symbolic"):
            gvl.write(
                path=Path(tmp) / "ds.gvl",
                bed=bed_path,
                variants=SparseVar(svar_path),
                max_jitter=2,
            )
```

- [ ] **Step 5: Run it**

Run: `pixi run -e dev pytest "tests/integration/dataset/test_haps_property.py::test_symbolic_breakend_svar_is_rejected" -v`
Expected: PASS.

If `SparseVar.from_vcf` itself raises on the symbolic/breakend ALTs (refusing to materialize them), then the unfiltered-rejection scenario cannot occur and this test is moot: replace it with a `pytest.skip("SparseVar.from_vcf rejects symbolic/breakend at build time")` and add a one-line comment recording the observed behavior. The Task 2 unit tests already cover the `svar.index` validation logic directly.

- [ ] **Step 6: Probe vcfixture for symbolic/breakend violation support**

Run:
```bash
pixi run -e dev python -c "
import inspect, vcfixture.strategies as st
src = inspect.getsource(st.documents)
for label in ('symbolic', 'breakend'):
    print(label, label in src)
"
```
Expected: prints whether `st.documents(violations=...)` recognizes `symbolic` / `breakend`.

- [ ] **Step 7: If vcfixture supports them, add property tests; otherwise skip**

IF Step 6 shows BOTH labels are supported, extend the existing harness. Change `_ALL_VIOLATIONS` (line ~58) to:
```python
_ALL_VIOLATIONS = frozenset(
    {"multiallelic", "non_atomic", "non_left_aligned", "symbolic", "breakend"}
)
```
and add, mirroring `test_multiallelic_raw_is_rejected`:
```python
@settings(max_examples=10, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=_spec_and_violating_doc("symbolic"))
def test_symbolic_raw_is_rejected(case_inputs):
    """gvl.write rejects raw symbolic-allele input."""
    _spec, doc = case_inputs
    assume(_has_violation_label(doc, "symbolic"))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="symbolic"):
            _raw_write_vcf(doc, tmp)


@settings(max_examples=10, deadline=None, suppress_health_check=_SUPPRESS)
@given(case_inputs=_spec_and_violating_doc("breakend"))
def test_breakend_raw_is_rejected(case_inputs):
    """gvl.write rejects raw breakend-allele input."""
    _spec, doc = case_inputs
    assume(_has_violation_label(doc, "breakend"))
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="breakend"):
            _raw_write_vcf(doc, tmp)
```
IF Step 6 shows either label is NOT supported, do NOT add these property tests and do NOT change `_ALL_VIOLATIONS`; the hand-crafted tests from Steps 2–5 are the coverage. Add a brief comment at the rejection-tests section noting vcfixture lacks symbolic/breakend violation generation.

- [ ] **Step 8: Run the full rejection test group**

Run: `pixi run -e dev pytest "tests/integration/dataset/test_haps_property.py" -k "rejected" -v`
Expected: all selected tests PASS (and `gen` has been run so test data exists).

- [ ] **Step 9: Commit**

```bash
rtk git add tests/integration/dataset/test_haps_property.py
rtk git commit -m "test(write): cover symbolic/breakend rejection for VCF and SVAR"
```

---

### Task 7: Update the genvarloader skill

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

Per `CLAUDE.md`, any change to `gvl.write`'s input contract must update this skill.

- [ ] **Step 1: Locate the preprocessing/requirements and gotchas sections**

Run: `pixi run -e dev rtk grep -n "multi-allelic\|biallelic\|bcftools\|plink2\|atomi\|left-align\|gotcha" skills/genvarloader/SKILL.md`
Expected: line numbers for the variant-preprocessing requirements and the "Common gotchas" section.

- [ ] **Step 2: Add the symbolic/breakend requirement**

In the variant-preprocessing requirements (wherever bi-allelic/atomized/left-aligned are listed), add that inputs must also be **free of symbolic (`<DEL>`, `<INS>`, …) and breakend ALT alleles**, because gvl expands every ALT into literal sequence. State that `gvl.write` raises `ValueError` (with per-class counts) if any are present — for VCF, PGEN, and SVAR alike.

In "Common gotchas", add an entry:
> **Symbolic / breakend variants are rejected, not skipped.** Remove them before `gvl.write` — e.g. `bcftools view -e 'ALT~"<" || ALT~"\["'` or construct the genoray reader with `filter=genoray.exprs.is_biallelic & ~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend`. SVAR inputs must be built from an already-filtered source, since gvl validates but cannot re-filter a materialized `.svar`.

(Match the surrounding markdown style/formatting of SKILL.md when inserting.)

- [ ] **Step 3: Sanity-check the skill renders and references are correct**

Run: `pixi run -e dev rtk grep -n "symbolic\|breakend" skills/genvarloader/SKILL.md`
Expected: the new content appears in both the requirements and gotchas sections.

- [ ] **Step 4: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): document symbolic/breakend variant rejection in gvl.write"
```

---

### Task 8: Full verification

- [ ] **Step 1: Run the variant write/validation tests together**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_write_validation.py "tests/integration/dataset/test_haps_property.py" -k "rejected or validation or symbolic or breakend or multiallelic" -v
```
Expected: all PASS.

- [ ] **Step 2: Run the broader dataset test suite for regressions**

Run: `pixi run -e dev pytest tests/dataset tests/integration/dataset -m "not slow" -q`
Expected: PASS (no regressions in existing write/read behavior).

- [ ] **Step 3: Lint and type-check the changed Python**

Run:
```bash
pixi run -e dev ruff check python/genvarloader/_dataset/_write.py tests/dataset/test_write_validation.py
pixi run -e dev typecheck
```
Expected: ruff clean. `typecheck` (pyrefly) may still report the PRE-EXISTING unrelated `seqpro.rag` import errors; confirm no NEW errors originate from `_write.py` or the new test.

- [ ] **Step 4: Final review of the diff**

Run: `rtk git log --oneline main..HEAD` and `rtk git diff main..HEAD --stat`
Expected: commits for the dep bump, validator, three wiring changes, tests, and skill update; changes confined to `pyproject.toml`, `pixi.toml`, `pixi.lock`, `python/genvarloader/_dataset/_write.py`, `tests/...`, and `skills/genvarloader/SKILL.md`.

---

## Self-Review Notes (author)

- **Spec coverage:** dep bump (Task 1) ✓; consolidated validator incl. multi-allelic + symbolic + breakend (Task 2) ✓; VCF/PGEN/SVAR wiring incl. SVAR's previously-missing guard (Tasks 3–5) ✓; full-index check (validator runs on whole index) ✓; reject-with-counts message ✓; tests prefer vcfixture with hand-crafted fallback (Task 6) ✓; SKILL.md update (Task 7) ✓; "no on-the-fly filtering" honored (validator only reads) ✓.
- **Open verifications flagged in-task (not placeholders):** exact column requirements of `is_symbolic`/`is_breakend` (Task 1 Step 4); whether `_sei is None` is multi-allelic-specific (Task 4 Step 2); whether `SparseVar.from_vcf` materializes symbolic/breakend or rejects at build (Task 6 Step 5); whether vcfixture can generate symbolic/breakend violations (Task 6 Step 6). Each has a concrete probe command and a defined branch for either outcome.
- **Type/name consistency:** `_reject_unsupported_variants(index: pl.DataFrame, source: str)` and the import alias `_gexprs` are used identically across Tasks 2–5.
