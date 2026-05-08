# `get_splice_bed` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `gvl.get_splice_bed(gtf, ...)` returning a BED-compatible DataFrame derived from a GTF, suitable for passing to `gvl.write()` for splicing datasets.

**Architecture:** A single pure function in `python/genvarloader/_dataset/_write.py` that scans the GTF lazily via `seqpro.gtf.scan`, filters/transforms with polars, and returns a sorted `pl.DataFrame`. Exported from the package root.

**Tech Stack:** Python, polars, seqpro (`sp.gtf.scan`, `sp.gtf.attr`, `sp.bed.sort`). Tests use pytest and `tmp_path`.

**Spec:** `docs/superpowers/specs/2026-05-08-get-splice-bed-design.md`.

---

## File Structure

- **Modify** `python/genvarloader/_dataset/_write.py` — add `get_splice_bed` function next to `write` (top-level, alongside the existing public `write`). The file is ~25 KB but `get_splice_bed` is a small self-contained helper that belongs with `write` since users compose the two.
- **Modify** `python/genvarloader/__init__.py` — import and re-export `get_splice_bed`, add to `__all__`.
- **Create** `tests/dataset/test_get_splice_bed.py` — unit tests with synthetic GTF fixture written to `tmp_path`.

No other files change.

---

## Synthetic GTF fixture

The test file uses a shared fixture that writes a small GTF to `tmp_path / "tiny.gtf"`. The fixture content exercises every behavior we test — define it once and reuse.

GTF format (tab-separated, 1-based inclusive start/end): `seqname\tsource\tfeature\tstart\tend\tscore\tstrand\tframe\tattribute`.

Fixture content (use exactly this text in tests; ordering is intentionally scrambled to test sort):

```
1	test	exon	100	200	.	+	.	gene_id "G1"; gene_name "GENEA"; transcript_id "T1"; exon_number "1"; transcript_support_level "1";
1	test	CDS	300	308	.	+	0	gene_id "G1"; gene_name "GENEA"; transcript_id "T1"; exon_number "2"; transcript_support_level "1";
1	test	CDS	100	108	.	+	0	gene_id "G1"; gene_name "GENEA"; transcript_id "T1"; exon_number "1"; transcript_support_level "1";
2	test	CDS	500	506	.	-	0	gene_id "G2"; gene_name "GENEB"; transcript_id "T2"; exon_number "1"; transcript_support_level "1";
2	test	CDS	600	606	.	-	0	gene_id "G2"; gene_name "GENEB"; transcript_id "T2"; exon_number "2"; transcript_support_level "1";
3	test	CDS	700	705	.	+	0	gene_id "G3"; gene_name "GENEC"; transcript_id "T3"; exon_number "1"; transcript_support_level "2";
4	test	CDS	800	804	.	+	0	gene_id "G4"; transcript_id "T4"; exon_number "1"; transcript_support_level "1";
1	test	five_prime_utr	50	99	.	+	.	gene_id "G1"; gene_name "GENEA"; transcript_id "T1";
```

Per-row notes (do not put in fixture, just for plan readers):

- T1 on chr 1: two CDS rows lengths 9+9 = 18 (multiple of 3), TSL=1, gene_name=GENEA. Two rows are out of order vs. genomic position to verify sorting.
- T2 on chr 2: two CDS rows lengths 7+7 = 14 (NOT multiple of 3), TSL=1.
- T3 on chr 3: one CDS row length 6 (multiple of 3), TSL=**2** (filtered out by default TSL=1).
- T4 on chr 4: one CDS row length 5 (NOT multiple of 3), TSL=1, **no gene_name attribute** (verifies nulls preserved when filter not applied; also filtered out by default require_multiple_of_3).
- One `exon` row and one `five_prime_utr` row to verify non-CDS rows are dropped.

Helper to write the fixture:

```python
GTF_TEXT = "\t".join  # placeholder marker; the real fixture uses the literal string above
```

In the test file, write the string verbatim. Use tabs (not spaces) between fields.

---

### Task 1: Test scaffolding and failing test suite

**Files:**
- Create: `tests/dataset/test_get_splice_bed.py`

- [ ] **Step 1: Create the test file with the GTF fixture and full test suite**

```python
from pathlib import Path

import polars as pl
import pytest

import genvarloader as gvl


GTF_TEXT = (
    "1\ttest\texon\t100\t200\t.\t+\t.\tgene_id \"G1\"; gene_name \"GENEA\"; transcript_id \"T1\"; exon_number \"1\"; transcript_support_level \"1\";\n"
    "1\ttest\tCDS\t300\t308\t.\t+\t0\tgene_id \"G1\"; gene_name \"GENEA\"; transcript_id \"T1\"; exon_number \"2\"; transcript_support_level \"1\";\n"
    "1\ttest\tCDS\t100\t108\t.\t+\t0\tgene_id \"G1\"; gene_name \"GENEA\"; transcript_id \"T1\"; exon_number \"1\"; transcript_support_level \"1\";\n"
    "2\ttest\tCDS\t500\t506\t.\t-\t0\tgene_id \"G2\"; gene_name \"GENEB\"; transcript_id \"T2\"; exon_number \"1\"; transcript_support_level \"1\";\n"
    "2\ttest\tCDS\t600\t606\t.\t-\t0\tgene_id \"G2\"; gene_name \"GENEB\"; transcript_id \"T2\"; exon_number \"2\"; transcript_support_level \"1\";\n"
    "3\ttest\tCDS\t700\t705\t.\t+\t0\tgene_id \"G3\"; gene_name \"GENEC\"; transcript_id \"T3\"; exon_number \"1\"; transcript_support_level \"2\";\n"
    "4\ttest\tCDS\t800\t804\t.\t+\t0\tgene_id \"G4\"; transcript_id \"T4\"; exon_number \"1\"; transcript_support_level \"1\";\n"
    "1\ttest\tfive_prime_utr\t50\t99\t.\t+\t.\tgene_id \"G1\"; gene_name \"GENEA\"; transcript_id \"T1\";\n"
)


@pytest.fixture
def gtf_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.gtf"
    p.write_text(GTF_TEXT)
    return p


def test_default_keeps_only_t1(gtf_path: Path):
    """Defaults: TSL=='1', require_multiple_of_3=True. Only T1 (chr 1) survives."""
    bed = gvl.get_splice_bed(gtf_path)
    assert set(bed.columns) == {
        "chrom",
        "chromStart",
        "chromEnd",
        "strand",
        "gene_name",
        "transcript_id",
        "exon_number",
    }
    assert bed["transcript_id"].unique().to_list() == ["T1"]
    assert bed.height == 2


def test_zero_based_start(gtf_path: Path):
    """GTF starts (1-based) become BED chromStart (0-based) by subtracting 1."""
    bed = gvl.get_splice_bed(gtf_path)
    starts = bed.sort("chromStart")["chromStart"].to_list()
    # T1 had GTF starts 100 and 300 -> 99 and 299
    assert starts == [99, 299]


def test_chrom_end_unchanged(gtf_path: Path):
    """GTF end (1-based inclusive) numerically equals BED chromEnd (0-based exclusive)."""
    bed = gvl.get_splice_bed(gtf_path)
    ends = bed.sort("chromStart")["chromEnd"].to_list()
    assert ends == [108, 308]


def test_dropped_non_cds_rows(gtf_path: Path):
    """exon and five_prime_utr rows are removed."""
    bed = gvl.get_splice_bed(gtf_path, transcript_support_level=None, require_multiple_of_3=False)
    # Every surviving row corresponds to a CDS feature; we have 6 CDS rows in fixture.
    assert bed.height == 6


def test_sorted_output(gtf_path: Path):
    """Output is sorted by chrom (natural), then chromStart."""
    bed = gvl.get_splice_bed(gtf_path, transcript_support_level=None, require_multiple_of_3=False)
    chroms = bed["chrom"].to_list()
    starts = bed["chromStart"].to_list()
    assert chroms == sorted(chroms, key=lambda c: (len(c), c))  # natural order
    # Within each chrom, starts are non-decreasing
    for c in set(chroms):
        sub = [s for ch, s in zip(chroms, starts) if ch == c]
        assert sub == sorted(sub)


def test_multiple_of_3_filter_off_keeps_t2(gtf_path: Path):
    """T2 (length 14, not multiple of 3) is kept when require_multiple_of_3=False."""
    bed = gvl.get_splice_bed(gtf_path, require_multiple_of_3=False)
    assert "T2" in bed["transcript_id"].unique().to_list()


def test_tsl_none_keeps_t3(gtf_path: Path):
    """T3 (TSL=='2') is kept when transcript_support_level=None."""
    bed = gvl.get_splice_bed(gtf_path, transcript_support_level=None)
    # T3 length is 6 (multiple of 3), so default require_multiple_of_3 still keeps it.
    assert "T3" in bed["transcript_id"].unique().to_list()


def test_tsl_explicit_value(gtf_path: Path):
    """transcript_support_level='2' selects only T3 among multiple-of-3 transcripts."""
    bed = gvl.get_splice_bed(gtf_path, transcript_support_level="2")
    assert bed["transcript_id"].unique().to_list() == ["T3"]


def test_contigs_filter(gtf_path: Path):
    """contigs=['1'] restricts to chr 1 rows."""
    bed = gvl.get_splice_bed(
        gtf_path, contigs=["1"], transcript_support_level=None, require_multiple_of_3=False
    )
    assert bed["chrom"].unique().to_list() == ["1"]


def test_gene_name_nulls_preserved(gtf_path: Path):
    """T4 has no gene_name attribute -> gene_name is null and the row is retained."""
    bed = gvl.get_splice_bed(
        gtf_path, transcript_support_level=None, require_multiple_of_3=False
    )
    t4 = bed.filter(pl.col("transcript_id") == "T4")
    assert t4.height == 1
    assert t4["gene_name"].to_list() == [None]


def test_dtypes(gtf_path: Path):
    """exon_number is Int32; chromStart/chromEnd are integers."""
    bed = gvl.get_splice_bed(
        gtf_path, transcript_support_level=None, require_multiple_of_3=False
    )
    assert bed.schema["exon_number"] == pl.Int32
    assert bed.schema["chromStart"].is_integer()
    assert bed.schema["chromEnd"].is_integer()
    assert bed.schema["chrom"] == pl.Utf8
    assert bed.schema["strand"] == pl.Utf8
    assert bed.schema["gene_name"] == pl.Utf8
    assert bed.schema["transcript_id"] == pl.Utf8
```

- [ ] **Step 2: Run the suite and confirm failures**

Run: `pixi run -e dev pytest tests/dataset/test_get_splice_bed.py -v`
Expected: every test fails with `AttributeError: module 'genvarloader' has no attribute 'get_splice_bed'`.

- [ ] **Step 3: Commit the failing tests**

```bash
rtk git add tests/dataset/test_get_splice_bed.py
rtk git commit -m "test: add failing tests for get_splice_bed"
```

---

### Task 2: Implement `get_splice_bed`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (append a new top-level function below `write`).

- [ ] **Step 1: Add the implementation**

Append at the end of `python/genvarloader/_dataset/_write.py` (after all existing top-level definitions; preserve the existing imports — `polars as pl` and `seqpro as sp` are already imported at the top of the file):

```python
def get_splice_bed(
    gtf: str | Path,
    contigs: list[str] | None = None,
    transcript_support_level: str | None = "1",
    require_multiple_of_3: bool = True,
) -> pl.DataFrame:
    """Process a GTF into a BED-compatible DataFrame for splicing datasets.

    The result has columns ``chrom``, ``chromStart`` (0-based), ``chromEnd``,
    ``strand``, ``gene_name``, ``transcript_id``, and ``exon_number``, sorted by
    chromosome (natural order) and ``chromStart``. Pass it directly to
    :func:`gvl.write` for splicing datasets.

    Parameters
    ----------
    gtf
        Path to a GTF file (gzipped or plain) accepted by :func:`seqpro.gtf.scan`.
    contigs
        If provided, keep only rows whose ``seqname`` is in this list.
    transcript_support_level
        If a string, require the GTF ``transcript_support_level`` attribute to
        equal it. ``None`` disables the filter.
    require_multiple_of_3
        If ``True``, keep only transcripts whose summed CDS length is a
        multiple of 3.
    """
    lf = sp.gtf.scan(gtf)

    if contigs is not None:
        lf = lf.filter(pl.col("seqname").is_in(contigs))

    lf = lf.filter(pl.col("feature") == "CDS").rename(
        {"seqname": "chrom", "start": "chromStart", "end": "chromEnd"}
    )

    lf = lf.with_columns(
        pl.col("chromStart") - 1,
        sp.gtf.attr("gene_name"),
        sp.gtf.attr("transcript_id"),
        sp.gtf.attr("exon_number").cast(pl.Int32),
    )

    drop_cols = ["source", "score", "frame", "feature", "attribute"]

    if require_multiple_of_3:
        lf = lf.with_columns(
            transcript_len=(pl.col("chromEnd") - pl.col("chromStart"))
            .sum()
            .over("transcript_id")
        ).filter(pl.col("transcript_len") % 3 == 0)
        drop_cols.append("transcript_len")

    if transcript_support_level is not None:
        lf = lf.filter(sp.gtf.attr("transcript_support_level") == transcript_support_level)

    df = lf.drop(drop_cols).collect()
    return sp.bed.sort(df)
```

- [ ] **Step 2: Run the test suite**

Run: `pixi run -e dev pytest tests/dataset/test_get_splice_bed.py -v`
Expected: tests still fail with `AttributeError` because the function has not been re-exported from the package yet — that happens in Task 3. (The function exists at `genvarloader._dataset._write.get_splice_bed`, but tests reference `gvl.get_splice_bed`.)

If you want a quick sanity check that the implementation itself works before Task 3, run:

```bash
pixi run -e dev python -c "from genvarloader._dataset._write import get_splice_bed; from pathlib import Path; import tempfile, os; \
    p = Path(tempfile.mkdtemp()) / 'tiny.gtf'; \
    p.write_text('1\ttest\tCDS\t100\t108\t.\t+\t0\tgene_id \"G1\"; transcript_id \"T1\"; exon_number \"1\"; transcript_support_level \"1\";\n'); \
    print(get_splice_bed(p))"
```
Expected: a 1-row DataFrame with `chromStart=99`, `chromEnd=108`.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "feat: add get_splice_bed for GTF→splicing-BED conversion"
```

---

### Task 3: Export `get_splice_bed` from the package root

**Files:**
- Modify: `python/genvarloader/__init__.py`

- [ ] **Step 1: Add the import and `__all__` entry**

Edit `python/genvarloader/__init__.py`:

Replace the line:

```python
from ._dataset._write import write
```

with:

```python
from ._dataset._write import get_splice_bed, write
```

And add `"get_splice_bed",` to the `__all__` list (e.g. immediately after `"write"`):

```python
__all__ = [
    "write",
    "get_splice_bed",
    "Dataset",
    ...
]
```

- [ ] **Step 2: Run the test suite — expect all green**

Run: `pixi run -e dev pytest tests/dataset/test_get_splice_bed.py -v`
Expected: 11 passed.

If a test fails, read the failure, fix the implementation in `_write.py`, rerun. Do not modify the tests unless the test itself is wrong (in which case explain why before changing it).

- [ ] **Step 3: Run the broader test suite to check no regressions**

Run: `pixi run -e dev pytest tests/dataset/ -v`
Expected: all previously passing tests still pass (the existing test_write.py has `@mark.skip` so it won't run; that's expected).

- [ ] **Step 4: Lint check**

Run: `pixi run -e dev ruff check python/genvarloader/_dataset/_write.py python/genvarloader/__init__.py tests/dataset/test_get_splice_bed.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/__init__.py
rtk git commit -m "feat: export get_splice_bed from package root"
```

---

## Self-review checklist (already performed)

- **Spec coverage:** every spec section is implemented — API signature (Task 2), pipeline steps 1–9 (Task 2), export (Task 3), all test cases from the spec's "Tests" section (Task 1).
- **Placeholders:** none — every step contains the literal code or command.
- **Type consistency:** function name `get_splice_bed`, parameter names `gtf, contigs, transcript_support_level, require_multiple_of_3`, return type `pl.DataFrame`, output columns `chrom, chromStart, chromEnd, strand, gene_name, transcript_id, exon_number` — all consistent across spec, tests, and implementation.
