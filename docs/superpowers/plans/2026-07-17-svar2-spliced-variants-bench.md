# SVAR2 spliced-variant decode benchmark + optimize — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Phase 4 (optimization) is measurement-driven — its concrete changes are produced
> by Task 5's profile, not pre-specified here.

**Goal:** Give PR #286's spliced SVAR2 variant decode a reproducible throughput
benchmark fed by vcfixture-rs bulk cohorts, then run the performant-py-rust
measure→optimize loop and land the winning change onto #286.

**Architecture:** A dev-only harness under `benchmarking/svar2_spliced/` that
(1) generates realistic cohorts with the `vcfixture bulk` CLI, (2) normalizes +
converts them to an SVAR2 `.gvl` store with a synthetic reference and splice BED,
(3) times the spliced `with_seqs("variants")` getitem while sweeping cohort size
and transcript count, gated by a byte-identical correctness oracle frozen from the
current PR #286 output.

**Tech Stack:** Python (numpy, polars, genoray, genvarloader), Rust
(`src/svar2/mod.rs`), vcfixture-rs bulk CLI, pyinstrument, samply/perf,
cargo-show-asm, bcftools/samtools.

## Global Constraints

- Branch `perf/svar2-spliced-variants-bench`, worktree
  `.claude/worktrees/svar2-spliced-bench`, stacked on PR #286 head `deb76d6`.
- Fresh pixi env for this worktree; do NOT share `.pixi` (memory:
  gvl-parallel-worktrees-fresh-pixi-env).
- After ANY `src/` change: `pixi run -e dev maturin develop --release` before
  running Python benchmarks (memory: pytest imports the stale `.so` otherwise).
- Correctness is byte-identical to current PR #286 output at every optimization
  step (`np.array_equal` on each RaggedVariants field's data + offsets).
- No new public GVL API; no change to PR #286 semantics.
- vcfixture bulk CLI flags (verbatim): `--profile germline-1kgp`, `--samples N`
  (names `s0..s{n-1}`), `--contigs chr1,...`, one of `--target-size 100MB` /
  `--records N` / `--records-per-contig N`, `--payload gt-only|gt-vaf|gatk|mutect2`,
  `--format bcf|vcf-gz|vcf`, `--seed N`, `-o out.bcf` (writes `.csi` + `.summary.json`).
- SVAR2 build recipe (verbatim from tests/dataset/test_svar2_dataset.py):
  `genoray._core.run_conversion_pipeline(str(bcf), str(ref), ["chr1"], str(out),
  sample_list, 25_000, 2, 1, 8*1024*1024)` → `gvl.write(d, splice_bed,
  variants=SparseVar2(out), samples=None, overwrite=True)` →
  `gvl.Dataset.open(d, reference=ref)`.
- Spliced read: `.with_settings(splice_info=("transcript_id","exon_number"),
  var_filter="exonic").with_seqs("variants")[rows, samples]`.
- splice BED columns: `chrom, chromStart, chromEnd, strand, transcript_id,
  exon_number`.

---

## File Structure

- `benchmarking/svar2_spliced/gen_cohort.py` — wrap `vcfixture bulk`; produce a
  normalized (`bcftools norm`) bi-allelic BCF + record its summary/provenance.
- `benchmarking/svar2_spliced/build_fixture.py` — synthetic reference FASTA +
  splice BED sized to the cohort's populated contig span; SVAR2 convert + gvl.write;
  content-addressed cache.
- `benchmarking/svar2_spliced/bench_spliced.py` — open dataset, build transcript
  queries, warmup + best-of-k timing over the (S, T) sweep; emit CSV.
- `benchmarking/svar2_spliced/oracle.py` — freeze/compare RaggedVariants output.
- `benchmarking/svar2_spliced/README.md` — how to run + interpret.

---

## Task 0: Worktree env + vcfixture CLI build

**Files:** none (environment setup).

- [ ] **Step 1: Create fresh pixi env for the worktree**

Run: `cd .claude/worktrees/svar2-spliced-bench && pixi install -e dev`
Expected: env resolves; `pixi run -e dev python -c "import genvarloader"` works.

- [ ] **Step 2: Build the Rust extension from PR #286 head**

Run: `pixi run -e dev maturin develop --release`
Expected: builds `genvarloader` extension from `src/` at `deb76d6`.

- [ ] **Step 3: Build the vcfixture bulk CLI once**

Run (in sibling repo, its own env):
`cd /carter/users/dlaub/projects/vcfixture-rs && cargo build --release --features cli`
Expected: `target/release/vcfixture` exists. Record its `--version`/git sha.

- [ ] **Step 4: Smoke the CLI**

Run: `.../vcfixture bulk --profile germline-1kgp --samples 50 --contigs chr1 --records 2000 --seed 42 -o /tmp/smoke.bcf`
Expected: `/tmp/smoke.bcf`, `.csi`, `.summary.json` written; note the declared
`##contig=<ID=chr1,length=...>` populated span (via `bcftools view -h`).

- [ ] **Step 5: Commit** (nothing to commit; setup only — skip.)

---

## Task 1: Cohort generator (`gen_cohort.py`)

**Files:**
- Create: `benchmarking/svar2_spliced/gen_cohort.py`

**Interfaces:**
- Produces: `gen_cohort(samples: int, records: int, *, contig="chr1", seed=42,
  profile="germline-1kgp", payload="gt-only", vcfixture_bin: Path, out_dir: Path)
  -> CohortResult` where `CohortResult` has `.bcf: Path` (normalized, bi-allelic),
  `.contig: str`, `.span: int` (populated span from the declared contig length),
  `.n_samples: int`, `.sample_names: list[str]`.

- [ ] **Step 1: Write `gen_cohort`** — shell out to the CLI, then normalize so
  gvl's bi-allelic/atomized/left-aligned requirement holds:

```python
import json, subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CohortResult:
    bcf: Path
    contig: str
    span: int
    n_samples: int
    sample_names: tuple[str, ...]

def gen_cohort(samples, records, *, contig="chr1", seed=42,
               profile="germline-1kgp", payload="gt-only",
               vcfixture_bin, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"cohort_s{samples}_r{records}_seed{seed}.raw.bcf"
    subprocess.run([str(vcfixture_bin), "bulk", "--profile", profile,
                    "--samples", str(samples), "--contigs", contig,
                    "--records", str(records), "--payload", payload,
                    "--seed", str(seed), "-o", str(raw)], check=True)
    # gvl requires bi-allelic, left-aligned, atomized (docs: bcftools norm).
    norm = out_dir / f"cohort_s{samples}_r{records}_seed{seed}.bcf"
    subprocess.run(["bcftools", "norm", "-m-", "-Ob", "-o", str(norm), str(raw)],
                   check=True)
    subprocess.run(["bcftools", "index", "-f", str(norm)], check=True)
    summary = json.loads((raw.with_suffix(".summary.json")).read_text())
    # populated span = declared contig length (vcfixture sets length = last POS).
    span = int(summary["contigs"][0]["length"])
    names = tuple(f"s{i}" for i in range(samples))
    return CohortResult(norm, contig, span, samples, names)
```

- [ ] **Step 2: Verify the summary schema** — run Task 0's smoke BCF through and
  `print` the `.summary.json` keys; adjust `summary["contigs"][0]["length"]` to the
  real key names if they differ (do NOT guess — read the emitted JSON).

Run: `pixi run -e dev python -c "import json,glob;print(json.load(open('/tmp/smoke.summary.json')))"`
Expected: locate the populated-span field; fix the accessor to match.

- [ ] **Step 3: Commit**

```bash
git add benchmarking/svar2_spliced/gen_cohort.py
git commit -m "bench(svar2): cohort generator wrapping vcfixture bulk"
```

---

## Task 2: Fixture builder (`build_fixture.py`) + end-to-end smoke

**Files:**
- Create: `benchmarking/svar2_spliced/build_fixture.py`

**Interfaces:**
- Consumes: `CohortResult` from Task 1.
- Produces: `build_fixture(cohort: CohortResult, *, n_transcripts: int,
  exons_per_tx=3, exon_len=200, cache_dir: Path) -> Fixture` where `Fixture` has
  `.gvl_path: Path`, `.reference: Path`, `.splice_bed: pl.DataFrame`,
  `.n_transcripts: int`. Content-addressed by `(n_samples, records, seed,
  n_transcripts, exons_per_tx, exon_len)`.

- [ ] **Step 1: Write the reference synthesizer** — a FASTA named for the contig,
  length ≥ span, deterministic bases:

```python
import numpy as np, polars as pl, subprocess
from pathlib import Path

def _write_reference(path: Path, contig: str, length: int, seed=0):
    rng = np.random.default_rng(seed)
    seq = rng.choice(np.frombuffer(b"ACGT", "S1"), size=length).tobytes().decode()
    path.write_text(f">{contig}\n{seq}\n")
    subprocess.run(["samtools", "faidx", str(path)], check=True)
```

- [ ] **Step 2: Write the splice-BED synthesizer** — tile `n_transcripts`
  transcripts of `exons_per_tx` exons inside `[0, span)`, alternating strand so the
  RC path is exercised:

```python
def _splice_bed(contig, span, n_transcripts, exons_per_tx, exon_len):
    rows = []
    stride = max(exon_len * (exons_per_tx + 1), 1)
    usable = max(span - stride, stride)
    for t in range(n_transcripts):
        base = (t * stride) % usable
        strand = "+" if t % 2 == 0 else "-"
        for e in range(exons_per_tx):
            start = base + e * (exon_len + 10)
            end = min(start + exon_len, span)
            if start >= end:
                continue
            rows.append((contig, start, end, strand, f"T{t}", e + 1))
    return pl.DataFrame(rows, schema=["chrom","chromStart","chromEnd","strand",
                                      "transcript_id","exon_number"], orient="row")
```

- [ ] **Step 3: Write `build_fixture`** — reference + splice BED, then the verbatim
  SVAR2 convert + gvl.write recipe, with a content-addressed cache dir:

```python
import genvarloader as gvl
from genoray import _core, SparseVar2

def build_fixture(cohort, *, n_transcripts, exons_per_tx=3, exon_len=200, cache_dir):
    key = f"s{cohort.n_samples}_r_{n_transcripts}tx_{exons_per_tx}x{exon_len}"
    root = Path(cache_dir) / key
    gvl_path = root / "ds.gvl"
    ref = root / "ref.fa"
    bed = _splice_bed(cohort.contig, cohort.span, n_transcripts, exons_per_tx, exon_len)
    if gvl_path.exists():
        return Fixture(gvl_path, ref, bed, n_transcripts)
    root.mkdir(parents=True, exist_ok=True)
    _write_reference(ref, cohort.contig, cohort.span + exon_len)
    svar2 = root / "store.svar2"
    _core.run_conversion_pipeline(str(cohort.bcf), str(ref), [cohort.contig],
                                  str(svar2), list(cohort.sample_names),
                                  25_000, 2, 1, 8 * 1024 * 1024)
    gvl.write(gvl_path, bed, variants=SparseVar2(svar2), samples=None, overwrite=True)
    return Fixture(gvl_path, ref, bed, n_transcripts)
```

- [ ] **Step 4: End-to-end smoke (the real integration gate)** — a tiny cohort all
  the way to a spliced-variants read. This is where REF-mismatch / multiallelic /
  conversion issues surface; fix them here (e.g. adjust the norm flags, add
  `--payload gt-only`, or `bcftools norm -f ref`) before scaling.

```python
# scratch, run manually:
c = gen_cohort(50, 2000, vcfixture_bin=BIN, out_dir=TMP)
f = build_fixture(c, n_transcripts=8, cache_dir=TMP)
ds = gvl.Dataset.open(f.gvl_path, reference=f.reference).with_settings(
    splice_info=("transcript_id","exon_number"), var_filter="exonic"
).with_seqs("variants")
out = ds[:, :]   # must return a RaggedVariants without error
print(type(out), out.alt.shape)
```

Run: `pixi run -e dev python benchmarking/svar2_spliced/_smoke.py`
Expected: prints `RaggedVariants` and a non-degenerate shape. If conversion errors
on REF/allele mismatch, resolve here (documented in README) — do not scale until green.

- [ ] **Step 5: Commit**

```bash
git add benchmarking/svar2_spliced/build_fixture.py
git commit -m "bench(svar2): fixture builder (reference + splice bed + svar2 write)"
```

---

## Task 3: Bench driver + oracle (`bench_spliced.py`, `oracle.py`)

**Files:**
- Create: `benchmarking/svar2_spliced/oracle.py`
- Create: `benchmarking/svar2_spliced/bench_spliced.py`

**Interfaces:**
- Consumes: `Fixture` (Task 2).
- Produces: `freeze(out: RaggedVariants) -> dict[str, tuple[np.ndarray, np.ndarray]]`
  and `assert_equal(frozen, out)` in `oracle.py`; `bench(fixture, *, n_query_rows,
  reps, warmup) -> BenchRow` in `bench_spliced.py`.

- [ ] **Step 1: Write the oracle** — freeze every field's data + offsets:

```python
import numpy as np
def freeze(out):
    frozen = {}
    for name in out.fields:                       # alt/start/ref/ilen/dosage/...
        f = out[name]
        frozen[name] = (np.asarray(f.data).copy(),
                        np.asarray(f.offsets).copy())
    return frozen

def assert_equal(frozen, out):
    got = set(out.fields)
    assert got == set(frozen), f"field set changed: {got} vs {set(frozen)}"
    for name, (data, off) in frozen.items():
        f = out[name]
        assert np.array_equal(np.asarray(f.data), data), f"{name} data differ"
        assert np.array_equal(np.asarray(f.offsets), off), f"{name} offsets differ"
```

Adjust `out.fields` / `out[name]` accessors to the real RaggedVariants API once
Task 2's smoke prints the object (read it; don't guess).

- [ ] **Step 2: Write the bench driver** — warmup + best-of-k median/min/spread,
  both serial and `GVL_FORCE_PARALLEL=1`:

```python
import os, time, statistics as st
from dataclasses import dataclass, asdict

@dataclass
class BenchRow:
    n_samples: int; n_transcripts: int; parallel: bool
    median_ms: float; min_ms: float; spread_ms: float

def _time_once(ds, rows, samples):
    t0 = time.perf_counter(); ds[rows, samples]; return (time.perf_counter()-t0)*1e3

def bench(fixture, *, n_query_rows, n_samples, reps=7, warmup=2, parallel=False):
    import genvarloader as gvl
    os.environ["GVL_FORCE_PARALLEL"] = "1" if parallel else "0"
    ds = gvl.Dataset.open(fixture.gvl_path, reference=fixture.reference).with_settings(
        splice_info=("transcript_id","exon_number"), var_filter="exonic"
    ).with_seqs("variants")
    rows = slice(0, n_query_rows); samples = slice(0, n_samples)
    for _ in range(warmup): ds[rows, samples]
    ts = [_time_once(ds, rows, samples) for _ in range(reps)]
    return BenchRow(n_samples, n_query_rows, parallel,
                    st.median(ts), min(ts), max(ts)-min(ts))
```

- [ ] **Step 3: Write the sweep `main`** — sweep `n_samples ∈ {500, 3202, 25000,
  <largest that fits>}` and `n_transcripts ∈ {32, 128, 256}`, holding `records`
  fixed so per-transcript density is constant; write `results.csv`; freeze the
  oracle at one mid sweep-point.

- [ ] **Step 4: Run the baseline sweep**

Run: `pixi run -e dev python benchmarking/svar2_spliced/bench_spliced.py --baseline`
Expected: `results.csv` with serial + parallel medians across the sweep. RECORD
these numbers — they are the Phase-3 baseline every later change is measured against.

- [ ] **Step 5: Commit**

```bash
git add benchmarking/svar2_spliced/oracle.py benchmarking/svar2_spliced/bench_spliced.py \
        benchmarking/svar2_spliced/results.csv
git commit -m "bench(svar2): spliced-variant sweep driver + frozen correctness oracle"
```

---

## Task 4: README + provenance

**Files:**
- Create: `benchmarking/svar2_spliced/README.md`

- [ ] **Step 1: Document** how to run (build CLI, run sweep), what each dimension
  means, the vcfixture bin version/sha used, the norm step's rationale, and how to
  read `results.csv`. Note the non-goals (no LD; benchmark-only).

- [ ] **Step 2: Commit**

```bash
git add benchmarking/svar2_spliced/README.md
git commit -m "bench(svar2): document the spliced-variant benchmark harness"
```

---

## Task 5: Phase-4 optimization loop (measurement-driven)

This task has no pre-written code — its changes are produced by the profile. Follow
performant-py-rust Phase 4 exactly. Repeat until the Phase-0 target, the Amdahl
ceiling, or diminishing returns; state which stopped you.

- [ ] **Step 1: Profile the baseline hot spot.**
  - Python regroup: `pyinstrument` around `ds[rows, samples]` at the largest cohort.
  - Rust decode: `perf record` on the Python process (memory:
    gvl-profiling-perf-not-pyspy-native), `perf report --children`.
  - Record the dominating slice. Hypothesis from prior findings (memory:
    svar2-spliced-gather-bottleneck): the per-field `to_packed()` + `flat[permutation]`
    row-gather in `_fetch_spliced_variants` dominates, not the Rust kernel.

- [ ] **Step 2: One hypothesis → one change.** Examples the profile may point to
  (do NOT apply blind — only what the profile justifies):
  - replace the per-field Python row-gather with a slice+concatenate or a Rust
    reorder (cf. PR #272's 8x spliced-gather fix);
  - single-group fast path when a query block is one contiguous group;
  - avoid `to_packed()` copies where offsets already suffice.

- [ ] **Step 3: Re-run the oracle** (`assert_equal`) across the sweep + degenerate
  cases: 0 exonic variants in a transcript, single transcript, minus-strand,
  single-group. Revert immediately if it fails — it's a bug, not a speedup.

- [ ] **Step 4: Re-run the benchmark.** Keep the change only if median improves at
  the dominating cohort AND the oracle passes. For any `src/` change: rebuild
  (`maturin develop --release`) first, and confirm the mechanism with
  `cargo-show-asm` rather than asserting it.

- [ ] **Step 5: Re-profile** (the hot spot moves) and repeat from Step 2.

- [ ] **Step 6: Commit each kept change** with the before/after median in the body,
  e.g. `perf(svar2): <change> — 256tx×25k samples altered N ms → M ms`.

---

## Task 6: Finalize + PR

- [ ] **Step 1:** Run the full sweep once more; regenerate `results.csv` with the
  final numbers; update the README's before/after table.
- [ ] **Step 2:** Run the SVAR2 correctness suite unchanged:
  `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py tests/unit -q`
  (memory: scoped runs skip tests/unit — include it). Expected: green.
- [ ] **Step 3:** Install prek hooks (`prek install`) if not already; run `prek`.
- [ ] **Step 4:** Push branch to `origin`; open a **draft** PR stacked on / basing
  the same as #286, body carrying the before/after sweep table and the stop reason.

---

## Self-Review

- **Spec coverage:** cohort gen (T1), fixture/ref/splice (T2), bench+oracle (T3),
  README/provenance (T4), optimize loop (T5), delivery (T6) — every spec component
  mapped. ✓
- **Placeholder scan:** the only intentionally-open items are Phase-4 changes
  (correctly measurement-driven) and two accessor confirmations (summary-JSON span
  key, RaggedVariants field API) explicitly gated on reading real output, not
  guessing. ✓
- **Type consistency:** `CohortResult`→`build_fixture`→`Fixture`→`bench`/`freeze`
  names/fields consistent across tasks. ✓
```

