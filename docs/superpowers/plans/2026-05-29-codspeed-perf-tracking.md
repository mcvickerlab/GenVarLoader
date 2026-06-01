# codspeed Perf Tracking + Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pytest-codspeed benchmark suite (micro numba functions + end-to-end reconstructor) over a committed realistic chr22 1kGP + GEUVADIS data slice, plus a py-spy/memray profiling driver for the haplotype, track, and variant hot paths, to track and root-cause the ~18–20× tracks / ~10–22× haplotype regression in `REGRESSIONS.md`.

**Architecture:** A committed build script regenerates a small self-contained `.gvl` dataset (5 samples, chr22, real indels + GEUVADIS read-depth tracks) and a mostly-`N`-masked chr22 reference (compresses to a few MB). Micro-benchmarks get realistic numba-function arguments by **capture-and-replay**: a fixture monkeypatches each hot function (in every consumer-module namespace that imports it by name), runs one real reconstruction batch, records the first call's args, and replays them under the benchmark timer. End-to-end benchmarks drive `Dataset` eager indexing. A profiling driver runs each hot path single-threaded under py-spy and memray.

**Tech Stack:** Python, pytest, pytest-benchmark, pytest-codspeed, numba, numpy, polars, plink2/bcftools/samtools, py-spy, memray, pixi.

---

## Background facts (verified against the codebase)

**Hot functions and their defining + consumer modules** (paths relative to `python/genvarloader/_dataset/`):

| Function | Defined in | Imported/used by name in | Triggered by |
|---|---|---|---|
| `get_diffs_sparse` | `_genotypes.py:8` | `_haps.py:38` (used `:398`) | `with_seqs("haplotypes")` indexing |
| `reconstruct_haplotypes_from_sparse` | `_genotypes.py:113` | `_haps.py:39` (used `:764,795,855,892`) | `with_seqs("haplotypes")` indexing |
| `intervals_to_tracks` | `_intervals.py:9` | `_reconstruct.py:29` (`:189`), `_tracks.py:21` (`:608,654`) | `with_tracks(...)` indexing |
| `shift_and_realign_tracks_sparse` | `_tracks.py:142` | `_reconstruct.py:34` (`:201`) | `with_tracks(...)` indexing with indels present |
| `_infer_germline_ccfs` | `_rag_variants.py:389` | same module, `:187` (`RaggedVariants.infer_germline_ccfs_`) | `with_seqs("variants")` indexing |

**Why patch the consumer namespace:** `from ._genotypes import reconstruct_haplotypes_from_sparse` binds the name into `_haps`'s globals. Replacing `genvarloader._dataset._genotypes.reconstruct_haplotypes_from_sparse` would NOT affect `_haps`'s already-bound reference. The capture helper therefore patches each `(module, attr)` listed in the "Imported/used by name in" column.

**Public read/write API (verified):**
- `gvl.write(path, bed, variants=None, tracks=None, samples=None, max_jitter=None, overwrite=False, max_mem="4g", extend_to_length=True)`
- `gvl.BigWigs(name: str, paths: dict[str, str])` and `gvl.BigWigs.from_table(name, table)` — `_bigwig.py:31,62`
- `gvl.Dataset.open(path, reference=None, ...)` — `_impl.py:130`
- `ds.with_seqs(kind: "reference"|"haplotypes"|"annotated"|"variants"|None)` — `_impl.py:485`
- `ds.with_tracks(tracks: str|list[str]|False|None, kind: "tracks"|"intervals"|None)` — `_impl.py:566`
- `ds.with_len(length)` — `_impl.py:408`
- Eager indexing `ds[region_idx, sample_idx]` triggers reconstruction (`__getitem__`, `_impl.py:1551`). Use this rather than `to_dataloader` to avoid a hard torch dependency in benches.

**Data sources (on `/carter`):**
- Genotypes: `/carter/users/dlaub/data/1kGP/plink2/hg38.norm.{pgen,psam,pvar.zst}`
- Tracks: `/carter/users/dlaub/data/1kGP-rna-seq/bw_chr22/*.bw`
- Sample→bigwig map: `/carter/users/dlaub/data/1kGP-rna-seq/sample_id_to_bigwig.csv` (columns `sample,path,read_count`; the `path` points at full bigwigs under `bigwigs/`, but a matching `bw_chr22/<same-basename>` exists)
- Regions: `/carter/users/dlaub/data/1kGP-rna-seq/chr22_egenes.bed`
- Reference for masking: `tests/data/fasta/hg38.fa.bgz` (verified to contain full `chr22`, 50,818,468 bp; fetched by `pixi run -e dev gen`)

**Build base dir:** `tests/benchmarks/data/`. Committed artifacts: `samples.txt`, `chr22_egenes.bed`, `chr22.masked.fa.gz` (+ `.fai`/`.gzi`), `chr22_geuv.gvl/`. Build script `build_realistic.py` is committed but only runs against `/carter`.

---

## File Structure

```
tests/benchmarks/
├── __init__.py                # (empty) make it a package for imports
├── _capture.py               # capture-and-replay helper (the one non-trivial unit)
├── conftest.py               # session fixtures: open dataset, captured-arg fixtures
├── test_micro.py             # micro-benchmarks (5 numba fns)
├── test_e2e.py               # end-to-end reconstructor benchmarks
├── data/
│   ├── build_realistic.py    # regenerate committed data from /carter
│   ├── samples.txt           # committed: chosen 5 sample IDs (one per line)
│   ├── chr22_egenes.bed      # committed: benchmark regions
│   ├── chr22.masked.fa.gz    # committed: masked chr22 reference (+ .fai, .gzi)
│   └── chr22_geuv.gvl/        # committed: built dataset
└── profiling/
    └── profile.py            # py-spy/memray driver (--mode haplotypes|tracks|variants)
```

`pixi.toml` gains `pytest-codspeed` and tasks `bench`, `profile-{haps,tracks,variants}`, `memray-{haps,tracks,variants}`.

---

## Task 1: Add pytest-codspeed dependency and pixi tasks

**Files:**
- Modify: `pixi.toml:22-43` (`[dependencies]`), `pixi.toml:118-132` (`[tasks]`)

- [ ] **Step 1: Add `pytest-codspeed` to `[dependencies]`**

In `pixi.toml`, in the `[dependencies]` table (after `pytest-benchmark = "*"` at line 35), add:

```toml
pytest-codspeed = "*"
```

- [ ] **Step 2: Add bench and profiling tasks to `[tasks]`**

In `pixi.toml`, in the `[tasks]` table (after `typecheck` at line 132), add:

```toml
bench = { cmd = "pytest tests/benchmarks --codspeed -p no:cov" }
bench-local = { cmd = "pytest tests/benchmarks --benchmark-only -p no:cov" }
profile-haps = { cmd = "py-spy record -o tests/benchmarks/profiling/haps.speedscope.json -f speedscope -- python tests/benchmarks/profiling/profile.py --mode haplotypes" }
profile-tracks = { cmd = "py-spy record -o tests/benchmarks/profiling/tracks.speedscope.json -f speedscope -- python tests/benchmarks/profiling/profile.py --mode tracks" }
profile-variants = { cmd = "py-spy record -o tests/benchmarks/profiling/variants.speedscope.json -f speedscope -- python tests/benchmarks/profiling/profile.py --mode variants" }
memray-haps = { cmd = "memray run -fo tests/benchmarks/profiling/haps.memray.bin python tests/benchmarks/profiling/profile.py --mode haplotypes" }
memray-tracks = { cmd = "memray run -fo tests/benchmarks/profiling/tracks.memray.bin python tests/benchmarks/profiling/profile.py --mode tracks" }
memray-variants = { cmd = "memray run -fo tests/benchmarks/profiling/variants.memray.bin python tests/benchmarks/profiling/profile.py --mode variants" }
```

- [ ] **Step 3: Install and verify the new dependency resolves**

Run: `pixi install -e dev`
Expected: completes without error; `pytest-codspeed` appears in the solve.

Run: `pixi run -e dev python -c "import pytest_codspeed; print(pytest_codspeed.__version__)"`
Expected: prints a version string (no ImportError).

- [ ] **Step 4: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build(bench): add pytest-codspeed dep and bench/profile pixi tasks"
```

---

## Task 2: Capture-and-replay helper

This is the one piece of real logic. It records the first call's positional+keyword args of a target function (wrapped in every consumer namespace), then restores. We TDD it against a trivial stand-in before wiring it to gvl internals.

**Files:**
- Create: `tests/benchmarks/__init__.py` (empty)
- Create: `tests/benchmarks/_capture.py`
- Test: `tests/benchmarks/test_capture.py`

- [ ] **Step 1: Create the empty package marker**

Create `tests/benchmarks/__init__.py` with no content (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/benchmarks/test_capture.py`:

```python
import types

from tests.benchmarks._capture import capture_first_call


def test_capture_records_first_call_args_across_namespaces():
    # Two modules that both bound the same function by name.
    defining = types.ModuleType("defining")
    consumer_a = types.ModuleType("consumer_a")
    consumer_b = types.ModuleType("consumer_b")

    def target(x, y, *, z=0):
        return x + y + z

    defining.target = target
    consumer_a.target = target  # `from defining import target`
    consumer_b.target = target

    calls = []

    def run():
        # Only the consumer namespaces are exercised at runtime.
        calls.append(consumer_a.target(1, 2, z=3))
        calls.append(consumer_b.target(10, 20, z=30))

    captured = capture_first_call(
        targets=[(consumer_a, "target"), (consumer_b, "target")],
        thunk=run,
    )

    # The real function still ran (capture is transparent).
    assert calls == [6, 60]
    # Only the first invocation was recorded.
    assert captured.args == (1, 2)
    assert captured.kwargs == {"z": 3}
    # Originals restored.
    assert consumer_a.target is target
    assert consumer_b.target is target


def test_capture_raises_if_never_called():
    mod = types.ModuleType("mod")
    mod.target = lambda: None

    import pytest

    with pytest.raises(RuntimeError, match="was never called"):
        capture_first_call(targets=[(mod, "target")], thunk=lambda: None)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/benchmarks/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` for `tests.benchmarks._capture`.

- [ ] **Step 4: Implement the helper**

Create `tests/benchmarks/_capture.py`:

```python
"""Capture-and-replay: record the first call's arguments of a hot function so a
micro-benchmark can replay them with realistic inputs.

The hot numba functions are imported *by name* into consumer modules
(``from ._genotypes import reconstruct_haplotypes_from_sparse``), so we must
patch each consumer namespace, not the defining module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CapturedCall:
    """The first observed call's positional and keyword arguments."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def capture_first_call(
    targets: list[tuple[Any, str]],
    thunk: Callable[[], Any],
) -> CapturedCall:
    """Run ``thunk``; record the first call to the patched function; restore.

    Parameters
    ----------
    targets
        ``(module_or_namespace, attribute_name)`` pairs that all hold a
        reference to the *same* function. Every pair is patched so the call is
        recorded no matter which namespace invokes it.
    thunk
        Zero-arg callable that triggers exactly one (or more) calls to the
        target. Only the first call's arguments are kept.

    Returns
    -------
    CapturedCall

    Raises
    ------
    RuntimeError
        If the target was never called while running ``thunk``.
    """
    captured: list[CapturedCall] = []
    original = getattr(targets[0][0], targets[0][1])

    def recorder(*args: Any, **kwargs: Any) -> Any:
        if not captured:
            captured.append(CapturedCall(args=args, kwargs=dict(kwargs)))
        return original(*args, **kwargs)

    for namespace, attr in targets:
        setattr(namespace, attr, recorder)
    try:
        thunk()
    finally:
        for namespace, attr in targets:
            setattr(namespace, attr, original)

    if not captured:
        name = getattr(original, "__name__", str(original))
        raise RuntimeError(f"{name} was never called while running the thunk")
    return captured[0]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/benchmarks/test_capture.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/benchmarks/__init__.py tests/benchmarks/_capture.py tests/benchmarks/test_capture.py
git commit -m "test(bench): add capture-and-replay helper for micro-benchmark inputs"
```

---

## Task 3: Realistic-data build script

Produces the committed slice from `/carter`. Heavy and `/carter`-dependent, so it is verified by run-and-inspect rather than unit-tested. Run it once to materialize the committed artifacts.

**Files:**
- Create: `tests/benchmarks/data/build_realistic.py`

- [ ] **Step 1: Write the build script**

Create `tests/benchmarks/data/build_realistic.py`:

```python
"""Regenerate the committed chr22 1kGP + GEUVADIS benchmark slice.

Run once on a host with /carter mounted:

    pixi run -e dev python tests/benchmarks/data/build_realistic.py

Produces (all under tests/benchmarks/data/):
  - samples.txt           the 5 chosen sample IDs
  - chr22_egenes.bed      benchmark regions (copied)
  - chr22.masked.fa.gz     masked chr22 reference (+ .fai, .gzi)
  - chr22_geuv.gvl/        the gvl dataset (5 samples, chr22, read-depth tracks)

Requires the test reference fasta (run `pixi run -e dev gen` first to fetch it).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent

PLINK_PREFIX = Path("/carter/users/dlaub/data/1kGP/plink2/hg38.norm")
RNA_DIR = Path("/carter/users/dlaub/data/1kGP-rna-seq")
SAMPLE_MAP = RNA_DIR / "sample_id_to_bigwig.csv"
BW_CHR22_DIR = RNA_DIR / "bw_chr22"
EGENES_BED = RNA_DIR / "chr22_egenes.bed"
REF_FASTA = REPO / "tests" / "data" / "fasta" / "hg38.fa.bgz"

N_SAMPLES = 5
N_REGIONS = 300  # cap region count to keep the committed dataset small


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def choose_samples() -> list[str]:
    """Deterministically pick N samples present in both genotypes and bigwigs."""
    psam = pl.read_csv(
        PLINK_PREFIX.with_suffix(".psam"), separator="\t"
    )
    # plink psam first column is "#IID" or "IID".
    iid_col = "#IID" if "#IID" in psam.columns else "IID"
    geno_samples = set(psam[iid_col].to_list())

    smap = pl.read_csv(SAMPLE_MAP)
    bw_samples = set(smap["sample"].to_list())

    overlap = sorted(geno_samples & bw_samples)
    if len(overlap) < N_SAMPLES:
        raise SystemExit(
            f"Only {len(overlap)} samples overlap genotypes+bigwigs; need {N_SAMPLES}."
        )
    chosen = overlap[:N_SAMPLES]
    (DATA / "samples.txt").write_text("\n".join(chosen) + "\n")
    print(f"Chosen samples: {chosen}")
    return chosen


def slice_pgen(samples: list[str]) -> Path:
    """plink2 slice: chr22, chosen samples, -> tests/benchmarks/data/chr22_5s.pgen."""
    keep = DATA / "_keep.txt"
    keep.write_text("#IID\n" + "\n".join(samples) + "\n")
    out_prefix = DATA / "chr22_5s"
    run(
        [
            "plink2",
            "--pfile", str(PLINK_PREFIX),
            "--chr", "chr22",
            "--keep", str(keep),
            "--make-pgen",
            "--out", str(out_prefix),
        ]
    )
    keep.unlink()
    return out_prefix.with_suffix(".pgen")


def copy_regions() -> Path:
    """Copy the chr22 egenes BED, capped to N_REGIONS rows."""
    bed = pl.read_csv(
        EGENES_BED,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    )
    if bed.height > N_REGIONS:
        bed = bed.head(N_REGIONS)
    out = DATA / "chr22_egenes.bed"
    bed.write_csv(out, include_header=False, separator="\t")
    print(f"Wrote {bed.height} regions to {out}")
    return out


def build_masked_reference(bed_path: Path) -> Path:
    """Extract chr22, mask everything outside benchmark regions to N, bgzip.

    A mostly-N chr22 compresses to a few MB while staying coordinate-correct
    over the benchmark windows used for haplotype reconstruction.
    """
    import numpy as np
    import pysam

    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    )

    fa = pysam.FastaFile(str(REF_FASTA))
    chr22 = np.frombuffer(fa.fetch("chr22").encode("ascii"), dtype="S1").copy()
    masked = np.full_like(chr22, b"N")
    pad = 0  # regions already include flanks; no extra pad needed
    for start, end in bed.select("start", "end").iter_rows():
        s = max(0, int(start) - pad)
        e = min(chr22.size, int(end) + pad)
        masked[s:e] = chr22[s:e]

    out_plain = DATA / "chr22.masked.fa"
    with open(out_plain, "w") as f:
        f.write(">chr22\n")
        seq = masked.tobytes().decode("ascii")
        for i in range(0, len(seq), 60):
            f.write(seq[i : i + 60] + "\n")

    out_bgz = DATA / "chr22.masked.fa.gz"
    if out_bgz.exists():
        out_bgz.unlink()
    run(["bgzip", str(out_plain)])  # -> chr22.masked.fa.gz
    run(["samtools", "faidx", str(out_bgz)])  # -> .fai + .gzi
    print(f"Wrote masked reference {out_bgz}")
    return out_bgz


def build_dataset(samples: list[str], pgen: Path, bed_path: Path) -> Path:
    import genvarloader as gvl
    from genoray import PGEN

    smap = pl.read_csv(SAMPLE_MAP)
    paths: dict[str, str] = {}
    for sample, full_path in smap.select("sample", "path").iter_rows():
        if sample not in samples:
            continue
        bw = BW_CHR22_DIR / Path(full_path).name
        if not bw.exists():
            raise SystemExit(f"Missing chr22 bigwig for {sample}: {bw}")
        paths[sample] = str(bw)
    assert set(paths) == set(samples), (set(samples) - set(paths))

    tracks = gvl.BigWigs("read-depth", paths)

    out = DATA / "chr22_geuv.gvl"
    if out.exists():
        shutil.rmtree(out)
    gvl.write(
        path=out,
        bed=bed_path,
        variants=PGEN(pgen),
        tracks=tracks,
        samples=samples,
        overwrite=True,
    )
    print(f"Wrote dataset {out}")
    return out


def main() -> None:
    if not REF_FASTA.exists():
        raise SystemExit(
            f"Reference {REF_FASTA} not found. Run `pixi run -e dev gen` first."
        )
    DATA.mkdir(parents=True, exist_ok=True)
    samples = choose_samples()
    pgen = slice_pgen(samples)
    bed_path = copy_regions()
    build_masked_reference(bed_path)
    build_dataset(samples, pgen, bed_path)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the build script**

Run: `pixi run -e dev python tests/benchmarks/data/build_realistic.py`
Expected: prints chosen samples, region count, "Wrote masked reference", "Wrote dataset", "Done." with no exception. Creates `tests/benchmarks/data/{samples.txt,chr22_egenes.bed,chr22.masked.fa.gz,chr22.masked.fa.gz.fai,chr22.masked.fa.gz.gzi,chr22_geuv.gvl/}`.

- [ ] **Step 3: Verify the dataset opens and yields the regression hot paths**

Run:
```bash
pixi run -e dev python -c "
import genvarloader as gvl
ref = 'tests/benchmarks/data/chr22.masked.fa.gz'
ds = gvl.Dataset.open('tests/benchmarks/data/chr22_geuv.gvl', ref)
print('shape', ds.shape)
t = ds.with_seqs(None).with_tracks('read-depth').with_len(16384)[0, 0]
print('tracks-only ok', type(t))
h = ds.with_seqs('haplotypes').with_tracks(False).with_len(16384)[0, 0]
print('haps ok', type(h))
v = ds.with_seqs('variants').with_tracks(False)[0, 0]
print('variants ok', type(v))
"
```
Expected: prints a shape, then `tracks-only ok`, `haps ok`, `variants ok` lines with no exception. (If `with_tracks(False)` errors because the dataset would return nothing, drop it — `with_seqs("haplotypes")` already deactivates other outputs as needed; adjust per the error message.)

- [ ] **Step 4: Check committed footprint**

Run: `du -sh tests/benchmarks/data/chr22_geuv.gvl tests/benchmarks/data/chr22.masked.fa.gz`
Expected: combined size in the low tens of MB. If `chr22_geuv.gvl` is larger, lower `N_REGIONS` in the script (e.g. 150), re-run Step 2, and re-check. Record the final size in the commit message.

- [ ] **Step 5: Remove build intermediates not meant to be committed**

Run: `rm -f tests/benchmarks/data/chr22_5s.log`
(Keep `chr22_5s.{pgen,pvar,psam}` only if you want the raw variant slice committed too; the dataset embeds the genotypes it needs, so these can be removed. Remove them: `rm -f tests/benchmarks/data/chr22_5s.*`)

- [ ] **Step 6: Commit the build script and committed data artifacts**

```bash
git add tests/benchmarks/data/build_realistic.py \
        tests/benchmarks/data/samples.txt \
        tests/benchmarks/data/chr22_egenes.bed \
        tests/benchmarks/data/chr22.masked.fa.gz \
        tests/benchmarks/data/chr22.masked.fa.gz.fai \
        tests/benchmarks/data/chr22.masked.fa.gz.gzi \
        tests/benchmarks/data/chr22_geuv.gvl
git commit -m "test(bench): add committed chr22 1kGP+GEUVADIS realistic slice + build script"
```

---

## Task 4: Session fixtures (conftest)

Provides the opened dataset and the captured-arg fixtures used by the micro-benchmarks. The captured-arg fixtures use Task 2's helper with the consumer-namespace targets from the Background table.

**Files:**
- Create: `tests/benchmarks/conftest.py`

- [ ] **Step 1: Write conftest**

Create `tests/benchmarks/conftest.py`:

```python
"""Session fixtures for the benchmark suite.

- ``bench_dataset``: the committed chr22 GEUVADIS dataset, opened once.
- ``captured_*``: realistic numba-function arguments, recorded once by running
  a real reconstruction batch and capturing the first call (see _capture.py).

All fixtures skip the whole module if the committed dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import genvarloader as gvl
from genvarloader._dataset import _haps, _rag_variants, _reconstruct
from tests.benchmarks._capture import capture_first_call

DATA = Path(__file__).resolve().parent / "data"
DS_PATH = DATA / "chr22_geuv.gvl"
REF_PATH = DATA / "chr22.masked.fa.gz"
SEQLEN = 16384
BATCH = 32  # number of (region, sample) pairs to drive per capture


@pytest.fixture(scope="session")
def bench_dataset():
    if not DS_PATH.exists():
        pytest.skip(
            f"Benchmark dataset {DS_PATH} not built. "
            "Run: pixi run -e dev python tests/benchmarks/data/build_realistic.py"
        )
    return gvl.Dataset.open(DS_PATH, REF_PATH)


def _batch_indices(ds, n: int):
    """A flat list of (region_idx, sample_idx) within the dataset bounds."""
    n_regions, n_samples = ds.shape[0], ds.shape[1]
    pairs = []
    r = s = 0
    for _ in range(min(n, n_regions * n_samples)):
        pairs.append((r % n_regions, s % n_samples))
        r += 1
        if r % n_regions == 0:
            s += 1
    regions = [p[0] for p in pairs]
    samples = [p[1] for p in pairs]
    return regions, samples


@pytest.fixture(scope="session")
def captured_haplotypes(bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    recon = capture_first_call(
        targets=[(_haps, "reconstruct_haplotypes_from_sparse")],
        thunk=lambda: ds[r, s],
    )
    return recon


@pytest.fixture(scope="session")
def captured_diffs(bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[(_haps, "get_diffs_sparse")],
        thunk=lambda: ds[r, s],
    )


@pytest.fixture(scope="session")
def captured_intervals_to_tracks(bench_dataset):
    ds = bench_dataset.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[
            (_reconstruct, "intervals_to_tracks"),
        ],
        thunk=lambda: ds[r, s],
    )


@pytest.fixture(scope="session")
def captured_realign_tracks(bench_dataset):
    ds = bench_dataset.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[(_reconstruct, "shift_and_realign_tracks_sparse")],
        thunk=lambda: ds[r, s],
    )


@pytest.fixture(scope="session")
def captured_germline_ccfs(bench_dataset):
    ds = bench_dataset.with_seqs("variants").with_len(SEQLEN)
    r, s = _batch_indices(ds, BATCH)
    return capture_first_call(
        targets=[(_rag_variants, "_infer_germline_ccfs")],
        thunk=lambda: ds[r, s],
    )
```

- [ ] **Step 2: Verify fixtures import cleanly (collection only)**

Run: `pixi run -e dev pytest tests/benchmarks/conftest.py --collect-only -q`
Expected: no import errors (conftest is imported during collection). No tests collected from conftest itself is fine.

- [ ] **Step 3: Smoke-check each capture fixture resolves**

Create a temporary check (do NOT commit) `tests/benchmarks/_smoke.py`:

```python
def test_smoke(captured_haplotypes, captured_diffs, captured_intervals_to_tracks,
               captured_realign_tracks, captured_germline_ccfs):
    for c in (captured_haplotypes, captured_diffs, captured_intervals_to_tracks,
              captured_realign_tracks, captured_germline_ccfs):
        assert c.args or c.kwargs
```

Run: `pixi run -e dev pytest tests/benchmarks/_smoke.py -v`
Expected: PASS. If `captured_germline_ccfs` raises `"_infer_germline_ccfs was never called"`, the variant path didn't infer CCFs for this data; in that case change its `targets` to capture `RaggedVariants.infer_germline_ccfs_` is not viable (method), so instead skip that micro-bench — document in Task 5 Step 5 and remove the `captured_germline_ccfs` fixture + its micro-bench, keeping the e2e `variants` bench as the variant-generation signal.

- [ ] **Step 4: Delete the smoke file**

Run: `rm tests/benchmarks/_smoke.py`

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/conftest.py
git commit -m "test(bench): add session fixtures for dataset + captured numba args"
```

---

## Task 5: Micro-benchmarks

One benchmark per hot function, replaying captured args under the `benchmark` fixture (works under both `pytest-benchmark` and `pytest --codspeed`). Each warms up the JIT once and asserts non-degenerate output.

**Files:**
- Create: `tests/benchmarks/test_micro.py`

- [ ] **Step 1: Write the micro-benchmarks**

Create `tests/benchmarks/test_micro.py`:

```python
"""Micro-benchmarks: isolated numba hot functions, replayed with realistic
arguments captured from a real reconstruction (see conftest.py)."""

from __future__ import annotations

import numpy as np

from genvarloader._dataset._genotypes import (
    get_diffs_sparse,
    reconstruct_haplotypes_from_sparse,
)
from genvarloader._dataset._intervals import intervals_to_tracks
from genvarloader._dataset._tracks import shift_and_realign_tracks_sparse


def _warm_and_run(benchmark, fn, captured):
    """Warm up JIT once, then benchmark replaying the captured call."""
    args, kwargs = captured.args, captured.kwargs
    fn(*args, **kwargs)  # JIT link / warmup (not timed)
    return benchmark(lambda: fn(*args, **kwargs))


def test_get_diffs_sparse(benchmark, captured_diffs):
    result = _warm_and_run(benchmark, get_diffs_sparse, captured_diffs)
    # diffs is (n_queries, ploidy) int32; must be a real array.
    assert isinstance(result, np.ndarray)
    assert result.size > 0


def test_reconstruct_haplotypes_from_sparse(benchmark, captured_haplotypes):
    # This kernel writes into a preallocated `out` buffer (returns None).
    out = captured_haplotypes.kwargs.get("out")
    if out is None:
        out = captured_haplotypes.args[0]
    before = np.asarray(out).copy()
    _warm_and_run(benchmark, reconstruct_haplotypes_from_sparse, captured_haplotypes)
    after = np.asarray(out)
    # The reconstruction wrote something (haplotype bytes are non-zero/non-pad).
    assert after.size > 0
    assert not np.array_equal(before, np.zeros_like(before)) or after.size > 0


def test_intervals_to_tracks(benchmark, captured_intervals_to_tracks):
    result = _warm_and_run(
        benchmark, intervals_to_tracks, captured_intervals_to_tracks
    )
    # intervals_to_tracks returns a dense track array.
    assert result is not None


def test_shift_and_realign_tracks_sparse(benchmark, captured_realign_tracks):
    result = _warm_and_run(
        benchmark, shift_and_realign_tracks_sparse, captured_realign_tracks
    )
    assert result is not None
```

> **Note on assertions:** `reconstruct_haplotypes_from_sparse` and
> `shift_and_realign_tracks_sparse` write into preallocated output buffers and may
> return `None`. The "honest" check is that the timed function ran with the captured
> args without error and (where a buffer is passed) the buffer is non-empty. Keep the
> assertions loose enough to pass for buffer-writing kernels but strict enough to fail
> if a fixture degenerates to empty inputs (`.size > 0`).

- [ ] **Step 2: Run the micro-benchmarks (local pytest-benchmark mode)**

Run: `pixi run -e dev pytest tests/benchmarks/test_micro.py --benchmark-only -p no:cov -v`
Expected: each `test_*` PASSES and prints a benchmark timing row. If `test_intervals_to_tracks`/`test_shift_and_realign_tracks_sparse` assert on a `None` return, relax to `assert result is None or result is not None` is wrong — instead remove the return assertion and assert on a captured input buffer's `.size > 0` (mirror the reconstruct test). Adjust based on the actual return type observed.

- [ ] **Step 3: If the germline-CCF micro-bench is viable, add it**

If Task 4 Step 3 confirmed `captured_germline_ccfs` resolves, append to `test_micro.py`:

```python
from genvarloader._dataset._rag_variants import _infer_germline_ccfs


def test_infer_germline_ccfs(benchmark, captured_germline_ccfs):
    result = _warm_and_run(benchmark, _infer_germline_ccfs, captured_germline_ccfs)
    assert result is not None
```

Run: `pixi run -e dev pytest tests/benchmarks/test_micro.py::test_infer_germline_ccfs --benchmark-only -p no:cov -v`
Expected: PASS. If the fixture was removed in Task 4, skip this step.

- [ ] **Step 4: Verify it also runs under codspeed walltime mode**

Run: `pixi run -e dev pytest tests/benchmarks/test_micro.py --codspeed -p no:cov`
Expected: completes; codspeed reports walltime measurements for each bench (no collection/run errors).

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/test_micro.py
git commit -m "test(bench): add micro-benchmarks for hot numba reconstruction kernels"
```

---

## Task 6: End-to-end benchmarks

Drives the reconstructor via eager indexing for the four generation modes, at the regression seqlen.

**Files:**
- Create: `tests/benchmarks/test_e2e.py`

- [ ] **Step 1: Write the end-to-end benchmarks**

Create `tests/benchmarks/test_e2e.py`:

```python
"""End-to-end benchmarks: reconstructor via eager Dataset indexing, at the
regression seqlen (16384). Covers haplotype, annotated, variant, track, and the
tracks-only path that REGRESSIONS.md fingered."""

from __future__ import annotations

import pytest

SEQLEN = 16384
BATCH = 32


def _indices(ds, n=BATCH):
    n_regions, n_samples = ds.shape[0], ds.shape[1]
    n = min(n, n_regions * n_samples)
    regions = [i % n_regions for i in range(n)]
    samples = [(i // n_regions) % n_samples for i in range(n)]
    return regions, samples


def _bench_indexing(benchmark, ds):
    r, s = _indices(ds)
    ds[r, s]  # warmup (JIT link, caches)
    result = benchmark(lambda: ds[r, s])
    assert result is not None


def test_e2e_haplotypes(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("haplotypes").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_annotated(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("annotated").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_variants(benchmark, bench_dataset):
    ds = bench_dataset.with_seqs("variants").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_tracks(benchmark, bench_dataset):
    ds = bench_dataset.with_tracks("read-depth").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)


def test_e2e_tracks_only(benchmark, bench_dataset):
    # The exact regression path: tracks only, no sequences.
    ds = bench_dataset.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    _bench_indexing(benchmark, ds)
```

- [ ] **Step 2: Run the end-to-end benchmarks**

Run: `pixi run -e dev pytest tests/benchmarks/test_e2e.py --benchmark-only -p no:cov -v`
Expected: all 5 PASS with timing rows. If `with_seqs("annotated")` or any mode raises a validation error from `_impl.py` (e.g. "Dataset has no genotypes"), the committed dataset lacks that capability — that should not happen for this dataset (it has genotypes + tracks + reference), so treat an error as a real bug to debug, not to skip.

- [ ] **Step 3: Verify under codspeed mode**

Run: `pixi run -e dev pytest tests/benchmarks/test_e2e.py --codspeed -p no:cov`
Expected: completes with codspeed walltime measurements, no errors.

- [ ] **Step 4: Run the whole suite together**

Run: `pixi run -e dev bench-local`
Expected: micro + e2e benches all collected and pass.

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/test_e2e.py
git commit -m "test(bench): add end-to-end reconstructor benchmarks"
```

---

## Task 7: Profiling driver

A single script with three modes, single numba thread, for py-spy and memray. Driven by the pixi tasks from Task 1.

**Files:**
- Create: `tests/benchmarks/profiling/profile.py`
- Create: `tests/benchmarks/profiling/__init__.py` (empty)

- [ ] **Step 1: Create the package marker and driver**

Create `tests/benchmarks/profiling/__init__.py` (empty).

Create `tests/benchmarks/profiling/profile.py`:

```python
"""Profiling driver for the haplotype, track, and variant hot paths.

Run single-threaded under py-spy or memray via the pixi tasks, e.g.:

    pixi run -e dev profile-tracks
    pixi run -e dev memray-haps

Modes:
  haplotypes  with_seqs("haplotypes")               (heavily-used path)
  tracks      with_seqs(None).with_tracks(...)       (REGRESSIONS.md target)
  variants    with_seqs("variants")                  (RaggedVariants assembly)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Force single numba thread BEFORE importing numba-backed code.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

DATA = Path(__file__).resolve().parents[1] / "data"
DS_PATH = DATA / "chr22_geuv.gvl"
REF_PATH = DATA / "chr22.masked.fa.gz"
SEQLEN = 16384
BATCH = 32
N_BATCHES = 200
BURN_IN = 5


def build(ds, mode: str):
    if mode == "haplotypes":
        return ds.with_seqs("haplotypes").with_len(SEQLEN)
    if mode == "tracks":
        return ds.with_seqs(None).with_tracks("read-depth").with_len(SEQLEN)
    if mode == "variants":
        return ds.with_seqs("variants").with_len(SEQLEN)
    raise SystemExit(f"unknown mode {mode!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["haplotypes", "tracks", "variants"], required=True)
    p.add_argument("--n-batches", type=int, default=N_BATCHES)
    args = p.parse_args()

    if not DS_PATH.exists():
        raise SystemExit(
            f"Dataset {DS_PATH} not built. Run "
            "`pixi run -e dev python tests/benchmarks/data/build_realistic.py`."
        )

    import genvarloader as gvl

    ds = build(gvl.Dataset.open(DS_PATH, REF_PATH), args.mode)
    n_regions, n_samples = ds.shape[0], ds.shape[1]
    n = min(BATCH, n_regions * n_samples)
    regions = [i % n_regions for i in range(n)]
    samples = [(i // n_regions) % n_samples for i in range(n)]

    print(f"mode={args.mode} threads={os.environ['NUMBA_NUM_THREADS']} "
          f"batches={args.n_batches} batch={n}")
    for i in range(args.n_batches + BURN_IN):
        _ = ds[regions, samples]
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the driver runs standalone for each mode**

Run: `NUMBA_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile.py --mode tracks --n-batches 5`
Expected: prints the `mode=tracks ...` line then `done`, no exception. Repeat with `--mode haplotypes` and `--mode variants`.

- [ ] **Step 3: Verify py-spy capture works for one mode**

Run: `pixi run -e dev profile-tracks`
Expected: produces `tests/benchmarks/profiling/tracks.speedscope.json` (a non-empty file). py-spy prints a summary.

- [ ] **Step 4: Verify memray capture works for one mode**

Run: `pixi run -e dev memray-tracks`
Expected: produces `tests/benchmarks/profiling/tracks.memray.bin`. Generate a quick summary: `pixi run -e dev memray summary tests/benchmarks/profiling/tracks.memray.bin` prints a peak-memory table.

- [ ] **Step 5: Ignore profiling outputs in git**

Append to `.gitignore` (create the entry if not present):

```
tests/benchmarks/profiling/*.speedscope.json
tests/benchmarks/profiling/*.memray.bin
```

- [ ] **Step 6: Commit the driver**

```bash
git add tests/benchmarks/profiling/__init__.py tests/benchmarks/profiling/profile.py .gitignore
git commit -m "test(bench): add py-spy/memray profiling driver for hot paths"
```

---

## Task 8: Run profiling and write up results

Produce the profiling artifacts for all three modes, analyze, and append a results section to `REGRESSIONS.md`.

**Files:**
- Modify: `docs/superpowers/REGRESSIONS.md` (append a section)

- [ ] **Step 1: Capture py-spy + memray for all three modes**

Run each:
```bash
pixi run -e dev profile-haps
pixi run -e dev profile-tracks
pixi run -e dev profile-variants
pixi run -e dev memray-haps
pixi run -e dev memray-tracks
pixi run -e dev memray-variants
```
Expected: six artifact files under `tests/benchmarks/profiling/`.

- [ ] **Step 2: Extract the top hot frames and peak memory per mode**

For each mode, summarize: open the `*.speedscope.json` (or run `py-spy` interactively) to find the top self-time frames, and run `pixi run -e dev memray summary tests/benchmarks/profiling/<mode>.memray.bin` for peak allocation sites. Record the top 5 frames and peak RSS per mode.

- [ ] **Step 3: Append the results section to REGRESSIONS.md**

Add a new section to `docs/superpowers/REGRESSIONS.md` (after "Suggested upstream investigation"):

```markdown
## Profiling results (local, chr22 GEUVADIS slice)

Profiled on the committed `tests/benchmarks/data/chr22_geuv.gvl` slice (5 samples,
chr22, GEUVADIS read-depth tracks, real 1kGP indels), `NUMBA_NUM_THREADS=1`,
seqlen 16384, via `pixi run -e dev profile-{haps,tracks,variants}` (py-spy) and
`memray-{haps,tracks,variants}`. Absolute parity numbers still require the
cluster-scale dataset; this localizes *where* time and memory go.

### Haplotypes (heavily-used path)
- Top frames: <fill from Step 2>
- Peak RSS: <fill>
- Maps to hypothesis: <serial bottleneck #1 / memory #4>

### Tracks (regression headline)
- Top frames: <fill>
- Peak RSS: <fill>
- Maps to hypothesis: <serial bottleneck #1 / memory #4>

### Variants
- Top frames: <fill>
- Peak RSS: <fill>

### Takeaway
<1–3 sentences: which function(s) dominate, whether haplotype and track hot
paths share a bottleneck, and the most promising optimization target.>
```

Replace every `<...>` placeholder with the actual measured frames/numbers from Step 2. Do not commit with placeholders remaining.

- [ ] **Step 4: Commit the write-up**

```bash
git add docs/superpowers/REGRESSIONS.md
git commit -m "docs(REGRESSIONS): add local py-spy/memray profiling results for hot paths"
```

---

## Self-Review notes

- **Spec coverage:** committed realistic chr22 slice (Task 3) ✓; masked reference (Task 3) ✓; both benchmark layers — micro (Task 5) + e2e (Task 6) ✓; micro inputs sliced from the real `.gvl` via capture-and-replay (Tasks 2/4) ✓; pytest-codspeed dep + tasks, local-only (Task 1) ✓; profiling of haplotypes/tracks/variants co-equal (Task 7) ✓; REGRESSIONS.md write-up headlining haps + tracks (Task 8) ✓; size gate (Task 3 Step 4) ✓.
- **Capture-namespace correctness:** verified each hot function's consumer modules; fixtures patch `_haps`, `_reconstruct`, `_rag_variants` accordingly, not the defining modules.
- **Known risk (documented inline):** `_infer_germline_ccfs` may not be triggered on this data; Task 4 Step 3 / Task 5 Step 3 give the fallback (drop the micro-bench; keep the e2e `variants` bench).
- **Known risk:** exact return/None semantics of the numba kernels — Task 5 Step 2 instructs adjusting assertions to buffer-`.size > 0` when a kernel returns `None`.
```
