# SVAR2 Write-Path Memory & `find_ranges` Complexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gvl.write(..., variants=SparseVar2(...), max_mem=...)` complete at population scale (414k samples, ~4k regions) with bounded memory and visible progress, by fixing genoray's `find_ranges` from `O(regions x total_variants)` to `O(total_variants)` and adding a memory-bounded chunked API.

**Architecture:** genoray's `find_ranges` currently rebuilds a `SearchTree` and a `v_ends` vector for every `(region, column)` pair. Hoist that region-independent state into a `VkColumnIndex` built once per column, invert the loop to column-outer, and parallelize with rayon over disjoint output slices. Fold per-region max-end computation into the same sweep so GenVarLoader can stop decoding every sample. Expose a chunked Python API (`_find_ranges_chunked`) that yields hap-slices sized from `max_mem`, and have the GenVarLoader writer consume it with per-chunk memmap writes, a disk preflight, and fractional progress.

**Tech Stack:** Rust (pyo3 0.29, numpy 0.29, ndarray 0.17, rayon 1.11), Python 3.10+ (numpy, polars, loguru, tqdm), pixi for both repos.

Design spec: `docs/superpowers/specs/2026-07-30-svar2-write-memory-design.md`
Issue: [gvl#333](https://github.com/mcvickerlab/GenVarLoader/issues/333)

## Global Constraints

- **Two repositories.** genoray (`/carter/users/dlaub/projects/genoray`) and GenVarLoader. genoray Tasks 1–4 land and release first; GenVarLoader Task 6 depends on that release.
- **Worktrees.** Work in `.claude/worktrees/` under each repo root. genoray needs its own worktree — GenVarLoader's worktree already exists at `.claude/worktrees/issue-333-svar2-write-mem`.
- **Branch targets.** Both repos: `main`. This is the file-backed `gvl.write` path, **not** StreamingDataset-board work, so it does **not** target the `streaming` branch.
- **Conventional Commits** in both repos (commitizen enforces this via a `commit-msg` hook).
- **Never edit genoray's `CHANGELOG.md`** — commitizen owns and regenerates it.
- **Composite max-end key packing:** `key = (pos << 21) | ext`, where `pos` is the 0-based variant position and `ext = 1 + deletion_len` (so `end = pos + ext`). `SHIFT = 21`. A key of `0` means "no variant in this region". This encoding is fixed by existing GenVarLoader behavior at `python/genvarloader/_dataset/_write.py:1101-1120` and must be reproduced exactly.
- **genoray version floor after release:** `genoray>=3.4.0,<4` (a `feat:` commit yields a minor bump from 3.3.0).
- **genoray Rust tests:** `pixi run -e lint test-rust`. Python tests: `pixi run pytest tests/<file>`. After **any** Rust change, rebuild the editable extension with `pixi run maturin develop --release` before running Python tests, or the stale `.so` is imported silently.
- **GenVarLoader tests:** `pixi run -e dev pytest <path> -q`. Before pushing, run the full tree: `pixi run -e dev pytest tests -q`.
- **Pre-commit hooks:** installed in both repos (`prek install`). In a fresh GenVarLoader worktree the `pyrefly-check` hook shells out to `pixi run -e dev`, which provisions a whole dev environment and can take >10 minutes on first run. Either provision the env first or use `SKIP=pyrefly-check` for commits that touch no Python.

---

## File Structure

**genoray**

| File | Responsibility | Change |
|---|---|---|
| `src/query/reader.rs` | Per-column search state | Replace `vk_snp_overlap`/`vk_indel_overlap` with `VkColumnIndex` + `vk_snp_index`/`vk_indel_index`; add `max_deletion_len` |
| `src/query/gather.rs` | Batch search core | Add `find_ranges_haps` (column-outer, rayon); rewire `find_ranges` |
| `src/query/union.rs` | Dense union | Add `dense_max_end_keys`; expose `DenseUnion::max_del` |
| `src/py_query_ranges.rs` | pyo3 bindings | Add `find_ranges_header`, `find_ranges_chunk` |
| `python/genoray/_svar2_batch.py` | Python query surface | Add `RangesChunk`, `RangesStream`, `_find_ranges_chunked`, `MAX_END_SHIFT` |
| `tests/test_ranges_split.rs` | Rust core tests | Add complexity + max-end-key tests |
| `tests/test_svar2_ranges.py` | Python API tests | Add chunk-equivalence + max-end tests |

**GenVarLoader**

| File | Responsibility | Change |
|---|---|---|
| `python/genvarloader/_dataset/_write.py` | Write pipeline | Add `_svar2_ranges_cache_bytes`, `_svar2_preflight`; rewrite `_write_from_svar2`; delete `_svar2_region_max_ends` |
| `pyproject.toml`, `pixi.toml` | Dependency floor | Bump genoray pin |
| `tests/dataset/test_write_svar2.py` | Write-path tests | Add preflight + chunking tests |
| `docs/source/format.md`, `docs/source/write.md`, `skills/genvarloader/SKILL.md` | User docs | Correct the "small" claim; document `max_mem` for SVAR2 |

## Task Dependency Graph

```
Task 0 (setup)
   |
   +-- Task 1 -> Task 2 -> Task 3 -> Task 4   [genoray chain]
   |
   +-- Task 5                                  [GenVarLoader, independent]
   |
   +-- Task 7                                  [docs, independent]

Task 4 + Task 5 -> Task 6                       [GenVarLoader integration]
```

**Parallel execution:** Tasks 1–4 (genoray chain), Task 5, and Task 7 are mutually independent and should be dispatched concurrently using superpowers:dispatching-parallel-agents with superpowers:subagent-driven-development. Task 6 waits for Task 4 and Task 5. Use Sonnet or weaker for implementation agents; reserve stronger models for second-pass fixes.

---

### Task 0: Setup — genoray tracking issue and worktree

**Files:**
- Create: genoray worktree at `/carter/users/dlaub/projects/genoray/.claude/worktrees/issue-333-find-ranges`

**Interfaces:**
- Produces: a genoray worktree on branch `worktree-issue-333-find-ranges`, and a genoray issue number to reference in commits.

- [ ] **Step 1: File the genoray tracking issue**

```bash
cd /carter/users/dlaub/projects/genoray
gh issue create \
  --title "find_ranges rebuilds a SearchTree per (region, column): O(regions x total_variants)" \
  --body "\`query::find_ranges\` loops region-outer / hap-inner and calls \`vk_snp_overlap\` / \`vk_indel_overlap\` per (region, column). Each call rebuilds region-independent state from scratch (\`src/query/reader.rs:244\`, \`:260\`):

\`\`\`rust
let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
let tree = SearchTree::new(positions);
\`\`\`

\`SearchTree::new\` is O(n) and allocates two Vecs sized to the column, so a batch of R regions over H columns does R*H*2 full tree builds and sweeps the packed store R times instead of once.

At the scale reported in mcvickerlab/GenVarLoader#333 (3,964 regions, 414,830 samples, ploidy 2) that is 6.6e9 tree builds. The caller sat at 0% for hours before being OOM-killed.

Fix: hoist the per-column state into a \`VkColumnIndex\` built once per column, invert to column-outer, parallelize with rayon. Also add a chunked API so the R*S*P payload can be produced under a memory budget, and fold per-region max-end computation into the same sweep.

Cross-repo: mcvickerlab/GenVarLoader#333"
```

Record the issue number; the genoray commits below reference it as `#<genoray-issue>`.

- [ ] **Step 2: Create the genoray worktree**

```bash
cd /carter/users/dlaub/projects/genoray
git fetch origin
git worktree add -b worktree-issue-333-find-ranges \
  .claude/worktrees/issue-333-find-ranges origin/main
```

- [ ] **Step 3: Install hooks and verify the baseline is green**

```bash
cd /carter/users/dlaub/projects/genoray/.claude/worktrees/issue-333-find-ranges
pixi run prek-install
pixi run -e lint test-rust 2>&1 | tail -20
```

Expected: all Rust tests pass. If `cargo test` fails to load `libpython`, prepend
`LD_LIBRARY_PATH=$PWD/.pixi/envs/lint/lib` to the command.

- [ ] **Step 4: Verify the Python baseline is green**

```bash
pixi run pytest tests/test_svar2_ranges.py tests/test_svar2_batch.py -q 2>&1 | tail -10
```

Expected: all pass. This is the byte-identity oracle for Task 1.

---

### Task 1: Hoist per-column search state; invert `find_ranges` to column-outer

**Files:**
- Modify: `src/query/reader.rs:241-284` (replace `vk_snp_overlap` / `vk_indel_overlap`)
- Modify: `src/query/gather.rs:338-393` (`find_ranges`)
- Test: `tests/test_ranges_split.rs`

**Interfaces:**
- Consumes: existing `ContigReader` fields `vk_snp`, `vk_indel`, `vk_indel_max_del`, `ploidy`, `n_samples`; `search::{SearchTree, overlap_range}`; `rvk::deletion_len`.
- Produces:
  - `pub(crate) struct VkColumnIndex` with `pub(crate) o0: usize` and `pub(crate) fn overlap(&self, q_start: u32, q_end: u32) -> Range<usize>`
  - `ContigReader::vk_snp_index(&self, col: usize) -> VkColumnIndex`
  - `ContigReader::vk_indel_index(&self, sample: usize, p: usize) -> VkColumnIndex`
  - `pub fn find_ranges_haps(reader: &ContigReader, regions: &[(u32, u32)], sample_cols: &[usize], hap_lo: usize, hap_hi: usize, out_snp: &mut [i64], out_indel: &mut [i64])`
  - `pub const PAR_COLUMN_THRESHOLD: usize = 64`
  - `find_ranges` keeps its exact existing signature and `RangesBundle` return contract.

- [ ] **Step 1: Write the failing complexity test**

Add to `tests/test_ranges_split.rs`:

```rust
/// `find_ranges` must build a bounded number of search trees regardless of how
/// many regions are queried. Before the column-outer rewrite this was
/// O(regions x columns): each `vk_*_overlap` call rebuilt the column's tree.
///
/// The fixture is deliberately small (2 samples x 2 ploidy = 4 columns, well
/// under `PAR_COLUMN_THRESHOLD`) so the serial path runs on this thread and
/// `search::search_tree_build_count` — a thread-local — stays observable.
#[test]
fn test_find_ranges_tree_builds_do_not_scale_with_regions() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    let one = vec![(0u32, 1_000_000u32)];
    let many: Vec<(u32, u32)> = (0..16).map(|i| (i * 20, i * 20 + 1_000_000)).collect();

    let b0 = search::search_tree_build_count();
    let _ = find_ranges(&reader, &one, None);
    let cost_one = search::search_tree_build_count() - b0;

    let b1 = search::search_tree_build_count();
    let _ = find_ranges(&reader, &many, None);
    let cost_many = search::search_tree_build_count() - b1;

    // Per-region dense-union/dense-snp/dense-indel trees are still built once
    // per region (3 per region, cheap and cohort-shared). The var_key channels
    // — the R*H term that made this O(regions x total_variants) — must not grow.
    let dense_growth = 3 * (many.len() - one.len());
    assert!(
        cost_many <= cost_one + dense_growth,
        "tree builds grew with region count: {cost_one} -> {cost_many} \
         (allowed growth {dense_growth})"
    );
}
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
cd /carter/users/dlaub/projects/genoray/.claude/worktrees/issue-333-find-ranges
pixi run -e lint cargo test --no-default-features --features conversion \
  --test test_ranges_split test_find_ranges_tree_builds_do_not_scale_with_regions -- --nocapture
```

Expected: FAIL — "tree builds grew with region count". With 4 columns and 2 channels the current code builds `R*8` var_key trees, so the count grows by 120 while only 45 is allowed.

- [ ] **Step 3: Add `VkColumnIndex` and the two constructors**

In `src/query/reader.rs`, delete `vk_snp_overlap` and `vk_indel_overlap` (lines 241–284) and add in their place:

```rust
/// Region-independent per-column search state for one var_key channel.
///
/// Built ONCE per column, then queried per region. Hoisting this out of the old
/// per-`(region, column)` `vk_*_overlap` methods is what turns `find_ranges`
/// from O(regions x columns) tree builds into O(columns): the packed store is
/// swept once instead of once per region.
pub(crate) struct VkColumnIndex {
    /// Absolute base offset of this column in the channel's packed arrays.
    pub(crate) o0: usize,
    /// `None` for an empty column — `SearchTree`/`overlap_range` are not
    /// defined over an empty position array, matching the old early return.
    inner: Option<(SearchTree, Vec<u32>)>,
    max_del: u32,
}

impl VkColumnIndex {
    /// Absolute `[start, end)` into the channel's packed positions/keys for one
    /// region. Every element of the returned range truly overlaps
    /// `[q_start, q_end)` — `overlap_range` does the left-overlap sub-scan.
    pub(crate) fn overlap(&self, q_start: u32, q_end: u32) -> Range<usize> {
        let Some((tree, v_ends)) = &self.inner else {
            return self.o0..self.o0;
        };
        let (s, e) = overlap_range(tree, v_ends, self.max_del, q_start, q_end);
        (self.o0 + s)..(self.o0 + e)
    }

    /// Absolute index of the highest-position variant overlapping the region,
    /// or `None` when the region is empty for this column. Positions are sorted
    /// within a column and the range is contiguous, so that is its last element.
    pub(crate) fn last_overlapping(&self, q_start: u32, q_end: u32) -> Option<usize> {
        let r = self.overlap(q_start, q_end);
        (r.end > r.start).then(|| r.end - 1)
    }
}

impl ContigReader {
    /// SNP-channel column index. SNP `v_end = pos + 1` and `max_region_length =
    /// 0`, since a SNP spans exactly one base.
    pub(crate) fn vk_snp_index(&self, col: usize) -> VkColumnIndex {
        let vk_range = self.vk_snp.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &self.vk_snp.positions()[o0..o1];
        if positions.is_empty() {
            return VkColumnIndex { o0, inner: None, max_del: 0 };
        }
        let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        VkColumnIndex {
            o0,
            inner: Some((SearchTree::new(positions), v_ends)),
            max_del: 0,
        }
    }

    /// Indel-channel column index for `(sample, p)`. `v_end = pos + 1 +
    /// deletion_len(key)`; the search bound is this column's `max_del`.
    pub(crate) fn vk_indel_index(&self, sample: usize, p: usize) -> VkColumnIndex {
        let col = sample * self.ploidy + p;
        let vk_range = self.vk_indel.column(col);
        let (o0, o1) = (vk_range.start, vk_range.end);
        let positions = &self.vk_indel.positions()[o0..o1];
        if positions.is_empty() {
            return VkColumnIndex { o0, inner: None, max_del: 0 };
        }
        let keys = &as_u32(&self.vk_indel.keys)[o0..o1];
        let v_ends: Vec<u32> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| pos + 1 + rvk::deletion_len(keys[i]))
            .collect();
        VkColumnIndex {
            o0,
            inner: Some((SearchTree::new(positions), v_ends)),
            max_del: self.vk_indel_max_del[[sample, p]],
        }
    }
}
```

- [ ] **Step 4: Add `find_ranges_haps` and rewire `find_ranges`**

In `src/query/gather.rs`, add `use rayon::prelude::*;` to the imports, then add before `find_ranges`:

```rust
/// Below this many columns the serial path runs instead of rayon's: fork/join
/// overhead dominates for small batches, and staying on the caller's thread
/// keeps `search::search_tree_build_count` (a thread-local) observable in tests.
pub const PAR_COLUMN_THRESHOLD: usize = 64;

/// Fill hap-major `[hap_lo, hap_hi)` slices of the two var_key range channels.
///
/// `out_snp` / `out_indel` are `(hap_hi - hap_lo, R, 2)` row-major `i64` — one
/// contiguous `R * 2` run per hap, which is exactly what lets rayon hand each
/// column a disjoint `par_chunks_mut` slice. The hap axis indexes the SELECTED
/// samples: hap `h` is `(sample_cols[h / ploidy], h % ploidy)`, matching the
/// sample-major-then-ploid order `find_ranges` has always produced.
///
/// Column-outer / region-inner, so each column's `VkColumnIndex` is built
/// exactly once.
pub fn find_ranges_haps(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    sample_cols: &[usize],
    hap_lo: usize,
    hap_hi: usize,
    out_snp: &mut [i64],
    out_indel: &mut [i64],
) {
    let ploidy = reader.ploidy;
    let r = regions.len();
    let n_haps = hap_hi - hap_lo;
    assert_eq!(out_snp.len(), n_haps * r * 2, "out_snp must be (n_haps, R, 2)");
    assert_eq!(out_indel.len(), n_haps * r * 2, "out_indel must be (n_haps, R, 2)");
    if n_haps == 0 || r == 0 {
        return;
    }

    let fill = |h_off: usize, snp_row: &mut [i64], indel_row: &mut [i64]| {
        let h = hap_lo + h_off;
        let s = sample_cols[h / ploidy];
        let p = h % ploidy;
        let snp_ix = reader.vk_snp_index(s * ploidy + p);
        let indel_ix = reader.vk_indel_index(s, p);
        for (ri, &(qs, qe)) in regions.iter().enumerate() {
            let a = snp_ix.overlap(qs, qe);
            snp_row[ri * 2] = a.start as i64;
            snp_row[ri * 2 + 1] = a.end as i64;
            let b = indel_ix.overlap(qs, qe);
            indel_row[ri * 2] = b.start as i64;
            indel_row[ri * 2 + 1] = b.end as i64;
        }
    };

    if n_haps < PAR_COLUMN_THRESHOLD {
        for (h_off, (snp_row, indel_row)) in out_snp
            .chunks_mut(r * 2)
            .zip(out_indel.chunks_mut(r * 2))
            .enumerate()
        {
            fill(h_off, snp_row, indel_row);
        }
    } else {
        out_snp
            .par_chunks_mut(r * 2)
            .zip(out_indel.par_chunks_mut(r * 2))
            .enumerate()
            .for_each(|(h_off, (snp_row, indel_row))| fill(h_off, snp_row, indel_row));
    }
}
```

Then replace the `vk_snp_range` / `vk_indel_range` construction in `find_ranges`
(currently `src/query/gather.rs:369-379`) with:

```rust
    // `find_ranges_haps` fills hap-major because that is the layout rayon can
    // split into disjoint slices. `RangesBundle` is region-major and is replayed
    // by `gather_ranges` unchanged, so transpose here. This costs one extra copy
    // of the payload; `find_ranges` is the small-batch read-path entry point,
    // while the population-scale writer uses the chunked API and never builds a
    // bundle at all.
    let mut snp_flat = vec![0i64; h * n_regions * 2];
    let mut indel_flat = vec![0i64; h * n_regions * 2];
    find_ranges_haps(
        reader, regions, &sample_cols, 0, h, &mut snp_flat, &mut indel_flat,
    );

    let mut vk_snp_range: Vec<Range<usize>> = Vec::with_capacity(n_regions * h);
    let mut vk_indel_range: Vec<Range<usize>> = Vec::with_capacity(n_regions * h);
    for ri in 0..n_regions {
        for hh in 0..h {
            let k = (hh * n_regions + ri) * 2;
            vk_snp_range.push(snp_flat[k] as usize..snp_flat[k + 1] as usize);
            vk_indel_range.push(indel_flat[k] as usize..indel_flat[k + 1] as usize);
        }
    }
```

Export the new items from `src/query/mod.rs` alongside the existing `find_ranges`
export: `pub use gather::{..., find_ranges_haps, PAR_COLUMN_THRESHOLD};`

- [ ] **Step 5: Run the complexity test — expect PASS**

```bash
pixi run -e lint cargo test --no-default-features --features conversion \
  --test test_ranges_split test_find_ranges_tree_builds_do_not_scale_with_regions -- --nocapture
```

Expected: PASS.

- [ ] **Step 6: Run the full Rust suite — nothing else may change**

```bash
pixi run -e lint test-rust 2>&1 | tail -20
```

Expected: all pass, including the pre-existing `test_find_ranges_dense_range_matches_overlap_batch`, `test_gather_ranges_reproduces_overlap_batch_field_for_field`, and the `test_readbound_gather.rs` tree-count assertions. These are the byte-identity oracle: `find_ranges` output must be unchanged.

- [ ] **Step 7: Rebuild and run the Python suite**

```bash
pixi run maturin develop --release 2>&1 | tail -5
pixi run pytest tests/test_svar2_ranges.py tests/test_svar2_batch.py \
  tests/test_py_ranges_readbound.py -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/query/reader.rs src/query/gather.rs src/query/mod.rs tests/test_ranges_split.rs
git commit -m "perf(query): build each var_key column's search tree once

find_ranges looped region-outer and rebuilt a SearchTree plus a v_ends
vector inside vk_snp_overlap/vk_indel_overlap for every (region, column)
pair, making a batch O(regions x total_variants) and sweeping the packed
store once per region.

Hoist the region-independent state into VkColumnIndex, invert the loop to
column-outer in the new find_ranges_haps, and parallelize with rayon over
disjoint par_chunks_mut slices above PAR_COLUMN_THRESHOLD columns.
find_ranges keeps its exact RangesBundle contract by transposing the
hap-major fill back to region-major.

Closes #<genoray-issue>
Relates to mcvickerlab/GenVarLoader#333"
```

---

### Task 2: Per-region max-end composite keys in the same sweep

**Files:**
- Modify: `src/query/gather.rs` (`find_ranges_haps` signature and body)
- Modify: `src/query/reader.rs` (`ContigReader::max_deletion_len`)
- Modify: `src/query/union.rs` (`DenseUnion::max_del` accessor, `dense_max_end_keys`)
- Modify: `src/query/mod.rs` (exports)
- Test: `tests/test_ranges_split.rs`

**Interfaces:**
- Consumes: `VkColumnIndex::last_overlapping`, `ContigReader::{vk_snp, vk_indel, dense_union, dense_view, ploidy}`, `DenseUnion::{refs, src, v_ends}`, `rvk::deletion_len`.
- Produces:
  - `pub const MAX_END_SHIFT: u32 = 21;`
  - `find_ranges_haps` now **returns** `Vec<u64>` of length `regions.len()` — the per-region max composite key over this hap slice. `0` means no variant. Signature otherwise unchanged.
  - `pub fn dense_max_end_keys(reader: &ContigReader, regions: &[(u32, u32)], dense_range: &[Range<usize>], sample_cols: &[usize], all_samples: bool) -> Vec<u64>`
  - `ContigReader::max_deletion_len(&self) -> u32`

- [ ] **Step 1: Write the failing max-end test**

Add to `tests/test_ranges_split.rs`. `synth_reader` builds chr1 with SNP@100 (S0 hap0), INS@200 (S0 hap1, S1 both), DEL@300 `AT>A` (S0 both, S1 hap1) — so `deletion_len = 1` and the DEL's end is `300 + 1 + 1 = 302`.

```rust
use genoray_core::query::{dense_max_end_keys, find_ranges_haps, MAX_END_SHIFT};

fn unpack_end(key: u64) -> u32 {
    ((key >> MAX_END_SHIFT) + (key & ((1 << MAX_END_SHIFT) - 1))) as u32
}

/// The per-region max end must be the end of the HIGHEST-POSITION overlapping
/// variant (ties broken by the larger end), not the largest end overall — this
/// is the SVAR1-parity rule GenVarLoader's `_svar2_region_max_ends` implements.
#[test]
fn test_max_end_keys_pick_highest_position_variant() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    // Region covering all three variants; the DEL at 300 is highest-position.
    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy;
    let mut snp = vec![0i64; h * 2];
    let mut indel = vec![0i64; h * 2];
    let vk_keys = find_ranges_haps(
        &reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel,
    );

    let dense = reader.dense_union();
    let dense_range: Vec<_> = regions.iter().map(|&(a, b)| dense.overlap(a, b)).collect();
    let dense_keys =
        dense_max_end_keys(&reader, &regions, &dense_range, &sample_cols, true);

    let key = vk_keys[0].max(dense_keys[0]);
    assert_ne!(key, 0, "region has variants, so the key must be non-zero");
    assert_eq!(unpack_end(key), 302, "DEL@300 with deletion_len 1 ends at 302");
}

/// A region containing only the SNP must report that SNP's end, and an empty
/// region must report the 0 sentinel so the caller keeps its original chromEnd.
#[test]
fn test_max_end_keys_snp_only_and_empty_region() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    let regions = vec![(90u32, 110u32), (900u32, 950u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy;
    let mut snp = vec![0i64; h * regions.len() * 2];
    let mut indel = vec![0i64; h * regions.len() * 2];
    let vk_keys = find_ranges_haps(
        &reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel,
    );

    let dense = reader.dense_union();
    let dense_range: Vec<_> = regions.iter().map(|&(a, b)| dense.overlap(a, b)).collect();
    let dense_keys =
        dense_max_end_keys(&reader, &regions, &dense_range, &sample_cols, true);

    let k0 = vk_keys[0].max(dense_keys[0]);
    assert_eq!(unpack_end(k0), 101, "SNP@100 ends at 101");
    assert_eq!(vk_keys[1].max(dense_keys[1]), 0, "no variants in [900, 950)");
}

/// Splitting the hap axis must not change the reduced result — the writer
/// reduces per-chunk keys with an elementwise max.
#[test]
fn test_max_end_keys_reduce_across_hap_slices() {
    let tmp = tempdir().unwrap();
    let out = tmp.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let reader = synth_reader(&out);

    let regions = vec![(0u32, 1_000u32)];
    let sample_cols: Vec<usize> = (0..2).collect();
    let h = 2 * reader.ploidy;

    let mut snp = vec![0i64; h * 2];
    let mut indel = vec![0i64; h * 2];
    let whole = find_ranges_haps(
        &reader, &regions, &sample_cols, 0, h, &mut snp, &mut indel,
    );

    let mut reduced = vec![0u64; 1];
    for lo in (0..h).step_by(1) {
        let mut s = vec![0i64; 2];
        let mut i = vec![0i64; 2];
        let part = find_ranges_haps(
            &reader, &regions, &sample_cols, lo, lo + 1, &mut s, &mut i,
        );
        reduced[0] = reduced[0].max(part[0]);
    }
    assert_eq!(whole, reduced);
}
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pixi run -e lint cargo test --no-default-features --features conversion \
  --test test_ranges_split max_end 2>&1 | tail -20
```

Expected: FAIL to compile — `dense_max_end_keys`, `MAX_END_SHIFT` unresolved, and `find_ranges_haps` returns `()`.

- [ ] **Step 3: Make `find_ranges_haps` return per-region max keys**

In `src/query/gather.rs`, add the constant and change `find_ranges_haps`:

```rust
/// Bit width reserved for `ext` in the packed max-end key `(pos << SHIFT) | ext`,
/// where `ext = 1 + deletion_len` so that `end = pos + ext`. Packing the small,
/// bounded `ext` (rather than the absolute end) makes an integer max over the
/// key order by position first and end second — the SVAR1 tie-break rule. Fixed
/// by GenVarLoader's existing `_svar2_region_max_ends`; do not change.
pub const MAX_END_SHIFT: u32 = 21;
```

Change the signature's return type to `-> Vec<u64>` and the body's tail. The
`fill` closure gains a `&mut [u64]` accumulator:

```rust
    let fill = |h_off: usize, snp_row: &mut [i64], indel_row: &mut [i64], acc: &mut [u64]| {
        let h = hap_lo + h_off;
        let s = sample_cols[h / ploidy];
        let p = h % ploidy;
        let snp_ix = reader.vk_snp_index(s * ploidy + p);
        let indel_ix = reader.vk_indel_index(s, p);
        let snp_pos = reader.vk_snp.positions();
        let indel_pos = reader.vk_indel.positions();
        let indel_keys = as_u32(&reader.vk_indel.keys);
        for (ri, &(qs, qe)) in regions.iter().enumerate() {
            let a = snp_ix.overlap(qs, qe);
            snp_row[ri * 2] = a.start as i64;
            snp_row[ri * 2 + 1] = a.end as i64;
            let b = indel_ix.overlap(qs, qe);
            indel_row[ri * 2] = b.start as i64;
            indel_row[ri * 2 + 1] = b.end as i64;

            // Positions are sorted within a column and the range is contiguous,
            // so the last element is the highest-position overlapping variant.
            let mut k = 0u64;
            if a.end > a.start {
                let pos = snp_pos[a.end - 1] as u64;
                k = k.max((pos << MAX_END_SHIFT) | 1); // SNP/INS: ext = 1
            }
            if b.end > b.start {
                let i = b.end - 1;
                let pos = indel_pos[i] as u64;
                let ext = 1 + rvk::deletion_len(indel_keys[i]) as u64;
                k = k.max((pos << MAX_END_SHIFT) | ext);
            }
            acc[ri] = acc[ri].max(k);
        }
    };

    if n_haps < PAR_COLUMN_THRESHOLD {
        let mut acc = vec![0u64; r];
        for (h_off, (snp_row, indel_row)) in out_snp
            .chunks_mut(r * 2)
            .zip(out_indel.chunks_mut(r * 2))
            .enumerate()
        {
            fill(h_off, snp_row, indel_row, &mut acc);
        }
        acc
    } else {
        out_snp
            .par_chunks_mut(r * 2)
            .zip(out_indel.par_chunks_mut(r * 2))
            .enumerate()
            .fold(
                || vec![0u64; r],
                |mut acc, (h_off, (snp_row, indel_row))| {
                    fill(h_off, snp_row, indel_row, &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![0u64; r],
                |mut a, b| {
                    for i in 0..r {
                        a[i] = a[i].max(b[i]);
                    }
                    a
                },
            )
    }
```

Also change the two early returns: `if n_haps == 0 || r == 0 { return vec![0u64; r]; }`.

Add `use crate::rvk;` and the `as_u32` import to `gather.rs` if not already present.

- [ ] **Step 4: Add `dense_max_end_keys`**

In `src/query/union.rs`, add a `max_del` accessor on `DenseUnion`:

```rust
impl DenseUnion {
    /// The per-contig dense deletion bound, for the caller's overflow preflight.
    pub(crate) fn max_del(&self) -> u32 {
        self.max_del
    }
}
```

Then add, in the same file:

```rust
/// Per-region max `(pos << MAX_END_SHIFT) | ext` over the DENSE channel,
/// restricted to variants carried by at least one selected hap. `0` when the
/// region has no such variant.
///
/// The dense genotype matrix is hap-major (`hap * n_dense_variants + col`), so a
/// "is this variant carried by anyone selected?" probe is strided across haps.
/// Two things keep that cheap:
///
/// * The walk runs BACKWARD from the end of the region's dense window and stops
///   once it drops below the position of the first carried variant it found.
///   Dense variants are common by construction, so this almost always terminates
///   on the first index.
/// * `all_samples` skips the carriage probe entirely: every dense variant in the
///   store has at least one carrier among all samples, so the last truly
///   overlapping variant is the answer. This is the path `gvl.write` takes.
///
/// The whole tied run at the winning position is scanned rather than stopping at
/// the first hit: within a class the table's order is not by `ext`, so a later
/// same-position variant can carry a longer deletion.
pub fn dense_max_end_keys(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    dense_range: &[Range<usize>],
    sample_cols: &[usize],
    all_samples: bool,
) -> Vec<u64> {
    let ploidy = reader.ploidy;
    let dense = reader.dense_union();
    let mut out = vec![0u64; regions.len()];

    for (ri, &(qs, _)) in regions.iter().enumerate() {
        let (ds, de) = (dense_range[ri].start, dense_range[ri].end);
        let mut best = 0u64;
        let mut best_pos: Option<u32> = None;
        let mut j = de;
        while j > ds {
            j -= 1;
            let pos = dense.refs[j].position;
            if let Some(bp) = best_pos {
                if pos < bp {
                    break; // every remaining index has a lower position
                }
            }
            if dense.v_ends[j] <= qs {
                continue; // no true left-overlap
            }
            let carried = all_samples || {
                let (class, dcol) = dense.src[j];
                let view = reader
                    .dense_view(class)
                    .expect("dense src implies table");
                sample_cols
                    .iter()
                    .any(|&s| (0..ploidy).any(|p| view.carried(s * ploidy + p, dcol)))
            };
            if !carried {
                continue;
            }
            let ext = (dense.v_ends[j] - pos) as u64;
            best = best.max(((pos as u64) << MAX_END_SHIFT) | ext);
            best_pos = Some(pos);
        }
        out[ri] = best;
    }
    out
}
```

Add `use crate::query::gather::MAX_END_SHIFT;` to `union.rs`.

- [ ] **Step 5: Add the overflow preflight accessor**

In `src/query/reader.rs`:

```rust
impl ContigReader {
    /// The largest deletion span on this contig across both the per-hap indel
    /// channel and the dense union. Callers packing max-end keys must check
    /// `1 + max_deletion_len() < (1 << MAX_END_SHIFT)` before doing so — a
    /// pathological >~2 Mb deletion footprint would otherwise silently corrupt
    /// the packed key.
    pub fn max_deletion_len(&self) -> u32 {
        let vk = self.vk_indel_max_del.iter().copied().max().unwrap_or(0);
        vk.max(self.dense_union().max_del())
    }
}
```

Export from `src/query/mod.rs`: `pub use gather::MAX_END_SHIFT;` and
`pub use union::dense_max_end_keys;`.

- [ ] **Step 6: Run the new tests — expect PASS**

```bash
pixi run -e lint cargo test --no-default-features --features conversion \
  --test test_ranges_split 2>&1 | tail -20
```

Expected: all pass, including Task 1's tests.

- [ ] **Step 7: Run the full Rust suite**

```bash
pixi run -e lint test-rust 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/query/gather.rs src/query/union.rs src/query/reader.rs \
        src/query/mod.rs tests/test_ranges_split.rs
git commit -m "feat(query): compute per-region max-end keys during the range sweep

find_ranges_haps now also returns the per-region max
(pos << 21) | (1 + deletion_len) composite key over its hap slice, taken
from the last element of each channel's range -- free, since the range was
just computed. dense_max_end_keys does the same for the cohort-shared dense
channel via a backward walk with an all-samples fast path.

Packed keys (not unpacked ends) are the reduction unit: the SVAR1 rule is
max by position THEN end, so reducing ends across hap slices would pick the
wrong variant when a lower-position deletion reaches further.

Lets consumers stop decoding every sample just to extend chromEnd.

Relates to #<genoray-issue>, mcvickerlab/GenVarLoader#333"
```

---

### Task 3: Chunked pyo3 bindings

**Files:**
- Modify: `src/py_query_ranges.rs`
- Test: `tests/test_svar2_ranges.py`

**Interfaces:**
- Consumes: `find_ranges_haps`, `dense_max_end_keys`, `MAX_END_SHIFT`, `ContigReader::max_deletion_len`, existing `bundle_to_dict` helpers.
- Produces two new `PyContigReader` methods:
  - `find_ranges_header(regions, samples) -> dict` with keys `region_starts` (i32, R), `dense_range` (i32, (R,2)), `dense_snp_range` (i32, (R,2)), `dense_indel_range` (i32, (R,2)), `sample_cols` (i64, S), `dense_max_end_keys` (i64, R), `n_regions`, `n_samples`, `ploidy`
  - `find_ranges_chunk(regions, samples, hap_lo, hap_hi) -> dict` with keys `vk_snp_range` (i64, (n_haps*R, 2)), `vk_indel_range` (i64, (n_haps*R, 2)), `max_end_keys` (i64, R), `hap_lo`, `hap_hi`

- [ ] **Step 1: Write the failing Python test**

Add to `tests/test_svar2_ranges.py`:

```python
def test_find_ranges_chunk_matches_find_ranges(svar2_store: Path):
    """Chunked hap slices must reassemble into the region-major bundle exactly."""
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    reg = list(zip(starts, ends))
    reader = sv._reader("chr1")
    bundle = sv._find_ranges("chr1", starts, ends)

    R = len(reg)
    P = sv.ploidy
    S = sv.n_samples
    H = S * P

    header = reader.find_ranges_header(reg, None)
    np.testing.assert_array_equal(
        np.asarray(header["dense_snp_range"]), np.asarray(bundle["dense_snp_range"])
    )
    np.testing.assert_array_equal(
        np.asarray(header["sample_cols"]), np.asarray(bundle["sample_cols"])
    )

    # One hap per call: the most adversarial chunking.
    snp = np.empty((H, R, 2), np.int64)
    indel = np.empty((H, R, 2), np.int64)
    for h in range(H):
        d = reader.find_ranges_chunk(reg, None, h, h + 1)
        snp[h] = np.asarray(d["vk_snp_range"]).reshape(1, R, 2)
        indel[h] = np.asarray(d["vk_indel_range"]).reshape(1, R, 2)

    # bundle vk ranges are region-major (R*H, 2); ours are hap-major (H, R, 2).
    np.testing.assert_array_equal(
        snp.transpose(1, 0, 2).reshape(R * H, 2),
        np.asarray(bundle["vk_snp_range"]),
    )
    np.testing.assert_array_equal(
        indel.transpose(1, 0, 2).reshape(R * H, 2),
        np.asarray(bundle["vk_indel_range"]),
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pixi run pytest tests/test_svar2_ranges.py::test_find_ranges_chunk_matches_find_ranges -q
```

Expected: FAIL — `PyContigReader` has no attribute `find_ranges_header`.

- [ ] **Step 3: Implement the two bindings**

In `src/py_query_ranges.rs`, extend the imports and add to the `#[pymethods]`
block:

```rust
use crate::query::{
    BatchResult, MAX_END_SHIFT, RangesBundle, dense_max_end_keys, find_ranges,
    find_ranges_haps, gather_ranges, read_ranges,
};
use pyo3::exceptions::PyValueError;

// ... inside impl PyContigReader ...

    /// Region-level half of a chunked `find_ranges`: everything whose size is
    /// O(regions) rather than O(regions * samples * ploidy), plus the dense
    /// channel's max-end contribution. Cheap enough to compute eagerly.
    pub fn find_ranges_header<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Fail fast rather than silently corrupting a packed key. `ext` is
        // 1 + deletion_len and must fit below the position field.
        let max_del = self.inner.max_deletion_len();
        if (1u64 + max_del as u64) >= (1u64 << MAX_END_SHIFT) {
            return Err(PyValueError::new_err(
                "variant footprint exceeds tie-break packing width",
            ));
        }

        let all_samples = samples.is_none();
        let sample_cols: Vec<usize> = match &samples {
            Some(s) => s.clone(),
            None => (0..self.inner.n_samples).collect(),
        };

        let dense = self.inner.dense_union();
        let dense_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| dense.overlap(qs, qe))
            .collect();
        let dense_snp_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| self.inner.dense_snp_overlap(qs, qe))
            .collect();
        let dense_indel_range: Vec<Range<usize>> = regions
            .iter()
            .map(|&(qs, qe)| self.inner.dense_indel_overlap(qs, qe))
            .collect();
        let region_starts: Vec<u32> = regions.iter().map(|&(qs, _)| qs).collect();
        let dmax = dense_max_end_keys(
            &self.inner, &regions, &dense_range, &sample_cols, all_samples,
        );

        let pairs_i32 = |v: &[Range<usize>]| -> Vec<i32> {
            let mut o = Vec::with_capacity(v.len() * 2);
            for r in v {
                o.push(r.start as i32);
                o.push(r.end as i32);
            }
            o
        };
        let to2d = |v: Vec<i32>| {
            Array2::from_shape_vec((regions.len(), 2), v)
                .expect("region pair shape")
                .to_pyarray(py)
        };

        let d = PyDict::new(py);
        d.set_item("dense_range", to2d(pairs_i32(&dense_range)))?;
        d.set_item("dense_snp_range", to2d(pairs_i32(&dense_snp_range)))?;
        d.set_item("dense_indel_range", to2d(pairs_i32(&dense_indel_range)))?;
        d.set_item("region_starts", u32_to_i32_pyarray(py, &region_starts))?;
        let cols: Vec<i64> = sample_cols.iter().map(|&x| x as i64).collect();
        d.set_item("sample_cols", PyArray1::from_slice(py, &cols))?;
        let dmax_i64: Vec<i64> = dmax.iter().map(|&x| x as i64).collect();
        d.set_item("dense_max_end_keys", PyArray1::from_slice(py, &dmax_i64))?;
        d.set_item("n_regions", regions.len())?;
        d.set_item("n_samples", sample_cols.len())?;
        d.set_item("ploidy", self.inner.ploidy)?;
        Ok(d)
    }

    /// One hap slice `[hap_lo, hap_hi)` of a chunked `find_ranges`. Fills freshly
    /// allocated numpy arrays IN PLACE, so the payload exists exactly once —
    /// unlike `find_ranges`, whose `Vec<Range<usize>>` -> `Vec<i64>` ->
    /// `ToPyArray` chain holds three copies at peak. Releases the GIL for the
    /// search so rayon and the caller's progress bar can both run.
    ///
    /// `vk_snp_range` / `vk_indel_range` come back hap-major, shape
    /// `(n_haps * R, 2)`; reshape to `(n_haps_samples, ploidy, R, 2)` in Python.
    pub fn find_ranges_chunk<'py>(
        &self,
        py: Python<'py>,
        regions: Vec<(u32, u32)>,
        samples: Option<Vec<usize>>,
        hap_lo: usize,
        hap_hi: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sample_cols: Vec<usize> = match &samples {
            Some(s) => s.clone(),
            None => (0..self.inner.n_samples).collect(),
        };
        let h_total = sample_cols.len() * self.inner.ploidy;
        if hap_lo > hap_hi || hap_hi > h_total {
            return Err(PyValueError::new_err(format!(
                "hap slice [{hap_lo}, {hap_hi}) out of bounds for {h_total} haps"
            )));
        }
        let n_haps = hap_hi - hap_lo;
        let r = regions.len();

        let snp = PyArray2::<i64>::zeros(py, [n_haps * r, 2], false);
        let indel = PyArray2::<i64>::zeros(py, [n_haps * r, 2], false);
        let max_keys = {
            let mut snp_rw = snp.readwrite();
            let mut indel_rw = indel.readwrite();
            let snp_s = snp_rw.as_slice_mut()?;
            let indel_s = indel_rw.as_slice_mut()?;
            py.detach(|| {
                find_ranges_haps(
                    &self.inner, &regions, &sample_cols, hap_lo, hap_hi, snp_s, indel_s,
                )
            })
        };

        let keys_i64: Vec<i64> = max_keys.iter().map(|&x| x as i64).collect();
        let d = PyDict::new(py);
        d.set_item("vk_snp_range", snp)?;
        d.set_item("vk_indel_range", indel)?;
        d.set_item("max_end_keys", PyArray1::from_slice(py, &keys_i64))?;
        d.set_item("hap_lo", hap_lo)?;
        d.set_item("hap_hi", hap_hi)?;
        Ok(d)
    }
```

**If `py.detach` fails to compile** because the `&mut [i64]` slices are not
`Ungil`: compute into local `Vec<i64>`s inside `py.detach`, then copy into the
numpy arrays afterward. That costs one extra chunk-sized copy — still bounded,
still far better than today's three copies of the whole contig. Do not abandon
the GIL release to keep the in-place fill; the GIL release is the more important
of the two.

- [ ] **Step 4: Rebuild and run the test — expect PASS**

```bash
pixi run maturin develop --release 2>&1 | tail -5
pixi run pytest tests/test_svar2_ranges.py::test_find_ranges_chunk_matches_find_ranges -q
```

Expected: PASS.

- [ ] **Step 5: Run the full genoray suite**

```bash
pixi run -e lint test-rust 2>&1 | tail -10
pixi run test 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/py_query_ranges.rs tests/test_svar2_ranges.py
git commit -m "feat(query): add chunked find_ranges bindings

find_ranges_header returns the O(regions) arrays plus the dense channel's
max-end keys; find_ranges_chunk returns one hap slice of the var_key
payload. The chunk binding fills freshly allocated numpy arrays in place
and releases the GIL, so peak memory is one copy of the chunk instead of
three copies of the whole contig.

Also preflights the max-end key packing width against the contig's largest
deletion, raising ValueError rather than silently corrupting a key.

Relates to #<genoray-issue>, mcvickerlab/GenVarLoader#333"
```

---

### Task 4: Python `RangesStream` / `_find_ranges_chunked`

**Files:**
- Modify: `python/genoray/_svar2_batch.py`
- Test: `tests/test_svar2_ranges.py`

**Interfaces:**
- Consumes: `PyContigReader.find_ranges_header`, `PyContigReader.find_ranges_chunk`.
- Produces (all importable from `genoray._svar2_batch`):
  - `MAX_END_SHIFT: int = 21`
  - `RangesChunk` frozen dataclass: `sample_start: int`, `n_samples: int`, `vk_snp_range: NDArray[np.int64]` shape `(n_samples, ploidy, R, 2)`, `vk_indel_range` same shape, `max_end_keys: NDArray[np.int64]` shape `(R,)`
  - `RangesStream` frozen dataclass: `n_regions`, `n_samples`, `ploidy`, `samples_per_chunk`, `region_starts`, `dense_range`, `dense_snp_range`, `dense_indel_range`, `sample_cols`, `dense_max_end_keys`, `chunks: Iterator[RangesChunk]`
  - `SparseVar2._find_ranges_chunked(contig, starts, ends, samples=None, *, max_mem=None) -> RangesStream`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_svar2_ranges.py`:

```python
import pytest

from genoray._svar2_batch import MAX_END_SHIFT


def _reassemble(stream) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R, P, S = stream.n_regions, stream.ploidy, stream.n_samples
    snp = np.empty((S, P, R, 2), np.int64)
    indel = np.empty((S, P, R, 2), np.int64)
    keys = stream.dense_max_end_keys.copy()
    for ch in stream.chunks:
        s0, s1 = ch.sample_start, ch.sample_start + ch.n_samples
        snp[s0:s1] = ch.vk_snp_range
        indel[s0:s1] = ch.vk_indel_range
        np.maximum(keys, ch.max_end_keys, out=keys)
    return snp, indel, keys


@pytest.mark.parametrize("max_mem", [None, 1 << 30, 1])
def test_chunked_matches_find_ranges(svar2_store: Path, max_mem):
    """Every chunking, including one sample per chunk, reassembles identically."""
    sv = SparseVar2(svar2_store)
    starts, ends = [0, 5], [40, 20]
    bundle = sv._find_ranges("chr1", starts, ends)
    R, P, S = 2, sv.ploidy, sv.n_samples

    if max_mem == 1:
        # 1 byte cannot fit a sample; the API must say so rather than silently
        # producing a zero-sized chunk.
        with pytest.raises(ValueError, match="max_mem"):
            sv._find_ranges_chunked("chr1", starts, ends, max_mem=max_mem)
        return

    stream = sv._find_ranges_chunked("chr1", starts, ends, max_mem=max_mem)
    snp, indel, _ = _reassemble(stream)
    np.testing.assert_array_equal(
        snp.reshape(S * P, R, 2).transpose(1, 0, 2).reshape(R * S * P, 2),
        np.asarray(bundle["vk_snp_range"]),
    )
    np.testing.assert_array_equal(
        indel.reshape(S * P, R, 2).transpose(1, 0, 2).reshape(R * S * P, 2),
        np.asarray(bundle["vk_indel_range"]),
    )


def test_chunked_max_end_keys_unpack_to_variant_ends(svar2_store: Path):
    """The reduced key unpacks to the end of the highest-position variant.

    The fixture's chr1 carries SNP@2, INS@6 and DEL@11 (ilen -2, so it ends at
    11 + 1 + 2 = 14). Region [0, 40) therefore ends at 14; region [0, 5) sees
    only SNP@2, which ends at 3.
    """
    sv = SparseVar2(svar2_store)
    stream = sv._find_ranges_chunked("chr1", [0, 0], [40, 5])
    _, _, keys = _reassemble(stream)
    mask = (1 << MAX_END_SHIFT) - 1
    ends = (keys >> MAX_END_SHIFT) + (keys & mask)
    assert keys[0] != 0 and keys[1] != 0
    assert int(ends[0]) == 14
    assert int(ends[1]) == 3


def test_chunked_sample_subset(svar2_store: Path):
    """A sample subset takes the carriage-probing dense path, not the fast path."""
    sub = [SparseVar2(svar2_store).available_samples[1]]
    sv = SparseVar2(svar2_store)
    bundle = sv._find_ranges("chr1", [0], [40], samples=sub)
    stream = sv._find_ranges_chunked("chr1", [0], [40], samples=sub)
    assert stream.n_samples == 1
    snp, _, _ = _reassemble(stream)
    np.testing.assert_array_equal(
        snp.reshape(-1, 2), np.asarray(bundle["vk_snp_range"])
    )
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pixi run pytest tests/test_svar2_ranges.py -q -k chunked
```

Expected: FAIL — `SparseVar2` has no attribute `_find_ranges_chunked`.

- [ ] **Step 3: Implement the Python layer**

In `python/genoray/_svar2_batch.py`, add near the top:

```python
from collections.abc import Iterator
from dataclasses import dataclass

#: Bit width reserved for ``ext`` in a packed max-end key. Mirrors Rust's
#: ``query::MAX_END_SHIFT``; consumers unpack with
#: ``end = (key >> MAX_END_SHIFT) + (key & ((1 << MAX_END_SHIFT) - 1))``.
MAX_END_SHIFT = 21


@dataclass(frozen=True)
class RangesChunk:
    """One hap slice of a chunked ``_find_ranges``.

    Attributes:
        sample_start: Offset of this chunk on the SELECTED sample axis.
        n_samples: Number of selected samples in this chunk.
        vk_snp_range: Shape ``(n_samples, ploidy, n_regions, 2)``, hap-major.
        vk_indel_range: Shape ``(n_samples, ploidy, n_regions, 2)``, hap-major.
        max_end_keys: Shape ``(n_regions,)``. Packed ``(pos << MAX_END_SHIFT) |
            ext`` maxima over this chunk's haps; ``0`` means no variant. Reduce
            across chunks with an elementwise maximum BEFORE unpacking -- the
            ordering rule is position first, end second, so reducing unpacked
            ends would pick the wrong variant.
    """

    sample_start: int
    n_samples: int
    vk_snp_range: "np.ndarray"
    vk_indel_range: "np.ndarray"
    max_end_keys: "np.ndarray"


@dataclass(frozen=True)
class RangesStream:
    """Memory-bounded, chunked form of ``_find_ranges``.

    The ``O(n_regions)`` arrays are computed eagerly; the
    ``O(n_regions * n_samples * ploidy)`` payload arrives via ``chunks``.
    ``n_samples`` is the progress denominator and each ``RangesChunk`` reports
    how many samples it advanced by.
    """

    n_regions: int
    n_samples: int
    ploidy: int
    samples_per_chunk: int
    region_starts: "np.ndarray"
    dense_range: "np.ndarray"
    dense_snp_range: "np.ndarray"
    dense_indel_range: "np.ndarray"
    sample_cols: "np.ndarray"
    dense_max_end_keys: "np.ndarray"
    chunks: "Iterator[RangesChunk]"
```

Then add the method to `_BatchQueryMixin`:

```python
    def _find_ranges_chunked(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        samples: "ArrayLike | None" = None,
        *,
        max_mem: int | None = None,
    ) -> RangesStream:
        """Chunked, memory-bounded ``_find_ranges``.

        ``starts``/``ends`` and ``samples`` behave as in :meth:`read_ranges`.

        The var_key payload is ``n_regions * n_samples * ploidy * 2`` int64
        pairs per channel, which is tens of GiB at cohort scale. This splits it
        along the SAMPLE axis -- not the region axis -- because the search is
        column-outer: chunking regions would re-sweep the whole packed store per
        chunk, while chunking samples keeps a single sweep.

        Args:
            contig: Contig name.
            starts: 0-based start positions of the query regions.
            ends: 0-based, exclusive end positions of the query regions.
            samples: Sample names selecting (and reordering) a subset.
            max_mem: Approximate byte budget for one chunk's payload. ``None``
                yields a single chunk covering every sample.

        Returns:
            A :class:`RangesStream` whose ``chunks`` generator yields
            :class:`RangesChunk` in ascending ``sample_start`` order.

        Raises:
            ValueError: If ``max_mem`` cannot fit a single sample's payload, or
                if the contig's largest deletion overflows the max-end key
                packing width.
        """
        reg = self._regions(starts, ends)
        sample_idxs = self._sample_idxs(samples)
        reader = self._reader(contig)
        header = reader.find_ranges_header(reg, sample_idxs)

        n_regions = int(header["n_regions"])
        n_samples = int(header["n_samples"])
        ploidy = int(header["ploidy"])

        # Both channels, 2 endpoints, int64. The 2x is slop for the transient
        # the binding holds while handing the arrays back.
        bytes_per_sample = n_regions * ploidy * 2 * 8 * 2
        if max_mem is None:
            per = max(n_samples, 1)
        else:
            per = int(max_mem) // (2 * bytes_per_sample) if bytes_per_sample else n_samples
            if per < 1:
                raise ValueError(
                    f"max_mem ({int(max_mem)} bytes) is too small for even one "
                    f"sample of {n_regions} regions at ploidy {ploidy}: needs at "
                    f"least {2 * bytes_per_sample} bytes."
                )
            per = min(per, max(n_samples, 1))

        def _gen() -> "Iterator[RangesChunk]":
            for s0 in range(0, n_samples, per):
                s1 = min(s0 + per, n_samples)
                d = reader.find_ranges_chunk(
                    reg, sample_idxs, s0 * ploidy, s1 * ploidy
                )
                cs = s1 - s0
                shape = (cs, ploidy, n_regions, 2)
                yield RangesChunk(
                    sample_start=s0,
                    n_samples=cs,
                    vk_snp_range=np.asarray(d["vk_snp_range"]).reshape(shape),
                    vk_indel_range=np.asarray(d["vk_indel_range"]).reshape(shape),
                    max_end_keys=np.asarray(d["max_end_keys"], np.int64),
                )

        return RangesStream(
            n_regions=n_regions,
            n_samples=n_samples,
            ploidy=ploidy,
            samples_per_chunk=per,
            region_starts=np.asarray(header["region_starts"]),
            dense_range=np.asarray(header["dense_range"]),
            dense_snp_range=np.asarray(header["dense_snp_range"]),
            dense_indel_range=np.asarray(header["dense_indel_range"]),
            sample_cols=np.asarray(header["sample_cols"]),
            dense_max_end_keys=np.asarray(header["dense_max_end_keys"], np.int64),
            chunks=_gen(),
        )
```

- [ ] **Step 4: Run the tests — expect PASS**

```bash
pixi run pytest tests/test_svar2_ranges.py -q
```

Expected: all pass.

- [ ] **Step 5: Lint and typecheck**

```bash
pixi run -e lint ruff check python/genoray tests
pixi run -e lint ruff format --check python/genoray tests
pixi run typecheck
```

Expected: clean.

- [ ] **Step 6: Run the full genoray suite**

```bash
pixi run test 2>&1 | tail -15
pixi run -e lint test-rust 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 7: Commit and push**

```bash
git add python/genoray/_svar2_batch.py tests/test_svar2_ranges.py
git commit -m "feat(svar2): add _find_ranges_chunked memory-bounded stream API

Returns a RangesStream: the O(regions) arrays eagerly, plus a generator of
per-sample-slice RangesChunk sized from max_mem. Chunking is along the
sample axis rather than the region axis because the search is column-outer
-- region chunks would re-sweep the packed store once per chunk.

Callers reduce max_end_keys across chunks with an elementwise maximum and
unpack once at the end.

Relates to #<genoray-issue>, mcvickerlab/GenVarLoader#333"
git push -u origin worktree-issue-333-find-ranges
```

- [ ] **Step 8: Open the genoray PR**

```bash
gh pr create --draft --base main \
  --title "perf(query): make find_ranges O(total_variants) and add a chunked API" \
  --body "Fixes the O(regions x total_variants) tree rebuild in \`find_ranges\`, adds a memory-bounded chunked API, and folds per-region max-end computation into the same sweep.

- Hoist per-column search state into \`VkColumnIndex\`; invert \`find_ranges\` to column-outer; parallelize with rayon. Tree builds drop from R*H*2 to H*2.
- \`find_ranges_haps\` also returns per-region packed max-end keys, so consumers no longer decode every sample to extend chromEnd.
- \`_find_ranges_chunked\` yields hap slices sized from \`max_mem\`; the chunk binding fills numpy in place and releases the GIL.

Closes #<genoray-issue>
Unblocks mcvickerlab/GenVarLoader#333

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

After review and merge, confirm the released version is >= 3.4.0 before Task 6.

---

### Task 5: GenVarLoader preflight and `max_mem` plumbing

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:324-328` (call site), `:1124-1135` (signature)
- Test: `tests/dataset/test_write_svar2.py`

**Interfaces:**
- Consumes: `write()`'s existing `effective_max_mem` local; `loguru.logger`; `shutil` (already imported at `_write.py:4`).
- Produces:
  - `_svar2_ranges_cache_bytes(n_regions: int, n_samples: int, ploidy: int) -> int`
  - `_svar2_preflight(out_dir: Path, n_regions: int, n_samples: int, ploidy: int) -> int` — logs and warns, returns the byte count
  - `_write_from_svar2(path, bed, svar2, samples, extend_to_length, max_mem)` — new trailing `max_mem: int` parameter

This task does **not** depend on the genoray release; it only changes GenVarLoader-internal plumbing.

- [ ] **Step 1: Write the failing tests**

Add to `tests/dataset/test_write_svar2.py`:

```python
def test_svar2_ranges_cache_bytes():
    """Both var-key channels: 2 * R * S * P * 2 endpoints * 8 bytes."""
    from genvarloader._dataset._write import _svar2_ranges_cache_bytes

    assert _svar2_ranges_cache_bytes(1, 1, 2) == 2 * 1 * 1 * 2 * 2 * 8
    # The scale from gvl#333: ~98 GiB for one chromosome/panel.
    big = _svar2_ranges_cache_bytes(3964, 414830, 2)
    assert 90 * 1024**3 < big < 110 * 1024**3


def test_svar2_preflight_warns_when_disk_is_short(tmp_path, monkeypatch):
    """A projected cache larger than free space must warn, not silently proceed."""
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        n = _write._svar2_preflight(tmp_path, 3964, 414830, 2)
    finally:
        logger.remove(sink)

    assert n == _write._svar2_ranges_cache_bytes(3964, 414830, 2)
    assert any("free" in m for m in msgs), msgs
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/issue-333-svar2-write-mem
pixi run -e dev pytest tests/dataset/test_write_svar2.py -q -k "cache_bytes or preflight"
```

Expected: FAIL — `cannot import name '_svar2_ranges_cache_bytes'`.

- [ ] **Step 3: Add the two helpers**

In `python/genvarloader/_dataset/_write.py`, immediately above `_write_from_svar2`:

```python
def _svar2_ranges_cache_bytes(n_regions: int, n_samples: int, ploidy: int) -> int:
    """Permanent on-disk size of the two ``svar2_ranges`` var-key caches.

    Each of ``vk_snp_range`` and ``vk_indel_range`` is a
    ``(regions, samples, ploidy, 2)`` int64 array. These are NOT small: one
    chromosome of a 414k-sample cohort over ~4k regions is ~98 GiB.

    Args:
        n_regions: Number of BED rows in the dataset.
        n_samples: Number of selected samples.
        ploidy: Ploidy of the variant source.

    Returns:
        Total bytes both channels will occupy on disk.
    """
    return 2 * n_regions * n_samples * ploidy * 2 * 8


def _svar2_preflight(
    out_dir: Path, n_regions: int, n_samples: int, ploidy: int
) -> int:
    """Log the projected ``svar2_ranges`` cache size and warn if disk is short.

    Warns rather than raising: free-space reporting is unreliable on some
    network filesystems, and a false refusal would block a valid large build.

    Args:
        out_dir: Directory the cache will be written to.
        n_regions: Number of BED rows in the dataset.
        n_samples: Number of selected samples.
        ploidy: Ploidy of the variant source.

    Returns:
        Projected total bytes of the two var-key caches.
    """
    n_bytes = _svar2_ranges_cache_bytes(n_regions, n_samples, ploidy)
    logger.info(
        f"svar2 range cache: {format_memory(n_bytes)} for {n_regions} regions "
        f"x {n_samples} samples x ploidy {ploidy}."
    )
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError:
        return n_bytes
    if n_bytes > free:
        logger.warning(
            f"svar2 range cache needs {format_memory(n_bytes)} but only "
            f"{format_memory(free)} is free at {out_dir}. The write will likely "
            f"fail with ENOSPC."
        )
    return n_bytes
```

- [ ] **Step 4: Run the tests — expect PASS**

```bash
pixi run -e dev pytest tests/dataset/test_write_svar2.py -q -k "cache_bytes or preflight"
```

Expected: PASS.

- [ ] **Step 5: Plumb `max_mem` and call the preflight**

Change the signature at `_write.py:1124`:

```python
def _write_from_svar2(
    path: Path,
    bed: pl.DataFrame,
    svar2: SparseVar2,
    samples: list[str],
    extend_to_length: bool,
    max_mem: int,
) -> tuple[pl.DataFrame, Svar2Link]:
```

Change the call site at `_write.py:324-327`:

```python
                elif isinstance(variants, SparseVar2):
                    gvl_bed, _svar2_link = _write_from_svar2(
                        path,
                        gvl_bed,
                        variants,
                        samples,
                        extend_to_length,
                        effective_max_mem,
                    )
```

Immediately after the `R, S, P = bed.height, len(samples), svar2.ploidy` line in
`_write_from_svar2`, and before the first `np.memmap(...)` call, insert:

```python
    _svar2_preflight(out_dir, R, S, P)
```

- [ ] **Step 6: Run the SVAR2 write tests**

```bash
pixi run -e dev pytest tests/dataset/test_write_svar2.py -q
```

Expected: all pass — behavior is unchanged, `max_mem` is accepted but not yet consumed.

- [ ] **Step 7: Lint**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
```

Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py
git commit -m "feat(write): preflight the svar2 range cache and accept max_mem

_write_from_svar2 now receives write()'s effective_max_mem (it previously
got no memory budget at all) and logs the projected on-disk size of the two
var-key range caches before allocating them, warning when it exceeds free
space. At 414k samples over ~4k regions that projection is ~98 GiB.

max_mem is accepted but not yet consumed; the chunked consumption lands
with the genoray 3.4 API.

Relates to #333"
```

---

### Task 6: Consume the chunked stream; delete `_svar2_region_max_ends`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:1067-1121` (delete `_svar2_region_max_ends`), `:1178-1200` (contig loop)
- Modify: `pyproject.toml:15`, `pixi.toml:107` (genoray pin)
- Test: `tests/dataset/test_write_svar2.py`

**Interfaces:**
- Consumes: `SparseVar2._find_ranges_chunked` and `genoray._svar2_batch.MAX_END_SHIFT` from Task 4; `_svar2_preflight` from Task 5.
- Produces: no new public symbols. `_svar2_region_max_ends` is removed.

**Prerequisite:** genoray >= 3.4.0 must be installed in the dev env.

- [ ] **Step 1: Write the failing tests**

Add to `tests/dataset/test_write_svar2.py`:

```python
def test_write_svar2_chunked_matches_unchunked(
    svar2_store: Path, vcf_and_ref, tmp_path
):
    """A tiny max_mem must force multiple chunks and produce identical output."""
    from genoray import SparseVar2

    _, ref = vcf_and_ref
    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 5], "chromEnd": [20, 30]}
    )

    calls: list[int] = []
    real = SparseVar2._find_ranges_chunked

    def spy(self, *args, **kwargs):
        stream = real(self, *args, **kwargs)
        calls.append(stream.samples_per_chunk)
        return stream

    big = tmp_path / "big.gvl"
    gvl.write(big, bed, SparseVar2(svar2_store), reference=ref, max_mem="4g")

    SparseVar2._find_ranges_chunked = spy
    try:
        small = tmp_path / "small.gvl"
        # 1 region-sample of payload is R*P*2*8*2 = 64 bytes; 2x slop -> 128.
        gvl.write(small, bed, SparseVar2(svar2_store), reference=ref, max_mem=128)
    finally:
        SparseVar2._find_ranges_chunked = real

    assert calls and all(c == 1 for c in calls), (
        f"expected one sample per chunk under a 128-byte budget, got {calls}"
    )

    for name in (
        "vk_snp_range.npy",
        "vk_indel_range.npy",
        "dense_snp_range.npy",
        "dense_indel_range.npy",
        "sample_cols.npy",
    ):
        a = (big / "genotypes" / "svar2_ranges" / name).read_bytes()
        b = (small / "genotypes" / "svar2_ranges" / name).read_bytes()
        assert a == b, name

    ra = pl.read_ipc(big / "input_regions.arrow")
    rb = pl.read_ipc(small / "input_regions.arrow")
    assert ra["chromEnd"].to_list() == rb["chromEnd"].to_list()


def test_write_svar2_max_ends_extend_chromend(svar2_store: Path, vcf_and_ref, tmp_path):
    """chromEnd must extend past a deletion that starts inside the region.

    The fixture's DEL is at 0-based POS 11 with ilen -2, so it ends at 14. A
    region of [0, 12) must be extended to 14.
    """
    from genoray import SparseVar2

    _, ref = vcf_and_ref
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [12]})
    out = tmp_path / "ext.gvl"
    gvl.write(out, bed, SparseVar2(svar2_store), reference=ref, max_mem="1g")
    regions = pl.read_ipc(out / "input_regions.arrow")
    assert regions["chromEnd"].to_list() == [14]
```

- [ ] **Step 2: Bump the genoray pin and install**

In `pyproject.toml:15`: `"genoray>=3.4.0,<4",`
In `pixi.toml:107`: `genoray = ">=3.4.0,<4"`

```bash
pixi install -e dev 2>&1 | tail -5
pixi run -e dev python -c "import genoray; print(genoray.__version__ if hasattr(genoray,'__version__') else 'ok'); from genoray import SparseVar2; assert hasattr(SparseVar2, '_find_ranges_chunked')"
```

Expected: no AssertionError.

- [ ] **Step 3: Run to confirm the tests fail**

```bash
pixi run -e dev pytest tests/dataset/test_write_svar2.py -q -k "chunked or max_ends_extend"
```

Expected: FAIL — `test_write_svar2_chunked_matches_unchunked` fails on `assert calls`, because `_write_from_svar2` still calls `_find_ranges`, not `_find_ranges_chunked`.

- [ ] **Step 4: Delete `_svar2_region_max_ends`**

Remove the entire function at `python/genvarloader/_dataset/_write.py:1067-1121`.

- [ ] **Step 5: Rewrite the contig loop**

Replace the loop body in `_write_from_svar2` (currently `_write.py:1178-1200`) with:

```python
    max_ends = np.empty(R, np.int32)
    contig_offset = 0
    pbar = tqdm(total=R, unit=" region")
    for (c,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        c = cast(str, c)
        pbar.set_description(f"Processing svar2 ranges for {df.height} regions on {c}")
        lo, hi = contig_offset, contig_offset + df.height
        rc = df.height
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        # extend_to_length is validated at function entry (False raises); the
        # read-bound kernel sizes haplotype output at read time.
        stream = svar2._find_ranges_chunked(
            c, starts, ends, samples=samples, max_mem=max_mem
        )
        dense_snp[lo:hi] = np.asarray(stream.dense_snp_range, np.int64).reshape(rc, 2)
        dense_indel[lo:hi] = np.asarray(stream.dense_indel_range, np.int64).reshape(
            rc, 2
        )

        # Packed (pos << SHIFT) | ext keys, NOT unpacked ends: SVAR1 parity picks
        # the highest-POSITION variant (ties by end), so a lower-position variant
        # with a longer deletion must not win the cross-chunk reduction.
        keys = stream.dense_max_end_keys.copy()
        for ch in stream.chunks:
            s0, s1 = ch.sample_start, ch.sample_start + ch.n_samples
            # Chunks are hap-major (samples, ploidy, regions, 2); the cache is
            # region-major. transpose() is a view -- numpy copies straight into
            # the memmap with no intermediate array.
            vk_snp[lo:hi, s0:s1] = ch.vk_snp_range.transpose(2, 0, 1, 3)
            vk_indel[lo:hi, s0:s1] = ch.vk_indel_range.transpose(2, 0, 1, 3)
            np.maximum(keys, ch.max_end_keys, out=keys)
            # Bound the dirty page cache: at cohort scale these memmaps are tens
            # of GiB and the kernel would otherwise reclaim at unpredictable times.
            vk_snp.flush()
            vk_indel.flush()
            pbar.update(rc * ch.n_samples / S)

        mask = (1 << MAX_END_SHIFT) - 1
        region_ends = np.asarray(ends, np.int64).copy()
        has = keys > 0  # 0 is the "no variant in this region" sentinel
        region_ends[has] = (keys[has] >> MAX_END_SHIFT) + (keys[has] & mask)
        max_ends[lo:hi] = region_ends.astype(np.int32)

        contig_offset += df.height
    pbar.close()
```

Add the import near the other genoray imports at the top of `_write.py`:

```python
from genoray._svar2_batch import MAX_END_SHIFT
```

- [ ] **Step 6: Run the SVAR2 write tests — expect PASS**

```bash
pixi run -e dev pytest tests/dataset/test_write_svar2.py -q
```

Expected: all pass.

- [ ] **Step 7: Verify no stale references to the deleted helper**

```bash
rg -n "_svar2_region_max_ends" python/ tests/ docs/
```

Expected: no matches.

- [ ] **Step 8: Run the full tree**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
pixi run -e dev pytest tests -q 2>&1 | tail -20
```

Expected: all clean and passing. A scoped run would miss `tests/unit/`, which this change's deleted symbol could reach.

- [ ] **Step 9: Commit**

```bash
git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py \
        pyproject.toml pixi.toml
git commit -m "fix(write): bound svar2 genotype-writing memory with max_mem

_write_from_svar2 called _find_ranges once per contig, materializing two
O(regions x samples x ploidy) int64 arrays plus Rust and numpy transients --
~245 GiB at 414k samples -- and then called _svar2_region_max_ends, which
decoded ALL samples for the whole contig regardless of the caller's
selection. Together those OOM-killed a real All of Us chr22 build.

Consume genoray 3.4's _find_ranges_chunked instead: per-sample-slice chunks
sized from max_mem, written straight into the memmaps via a transposed
view, flushed per chunk. max_ends now comes from the same sweep as packed
composite keys, reduced across chunks and unpacked once, so no decode pass
happens at all.

Progress is now fractional within a contig, so a single-contig BED no longer
sits at 0% for the entire run.

Closes #333"
```

- [ ] **Step 10: Push and open the PR**

```bash
git push -u origin worktree-issue-333-svar2-write-mem
gh pr create --draft --base main \
  --title "fix(write): bound SVAR2 genotype-writing memory with max_mem" \
  --body "Closes #333.

Depends on genoray >= 3.4.0 (d-laub/genoray#<genoray-issue>).

Three defects, only one named in the issue:

1. genoray \`find_ranges\` was O(regions x total_variants) -- a \`SearchTree\` was rebuilt per (region, column). Fixed in genoray 3.4.
2. \`_svar2_region_max_ends\` decoded ALL samples for the whole contig, ignoring the sample selection. Deleted; genoray now returns packed max-end keys from the range sweep.
3. \`_write_from_svar2\` received no \`max_mem\` and materialized a whole contig's ranges at once. Now consumes the chunked stream with per-chunk memmap writes and flushes.

Also preflights and logs the ~98 GiB permanent range cache, and makes progress fractional within a contig.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/source/format.md` (the `genotypes/svar2_ranges` section)
- Modify: `docs/source/write.md` and the `gvl.write` docstring at `python/genvarloader/_dataset/_write.py:141-148`
- Modify: `skills/genvarloader/SKILL.md`

This task is independent of Tasks 1–6 and can run concurrently.

- [ ] **Step 1: Find the false "small" claim**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/issue-333-svar2-write-mem
rg -n "small" docs/source/format.md
rg -n "svar2_ranges" -A 25 docs/source/format.md
```

- [ ] **Step 2: Correct `format.md`**

Replace the sentence describing the per-`(region, sample, ploidy)` arrays as
"small" with:

```markdown
`vk_snp_range.npy` and `vk_indel_range.npy` are each
`(regions, samples, ploidy, 2)` int64, so the two together occupy

```
2 x regions x samples x ploidy x 2 x 8 bytes
```

This grows linearly in **both** the number of BED rows and the number of
selected samples. It is not small at cohort scale: ~4,000 regions over 414,830
diploid samples is approximately **98 GiB** for a single chromosome/panel.
`gvl.write` logs the projected size before allocating and warns when it exceeds
free disk. Budget disk accordingly, or reduce the region count or sample
selection.
```

- [ ] **Step 3: Update the `max_mem` docstring**

At `python/genvarloader/_dataset/_write.py:141-148`, extend the `max_mem`
description with:

```
            For a ``.svar2`` variant source this also bounds the genotype
            range-cache write: ranges are produced in per-sample chunks sized to
            fit the budget rather than a whole contig at once.
```

- [ ] **Step 4: Update `write.md`**

```bash
rg -n "max_mem" docs/source/write.md
```

Add, in the `max_mem` discussion, a sentence noting that the SVAR2 genotype
branch honours it as of this release, and that the permanent range cache is
governed by disk (see `format.md`), not `max_mem`.

- [ ] **Step 5: Update the skill**

In `skills/genvarloader/SKILL.md`, under the `gvl.write` section add a `max_mem`
note, and under "Common gotchas" add:

```markdown
- **SVAR2 range caches scale with `regions x samples x ploidy`.** `gvl.write`
  with a `.svar2` source writes a permanent
  `2 x regions x samples x ploidy x 2 x 8` byte cache under
  `genotypes/svar2_ranges/`. That is ~98 GiB for ~4,000 regions over 414,830
  diploid samples. `max_mem` bounds RAM during the write; it does not bound this
  on-disk cache.
```

- [ ] **Step 6: Verify the api.md/`__all__` invariant still holds**

```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```

Expected: `MISSING: none`. This change adds no public symbols, so this is a
regression check rather than an edit.

- [ ] **Step 7: Build the docs**

```bash
pixi run -e docs doc 2>&1 | tail -20
```

Expected: builds without new warnings.

- [ ] **Step 8: Commit**

```bash
git add docs/source/format.md docs/source/write.md skills/genvarloader/SKILL.md \
        python/genvarloader/_dataset/_write.py
git commit -m "docs: correct the svar2 range-cache size claim

format.md described the per-(region, sample, ploidy) range arrays as
'small'. They are 2 * R * S * P * 2 * 8 bytes -- ~98 GiB for one chromosome
of a 414k-sample cohort. Give the formula, a worked population-scale
example, and the disk-vs-max_mem distinction, and note that max_mem now
governs the SVAR2 genotype write.

Relates to #333"
```

---

## Follow-up issues to file after merge

- [ ] **genoray/GenVarLoader:** shrink the on-disk SVAR2 range cache below 16 bytes/entry (e.g. `start: int64` + `len: int32`). Format change, needs read-path and version-compat work.
- [ ] **GenVarLoader:** `_write_from_svar` (SVAR1) also ignores `max_mem`. Different mechanism (`_find_starts_ends_with_length(..., out=)`), no transient amplification, so lower priority.

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| genoray Rust core — hoist per-column index | Task 1 |
| genoray Rust core — column-outer `find_ranges_haps` + rayon | Task 1 |
| genoray Rust core — output order / region-major bundle preserved | Task 1, Step 4 |
| genoray chunked Python API — `RangesStream`/`RangesChunk` | Task 4 |
| genoray chunked Python API — chunk sizing from `max_mem` | Task 4, Step 3 |
| genoray chunked Python API — in-place numpy fill, GIL release | Task 3, Step 3 |
| genoray `max_ends` — vk channels | Task 2, Step 3 |
| genoray `max_ends` — dense channel backward walk + `samples=None` fast path | Task 2, Step 4 |
| genoray `max_ends` — packing-width overflow guard | Task 2 Step 5 + Task 3 Step 3 |
| GenVarLoader — preflight | Task 5 |
| GenVarLoader — `max_mem` plumbing | Task 5, Step 5 |
| GenVarLoader — per-contig chunked loop, flush, fractional progress | Task 6, Step 5 |
| GenVarLoader — delete `_svar2_region_max_ends` | Task 6, Step 4 |
| Testing — loop-inversion byte-identity guard | Task 1, Steps 6–7 |
| Testing — complexity regression guard via `TREE_BUILDS` | Task 1, Step 1 |
| Testing — chunk equivalence property | Task 4, Step 1 |
| Testing — `max_ends` parity | Task 2 Step 1, Task 4 Step 1, Task 6 Step 1 |
| Testing — `samples` subset (non-fast-path dense walk) | Task 4, Step 1 |
| Testing — GenVarLoader scale guard (#333 §4) | Task 6, Step 1 |
| Docs — `format.md`, `write.md`, SKILL.md, CHANGELOG | Task 7 |
| Follow-up issues | end of plan |

No spec requirement is unassigned.

**Deviations from the spec, deliberate**

1. **`max_ends` is carried as packed composite keys, not unpacked ends.** The spec's field names were `max_ends: NDArray[np.int32]` / `dense_max_ends`. Reducing unpacked ends across hap chunks is *wrong*: the SVAR1 rule orders by position first and end second, so a lower-position variant with a longer deletion would win a naive `np.maximum` over ends. The fields are therefore `max_end_keys` / `dense_max_end_keys` (int64 packed), reduced with `np.maximum` and unpacked once by the consumer. The spec has been amended to match.
2. **`find_ranges` gains one transposed copy of its payload.** The spec claimed the region-major reorder would be free inside `bundle_to_dict`. Keeping the rayon-safe hap-major fill means `find_ranges` transposes into its region-major `Vec<Range<usize>>`. This only affects the small-batch read path; the population-scale writer uses the chunked API and never builds a bundle. Noted in the Task 1 code comment and amended in the spec.
3. **`PAR_COLUMN_THRESHOLD` was not in the spec.** Small batches stay serial so rayon's fork/join overhead is avoided *and* so `search::search_tree_build_count` — a thread-local that existing tests already depend on — stays observable from the caller's thread. Without this the complexity guard test could not be written.

**Placeholder scan:** no TBD/TODO. `#<genoray-issue>` is a deliberate, explained substitution produced by Task 0, Step 1.

**Type consistency:** `find_ranges_haps` returns `Vec<u64>` after Task 2 and every later call site (Task 3's binding, Task 2's tests) uses that return. `MAX_END_SHIFT` is `u32` in Rust and `int` in Python, both 21. `RangesChunk.vk_snp_range` is `(n_samples, ploidy, R, 2)` at its definition (Task 4) and is transposed with `(2, 0, 1, 3)` at its only consumer (Task 6) into `(R, n_samples, ploidy, 2)`, matching the `vk_snp[lo:hi, s0:s1]` memmap slice. `_svar2_preflight` returns `int` and Task 5's test asserts against `_svar2_ranges_cache_bytes`.
