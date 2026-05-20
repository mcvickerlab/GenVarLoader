# Index-aware `max_mem` accounting in `gvl.write`

**Date:** 2026-05-19

## Problem

`gvl.write(..., max_mem=...)` passes `max_mem` directly to genoray's
`_chunk_ranges_with_length`, which divides it by per-variant genotype memory to
size chunks. The genoray variant index (CHROM/POS/REF/ALT/ILEN), which is
already resident in memory when `write()` runs, is **not counted** against
`max_mem`. For large cohorts (1000G, UKBB-scale), the index alone can be GBs,
so actual peak memory significantly exceeds the user-specified `max_mem`.

## Goal

Make `max_mem` a true total cap: subtract the genoray reader's in-memory
footprint from `max_mem` before passing the remainder to genoray's chunking
logic, and warn the user when the resident footprint is large relative to
`max_mem`.

## Design

This work spans two repositories. Genoray gets a new public property; GVL
consumes it.

### Part 1 — genoray (`feat/nbytes` branch)

Add a public `nbytes: int` property on each reader class. Name matches the
numpy convention (e.g. `ndarray.nbytes`). Returns the total in-memory footprint
of resident data structures (i.e. data not backed by mmap).

- **`VCF.nbytes`** — bytes held by the loaded variant index:
  ```python
  @property
  def nbytes(self) -> int:
      if self._index is None:
          return 0
      return self._index.estimated_size()
  ```

- **`PGEN.nbytes`** — index dataframe plus the `StartsEndsIlens` cache:
  ```python
  @property
  def nbytes(self) -> int:
      n = 0
      if self._index is not None:
          n += self._index.estimated_size()
      if self._sei is not None:
          n += (
              self._sei.v_starts.nbytes
              + self._sei.v_ends.nbytes
              + self._sei.ilens.nbytes
              + self._sei.alt.estimated_size()
          )
      return n
  ```

- **`SparseVar.nbytes`** — index only. Genotypes (`self.genos`) and fields
  (`self.fields`) are memory-mapped and therefore do not count as resident:
  ```python
  @property
  def nbytes(self) -> int:
      return self.index.estimated_size()
  ```

**Tests:** for each class, assert `nbytes > 0` after the index is loaded and
that it scales with the number of variants. Use existing genoray fixtures.

**Release:** minor version bump.

### Part 2 — GVL consumer (depends on Part 1 released)

In `python/genvarloader/_dataset/_write.py::write()`, immediately after the
variant reader has been instantiated and (for VCF) `_load_index()` has run,
adjust `max_mem`:

```python
from genoray._utils import format_memory  # already used elsewhere

idx_bytes = variants.nbytes
effective_max_mem = max_mem - idx_bytes

logger.info(
    f"Variant reader resident size: {format_memory(idx_bytes)}; "
    f"max_mem budget: {format_memory(max_mem)}; "
    f"available for chunking: {format_memory(max(effective_max_mem, 0))}"
)
if idx_bytes > max_mem // 2:
    warnings.warn(
        f"Variant index resident size ({format_memory(idx_bytes)}) "
        f"exceeds 50% of max_mem ({format_memory(max_mem)}). "
        f"Consider increasing max_mem.",
        stacklevel=2,
    )
```

Then pass `effective_max_mem` (not the original `max_mem`) into:

- `_write_from_vcf(path, gvl_bed, variants, effective_max_mem, ...)`
- `_write_from_pgen(path, gvl_bed, variants, effective_max_mem, ...)`
- `_write_from_svar(...)` — does not currently take `max_mem`, no signature
  change required, but the warning still fires.

If `effective_max_mem <= 0`, do not raise here — genoray's existing
"Maximum memory ... insufficient to read a single variant" check will surface
a more precise error including `mem_per_variant`.

The downstream track writers (`_write_track`) continue to receive the original
`max_mem` unchanged; tracks do not share the genoray index.

**Dependency bump:** raise the minimum `genoray` version in
`pyproject.toml` / `pixi.toml` to the release containing `.nbytes`.

### Why split this across two PRs

The genoray property is a small, isolated addition that can land and be
released independently. GVL then takes the version bump in a follow-up PR.
This keeps the changes reviewable and avoids a coordinated release.

## Testing plan

### genoray

- Unit test per reader (`VCF`, `PGEN`, `SparseVar`) using existing test
  fixtures: assert `nbytes > 0` after construction (or after `_load_index()`
  for `VCF` when the gvi index exists), and assert that `nbytes` grows when
  more variants are present (fixture-with-more-variants comparison).

### GVL

- Integration test in `tests/dataset/` that:
  - calls `gvl.write` with a small fixture and a generous `max_mem`,
  - patches/monkey-patches `variants.nbytes` to return a value > `max_mem // 2`,
  - asserts the warning is emitted (via `pytest.warns`),
  - asserts the dataset still writes successfully.
- Verify existing `pixi run -e dev test` suite still passes — chunk sizing is
  effectively the same for small fixtures where `nbytes ≪ max_mem`.

## Out of scope

- Track writers (`_write_track`): they already accept `max_mem` and don't
  share the genoray index. Could be revisited if BigWig/Table resident state
  is found to be material.
- Process-level RSS tracking: not pursued (noisy, platform-dependent).

## Open questions

None at design time.
