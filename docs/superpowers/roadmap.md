# GenVarLoader internal roadmap

Living document tracking **deferred features and intentional gaps** in
GenVarLoader. Entries here are notes-to-future-us — ideas worth keeping alive
without cluttering source with dead stubs or speculative scaffolding. They are
**not promises** and have no scheduled timeline.

Use this when you remove unimplemented code, defer a feature, or notice an
obvious extension worth recording. If you implement an item, link the PR and
move the entry to a "Shipped" section at the bottom (or delete it).

When adding an entry:

- **What** the feature would do (with an API sketch if relevant)
- **Why** it's interesting (concrete use case)
- **Status** — deferred / blocked-on-X / under consideration
- **What's needed** to ship it
- **Reference** — commit, PR, or file where prior work lived

---

## Deferred features

### Transformed track writing

**What:** apply a user-supplied transform to an existing on-disk track's
values and write the result as a derived track on the same dataset. API
sketch (the `Dataset.write_transformed_track` method already exists as a
`NotImplementedError` stub):

```python
ds.write_transformed_track(
    new_track="my_track_log",
    existing_track="my_track",
    transform=lambda r_idx, s_idx, tracks: ragged_log(tracks),
    max_mem=2**30,
    overwrite=False,
)
```

**Why interesting:** lets users derive normalized / smoothed / residualized
tracks (log-transform, z-score, deconvolution, baseline subtraction) without
round-tripping through Python via the full `gvl.write` pipeline. Useful for
ML preprocessing where the dataset itself is the source of truth.

**Status:** deferred. A pre-Awkward-Ragged implementation existed in
`Tracks.write_transformed_track` (chunked `intervals_to_tracks` → user
transform → `tracks_to_intervals`); it did not survive the migration of
`Ragged` to its Awkward-backed form and has been a `raise NotImplementedError`
stub ever since. The dead body was removed in the PR5c refactor pass (see
**Reference** below).

**What's needed to ship:**

1. Port the chunked decompress → transform → recompress loop to the
   awkward-backed `Ragged` API (`seqpro.rag.Ragged`). The numba kernels
   `intervals_to_tracks` and `tracks_to_intervals` still exist and operate
   on raw arrays; only the Python wrapper that wraps results in `Ragged`
   needs updating.
2. Verify the resulting on-disk layout matches what `gvl.write` produces
   for a track (so subsequent reads see no difference).
3. Add a test using a trivial transform (identity, then log) and assert
   numerical parity with the pre-transform values read back as tracks.
4. Decide whether the haplotype-extended path (extend region ends by max
   `_haplotype_ilens`) is in scope for v1 or deferred — the previous
   implementation supported it via an optional `haps` argument.

**Reference:** removed from `python/genvarloader/_dataset/_reconstruct.py`
(method `Tracks.write_transformed_track`) in PR5c. The previous implementation
is visible at:

```
git show 1f1b718:python/genvarloader/_dataset/_reconstruct.py
```

(commit `1f1b718` is the last point in `main` where the dead body existed.)
