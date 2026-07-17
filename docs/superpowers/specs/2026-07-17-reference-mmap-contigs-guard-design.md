# Fix #285 — `Reference.from_path(in_memory=False, contigs=...)` silently returns wrong bytes

## Problem

`Reference.from_path` (`python/genvarloader/_dataset/_reference.py`) only rebuilds
`reference`/`offsets` into the caller's `contigs` order inside its `if in_memory:`
branch (~:99–107). On the `in_memory=False` (memory-mapped) branch it accepts the
`contigs` argument, rebuilds `c_map` in the caller's order, but leaves `offsets` in
**full-FASTA order** over all contigs. Any caller that indexes `offsets` by its own
contig index — the documented convention, e.g. via `bed_to_regions(bed, ref.c_map)`
feeding the `get_reference` kernel — silently reads **the wrong contig's bytes**, with
no error.

Reproduction:

```python
Reference.from_path(multi_contig_fa, ["chr2", "chr1"], in_memory=False)
# offsets stays [0, len_chr1, len_chr1+len_chr2] but c_map order is [chr2, chr1],
# so contig index 0 (chr2) resolves to chr1's byte range.
```

Found during #275 review.

### Why it is latent today

- The only production caller that passes `contigs` is `_resolve_reference`
  (`_open.py:133`), which uses the default `in_memory=True` → correct branch.
- Every `in_memory=False` construction in the repo (conftest fixtures, parity tests)
  passes **no** `contigs` → `contigs=None` → resolved to full-FASTA order → correct.

So no current caller triggers the bug, but the combination is accepted and wrong. A
user who builds an `in_memory=False` Reference with reordered/subset contigs and passes
it to `Dataset.open` gets silently wrong sequence.

## Why we can't "just honor `contigs`" on the mmap path

The `ref_offsets` kernel contract is a single cumulative array of length `n+1`, where
contig `i` occupies `reference[offsets[i]:offsets[i+1]]`. That convention **requires the
selected contigs to be physically contiguous and in the requested order** in the backing
buffer. The `in_memory=True` path satisfies it by *copying* the selected contigs into
caller order. The mmap stays in full-FASTA order and cannot be reordered or subset
without that same copy — which is exactly the memory cost `in_memory=False` exists to
avoid.

Honoring an arbitrary reorder/subset over the mmap would require either:

- (a) copying into memory — defeats the flag, and silently spends the RAM the caller
  asked not to spend; or
- (b) changing the kernel contract to per-contig `(start, end)` pairs — a large,
  invasive change rippling through `_haps`, `_ref`, `_svar2_haps`, and the Rust
  `get_reference` kernel, for a capability no caller needs (YAGNI).

## Decision — hard-reject the invalid combination (fail fast)

Make the invalid state unrepresentable at construction. When `in_memory=False` and the
resolved `contigs` list is **not exactly equal to the full FASTA contig order**, raise
`ValueError`. This matches the codebase principle of failing fast over silently-wrong
output, costs zero on every current caller, and leaves the user in control of the memory
tradeoff (the alternative — auto-promoting to `in_memory=True` — was rejected because it
hides a memory cost behind an explicit flag).

### Behavior matrix

| `in_memory` | `contigs`                          | Result                          |
|-------------|------------------------------------|---------------------------------|
| `True`      | any valid subset/reorder / `None`  | works (existing copy path)      |
| `False`     | `None`                             | works (full FASTA, unchanged)   |
| `False`     | explicit list == full FASTA order  | works (no-op reorder)           |
| `False`     | reorder or subset of the FASTA     | **`ValueError`** (new)          |

"Equal to full FASTA order" is compared **after** contig-name normalization
(`c_map.norm`), so a user passing normalized-but-identical names (e.g. `"chr1"` vs `"1"`)
is still accepted.

## Implementation

`python/genvarloader/_dataset/_reference.py`, in `from_path`'s `else` branch:

```python
else:
    if contigs != list(full_contigs):
        raise ValueError(
            "in_memory=False cannot reorder or subset contigs: the memory-mapped "
            "reference stays in FASTA order, so a reordered/subset `contigs` would "
            "index the wrong contig's bytes. Pass in_memory=True to materialize the "
            "requested contig order in memory, or omit `contigs` to keep the full FASTA."
        )
    reference = ref_mmap
```

At that point `contigs` is always a resolved list (either `c_map.contigs` when the arg
was `None`, or the normalized list), and `full_contigs` is the FASTA's contig dict in
file order, so `list(full_contigs)` is the full order to compare against.

### Docstring updates

Note the restriction in the `contigs` and `in_memory` parameter docs of `from_path`:
`in_memory=False` requires `contigs` to be `None` or the full FASTA order; reordering or
subsetting requires `in_memory=True`.

## Tests

New unit test file (or extend an existing reference/fasta test) using the synthetic
multi-contig fixture (`ref_fasta`, chr1/chr2):

1. **Negative (the bug):** `from_path(ref_fasta, ["chr2", "chr1"], in_memory=False)`
   raises `ValueError`; likewise a strict subset `["chr1"]` (not full order) raises.
2. **Positive, still valid:** `in_memory=False` with `contigs=None` works, and with an
   explicit full-order list (`["chr1", "chr2"]`) works — both yield offsets/contigs in
   full order.
3. **Lock-in the correct branch:** `from_path(ref_fasta, ["chr2", "chr1"], in_memory=True)`
   returns chr2's bytes at contig index 0 (positive correctness guard on the path that
   already works, so a future refactor can't regress it).

## Docs audit

This is a bugfix that tightens an already-broken input combination; it does not add a
user-facing feature or change a working default. Public API surface (`__all__`,
signatures) is unchanged. Scope check per CLAUDE.md:

- `skills/genvarloader/SKILL.md`, `docs/source/*.md`: check whether any example shows
  `Reference.from_path(in_memory=False, contigs=...)`; if so, correct it. Expected: none.
- No `api.md`/`__all__` change (no symbol added/removed/renamed).

## Out of scope

- Per-contig `(start, end)` kernel contract to support reordered mmap views (option (b)
  above) — no caller needs it.
- Any change to the `in_memory=True` copy path beyond the correctness lock-in test.
