# Prefetching DataLoader — Design

**Status:** draft, awaiting user review
**Date:** 2026-05-28

## Motivation

gvl batches are produced by `Dataset[r_idx, s_idx]`, which internally multithreads over the requested instances. Throughput **scales with fetch size**: a single fetch of ~hundreds-of-MB to >1 GB worth of `(region, sample)` pairs is substantially faster (per-instance) than many small fetches, because internal threading amortizes setup, IO, and reconstruction overhead.

The current `Dataset.to_dataloader()` (`python/genvarloader/_torch.py:45`) wraps `torch.utils.data.DataLoader` and fetches one mini-batch at a time. With `num_workers=0` this leaves on-CPU bandwidth on the table; with `num_workers>0` it pays multiprocessing overhead that often dominates the savings, hence the existing warning at `_torch.py:63`.

The bottleneck we want to fix is **per-batch fetch latency starving the GPU**. We do that by fetching one large *chunk* per call (sized to a user budget, default 2 GB), then slicing the chunk into the user's training mini-batches. Two new modes differ only in whether the next chunk's fill overlaps with the current chunk's drain.

## Modes

| Mode | RAM | Producer | Refill latency visible? | When to use |
|---|---|---|---|---|
| `"buffered"` | 1 × `buffer_bytes` | Main process, blocking | Yes (amortized over many mini-batches) | Simplest path; single CPU sufficient |
| `"double_buffered"` | 2 × `buffer_bytes / 2` (= `buffer_bytes` total) | Subprocess via POSIX shm, ping-pong | No (hidden behind drain) | Want to fully hide fetch latency |

`mode=None` (default) preserves today's behavior — plain `torch.utils.data.DataLoader`.

### `buffer_bytes` → `slot_bytes` translation

`buffer_bytes` is the total RAM the user budgets for the loader. The internal planner works in per-slot terms:

- `mode="buffered"`: `slot_bytes = buffer_bytes` (one slot).
- `mode="double_buffered"`: `slot_bytes = buffer_bytes // 2` (two slots, total ≈ `buffer_bytes`).

The `ChunkPlanner` is parameterized by `slot_bytes`; all later capacity claims (`peak_chunk_bytes`, per-batch ceiling) are per-slot.

## Public API

Extend `Dataset.to_dataloader()` with new keyword arguments. No new top-level exports.

```python
def to_dataloader(
    self,
    *,
    mode: Literal["buffered", "double_buffered"] | None = None,
    buffer_bytes: int = 2 * 1024**3,        # 2 GiB default; total footprint across all slots
    copy: bool = True,                       # zero-copy opt-out; see §4
    heartbeat_seconds: float = 60.0,         # double_buffered only; see §5
    # ... existing args (batch_size, shuffle, sampler, ...) ...
) -> torch.utils.data.DataLoader: ...
```

Returned object is `torch.utils.data.DataLoader`-compatible: supports `for batch in loader`, `len(loader)`, `batch_size`. Internally, when `mode is not None`, the dataset passed to torch is a custom `IterableDataset` (`Buffered…` or `DoubleBuffered…`) and `num_workers=0` is enforced.

`num_workers > 0` with a non-`None` mode raises — the new loader is the concurrency strategy.

## Determinism preconditions

Per output mode (`with_seqs`):

- `"reference"` → no determinism requirement.
- `"haplotypes"` and `"annotated"` → require `dataset.deterministic` and `haplotype_lengths()` not `None`.
- `"variants"` → no determinism requirement.

In all cases, spliced datasets (`dataset._sp_idxer is not None`) are rejected with a clear message; both modes block on `haplotype_lengths()`'s spliced `NotImplementedError`.

## Exact footprint computation

The design pivots on knowing **exactly** how many bytes each `(region, sample)` instance will materialize to, given the current Dataset schema. Per-region upper bounds are rejected because variants-per-region are Zipf-distributed — bounds would routinely over-allocate.

### New `Dataset` method

```python
def _output_bytes_per_instance(
    self,
    regions: Idx | None = None,
    samples: Idx | str | Sequence[str] | None = None,
) -> NDArray[np.int64]:
    """Exact bytes one (region, sample) instance materializes to under the
    current schema (with_seqs, with_tracks, with_settings.var_fields).

    Returns shape (n_regions, n_samples). Raises NotImplementedError for
    spliced datasets. Raises for non-deterministic datasets when with_seqs
    is in {"haplotypes", "annotated"}.
    """
```

Internal dispatch:

- `reference`: `region_length × ref.dtype.itemsize` (per instance, same for all samples).
- `haplotypes`: `haplotype_lengths.sum(-1) × itemsize` (sum over ploidy axis).
- `annotated`: `hap_bytes + 4 × hap_len_sum (ref_coords int32) + 4 × n_variants_sum (var_idxs int32)`.
- `variants`: iterate `self._seqs.var_fields`; apply the per-field formula below.
- `+ tracks`: append `Σ_track hap_len_sum × n_tracks × track_dtype.itemsize`.

### `variants` formula

For each variant set `V_{r,s,p}`:

```
bytes(r, s, p) = Σ_{f in var_fields} field_bytes(f, V_{r,s,p})

field_bytes("start",   V) = |V| × itemsize(POS_TYPE)
field_bytes("ilen",    V) = |V| × 4
field_bytes("dosage",  V) = |V| × itemsize(DOSAGE_TYPE)
field_bytes("alt",     V) = Σ_{v ∈ V} len(ALT_v)        # allele scan
field_bytes("ref",     V) = Σ_{v ∈ V} len(REF_v)        # allele scan
field_bytes(info_col,  V) = |V| × itemsize(variants.info[info_col].dtype)
```

`|V|` comes from `Dataset.n_variants(r, s)`. INFO columns and POS/DOSAGE itemsizes are known from the on-disk schema attached to `Haps`/`Variants`.

### Allele-length scan

Computed in O(|V|) without touching allele payload bytes by exploiting the `ListOffsetArray` layout of `RaggedAlleles`:

```python
# Haps._allele_bytes_sum(idx, kind: Literal["alt", "ref"]) -> NDArray[int64]
v_idxs = genos.data                                 # selected variant indices
offsets = self.variants.<kind>.offsets              # ListOffsetArray offsets
v_lens = offsets[v_idxs + 1] - offsets[v_idxs]      # vectorized per-variant length
return np.add.reduceat(v_lens, genos.offsets[:-1])  # ragged-sum per (r, s, p)
```

Same data access pattern as `Haps._get_variants` (`_haps.py:634`); no payload read.

### Pre-pass cost

One-time `_output_bytes_per_instance(subset_regions, subset_samples)` at loader construction returns an `(n_regions, n_samples)` int64 table. Cost is dominated by `haplotype_lengths()` and `n_variants()` (one walk of the sparse genotype index for the subset). For typical 10⁴ regions × 10³ samples × ploidy 2: ~80 MB int32 working set, seconds to compute, dwarfed by epoch time.

## Components

Five units, each with a single responsibility. All new files in `python/genvarloader/`.

### `_chunked.py`

- `ChunkPlanner(sampler, batch_size, slot_bytes, bytes_per_instance)` — pure logic.
  - Walks the BatchSampler-resolved `(r_idx, s_idx)` sequence in order.
  - Greedy fill: accumulate `bytes_per_instance[r, s]` until adding the next mini-batch would exceed `slot_bytes`; close the chunk on that mini-batch boundary.
  - Yields `(chunk_r_idx, chunk_s_idx, n_batches_in_chunk)`.
  - Computes `peak_chunk_bytes` (max across the epoch) used to size slots.
- `slice_chunk(chunk_output, batch_size) -> iterator[batch]` — slices one gvl fetch result into mini-batches. Handles `ndarray`, `Ragged`, `AnnotatedHaps`, `RaggedVariants`, and tuples thereof.

No I/O, fully unit-testable.

### `_buffered_loader.py` — `mode="buffered"`

- `BufferedTorchDataset(td.IterableDataset)` — owns a `ChunkPlanner`. In `__iter__`:
  1. Pull next `(chunk_r_idx, chunk_s_idx)`.
  2. Block on `chunk = self.dataset[chunk_r_idx, chunk_s_idx]`.
  3. Yield from `slice_chunk(chunk, batch_size)`.
- No threads, no processes, no shm. ~80 lines.

### `_shm_layout.py` — IPC contract

See §4.

### `_double_buffered_loader.py` — `mode="double_buffered"`

- `DoubleBufferedTorchDataset(td.IterableDataset)` — owns:
  - Two `multiprocessing.shared_memory.SharedMemory` blocks, each of capacity `peak_chunk_bytes`.
  - Two `(free, ready)` `multiprocessing.Event` pairs, one per slot.
  - A small `multiprocessing.Queue` carrying `(slot_idx, r_idx, s_idx)` items from main → producer.
  - One-direction `multiprocessing.Pipe` carrying producer-side exceptions back.
  - A persistent producer subprocess handle.
- `__iter__` body, per chunk: wait `ready[i]` (with heartbeat) → read header + build views → yield slices → set `free[i]`, clear `ready[i]` → flip `i`.

### `_producer.py` — subprocess entrypoint

- `producer_main(dataset_path, schema_payload, shm_names, events, index_queue, exc_pipe)`.
- Re-opens the dataset via `Dataset.open(dataset_path)` and reapplies the subset/schema (`with_seqs`, `with_tracks`, `with_settings`) from `schema_payload`.
- Loop: wait `free[i]` → drain next item from `index_queue` → `chunk = dataset[r_idx, s_idx]` → `write_chunk(shm[i], chunk)` → set `ready[i]`, clear `free[i]` → flip `i`.
- On `Exception`: write `(type, value, traceback)` to `exc_pipe`, exit.

### Touched existing code

`_torch.py:get_dataloader` — add `mode`, `buffer_bytes`, `copy`, `heartbeat_seconds` args; dispatch to `BufferedTorchDataset` / `DoubleBufferedTorchDataset` / current path.

## §4 — Shared-memory slot layout

Each slot is a flat byte arena of size `peak_chunk_bytes`. Output composition is fixed at construction, so the producer and consumer share an implicit schema; the per-fill **header** carries only the variable parts (counts, offsets).

### Header (hand-rolled, fixed-size prefix)

```
struct ChunkHeader {
    u64 n_instances;
    u64 payload_bytes;
    u8  n_arrays;
    // followed by n_arrays of ArrayDescriptor:
    struct ArrayDescriptor {
        u8   kind;            // 0=dense, 1=ragged
        u8   dtype_code;      // numpy dtype tag (small enum)
        u8   ndim;
        u64  shape[ndim];     // ragged: last dim omitted
        u64  data_offset;
        u64  data_nbytes;
        u64  lengths_offset;  // ragged only; 0 otherwise
        u64  lengths_nbytes;  // ragged only
    };
}
```

Header schema is closed and stable across the epoch. Hand-rolled `struct.pack` keeps the hot path GIL-free and dependency-free.

### Per output mode

- `reference`: 1 array, dense or ragged depending on `output_length`.
- `haplotypes`: 1 ragged S1 array, `lengths` length = `n_instances × ploidy`.
- `annotated`: 3 ragged arrays — haps (S1), ref_coords (int32), var_idxs (int32). Each carries its own `lengths` (haps/ref_coords use hap lengths; var_idxs uses `n_variants`).
- `variants`: one ragged array per active `var_field`. `alt`/`ref` are S1 with a nested-offsets encoding (two parallel `lengths` arrays — outer `n_variants` per `(instance, ploid)`, inner allele lengths per variant). Numeric fields (`start`/`ilen`/`dosage`/info columns) are flat with the outer `lengths` only. Consumer reconstructs `RaggedVariants` via `lengths_to_offsets` (`_utils.py:13`) and `RaggedVariants.from_ak`.
- `+ tracks`: append one ragged array per active track, dtype = track dtype.

### Synchronization

Per slot, two `multiprocessing.Event`s. Initial: both `free` set, both `ready` clear. Producer waits `free[i]` → writes → sets `ready[i]`, clears `free[i]`. Consumer waits `ready[i]` → reads → sets `free[i]`, clears `ready[i]`. The data path uses no queues. A small `multiprocessing.Queue` carries chunk-index batches from main → producer.

### Zero-copy and the `copy` flag

Consumer views (numpy/`Ragged`/`ak.Array`) are constructed against `shm.buf`. They are valid only until the consumer signals `free[i]`. Holding a batch past the next iteration produces undefined behavior.

To make the safe path the default, `copy=True` (default) calls `.copy()` (or `.clone()` for tensors) on each yielded array before signaling `free[i]`. `copy=False` yields zero-copy views; users opting in must consume each batch before the next iteration. When `pin_memory=True`, copy happens anyway, so `copy` has no effect.

## §5 — Error handling, lifecycle, testing

### Construction-time preconditions (raise immediately)

1. Spliced dataset → reject.
2. `with_seqs in {"haplotypes", "annotated"}` and (not deterministic or `haplotype_lengths` is None) → reject with a pointer to `with_settings(deterministic=True)`.
3. `mode is not None` and `num_workers > 0` → reject.
4. `max(per_batch_bytes) > slot_bytes` (a single mini-batch exceeds slot capacity) → reject with the offending batch, its byte size, and the user knobs (`batch_size↓`, `buffer_bytes↑`).

All four name the user-actionable fix.

### Runtime errors — `mode="double_buffered"`

- **Producer exception**: producer writes `(type, value, traceback)` to a `multiprocessing.Pipe` and exits. Consumer polls the pipe alongside `ready[i].wait(timeout=heartbeat_seconds)`. On exception → raise `ProducerError(original)` from the iteration loop.
- **Heartbeat**: if `ready[i]` doesn't fire within `heartbeat_seconds`, check `producer.is_alive()`. Dead + no pipe message → `ProducerDied`. Alive → keep waiting (genuinely slow chunk).
- **KeyboardInterrupt**: `__exit__` / `__del__` / `atexit` cleanup terminates the producer, unlinks both shm blocks, drains the index queue.

### Shm naming and leaks

Names: `gvl-{pid}-{uuid4.hex[:8]}-{i}`. Always `close()` then `unlink()` from the owner. Main-process `atexit` hook unlinks any registered shm regardless of exit path. We do not GC orphans from prior runs; unique naming prevents collisions.

### Persistent producer across epochs

First `__iter__` spawns; subsequent calls reuse the producer and push a fresh epoch's chunks into the index queue. `close()` / `__del__` terminates.

### Testing

All tests under `tests/unit/` unless noted. New files except where stated.

1. **`test_chunk_planner.py`** — seeded index sequence + synthetic `bytes_per_instance` table. Asserts: every chunk ≤ `slot_bytes`; mini-batch boundaries preserved; single oversized batch raises. Pure CPU, fast.
2. **`test_output_bytes_per_instance.py`** — for every `(with_seqs × with_tracks × var_fields)` combination on the existing toy dataset, the method returns exactly `numpy.nbytes` of the actual `dataset[r, s]` output. Owns the exact-footprint invariant.
3. **`test_shm_layout.py`** — round-trip every dtype/raggedness combination through a `SharedMemory` block. Cross-process via `multiprocessing.Process` for at least one variant to confirm header layout is process-stable.
4. **`test_buffered_loader.py`** — iterate full epoch in `mode="buffered"`; compare every batch to direct `dataset[r, s]` output. Cover all `with_seqs` modes, with/without tracks, with/without jitter.
5. **`test_double_buffered_loader.py`** (marked `@pytest.mark.slow`) — same batch-equivalence test for `mode="double_buffered"`, plus three failure paths:
   - Inject a `raise` inside the producer's fetch → `ProducerError` re-raised in trainer.
   - `kill -9` the producer mid-epoch → `ProducerDied` within `heartbeat_seconds`.
   - `KeyboardInterrupt` mid-iter → no leaked shm (verify via `/dev/shm` on Linux).

## Out of scope for v1

- DDP / multi-GPU. Design leaves room (per-rank producer + sharded sampler) but isn't implemented.
- Non-deterministic / random-variant-selection datasets for `haplotypes`/`annotated`. Possible later via tight bound + reserved slack.
- Spliced datasets. Blocked on `haplotype_lengths` implementation.
- Auto mode selection. User picks; we don't guess.

## File-level change summary

New:

- `python/genvarloader/_chunked.py`
- `python/genvarloader/_buffered_loader.py`
- `python/genvarloader/_shm_layout.py`
- `python/genvarloader/_double_buffered_loader.py`
- `python/genvarloader/_producer.py`
- `tests/unit/test_chunk_planner.py`
- `tests/unit/test_output_bytes_per_instance.py`
- `tests/unit/test_shm_layout.py`
- `tests/unit/test_buffered_loader.py`
- `tests/unit/test_double_buffered_loader.py`

Modified:

- `python/genvarloader/_torch.py` — `mode`/`buffer_bytes`/`copy`/`heartbeat_seconds` args on `get_dataloader`, dispatch logic.
- `python/genvarloader/_dataset/_impl.py` — new `_output_bytes_per_instance` method.
- `python/genvarloader/_dataset/_haps.py` — new private `_allele_bytes_sum` helper.

Skill update required (per CLAUDE.md "Maintaining the `genvarloader` skill"): document the new `mode`, `buffer_bytes`, `copy`, `heartbeat_seconds` arguments on `to_dataloader` and the two modes.
