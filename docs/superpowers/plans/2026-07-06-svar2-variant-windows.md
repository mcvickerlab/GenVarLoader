# svar2 variant-windows (+ unphased_union) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Dataset.open(..., svar2=...).with_seqs("variant-windows", VarWindowOpt(...))[regions, samples]` work (currently `NotImplementedError`), and enable `unphased_union` for both svar2 decode modes (`variants` and `variant-windows`).

**Architecture:** svar2 variant-windows = the already-validated svar2 `variants` decode (`decode_variants_from_svar2_readbound`) composed with the already-validated window-assembly Rust kernel (`assemble_variant_buffers`, via the `_assemble_variant_buffers_rust` shim), per contig group, feeding the decoded per-variant arrays as the kernel's "global" arrays with an **identity gather** (`v_idxs = arange(n_var)`). No new Rust. `unphased_union` is a per-group offset fold (`row_offsets[::P]`, `eff_ploidy=1`) applied before assembly, identical to SVAR1's `get_variants_flat`.

**Tech Stack:** Python 3.10, numpy, `seqpro.rag.Ragged`, PyO3 Rust extension (unchanged), pytest, pixi (`-e dev`).

**Spec:** `docs/superpowers/specs/2026-07-06-svar2-variant-windows-design.md`

## Global Constraints

- **No Rust change** — this composes existing kernels. `maturin develop --release` is only needed if the branch's `.so` is stale from prior work; run it once at the start if in doubt (pytest imports the stale `.so` otherwise — CLAUDE.md).
- **Parity is a hard gate.** svar2 read output is byte-identical to its oracle. After every change run the relevant pytest; before pushing run the full tree `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`).
- **Deletion-ALT difference:** SVAR1 keeps a deletion's anchor base (e.g. `G` for `GTA>G`); svar2 decodes `""`. So `alt`/`alt_window` bytes are **not** SVAR1-identical — only `ref_window` is. Oracle strategy per task reflects this.
- **HPC gotcha:** pass `--basetemp=$(pwd)/.pytest_tmp` to pytest so the write path's `os.link` hardlink does not fail cross-device (Errno 18).
- **Coordinate convention:** svar2 decoded `pos` is 0-based within-contig, proven `==` SVAR1 `start`; it feeds the kernel's `v_starts` directly. The per-group reference is a single-contig slice (`_ref_for_contig`), so `v_contigs = zeros` and `ref_offsets = [0, len]`.
- **Commits:** conventional-commit style. Commit after each task's tests pass. The pre-commit `pyrefly` hook spuriously fails on docs-only commits in this worktree (finds no Python files → exit 1); for commits that DO touch Python it runs normally, so do not add `--no-verify` to code commits.
- **`ref="allele"` is already blocked** upstream (`_impl.py` ~line 720 raises `ValueError` when `window_opt.ref=="allele"` and `variants.ref is None`, always true for `Svar2Haps`). No new guard; a test pins it.

---

## File Structure

- **Modify** `python/genvarloader/_dataset/_svar2_haps.py`:
  - new method `Svar2Haps._reconstruct_variant_windows(idx, regions) -> _FlatVariantWindows`;
  - `Svar2Haps.__call__`: replace the `_FlatVariantWindows` `NotImplementedError` with the jitter guard + dispatch;
  - `Svar2Haps._reconstruct_variants`: add the `unphased_union` fold;
  - `Svar2Haps._guard_unsupported`: remove the `unphased_union` clause;
  - imports: add `_assemble_variant_buffers_rust`, `_FlatWindow` from `._flat_variants`; add `_Flat` is already imported.
- **Modify** `tests/dataset/test_svar2_dataset.py`: new variant-windows + union parity tests and guard tests (reuses the module's existing `_src`/`svar_fixture`/`svar2_fixture`/`bed` and `_src2`/`svar_fixture2`/`svar2_fixture2` fixtures).
- **Modify** docs: `skills/genvarloader/SKILL.md`, `docs/source/*` where svar2 mode support is enumerated, and the two related specs' out-of-scope notes.

---

## Task 1: `_reconstruct_variant_windows` core (non-union, incl. dummy fill) + wiring + guards

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py`
- Test: `tests/dataset/test_svar2_dataset.py`

**Interfaces:**
- Consumes (existing): `decode_variants_from_svar2_readbound`, `self._gather_inputs`, `self._contig_groups`, `self._ref_for_contig`, `self._inverse_row_perm`, `_ragged_arange_src`, `_ragged_arange_gather_2level`, `lengths_to_offsets`; `_assemble_variant_buffers_rust`, `_FlatWindow`, `_FlatVariantWindows` (from `_flat_variants`); `self.window_opt`, `self.token_lut`, `self.reference`, `self.var_fields`, `self.dummy_variant`, `self.unknown_token`.
- Produces: `Svar2Haps._reconstruct_variant_windows(idx, regions) -> _FlatVariantWindows` and the wired `__call__` branch. Task 3 adds a `p_eff` fold to this same method.

- [ ] **Step 1: Write the failing test — ref_window byte-identical to SVAR1 (single-contig)**

Add to `tests/dataset/test_svar2_dataset.py`. Place a shared opt + helper near the top of the file (after imports):

```python
from genvarloader import VarWindowOpt

_WIN_OPT = VarWindowOpt(
    flank_length=3, token_alphabet=b"ACGT", unknown_token=4, ref="window", alt="window"
)


def _open_windows_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref, opt=_WIN_OPT):
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    w1 = ds1.with_output_format("flat").with_seqs("variant-windows", opt)[:, :]
    w2 = ds2.with_output_format("flat").with_seqs("variant-windows", opt)[:, :]
    return w1, w2


def _assert_window_equal(a, b, name: str) -> None:
    """Flat-buffer equality of two _FlatWindow fields (data + both offset levels)."""
    assert np.array_equal(np.asarray(a.var_offsets), np.asarray(b.var_offsets)), (
        f"{name} var_offsets differ"
    )
    assert np.array_equal(np.asarray(a.seq_offsets), np.asarray(b.seq_offsets)), (
        f"{name} seq_offsets differ"
    )
    assert np.array_equal(np.asarray(a.data), np.asarray(b.data)), f"{name} data differ"
```

Then the test:

```python
def test_svar2_variant_windows_ref_window_matches_svar1(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """ref_window is a pure reference read over an identical variant SET, so it is
    byte-identical to SVAR1 (independent of the deletion-ALT encoding difference)."""
    _bcf, ref = _src
    w1, w2 = _open_windows_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    assert w2.ref_window is not None
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window")
    # scalar start field also identical (same variant SET) — compare _Flat buffers.
    assert np.array_equal(
        np.asarray(w2.fields["start"].data), np.asarray(w1.fields["start"].data)
    )
    assert np.array_equal(
        np.asarray(w2.fields["start"].offsets), np.asarray(w1.fields["start"].offsets)
    )
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py::test_svar2_variant_windows_ref_window_matches_svar1" -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL with `NotImplementedError: svar2 datasets do not support with_seqs('variant-windows') yet.`

- [ ] **Step 3: Add imports to `_svar2_haps.py`**

At the existing import of `_FlatVariantWindows` (line ~50), extend it:

```python
from ._flat_variants import (
    _FlatVariantWindows,
    _FlatWindow,
    _assemble_variant_buffers_rust,
)
```

(`_Flat` and `lengths_to_offsets` are already imported at the top of the file.)

- [ ] **Step 4: Implement `_reconstruct_variant_windows`**

Add this method to `Svar2Haps`, right after `_reconstruct_variants` (before `# ---- helpers ----`):

```python
def _reconstruct_variant_windows(
    self, idx: NDArray[np.integer], regions: NDArray[np.integer]
) -> _FlatVariantWindows:
    """Variant-windows for svar2: decode variants per contig group, then run the
    shared ``assemble_variant_buffers`` window kernel over the decoded arrays via
    an identity gather. ``ref="allele"`` is blocked upstream, so ref is always a
    reference-read window; ``alt`` follows ``window_opt.alt``.
    """
    assert self.window_opt is not None and self.token_lut is not None
    assert self.reference is not None
    from typing import Any

    opt = self.window_opt
    L = opt.flank_length
    ref_mode = 1  # ref="window" (ref="allele" rejected in with_seqs)
    alt_mode = 1 if opt.alt == "window" else 2
    include_ilen = "ilen" in self.var_fields

    regions = np.asarray(regions, np.int32)
    P = int(self.genotypes.shape[-2])
    b = len(idx)
    R_all, S_all = int(self.genotypes.shape[0]), int(self.genotypes.shape[1])
    r_q, si_q = np.unravel_index(np.asarray(idx), (R_all, S_all))
    contig_ids = regions[:, 0].astype(np.int64)
    groups = self._contig_groups(contig_ids)

    p_eff = P  # unphased_union fold (Task 3) sets this to 1 per group.

    cat_row_off: list[NDArray[np.int64]] = []  # per-group var boundaries
    cat_pos: list[NDArray[np.int32]] = []
    cat_ilen: list[NDArray[np.int32]] = []
    cat_query_order: list[NDArray[np.intp]] = []
    # name -> per-group (token_data, per-variant seq offsets)
    win_data: dict[str, list[NDArray]] = {}
    win_seq_off: dict[str, list[NDArray[np.int64]]] = {}

    for ci, qsel in groups:
        gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)
        pos, ilen, alt_bytes, str_off, var_off = (
            decode_variants_from_svar2_readbound(
                self.store,
                self.ds_contigs[ci],
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                P,
            )
        )
        pos = np.asarray(pos, np.int32)
        ilen = np.asarray(ilen, np.int32)
        alt_bytes = np.asarray(alt_bytes, np.uint8)
        str_off = np.asarray(str_off, np.int64)
        var_off = np.asarray(var_off, np.int64)

        row_off = var_off  # Task 3: fold to var_off[::P] under unphased_union.
        n_var = int(len(pos))
        ref_, ref_offsets = self._ref_for_contig(ci)
        bufs = _assemble_variant_buffers_rust(
            1,  # windows mode
            np.arange(n_var, dtype=np.int32),  # identity v_idxs (data pre-gathered)
            row_off,
            alt_bytes,  # alt_global
            str_off,  # alt_off_global
            None,  # ref_global (ref="window")
            None,  # ref_off_global
            False,  # want_ref_bytes
            False,  # want_flank
            ref_mode,
            alt_mode,
            L,
            self.token_lut,
            np.zeros(n_var, np.int32),  # v_contigs (single-contig ref slice)
            pos,  # v_starts
            ilen,  # ilens
            ref_,
            ref_offsets,
            self.reference.pad_char,
        )

        cat_row_off.append(row_off)
        cat_pos.append(pos)
        cat_ilen.append(ilen)
        cat_query_order.append(qsel)
        for name, (data, seq_off) in bufs.items():
            win_data.setdefault(name, []).append(np.asarray(data))
            win_seq_off.setdefault(name, []).append(np.asarray(seq_off, np.int64))

    shape: tuple[int | None, ...] = (b, p_eff, None)
    wshape: tuple[int | None, ...] = (b, p_eff, None, None)

    # Single contig group: grouped order already equals global (b, p_eff) order.
    if len(cat_pos) == 1:
        row_off = cat_row_off[0]
        fields: dict[str, Any] = {
            "start": _Flat.from_offsets(cat_pos[0], shape, row_off)
        }
        if include_ilen:
            fields["ilen"] = _Flat.from_offsets(cat_ilen[0], shape, row_off)
        win = _FlatVariantWindows(fields)
        for name in win_data:
            setattr(
                win,
                name,
                _FlatWindow(win_data[name][0], win_seq_off[name][0], row_off, wshape),
            )
    else:
        perm = self._inverse_row_perm(cat_query_order, b, p_eff)
        grouped_row_off = lengths_to_offsets(
            np.concatenate([np.diff(r) for r in cat_row_off]), np.int64
        )
        pos_c = np.concatenate(cat_pos)
        ilen_c = np.concatenate(cat_ilen)
        src, row_off_g = _ragged_arange_src(grouped_row_off, perm)
        if src.size == 0:
            pos_g = pos_c[:0].copy()
            ilen_g = ilen_c[:0].copy()
        else:
            pos_g = pos_c[src]
            ilen_g = ilen_c[src]
        fields = {"start": _Flat.from_offsets(pos_g, shape, row_off_g)}
        if include_ilen:
            fields["ilen"] = _Flat.from_offsets(ilen_g, shape, row_off_g)
        win = _FlatVariantWindows(fields)
        for name in win_data:
            data_c = np.concatenate(win_data[name])
            grouped_seq_off = lengths_to_offsets(
                np.concatenate([np.diff(s) for s in win_seq_off[name]]), np.int64
            )
            nd, nvar, nseq = _ragged_arange_gather_2level(
                data_c, grouped_row_off, grouped_seq_off, perm
            )
            setattr(win, name, _FlatWindow(nd, nseq, nvar, wshape))

    if self.dummy_variant is not None:
        win = win.fill_empty_groups(
            self.dummy_variant, unk=self.unknown_token, flank_length=L
        )
    return win
```

- [ ] **Step 5: Wire `__call__` — replace the NotImplementedError with the guard + dispatch**

In `Svar2Haps.__call__`, replace this block:

```python
        if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):
            if issubclass(self.kind, _FlatVariantWindows):
                raise NotImplementedError(
                    "svar2 datasets do not support with_seqs('variant-windows') yet."
                )
```

with:

```python
        if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):
            # variants AND variant-windows decode variants; the read-bound decode
            # has NO right-clip, so max_jitter>0 / jitter>0 would over-include
            # variants past the (unpadded) read window. Guard both modes.
            if self.max_jitter > 0 or jitter > 0:
                raise NotImplementedError(
                    "variants/variant-windows output for svar2 datasets written with"
                    f" max_jitter>0 (here max_jitter={self.max_jitter}) or read with"
                    f" jitter>0 (here jitter={jitter}) is not yet supported: the"
                    " read-bound decode does not right-clip to the post-jitter window."
                )
            if issubclass(self.kind, _FlatVariantWindows):
                return cast(_H, self._reconstruct_variant_windows(idx, regions))
```

Then DELETE the now-duplicated jitter guard that follows in the old `RaggedVariants` branch (the block starting `# ``decode_variants_from_svar2_readbound`` has NO right-clip` down through its `raise NotImplementedError(... "right-clip" ...)`), since the guard now runs once above for both kinds. Keep the trailing `return cast(_H, self._reconstruct_variants(idx, regions))`.

The resulting branch reads:

```python
        if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):
            if self.max_jitter > 0 or jitter > 0:
                raise NotImplementedError(
                    "variants/variant-windows output for svar2 datasets written with"
                    f" max_jitter>0 (here max_jitter={self.max_jitter}) or read with"
                    f" jitter>0 (here jitter={jitter}) is not yet supported: the"
                    " read-bound decode does not right-clip to the post-jitter window."
                )
            if issubclass(self.kind, _FlatVariantWindows):
                return cast(_H, self._reconstruct_variant_windows(idx, regions))
            # RaggedVariants: RC is applied by the caller (_getitem_unspliced),
            # so to_rc is intentionally ignored here (mirrors SVAR1 Haps).
            return cast(_H, self._reconstruct_variants(idx, regions))
```

- [ ] **Step 6: Run the ref_window test to confirm it passes**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py::test_svar2_variant_windows_ref_window_matches_svar1" -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS

- [ ] **Step 7: Add the alt-window decomposition test (correctness of the alt assembly)**

`alt_window` is NOT SVAR1-identical (deletion ALT differs), so verify it against svar2's own outputs: since tokenization is per-byte independent, `alt_window[j] == ref_window[j][:L] · alt_allele[j] · ref_window[j][-L:]`, where `alt_allele` is the bare tokenized alt (a second svar2 windows read with `alt="allele"`) and `ref_window` is already validated against SVAR1 in Step 1.

```python
def test_svar2_variant_windows_alt_window_decomposition(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """alt_window[j] == ref_window[j][:L] + tokenize(alt_j) + ref_window[j][-L:].
    Uses only svar2's own outputs; ref_window is separately pinned to SVAR1."""
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    L = _WIN_OPT.flank_length
    w_win = ds2.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    alt_opt = VarWindowOpt(
        flank_length=L, token_alphabet=b"ACGT", unknown_token=4, ref="window", alt="allele"
    )
    w_alt = ds2.with_output_format("flat").with_seqs("variant-windows", alt_opt)[:, :]

    rw = w_win.ref_window
    aw = w_win.alt_window
    ba = w_alt.alt  # bare tokenized alt (_FlatWindow)
    assert aw is not None and rw is not None and ba is not None

    # Same variant SET/order across the two reads.
    assert np.array_equal(np.asarray(aw.var_offsets), np.asarray(ba.var_offsets))
    n_var = len(np.asarray(aw.seq_offsets)) - 1
    rso, aso, bso = (
        np.asarray(rw.seq_offsets),
        np.asarray(aw.seq_offsets),
        np.asarray(ba.seq_offsets),
    )
    rd, ad, bd = np.asarray(rw.data), np.asarray(aw.data), np.asarray(ba.data)
    for j in range(n_var):
        rj = rd[rso[j] : rso[j + 1]]
        aj = ad[aso[j] : aso[j + 1]]
        bj = bd[bso[j] : bso[j + 1]]
        expected = np.concatenate([rj[:L], bj, rj[len(rj) - L :]])
        assert np.array_equal(aj, expected), f"alt_window variant {j} mismatch"
```

- [ ] **Step 8: Add the bare-alt tokenization test (pins alt bytes → tokens)**

Confirms the bare `alt` equals `tokenize(variants.alt)`, tying the window alt bytes to the validated variants decode.

```python
def test_svar2_variant_windows_bare_alt_tokenizes_variants_alt(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    import awkward as ak

    from genvarloader._dataset._flat_flanks import build_token_lut

    _bcf, ref = _src
    _, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    L = _WIN_OPT.flank_length
    alt_opt = VarWindowOpt(
        flank_length=L, token_alphabet=b"ACGT", unknown_token=4, ref="window", alt="allele"
    )
    w_alt = ds2.with_output_format("flat").with_seqs("variant-windows", alt_opt)[:, :]
    v = ds2.with_seqs("variants")[:, :]  # RaggedVariants (validated)

    lut, _ = build_token_lut(b"ACGT", 4)
    # Per (b,p) row, list of alt byte-strings, in variant order.
    alt_rows = ak.to_list(v.alt.to_ak())  # nested (b, p) -> [bytes,...]
    flat_alts: list[bytes] = []
    for per_ploid in alt_rows:
        for per_var in per_ploid:
            for a in per_var:
                flat_alts.append(bytes(a) if not isinstance(a, bytes) else a)

    ba = w_alt.alt
    bso, bd = np.asarray(ba.seq_offsets), np.asarray(ba.data)
    assert len(flat_alts) == len(bso) - 1
    for j, a in enumerate(flat_alts):
        toks = bd[bso[j] : bso[j + 1]]
        expected = np.array([lut[byte] for byte in a], dtype=toks.dtype)
        assert np.array_equal(toks, expected), f"bare alt variant {j} mismatch"
```

- [ ] **Step 9: Add multi-contig parity test**

```python
def test_svar2_variant_windows_multicontig(tmp_path, svar_fixture2, svar2_fixture2, _src2):
    """ref_window byte-identical to SVAR1 across an interleaved 2-contig bed
    (single-contig fast path bypassed -> exercises the group-stitch reorder)."""
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 10, 5],
            "chromEnd": [40, 40, 40, 20],
        }
    )
    d1 = tmp_path / "vw_mc1.gvl"
    d2 = tmp_path / "vw_mc2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True)
    gvl.write(d2, bed, variants=SparseVar2(svar2_fixture2), samples=None, overwrite=True)
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)
    w1 = ds1.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    w2 = ds2.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window")
    # alt_window decomposition holds across the stitch too.
    w2.alt_window.to_ragged()  # offsets/data consistent post-reorder
    w2.ref_window.to_ragged()
```

- [ ] **Step 10: Add the dummy-variant empty-group fill test**

The module `bed` has a variant-free tail (`chr1 [20, 40)`), so with a `dummy_variant` set, its rows must each get one all-`unknown_token` window of width `2L + len(dummy allele)`.

```python
def test_svar2_variant_windows_dummy_fills_empty_groups(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    from genvarloader import DummyVariant

    _bcf, ref = _src
    _, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    L = _WIN_OPT.flank_length
    dummy = DummyVariant(alt=b"N", ref=b"N")
    w = (
        ds2.with_output_format("flat")
        .with_settings(dummy_variant=dummy)
        .with_seqs("variant-windows", _WIN_OPT)[:, :]
    )
    # Every (b*p) row now has >= 1 variant (no empty rows).
    vo = np.asarray(w.ref_window.var_offsets)
    assert np.all(np.diff(vo) >= 1)
    # ref_window dummy width = 2L + len(dummy.ref); alt_window = 2L + len(dummy.alt).
    # (For a filled row the sole variant's window length equals the dummy width.)
    # Assert at least one dummy-width ref window exists (the tail region rows).
    rso = np.asarray(w.ref_window.seq_offsets)
    assert (np.diff(rso) == (2 * L + len(dummy.ref))).any()
    w.ref_window.to_ragged()
    w.alt_window.to_ragged()
```

- [ ] **Step 11: Add the guard tests (ref="allele" ValueError, jitter NotImplementedError)**

```python
def test_svar2_variant_windows_ref_allele_guard(tmp_path, bed, svar2_fixture, _src):
    """ref='allele' needs stored REF bytes svar2 lacks -> ValueError at with_seqs."""
    from genoray import SparseVar2

    _bcf, ref = _src
    d = tmp_path / "d.gvl"
    gvl.write(d, bed, variants=SparseVar2(svar2_fixture), samples=None, overwrite=True)
    ds = gvl.Dataset.open(d, reference=ref).with_output_format("flat")
    bad = VarWindowOpt(
        flank_length=3, token_alphabet=b"ACGT", unknown_token=4, ref="allele", alt="window"
    )
    with pytest.raises(ValueError, match="REF"):
        ds.with_seqs("variant-windows", bad)


def test_svar2_variant_windows_jitter_guard(tmp_path, svar2_fixture, _src):
    """variant-windows must raise when written with max_jitter>0 (no right-clip)."""
    from genoray import SparseVar2

    _bcf, ref = _src
    jbed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [5], "chromEnd": [20]})
    d = tmp_path / "d.gvl"
    gvl.write(
        d, jbed, variants=SparseVar2(svar2_fixture), samples=None, max_jitter=2, overwrite=True
    )
    ds = gvl.Dataset.open(d, reference=ref).with_output_format("flat")
    with pytest.raises(NotImplementedError, match="right-clip"):
        ds.with_seqs("variant-windows", _WIN_OPT)[:, :]
```

- [ ] **Step 12: Run the full Task-1 test set**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py" -v -k "variant_windows" --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (7 new tests). Also run the existing svar2 suite to confirm no regression:
`pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q --basetemp=$(pwd)/.pytest_tmp` → all pass.

- [ ] **Step 13: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_svar2_haps.py tests/dataset/test_svar2_dataset.py
git commit -m "feat(svar2): variant-windows read path (ref=window, alt window/allele)

Compose decode_variants_from_svar2_readbound with assemble_variant_buffers
(identity gather) per contig group. ref_window byte-identical to SVAR1;
alt validated via ref-flank decomposition + tokenized variants.alt. Wire
__call__ (jitter guard shared with variants), pin ref=allele + jitter guards.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: unphased_union for svar2 `variants` mode

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`_reconstruct_variants`, `_guard_unsupported`)
- Test: `tests/dataset/test_svar2_dataset.py`

**Interfaces:**
- Consumes: `self.unphased_union` (inherited `Haps` field, set via `with_settings`), the existing `_reconstruct_variants` structure.
- Produces: `_reconstruct_variants` honoring the ploidy-1 fold; `_guard_unsupported` no longer raises on `unphased_union`.

- [ ] **Step 1: Write the failing test — variants union vs SVAR1**

```python
def test_svar2_variants_unphased_union_matches_svar1(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """Ploidy-1 union: start/ilen byte-identical to SVAR1 union (order-preserving
    fold, no dedup). ALT differs by encoding, so ALT is not compared to SVAR1."""
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_seqs("variants").with_settings(unphased_union=True)[:, :]
    b = ds2.with_seqs("variants").with_settings(unphased_union=True)[:, :]
    # Ploidy axis folded 2 -> 1.
    assert a.start.shape[-2] == 1 and b.start.shape[-2] == 1
    _assert_ragged_equal(a.start.to_packed(), b.start.to_packed(), "start")
    _assert_ragged_equal(a.ilen.to_packed(), b.ilen.to_packed(), "ilen")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py::test_svar2_variants_unphased_union_matches_svar1" -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL with `NotImplementedError: unphased_union is not supported for svar2 datasets yet.`

- [ ] **Step 3: Remove the unphased_union clause from `_guard_unsupported`**

In `Svar2Haps._guard_unsupported`, delete:

```python
        if self.unphased_union:
            raise NotImplementedError(
                "unphased_union is not supported for svar2 datasets yet."
            )
```

(Haplotypes/annotated + union is still blocked at `with_seqs` in `_impl.py`, so the haplotypes/tracks paths can never reach here with the flag set.)

- [ ] **Step 4: Add the fold to `_reconstruct_variants`**

In `_reconstruct_variants`, after `P = int(self.genotypes.shape[-2])` and after the `groups = self._contig_groups(...)` line, introduce `p_eff`. Then in the per-group loop, fold `var_off` before appending, and use `p_eff` for the shape and stitch. Concretely:

1. After `groups = self._contig_groups(contig_ids)` add:
   ```python
   p_eff = 1 if self.unphased_union else P
   ```
2. In the loop, the group currently does `var_off = np.asarray(var_off, np.int64)` then `cat_var_lens.append(np.diff(var_off))`. Insert the fold immediately after `var_off = np.asarray(var_off, np.int64)`:
   ```python
   if self.unphased_union:
       var_off = np.ascontiguousarray(var_off[::P])
   ```
   (This keeps every P-th boundary: hap-0's then hap-1's variants per query, concatenated. `pos`/`ilen`/`alt` data and `str_off` are untouched — only row grouping changes.)
3. Replace both `shape = (b, P, None)` occurrences (single-group fast path and multi-group path) with `shape = (b, p_eff, None)`.
4. Replace the single `perm = self._inverse_row_perm(cat_query_order, b, P)` with `perm = self._inverse_row_perm(cat_query_order, b, p_eff)`.

- [ ] **Step 5: Run the union test + the existing variants parity tests**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py" -v -k "variants" --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS — the new union test plus the existing `test_svar2_variants_positions_match_svar1`, `test_svar2_variants_match_svar2_oracle`, `test_svar2_variants_jitter_guard_raises` all still pass (p_eff=P when the flag is off preserves the diploid path byte-for-byte).

- [ ] **Step 6: Commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_svar2_haps.py tests/dataset/test_svar2_dataset.py
git commit -m "feat(svar2): unphased_union for variants mode (ploidy-1 fold)

Fold row_offsets[::P], eff_ploidy=1 per contig group (order-preserving,
no dedup) — byte-identical to SVAR1 union for start/ilen. Drop the
unphased_union guard; haplotypes/annotated+union stays blocked upstream.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: unphased_union for svar2 `variant-windows`

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py` (`_reconstruct_variant_windows`)
- Test: `tests/dataset/test_svar2_dataset.py`

**Interfaces:**
- Consumes: `_reconstruct_variant_windows` from Task 1 (the `p_eff = P` placeholder and `row_off = var_off` comment mark the fold points), `self.unphased_union`.
- Produces: variant-windows honoring the ploidy-1 fold.

- [ ] **Step 1: Write the failing test — windows union**

```python
def test_svar2_variant_windows_unphased_union(
    tmp_path, bed, svar_fixture, svar2_fixture, _src
):
    """Union folds ploidy 2->1 for windows; ref_window still byte-identical to
    SVAR1 union, and the union row is hap-0's windows then hap-1's, concatenated."""
    _bcf, ref = _src
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    w1 = (
        ds1.with_output_format("flat")
        .with_settings(unphased_union=True)
        .with_seqs("variant-windows", _WIN_OPT)[:, :]
    )
    w2 = (
        ds2.with_output_format("flat")
        .with_settings(unphased_union=True)
        .with_seqs("variant-windows", _WIN_OPT)[:, :]
    )
    # Ploidy axis folded 2 -> 1. Scalar shape is (R,S,p_eff,None) so ploidy is at
    # [-2]; window shape is (R,S,p_eff,None,None) so ploidy is at [-3].
    assert w2.fields["start"].shape[-2] == 1
    assert w2.ref_window.shape[-3] == 1
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window")
    # Union row count == sum over haplotypes: compare to the non-union var counts.
    nu = np.asarray(w2.ref_window.var_offsets)
    w2_diploid = ds2.with_output_format("flat").with_seqs("variant-windows", _WIN_OPT)[:, :]
    nd = np.asarray(w2_diploid.ref_window.var_offsets)
    P = int(ds2._seqs.genotypes.shape[-2])
    # Folded per-row counts == sum of the P per-hap counts (rows q*P+p are contiguous).
    diploid_counts = np.diff(nd).reshape(-1, P).sum(1)
    union_counts = np.diff(nu)
    assert np.array_equal(union_counts, diploid_counts)
    w2.ref_window.to_ragged()
    w2.alt_window.to_ragged()
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py::test_svar2_variant_windows_unphased_union" -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — the ploidy axis is still 2 (`assert ... shape[-2] == 1` fails), because Task 1 hardcodes `p_eff = P` and `row_off = var_off`.

- [ ] **Step 3: Apply the fold in `_reconstruct_variant_windows`**

Two edits in the method:

1. Replace `p_eff = P  # unphased_union fold (Task 3) sets this to 1 per group.` with:
   ```python
   p_eff = 1 if self.unphased_union else P
   ```
2. Replace `row_off = var_off  # Task 3: fold to var_off[::P] under unphased_union.` with:
   ```python
   row_off = np.ascontiguousarray(var_off[::P]) if self.unphased_union else var_off
   ```

(The `v_idxs = arange(n_var)` identity, `pos`/`ilen`/`alt_bytes`, and the assemble call are unchanged — folding `row_off` only regroups the per-row variant boundaries the kernel emits. `shape`/`wshape` already use `p_eff`, and the multi-group stitch already uses `_inverse_row_perm(cat_query_order, b, p_eff)`.)

- [ ] **Step 4: Run the windows union test + the whole variant_windows set**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py" -v -k "variant_windows" --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (Task-1 tests unaffected — `p_eff=P` when the flag is off — plus the new union test).

- [ ] **Step 5: Add a multi-contig + union test (locks the p_eff=1 stitch)**

```python
def test_svar2_variant_windows_union_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    bed = pl.DataFrame(
        {"chrom": ["chr2", "chr1"], "chromStart": [0, 0], "chromEnd": [40, 40]}
    )
    d1 = tmp_path / "vwu_mc1.gvl"
    d2 = tmp_path / "vwu_mc2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True)
    gvl.write(d2, bed, variants=SparseVar2(svar2_fixture2), samples=None, overwrite=True)
    ds1 = gvl.Dataset.open(d1, reference=ref)
    ds2 = gvl.Dataset.open(d2, reference=ref)
    w1 = (ds1.with_output_format("flat").with_settings(unphased_union=True)
          .with_seqs("variant-windows", _WIN_OPT)[:, :])
    w2 = (ds2.with_output_format("flat").with_settings(unphased_union=True)
          .with_seqs("variant-windows", _WIN_OPT)[:, :])
    assert w2.ref_window.shape[-3] == 1  # window ploidy axis
    _assert_window_equal(w2.ref_window, w1.ref_window, "ref_window (union, multicontig)")
    w2.alt_window.to_ragged()
```

- [ ] **Step 6: Run + commit**

Run: `pixi run -e dev pytest "tests/dataset/test_svar2_dataset.py" -v -k "variant_windows" --basetemp=$(pwd)/.pytest_tmp` → PASS.

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_svar2_haps.py tests/dataset/test_svar2_dataset.py
git commit -m "feat(svar2): unphased_union for variant-windows (ploidy-1 fold)

Fold row_offsets[::P] before the window assemble call; p_eff=1 drives
shape + stitch. ref_window stays SVAR1-identical under union.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: docs, skill, and spec out-of-scope updates

**Files:**
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `docs/source/*.md` (whichever enumerate svar2 output-mode support — grep first)
- Modify: `docs/superpowers/specs/2026-07-05-svar2-readbound-getitem-perf-design.md` (out-of-scope note)

**Interfaces:** none (docs only).

- [ ] **Step 1: Find where svar2 mode support / variant-windows is documented**

Run:
```bash
grep -rn "variant-windows\|svar2\|SVAR2\|NotImplementedError" skills/genvarloader/SKILL.md docs/source/*.md | grep -i "svar2\|variant-windows"
```
Read each hit and note which claim "svar2 does not support variant-windows / unphased_union" (or list supported modes).

- [ ] **Step 2: Update the skill**

In `skills/genvarloader/SKILL.md`, wherever svar2's supported output modes or "not supported" gotchas are listed, state that svar2 now supports `variant-windows` (`ref="window"`, `alt ∈ {window, allele}`) and `unphased_union` (for `variants` and `variant-windows`). Keep the still-unsupported list accurate: `ref="allele"`, `min_af`/`max_af`, spliced, annotated, in-kernel RC, and `max_jitter>0`/`jitter>0` for the variants/variant-windows decode.

- [ ] **Step 3: Update user docs**

Apply the same correction to any `docs/source/*.md` hit from Step 1 (e.g. a support matrix in `dataset.md`/`faq.md`). If no doc enumerates svar2 mode support, note that in the commit message and skip.

- [ ] **Step 4: Update the read-bound perf spec's out-of-scope note**

In `docs/superpowers/specs/2026-07-05-svar2-readbound-getitem-perf-design.md` §2, the "Out of scope" line lists `variant-windows` and `unphased_union` among guarded modes. Add a parenthetical that these are now implemented (see `2026-07-06-svar2-variant-windows-design.md`); the perf/fusion work remains deferred.

- [ ] **Step 5: Verify api.md unchanged is correct**

No new public symbol is added, so `api.md`/`__all__` need no change. Confirm:
```bash
python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`.

- [ ] **Step 6: Commit**

```bash
git add skills/genvarloader/SKILL.md docs/source/ docs/superpowers/specs/2026-07-05-svar2-readbound-getitem-perf-design.md
git commit -m "docs(svar2): variant-windows + unphased_union now supported

Update skill + user docs mode matrix; note the read-bound perf spec's
out-of-scope items are implemented (fusion still deferred).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification (after all tasks)

- [ ] **Full svar2 suite**

Run: `pixi run -e dev pytest tests -k svar2 -q --basetemp=$(pwd)/.pytest_tmp`
Expected: all pass.

- [ ] **Full tree (catches stale references in tests/unit/)**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: all pass (matches the pre-change green baseline: ~1030 pytest).

- [ ] **Typecheck + cargo (insurance; no Rust changed)**

Run: `pixi run -e dev typecheck && pixi run -e dev cargo-test`
Expected: pass.

---

## Self-Review notes (for the executor)

- **Spec coverage:** Task 1 = windows core + wiring + guards (§3 core, §4 ref_window/alt oracle, §4.5 guards); Task 2 = variants union (§3 "_reconstruct_variants gets the same fold", §4.6); Task 3 = windows union (§3 step 1a, §4.6); Task 4 = docs (§6). `ref="allele"` (blocked upstream) and jitter guards are pinned by tests in Task 1.
- **Type consistency:** `_reconstruct_variant_windows` returns `_FlatVariantWindows`; `_FlatWindow(data, seq_offsets, var_offsets, shape)` positional order matches the dataclass in `_flat_variants.py`; `_assemble_variant_buffers_rust` arg order matches its definition (mode, v_idxs, row_offsets, alt_global, alt_off_global, ref_global, ref_off_global, want_ref_bytes, want_flank, ref_mode, alt_mode, flank_len, lut, v_contigs, v_starts, ilens, reference, ref_offsets, pad_char).
- **Union fold identity:** `p_eff` and `row_off = var_off[::P]` are the ONLY union-specific changes in each reconstruct method; when the flag is off both reduce to the pre-existing diploid path, so the existing parity tests must remain green (a regression signal).
