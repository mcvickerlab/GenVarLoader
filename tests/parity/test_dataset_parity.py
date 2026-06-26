"""Dataset read-path parity backstops for track kernels.

Covers three cases:

1. ``intervals_to_tracks`` only (track-only dataset, no variants):
   Proves that flipping GVL_BACKEND produces byte-identical tracks through
   the real Dataset.__getitem__ path.

2. ``shift_and_realign_tracks_sparse`` (haplotypes+tracks dataset with indels):
   Proves that the dispatch wiring for the realignment kernel is correct
   end-to-end, across every insertion-fill strategy.

3. Strand=−1 parity backstops (Task 7 — pre-wiring safety net):
   Proves that flipping GVL_BACKEND produces byte-identical output for datasets
   with mixed + and − strand regions, across all five output kinds
   (reference, haplotypes, annotated, tracks, tracks-seqs) in the UNSPLICED
   path, and across the four splice-capable kinds (reference, haplotypes,
   annotated, tracks) in the SPLICED path.  Both backends currently apply RC as
   a Python post-pass in ``_query._getitem_unspliced`` / ``_getitem_spliced``;
   these tests establish the regression net that Task 8 kernel-level RC wiring
   must keep green.  Each path also carries a non-vacuity assertion (output
   differs from the forward orientation AND equals the exact reverse-complement
   on a non-palindromic −strand region/transcript).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity._fixtures import (
    build_haps_tracks_dataset,
    build_strand_mixed_dataset,
    build_track_dataset,
)

pytestmark = pytest.mark.parity


def _read_track_array(
    ds, r_idx: np.ndarray, s_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (data, offsets) from the RaggedTracks produced by ds[r_idx, s_idx].

    Dataset.open with no reference and no variants + with_tracks("signal") returns
    a RaggedTracks directly from __getitem__.  RaggedTracks is a Ragged[np.float32]
    so it carries .data (flat float32 buffer) and .offsets (int64).
    """
    result = ds[r_idx, s_idx]
    # result is RaggedTracks (a seqpro Ragged[np.float32]) when no seqs are configured
    data = np.asarray(result.data, dtype=np.float32)
    offsets = np.asarray(result.offsets, dtype=np.int64)
    return data, offsets


def test_track_getitem_identical_across_backends(tmp_path, monkeypatch):
    ds_dir = build_track_dataset(tmp_path)

    import genvarloader as gvl
    import genvarloader._dataset._reconstruct as _recon_mod
    import genvarloader._dataset._tracks as _tracks_mod

    ds = gvl.Dataset.open(ds_dir)
    # tracks-only dataset: with_tracks enables the signal track explicitly
    ds = ds.with_tracks("signal")

    # Use slice(None) for both dims so Dataset uses "basic" indexing (cross-product)
    # which returns shape (n_regions, n_samples, n_tracks, ~length).
    r_idx = slice(None)
    s_idx = slice(None)

    # --- spy: assert intervals_to_tracks is actually called on the live read path ---
    calls: dict[str, int] = {"n": 0}

    def _make_spy(orig):
        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        return spy

    # Patch BOTH call-site modules; the track-only path uses _tracks_mod
    monkeypatch.setattr(
        _tracks_mod, "intervals_to_tracks", _make_spy(_tracks_mod.intervals_to_tracks)
    )
    monkeypatch.setattr(
        _recon_mod, "intervals_to_tracks", _make_spy(_recon_mod.intervals_to_tracks)
    )

    # --- numba read ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    data_n, off_n = _read_track_array(ds, r_idx, s_idx)

    # Backstop guard: kernel must have been called at least once
    assert calls["n"] > 0, (
        f"intervals_to_tracks was NEVER called during the numba read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the read path and confirm the track reconstructor is active."
    )

    # --- rust read ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    data_r, off_r = _read_track_array(ds, r_idx, s_idx)

    # --- byte-identical comparison ---
    np.testing.assert_array_equal(
        off_n, off_r, err_msg="offsets differ across backends"
    )
    assert data_n.dtype == data_r.dtype == np.float32, (
        f"dtype mismatch: numba={data_n.dtype}, rust={data_r.dtype}"
    )
    np.testing.assert_array_equal(
        data_n, data_r, err_msg="track data differs across backends"
    )

    # Sanity: the read painted real non-zero signal (not an all-zero vacuous match)
    assert np.any(data_n != 0.0), (
        "Track data is all-zero — regions may not overlap synthetic intervals. "
        "Non-zero signal is required to prove the comparison is meaningful."
    )


# ---------------------------------------------------------------------------
# Haplotypes+tracks realignment backstop
# ---------------------------------------------------------------------------


def test_tracks_realign_getitem_identical_across_backends(
    synthetic_case, tmp_path, monkeypatch
):
    """Spy-guarded backstop for tracks realignment dispatch wiring (Task 11/14).

    Proves that materialising a haplotypes+tracks dataset (with indel-bearing
    genotypes) via ``ds[:, :]`` produces byte-identical track output across
    GVL_BACKEND=rust and GVL_BACKEND=numba, for every insertion-fill strategy.

    After Task 14, the Rust path calls the fused entry
    ``intervals_and_realign_track_fused`` (one FFI crossing per track) instead
    of the composed ``shift_and_realign_tracks_sparse`` dispatch.  The spy
    targets ``intervals_and_realign_track_fused`` on the Rust path.

    The numba path continues to use the composed path (intervals_to_tracks
    → shift_and_realign_tracks_sparse via dispatch); the parity check
    (byte-identical output) remains the gate.

    Fixture geometry:
    - A fresh GVL dataset is built in tmp_path via gvl.write with both the
      session SparseVar variants (which contain indels on chr1/chr2) and a
      synthetic BigWig ``signal`` track for samples s0/s1/s2.
    - max_jitter=0 is used to avoid the pre-existing intervals_to_tracks
      landmine: with max_jitter>0, gvl.write clips BigWig intervals to the
      jitter-expanded region boundaries (chromStart - max_jitter), but
      Dataset.open derives _full_regions from the original chromStart.  The
      gap of max_jitter bp causes stored interval starts to precede the
      query start, violating the Rust kernel contract and triggering a
      PanicException.  With max_jitter=0 the boundaries match exactly.

    Fill strategies covered: all 5 (Repeat5p, Repeat5pNormalized, Constant,
    FlankSample, Interpolate).  Each is set via with_insertion_fill and the
    byte-identical comparison is re-run.
    """
    import genvarloader as gvl
    import genvarloader._dataset._reconstruct as _recon_mod
    from genvarloader._dataset._insertion_fill import (
        Constant,
        FlankSample,
        Interpolate,
        Repeat5p,
        Repeat5pNormalized,
    )

    # --- build fixture: fresh variants+tracks dataset with max_jitter=0 ---
    ds_dir = build_haps_tracks_dataset(tmp_path, synthetic_case.svar_path)

    # Open with the session reference so haplotype reconstruction runs.
    # Use synthetic_case.ref_path to get the same reference used to build
    # the variants, not the pre-committed tests/data/fasta reference.
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds_base = gvl.Dataset.open(ds_dir, reference=ref)
    ds_base = ds_base.with_seqs("haplotypes").with_tracks("signal")

    # --- install spy on the fused Rust entry ---
    # After Task 14 the Rust path calls intervals_and_realign_track_fused
    # directly (not via _dispatch), so we monkeypatch _recon_mod.
    orig_fused = getattr(_recon_mod, "intervals_and_realign_track_fused", None)
    assert orig_fused is not None, (
        "intervals_and_realign_track_fused not found on _recon_mod — "
        "ensure it is imported at module level in _reconstruct.py"
    )

    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

    # All 5 insertion-fill strategies to cover.
    fill_strategies = [
        Repeat5p(),
        Repeat5pNormalized(),
        Constant(0.0),
        FlankSample(flank_width=5),
        Interpolate(order=1),
    ]

    for strategy in fill_strategies:
        strategy_name = type(strategy).__name__
        ds = ds_base.with_insertion_fill(strategy)

        monkeypatch.setattr(_recon_mod, "intervals_and_realign_track_fused", _spy_fused)
        calls["n"] = 0  # reset per-strategy counter

        # --- rust read (fused path, spy active) ---
        monkeypatch.setenv("GVL_BACKEND", "rust")
        out_rust = ds[:, :]

        rust_call_count = calls["n"]

        # --- numba read (composed path — spy must NOT fire) ---
        monkeypatch.setenv("GVL_BACKEND", "numba")
        out_numba = ds[:, :]

        # Wiring guard: numba must NOT fire the fused spy.
        assert calls["n"] == rust_call_count, (
            f"[{strategy_name}] intervals_and_realign_track_fused spy fired during "
            f"the numba read (count went from {rust_call_count} to {calls['n']}) "
            "— spy is wired to the numba path, which is a bug."
        )

        # Anti-vacuous guard: fused entry must have been invoked.
        assert rust_call_count > 0, (
            f"[{strategy_name}] intervals_and_realign_track_fused was NEVER "
            f"invoked during the rust read (calls={rust_call_count}) — "
            "the backstop is vacuous. Inspect HapsTracks.__call__ to "
            "confirm intervals_and_realign_track_fused is called on the Rust path."
        )

        # --- extract track arrays from the (haps, tracks) tuple ---
        # out_rust and out_numba are (RaggedSeqs, RaggedTracks) tuples.
        _, tracks_rust = out_rust
        _, tracks_numba = out_numba
        data_r = np.asarray(tracks_rust.data, dtype=np.float32)
        off_r = np.asarray(tracks_rust.offsets, dtype=np.int64)
        data_n = np.asarray(tracks_numba.data, dtype=np.float32)
        off_n = np.asarray(tracks_numba.offsets, dtype=np.int64)

        # --- byte-identical comparison ---
        np.testing.assert_array_equal(
            off_n,
            off_r,
            err_msg=f"[{strategy_name}] track offsets differ across backends",
        )
        assert data_n.dtype == data_r.dtype == np.float32, (
            f"[{strategy_name}] dtype mismatch: numba={data_n.dtype}, "
            f"rust={data_r.dtype}"
        )
        np.testing.assert_array_equal(
            data_n,
            data_r,
            err_msg=f"[{strategy_name}] track data differs across backends",
        )

        # Non-triviality: at least some non-zero track values (not all-zero
        # vacuous match).  Signal values are drawn from N(0,1) so near-zero
        # is extremely unlikely but possible; we check the overall tensor.
        assert data_r.size > 0, (
            f"[{strategy_name}] Track output is empty — "
            "regions may not overlap stored intervals."
        )
        # At least one realigned haplotype must differ from the input track
        # values OR be non-zero — any non-zero value proves the track was
        # painted from the BigWig intervals.
        assert np.any(data_r != 0.0), (
            f"[{strategy_name}] All realigned track values are 0 — "
            "the BigWig intervals may not overlap the stored regions, "
            "making this comparison vacuous."
        )

        # Restore original between strategies.
        monkeypatch.setattr(_recon_mod, "intervals_and_realign_track_fused", orig_fused)


# ---------------------------------------------------------------------------
# variant-windows live-path spy
# ---------------------------------------------------------------------------


def test_assemble_variant_buffers_runs_on_live_windows_path(
    phased_svar_gvl, reference, monkeypatch
):
    """The rust mega-call must actually fire on the windows __getitem__ path.

    Installs a counting spy on the registered ``rust`` entry of
    ``assemble_variant_buffers``, opens a variant-windows dataset, indexes a
    batch, and asserts the spy was invoked at least once.  Guards against a
    vacuous parity pass caused by the kernel not being wired into the live
    ``__getitem__`` path (e.g. silently bypassed or short-circuited).
    """
    import genvarloader as gvl
    import genvarloader._dataset._flat_variants  # noqa: F401 — triggers register()
    import genvarloader._dispatch as _dispatch
    from genvarloader import VarWindowOpt

    ds = gvl.Dataset.open(phased_svar_gvl, reference=reference)
    ds = (
        ds.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )

    # Install a counting spy on the rust entry of assemble_variant_buffers.
    numba_fn, rust_fn = _dispatch.backends("assemble_variant_buffers")
    calls: dict[str, int] = {"n": 0}

    def _spy_rust(*a, **k):
        calls["n"] += 1
        return rust_fn(*a, **k)

    orig_entry = dict(_dispatch._REGISTRY["assemble_variant_buffers"])
    _dispatch.register(
        "assemble_variant_buffers", numba=numba_fn, rust=_spy_rust, default="rust"
    )
    try:
        monkeypatch.setenv("GVL_BACKEND", "rust")
        _ = ds[[0, 1], [0, 1]]
    finally:
        _dispatch._REGISTRY["assemble_variant_buffers"] = orig_entry

    assert calls["n"] > 0, (
        "assemble_variant_buffers was NEVER invoked on the live variant-windows "
        f"__getitem__ path (calls={calls['n']}) — the backstop is vacuous. "
        "Inspect get_variants_flat to confirm the kernel is called on the windows branch."
    )


# ---------------------------------------------------------------------------
# Strand=−1 parity backstops (Task 7 — pre-wiring safety net)
# ---------------------------------------------------------------------------
#
# Both backends currently apply reverse-complement as a Python post-pass
# (``_query._getitem_unspliced`` calls ``reverse_complement_ragged`` after the
# reconstructor returns).  These tests prove byte-identical output before any
# kernel-level RC wiring (Task 8) is done, establishing the regression net.
# Task 8 must keep every parametrize case below green.
#
# Kinds covered: reference, haplotypes, annotated, tracks, tracks-seqs.
# Spliced variants are excluded: the fixture has no transcript annotations.


def _compare_strand_outputs(numba_out, rust_out, kind: str) -> None:
    """Assert byte-identical output between backends.

    Handles Ragged (reference/haplotypes/tracks), RaggedAnnotatedHaps
    (annotated), and tuple[Ragged, Ragged] (tracks-seqs).
    """
    from genvarloader._ragged import RaggedAnnotatedHaps

    def _cmp_one(n, r, label: str) -> None:
        np.testing.assert_array_equal(
            np.asarray(n.data),
            np.asarray(r.data),
            err_msg=f"[{kind}] {label}: data differs across backends",
        )
        np.testing.assert_array_equal(
            np.asarray(n.offsets, dtype=np.int64),
            np.asarray(r.offsets, dtype=np.int64),
            err_msg=f"[{kind}] {label}: offsets differ across backends",
        )

    def _cmp(n, r, label: str) -> None:
        if isinstance(n, RaggedAnnotatedHaps):
            assert isinstance(r, RaggedAnnotatedHaps)
            _cmp_one(n.haps, r.haps, f"{label}.haps")
            _cmp_one(n.var_idxs, r.var_idxs, f"{label}.var_idxs")
            _cmp_one(n.ref_coords, r.ref_coords, f"{label}.ref_coords")
        else:
            _cmp_one(n, r, label)

    if isinstance(numba_out, tuple):
        assert isinstance(rust_out, tuple) and len(numba_out) == len(rust_out)
        for i, (n, r) in enumerate(zip(numba_out, rust_out)):
            _cmp(n, r, f"component[{i}]")
    else:
        _cmp(numba_out, rust_out, "output")


@pytest.mark.parametrize(
    "kind",
    ["reference", "haplotypes", "annotated", "tracks", "tracks-seqs", "haps-tracks"],
)
def test_neg_strand_parity(kind, tmp_path, synthetic_case, monkeypatch):
    """Mixed +/− strand regions produce byte-identical output across GVL_BACKEND.

    Covers six output kinds over a fresh variants+tracks+strand dataset with
    ``max_jitter=0``.  Both backends currently apply RC as a Python post-pass
    before kernel-level RC wiring (Task 8) lands.

    Spliced variants are excluded: the strand fixture has no transcript
    annotations (no GTF / transcript-ID column).  The non-vacuity assertion
    that RC genuinely fires and produces the correct complement+reverse lives in
    ``test_negative_strand_actually_reverse_complements``.

    The ``"haps-tracks"`` kind covers the ``HapsTracks`` reconstructor
    (``with_seqs("haplotypes").with_tracks("signal")``), which routes through
    ``intervals_and_realign_track_fused``.  That kernel performs an in-kernel
    f32 REVERSE for negative-strand rows (rust path); the numba oracle applies
    the reverse as a Python post-pass.  Byte-identical output across backends
    proves the two paths agree.
    """
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)

    # Open and configure the dataset for the kind under test.
    if kind == "tracks":
        # Open without reference so no seq mode is auto-activated by Dataset.open.
        ds = gvl.Dataset.open(ds_dir)
        ds = ds.with_seqs(None).with_tracks("signal")
    elif kind == "tracks-seqs":
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs("reference").with_tracks("signal")
    elif kind == "haps-tracks":
        # Haplotypes + realigned tracks: routes through HapsTracks reconstructor.
        # intervals_and_realign_track_fused reverses track values in-kernel on
        # the rust path for negative-strand rows; the numba oracle reverses via
        # the Python post-pass in _query._getitem_unspliced.
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs("haplotypes").with_tracks("signal")
    else:
        # "reference", "haplotypes", "annotated"
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs(kind).with_tracks(False)  # type: ignore[arg-type]

    # Non-vacuity guard: fixture must have -strand regions.
    neg_mask = ds._full_regions[:, 3] == -1
    assert np.any(neg_mask), (
        f"[{kind}] Fixture has no -strand regions; parity test is vacuous."
    )

    # --- numba read ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # --- rust read ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    # --- byte-identical comparison ---
    _compare_strand_outputs(out_numba, out_rust, kind)


def test_negative_strand_actually_reverse_complements(
    tmp_path, synthetic_case, monkeypatch
):
    """Non-vacuity: a −strand region's bytes differ from the forward-oriented
    bytes AND equal the exact reverse-complement.

    Uses reference mode so all samples share the same deterministic reference
    sequence, making the before/after comparison unambiguous.

    Fixture geometry: region 1 (chr1:1110686-1110706, strand=−1) carries the
    reference sequence GAATGTAAGACGCAGCGTGC — a non-palindrome whose RC is
    GCACGCTGCGTCTTACATTC — so both guards reliably fire.
    """
    import genvarloader as gvl
    from seqpro.rag import reverse_complement

    from genvarloader._ragged import _COMP

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)

    ds = gvl.Dataset.open(ds_dir, reference=ref)
    ds = ds.with_seqs("reference").with_tracks(False)

    neg_mask = ds._full_regions[:, 3] == -1
    assert np.any(neg_mask), (
        "No -strand regions in fixture; non-vacuity test is vacuous."
    )
    neg_idx = int(np.where(neg_mask)[0][0])  # first -strand region (index 1)

    monkeypatch.setenv("GVL_BACKEND", "rust")

    # Forward-oriented reference at the -strand region (RC disabled).
    ds_fwd = ds.with_settings(rc_neg=False)
    fwd = ds_fwd[neg_idx, 0]  # Ragged[S1], shape (None,)

    # RC-applied output (rc_neg=True by default).
    out = ds[neg_idx, 0]  # Ragged[S1], shape (None,)

    fwd_bytes = np.asarray(fwd.data).tobytes()
    out_bytes = np.asarray(out.data).tobytes()

    # Compute the reverse-complement of the forward sequence up front so the
    # palindrome self-check below can use it.
    # For a (None,)-shaped Ragged, rag_dim=0 → 1 row → mask has exactly one entry.
    mask = np.array([True], dtype=bool)
    rc_fwd = reverse_complement(fwd, _COMP, mask=mask, copy=True)
    rc_fwd_bytes = np.asarray(rc_fwd.data).tobytes()

    # Self-check: the anchor region must be non-palindromic, else Guard 1 is
    # silently unreliable (out == fwd would be expected even if RC fired).
    assert fwd_bytes != rc_fwd_bytes, (
        f"Anchor -strand region {neg_idx} is palindromic (fwd == rc(fwd)) — "
        "non-vacuity Guard 1 is unreliable; pick a different anchor region."
    )

    # Guard 1: RC must have changed bytes (non-palindrome check).
    assert out_bytes != fwd_bytes, (
        f"RC had NO effect on -strand region {neg_idx}: output is byte-identical "
        "to the forward-oriented sequence.  The region may be a palindrome, or "
        "rc_neg=True is not being applied on the read path."
    )

    # Guard 2: output must equal the exact reverse-complement of the forward seq.
    assert out_bytes == rc_fwd_bytes, (
        f"Output for -strand region {neg_idx} is NOT the exact reverse-complement "
        "of the forward-oriented sequence.\n"
        "  forward : "
        f"{bytes(np.asarray(fwd.data).view(np.uint8)).decode('ascii')!r}\n"
        "  rc(fwd) : "
        f"{bytes(np.asarray(rc_fwd.data).view(np.uint8)).decode('ascii')!r}\n"
        "  output  : "
        f"{bytes(np.asarray(out.data).view(np.uint8)).decode('ascii')!r}"
    )


# ---------------------------------------------------------------------------
# Strand=−1 SPLICED parity backstops (Task 7 — pre-wiring safety net)
# ---------------------------------------------------------------------------
#
# Splice mode is activated the same way as test_spliced_haplotypes_parity.py:
# inject a synthetic ``transcript_id`` column onto ``ds._full_bed`` and call
# ``with_settings(splice_info="transcript_id")`` — no GTF / transcript-ID
# storage is required.
#
# The 5 strand-mixed regions (strand [+,-,+,-,+]) are grouped into 4
# transcripts (BED order), arranged so the spliced negative-strand RC path is
# genuinely exercised:
#   T1: [0]    chr1 +          single-exon positive
#   T2: [1]    chr1 -          single-exon PURE NEGATIVE (non-vacuity anchor)
#   T3: [2,3]  chr1 +, chr2 -  multi-exon containing a negative exon
#   T4: [4]    chr2 +          single-exon positive
#
# RC is applied per-exon (``_query._getitem_spliced`` reverse-complements each
# element before regrouping into transcripts), so the spliced output of the
# single-exon T2 is the exact RC of its forward orientation — which makes the
# non-vacuity Guard 2 (output == revcomp(forward)) hold cleanly.  T3 exercises
# per-exon RC inside a genuine multi-exon (cross-contig) splice.
_SPLICE_TRANSCRIPT_IDS = ["T1", "T2", "T3", "T3", "T4"]
# T2 is the second transcript in BED order → spliced index 1.
_NEG_TRANSCRIPT_IDX = 1


def _open_strand_spliced(ds_dir, ref, kind: str):
    """Open the strand-mixed dataset in spliced mode for ``kind``.

    Returns the spliced Dataset (or raises if the kind cannot be spliced).
    """
    from dataclasses import replace

    import polars as pl

    import genvarloader as gvl

    if kind == "tracks":
        ds = gvl.Dataset.open(ds_dir)
        ds = ds.with_seqs(None).with_tracks("signal")
    else:
        # "reference", "haplotypes", "annotated"
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs(kind).with_tracks(False)  # type: ignore[arg-type]

    sub_bed = ds._full_bed.with_columns(
        pl.Series("transcript_id", _SPLICE_TRANSCRIPT_IDS)
    )
    ds = replace(ds, _full_bed=sub_bed).with_settings(splice_info="transcript_id")
    assert ds.is_spliced, f"[{kind}] dataset should be in spliced mode"
    return ds


@pytest.mark.parametrize(
    "kind",
    ["reference", "haplotypes", "annotated", "tracks"],
)
def test_neg_strand_spliced_parity(kind, tmp_path, synthetic_case, monkeypatch):
    """Spliced mixed +/− strand transcripts: byte-identical across GVL_BACKEND.

    Covers the four splice-capable output kinds (reference, haplotypes,
    annotated, tracks).  ``tracks-seqs`` is intentionally excluded: the splice
    path raises ``NotImplementedError`` for ``SeqsTracks`` ("Splicing of
    sequences + un-realigned tracks is not supported"), so there is no spliced
    tracks-seqs combo to compare.

    Both backends currently apply RC per-exon as a Python post-pass in
    ``_query._getitem_spliced`` before kernel-level RC wiring (Task 8) lands.
    """
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = _open_strand_spliced(ds_dir, ref, kind)

    # The negative-strand anchor transcript (T2) must really be -strand.
    neg_transcript = ds.spliced_regions[_NEG_TRANSCRIPT_IDX]
    assert "-" in neg_transcript["strand"].item(0), (
        f"[{kind}] anchor transcript is not negative-strand; test is vacuous."
    )

    # --- numba read ---
    monkeypatch.setenv("GVL_BACKEND", "numba")
    out_numba = ds[:, :]

    # --- rust read ---
    monkeypatch.setenv("GVL_BACKEND", "rust")
    out_rust = ds[:, :]

    # --- byte-identical comparison ---
    _compare_strand_outputs(out_numba, out_rust, f"spliced/{kind}")


def test_negative_strand_spliced_reverse_complements(
    tmp_path, synthetic_case, monkeypatch
):
    """Non-vacuity for the spliced path: a −strand transcript's bytes differ
    from the forward-oriented bytes AND equal the exact reverse-complement.

    Uses spliced reference mode and the single-exon pure-negative transcript T2
    (region chr1:1110686-1110706, reference GAATGTAAGACGCAGCGTGC, a
    non-palindrome).  Because T2 has exactly one exon, per-exon RC of the whole
    transcript equals the reverse-complement of its forward orientation, so the
    Guard 2 check is unambiguous.
    """
    import genvarloader as gvl
    from seqpro.rag import reverse_complement

    from genvarloader._ragged import _COMP

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds = _open_strand_spliced(ds_dir, ref, "reference")

    t_idx = _NEG_TRANSCRIPT_IDX
    assert "-" in ds.spliced_regions[t_idx]["strand"].item(0), (
        "Anchor spliced transcript is not negative-strand; test is vacuous."
    )

    monkeypatch.setenv("GVL_BACKEND", "rust")

    # Forward-oriented spliced transcript (RC disabled).
    ds_fwd = ds.with_settings(rc_neg=False)
    fwd = ds_fwd[t_idx, 0]  # Ragged[S1], shape (None,)

    # RC-applied spliced transcript (rc_neg=True by default).
    out = ds[t_idx, 0]  # Ragged[S1], shape (None,)

    fwd_bytes = np.asarray(fwd.data).tobytes()
    out_bytes = np.asarray(out.data).tobytes()

    # For a single-exon (None,)-shaped Ragged, rag_dim=0 → 1 row → 1 mask entry.
    mask = np.array([True], dtype=bool)
    rc_fwd = reverse_complement(fwd, _COMP, mask=mask, copy=True)
    rc_fwd_bytes = np.asarray(rc_fwd.data).tobytes()

    # Self-check: anchor transcript must be non-palindromic.
    assert fwd_bytes != rc_fwd_bytes, (
        f"Anchor spliced transcript {t_idx} is palindromic (fwd == rc(fwd)) — "
        "non-vacuity Guard 1 is unreliable; pick a different anchor transcript."
    )

    # Guard 1: RC must have changed bytes.
    assert out_bytes != fwd_bytes, (
        f"RC had NO effect on spliced -strand transcript {t_idx}: output is "
        "byte-identical to the forward-oriented sequence.  rc_neg=True may not "
        "be applied on the spliced read path."
    )

    # Guard 2: output must equal the exact reverse-complement of the forward seq.
    assert out_bytes == rc_fwd_bytes, (
        f"Output for spliced -strand transcript {t_idx} is NOT the exact "
        "reverse-complement of the forward-oriented sequence.\n"
        "  forward : "
        f"{bytes(np.asarray(fwd.data).view(np.uint8)).decode('ascii')!r}\n"
        "  rc(fwd) : "
        f"{bytes(np.asarray(rc_fwd.data).view(np.uint8)).decode('ascii')!r}\n"
        "  output  : "
        f"{bytes(np.asarray(out.data).view(np.uint8)).decode('ascii')!r}"
    )
