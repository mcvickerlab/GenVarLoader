"""Dataset read-path parity backstops for track kernels.

Covers three cases:

1. ``intervals_to_tracks`` only (track-only dataset, no variants):
   Proves that the rust backend produces output matching the frozen golden
   through the real Dataset.__getitem__ path.

2. ``shift_and_realign_tracks_sparse`` (haplotypes+tracks dataset with indels):
   Proves that the dispatch wiring for the realignment kernel is correct
   end-to-end, across every insertion-fill strategy.

3. Strand=−1 parity backstops (Task 7 — pre-wiring safety net):
   Proves that the rust backend produces byte-identical output matching the
   frozen golden for datasets with mixed + and − strand regions, across all
   five output kinds (reference, haplotypes, annotated, tracks, tracks-seqs)
   in the UNSPLICED path, and across the four splice-capable kinds in the
   SPLICED path.  Analytical non-vacuity tests (RC guard) are also included.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.parity import _golden
from tests.parity._fixtures import (
    _JITTER_SIGNAL_PER_SAMPLE,
    build_haps_tracks_dataset,
    build_strand_mixed_dataset,
    build_track_dataset,
    build_track_dataset_jittered,
)

pytestmark = pytest.mark.parity


def test_track_getitem_identical_across_backends(tmp_path, monkeypatch):
    import genvarloader as gvl
    import genvarloader._dataset._tracks as _tracks_mod

    ds_dir = build_track_dataset(tmp_path)
    ds = gvl.Dataset.open(ds_dir)
    ds = ds.with_tracks("signal")

    r_idx = slice(None)
    s_idx = slice(None)

    # --- spy: assert intervals_to_tracks is actually called on the live read path ---
    calls: dict[str, int] = {"n": 0}

    def _make_spy(orig):
        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        return spy

    # The track-only path calls intervals_to_tracks via _tracks_mod (the
    # haps+tracks path uses the fused intervals_and_realign_track_fused in
    # _reconstruct, which is covered by test_fused_tracks_parity).
    monkeypatch.setattr(
        _tracks_mod, "intervals_to_tracks", _make_spy(_tracks_mod.intervals_to_tracks)
    )

    # --- read (default rust backend) ---
    result = ds[r_idx, s_idx]

    # Backstop guard: kernel must have been called at least once
    assert calls["n"] > 0, (
        f"intervals_to_tracks was NEVER called during the read "
        f"(calls={calls['n']}) — the backstop is vacuous. "
        "Inspect the read path and confirm the track reconstructor is active."
    )

    # Sanity: the read painted real non-zero signal
    data = np.asarray(result.data, dtype=np.float32)
    assert np.any(data != 0.0), (
        "Track data is all-zero — regions may not overlap synthetic intervals. "
        "Non-zero signal is required to prove the comparison is meaningful."
    )

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(result, _golden.load_flat_golden("ds_tracks"))


# ---------------------------------------------------------------------------
# max_jitter > 0 end-to-end parity + oracle (#242 regression)
# ---------------------------------------------------------------------------


def test_tracks_max_jitter_intervals_parity_and_oracle(tmp_path):
    """End-to-end regression for #242: max_jitter>0 track reads match the golden
    and the hand-computed positional oracle.

    Bug #242 root cause
    -------------------
    ``gvl.write`` clips BigWig intervals to the jitter-expanded write window
    ``[chromStart - max_jitter, chromEnd + max_jitter]``, so stored interval
    starts equal ``chromStart - max_jitter``.  ``Dataset.open`` derives query
    starts from the ORIGINAL ``chromStart`` (``input_regions.arrow``), so
    ``itv_start - query_start = -max_jitter`` — a negative offset.
    Fix (PR #244): both kernels now clip ``s = max(itv_start - query_start, 0)``.

    Guards
    ------
    - **Non-vacuity**: at least one ``regions.npy[:,1]`` (stored start) is
      strictly ``<`` the corresponding ``input_regions.arrow`` chromStart
      (original start), proving the #242 boundary condition is exercised.
    - **Golden replay**: output matches the frozen golden.
    - **Positional oracle**: each individual (region, sample) track SLICE
      exactly equals ``np.full(REGION_LEN, sample_constant)`` — catches sample
      misordering / spatial misplacement that a count-based check would miss.
    - **Non-triviality**: at least one output value is non-zero.
    """
    import polars as pl

    import genvarloader as gvl

    MAX_JITTER = 4
    REGION_LEN = 20  # chromEnd - chromStart for every fixture region
    N_REGIONS = 3
    N_SAMPLES = 3  # s0, s1, s2

    ds_dir = build_track_dataset_jittered(tmp_path, max_jitter=MAX_JITTER)

    # --- Non-vacuity guard: stored start < original chromStart (#242 condition) ---
    regions = np.load(ds_dir / "regions.npy")  # shape (N_REGIONS, 4), int32
    input_bed = pl.read_ipc(ds_dir / "input_regions.arrow")
    r_idx_map = input_bed["r_idx_map"].to_numpy()  # original_row → sorted_pos
    orig_starts = input_bed["chromStart"].to_numpy()
    stored_starts_aligned = regions[r_idx_map, 1]  # stored starts per original row
    assert np.any(stored_starts_aligned < orig_starts), (
        "Non-vacuity guard FAILED: no stored region start is < the original chromStart. "
        f"stored (aligned)={stored_starts_aligned.tolist()}, orig={orig_starts.tolist()}. "
        "The max_jitter expansion is not exercising the #242 boundary condition."
    )

    # --- Open dataset ---
    ds = gvl.Dataset.open(ds_dir)
    ds = ds.with_tracks("signal")
    assert ds.jitter == 0, (
        f"Expected ds.jitter == 0 after Dataset.open (deterministic default), "
        f"got {ds.jitter}."
    )

    # --- Read (default rust backend) ---
    result = ds[:, :]
    tracks_t = result[1] if isinstance(result, tuple) else result
    data = np.asarray(tracks_t.data, dtype=np.float32)
    off = np.asarray(tracks_t.offsets, dtype=np.int64)

    # --- Golden replay ---
    _golden.assert_output_matches_golden(
        result, _golden.load_flat_golden("ds_tracks_jitter")
    )

    # --- Positional, hand-computed oracle ---
    sample_consts = [np.float32(v) for v in _JITTER_SIGNAL_PER_SAMPLE.values()]
    assert off.size - 1 == N_REGIONS * N_SAMPLES, (
        f"Expected {N_REGIONS * N_SAMPLES} track rows, got {off.size - 1}; "
        "the (region, sample) layout assumption is wrong."
    )
    for region in range(N_REGIONS):
        for sample in range(N_SAMPLES):
            row = region * N_SAMPLES + sample
            seg = data[off[row] : off[row + 1]]
            expected = np.full(REGION_LEN, sample_consts[sample], dtype=np.float32)
            np.testing.assert_array_equal(
                seg,
                expected,
                err_msg=(
                    f"Positional oracle mismatch at region {region}, sample "
                    f"{sample} (row {row}): expected constant "
                    f"{sample_consts[sample]} over {REGION_LEN} positions."
                ),
            )

    total_expected = N_REGIONS * N_SAMPLES * REGION_LEN  # 3 × 3 × 20 = 180
    assert data.size == total_expected, (
        f"Output data size {data.size} != expected {total_expected} "
        f"({N_REGIONS} regions × {N_SAMPLES} samples × {REGION_LEN} positions)."
    )

    # --- Non-triviality ---
    assert np.any(data != 0.0), (
        "All track values are 0.0 — constant BigWig signal is not reaching the output."
    )


# ---------------------------------------------------------------------------
# Haplotypes+tracks realignment backstop
# ---------------------------------------------------------------------------


def test_tracks_realign_getitem_identical_across_backends(
    synthetic_case, tmp_path, monkeypatch
):
    """Spy-guarded backstop for tracks realignment dispatch wiring (Task 11/14).

    Proves that materialising a haplotypes+tracks dataset (with indel-bearing
    genotypes) via ``ds[:, :]`` produces output matching the frozen golden,
    for every insertion-fill strategy.

    After Task 14, the Rust path calls the fused entry
    ``intervals_and_realign_track_fused`` (one FFI crossing per track).
    The spy targets this entry.
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

    ds_dir = build_haps_tracks_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)
    ds_base = gvl.Dataset.open(ds_dir, reference=ref)
    ds_base = ds_base.with_seqs("haplotypes").with_tracks("signal")

    orig_fused = getattr(_recon_mod, "intervals_and_realign_track_fused", None)
    assert orig_fused is not None, (
        "intervals_and_realign_track_fused not found on _recon_mod — "
        "ensure it is imported at module level in _reconstruct.py"
    )

    calls: dict[str, int] = {"n": 0}

    def _spy_fused(*a, **k):
        calls["n"] += 1
        return orig_fused(*a, **k)

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

        # --- read (default rust backend, spy active) ---
        out = ds[:, :]

        # Anti-vacuous guard
        assert calls["n"] > 0, (
            f"[{strategy_name}] intervals_and_realign_track_fused was NEVER "
            f"invoked during the read (calls={calls['n']}) — "
            "the backstop is vacuous. Inspect HapsTracks.__call__ to "
            "confirm intervals_and_realign_track_fused is called on the Rust path."
        )

        # --- extract tracks for non-triviality check ---
        _, tracks_out = out
        data_r = np.asarray(tracks_out.data, dtype=np.float32)
        assert data_r.size > 0, (
            f"[{strategy_name}] Track output is empty — "
            "regions may not overlap stored intervals."
        )
        assert np.any(data_r != 0.0), (
            f"[{strategy_name}] All realigned track values are 0 — "
            "the BigWig intervals may not overlap the stored regions, "
            "making this comparison vacuous."
        )

        # --- replay against frozen golden ---
        golden_name = f"ds_haps_tracks_{strategy_name}"
        _golden.assert_output_matches_golden(out, _golden.load_flat_golden(golden_name))

        # Restore original between strategies.
        monkeypatch.setattr(_recon_mod, "intervals_and_realign_track_fused", orig_fused)


# ---------------------------------------------------------------------------
# variant-windows live-path spy
# ---------------------------------------------------------------------------


def test_assemble_variant_buffers_runs_on_live_windows_path(phased_svar_gvl, reference):
    """The rust mega-call must actually fire on the windows __getitem__ path.

    Installs a counting spy on the registered ``rust`` entry of
    ``assemble_variant_buffers``, opens a variant-windows dataset, indexes a
    batch, and asserts the spy was invoked at least once.
    """
    import genvarloader as gvl
    import genvarloader._dataset._flat_variants  # noqa: F401 — triggers register()
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

    spy, calls, restore = _golden.make_kernel_spy("assemble_variant_buffers")
    try:
        _ = ds[[0, 1], [0, 1]]
    finally:
        restore()

    assert calls["n"] > 0, (
        "assemble_variant_buffers was NEVER invoked on the live variant-windows "
        f"__getitem__ path (calls={calls['n']}) — the backstop is vacuous. "
        "Inspect get_variants_flat to confirm the kernel is called on the windows branch."
    )


# ---------------------------------------------------------------------------
# Strand=−1 parity backstops (Task 7 — pre-wiring safety net)
# ---------------------------------------------------------------------------

_SPLICE_TRANSCRIPT_IDS = ["T1", "T2", "T3", "T3", "T4"]
_NEG_TRANSCRIPT_IDX = 1


def _open_strand_spliced(ds_dir, ref, kind: str):
    """Open the strand-mixed dataset in spliced mode for ``kind``."""
    from dataclasses import replace

    import polars as pl

    import genvarloader as gvl

    if kind == "tracks":
        ds = gvl.Dataset.open(ds_dir)
        ds = ds.with_seqs(None).with_tracks("signal")
    else:
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
    ["reference", "haplotypes", "annotated", "tracks", "tracks-seqs", "haps-tracks"],
)
def test_neg_strand_parity(kind, tmp_path, synthetic_case):
    """Mixed +/− strand regions produce output matching the frozen golden.

    Covers six output kinds over a fresh variants+tracks+strand dataset with
    ``max_jitter=0``.
    """
    import genvarloader as gvl

    ds_dir = build_strand_mixed_dataset(tmp_path, synthetic_case.svar_path)
    ref = gvl.Reference.from_path(synthetic_case.ref_path, in_memory=False)

    if kind == "tracks":
        ds = gvl.Dataset.open(ds_dir)
        ds = ds.with_seqs(None).with_tracks("signal")
    elif kind == "tracks-seqs":
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs("reference").with_tracks("signal")
    elif kind == "haps-tracks":
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs("haplotypes").with_tracks("signal")
    else:
        ds = gvl.Dataset.open(ds_dir, reference=ref)
        ds = ds.with_seqs(kind).with_tracks(False)  # type: ignore[arg-type]

    # Non-vacuity guard: fixture must have -strand regions.
    neg_mask = ds._full_regions[:, 3] == -1
    assert np.any(neg_mask), (
        f"[{kind}] Fixture has no -strand regions; parity test is vacuous."
    )

    # --- read (default rust backend) ---
    out = ds[:, :]

    # --- replay against frozen golden ---
    safe_kind = kind.replace("-", "_")
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden(f"ds_neg_strand_{safe_kind}")
    )


def test_negative_strand_actually_reverse_complements(tmp_path, synthetic_case):
    """Non-vacuity: a −strand region's bytes differ from the forward-oriented
    bytes AND equal the exact reverse-complement.
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

    # Forward-oriented reference at the -strand region (RC disabled).
    ds_fwd = ds.with_settings(rc_neg=False)
    fwd = ds_fwd[neg_idx, 0]  # Ragged[S1], shape (None,)

    # RC-applied output (rc_neg=True by default).
    out = ds[neg_idx, 0]  # Ragged[S1], shape (None,)

    fwd_bytes = np.asarray(fwd.data).tobytes()
    out_bytes = np.asarray(out.data).tobytes()

    mask = np.array([True], dtype=bool)
    rc_fwd = reverse_complement(fwd, _COMP, mask=mask, copy=True)
    rc_fwd_bytes = np.asarray(rc_fwd.data).tobytes()

    # Self-check: the anchor region must be non-palindromic.
    assert fwd_bytes != rc_fwd_bytes, (
        f"Anchor -strand region {neg_idx} is palindromic (fwd == rc(fwd)) — "
        "non-vacuity Guard 1 is unreliable; pick a different anchor region."
    )

    # Guard 1: RC must have changed bytes.
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


@pytest.mark.parametrize(
    "kind",
    ["reference", "haplotypes", "annotated", "tracks"],
)
def test_neg_strand_spliced_parity(kind, tmp_path, synthetic_case):
    """Spliced mixed +/− strand transcripts: output matches the frozen golden.

    Covers the four splice-capable output kinds (reference, haplotypes,
    annotated, tracks).
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

    # --- read (default rust backend) ---
    out = ds[:, :]

    # --- replay against frozen golden ---
    _golden.assert_output_matches_golden(
        out, _golden.load_flat_golden(f"ds_neg_strand_spliced_{kind}")
    )


def test_negative_strand_spliced_reverse_complements(tmp_path, synthetic_case):
    """Non-vacuity for the spliced path: a −strand transcript's bytes differ
    from the forward-oriented bytes AND equal the exact reverse-complement.
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

    # Forward-oriented spliced transcript (RC disabled).
    ds_fwd = ds.with_settings(rc_neg=False)
    fwd = ds_fwd[t_idx, 0]  # Ragged[S1], shape (None,)

    # RC-applied spliced transcript (rc_neg=True by default).
    out = ds[t_idx, 0]  # Ragged[S1], shape (None,)

    fwd_bytes = np.asarray(fwd.data).tobytes()
    out_bytes = np.asarray(out.data).tobytes()

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
