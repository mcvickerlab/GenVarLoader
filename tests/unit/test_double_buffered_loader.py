"""End-to-end tests for mode='double_buffered'."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")

import genvarloader as gvl
from seqpro.rag import to_padded


def _rv_eq(a, b):
    """Field-by-field equality for RaggedVariants.

    RaggedVariants is a record _core.Ragged wrapper (not an awkward array) and
    has no __eq__, so compare each field's values via to_ak().to_list().
    """
    from genvarloader._dataset._rag_variants import RaggedVariants

    assert isinstance(a, RaggedVariants) and isinstance(b, RaggedVariants), (
        f"expected RaggedVariants, got {type(a)} and {type(b)}"
    )
    assert set(a.fields) == set(b.fields), (
        f"field sets differ: {a.fields} vs {b.fields}"
    )
    for fname in a.fields:
        assert a[fname].to_ak().to_list() == b[fname].to_ak().to_list(), (
            f"field {fname!r} mismatch"
        )


# ---------------------------------------------------------------------------
# File-backed fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def file_backed_ds(phased_vcf_gvl, reference):
    """Open the phased-VCF test dataset with a reference genome.

    This is the smallest existing on-disk GVL dataset in the test suite
    (10 regions, 3 samples).  We open it in *reference* mode — no genotype
    reconstruction — so the test is fast and the producer subprocess can
    reopen the same path without any extra data on disk.
    """
    return gvl.Dataset.open(phased_vcf_gvl, reference=reference)


# ---------------------------------------------------------------------------
# T12: Happy path — double_buffered output matches buffered output
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes"])
def test_double_buffered_iter_matches_buffered(file_backed_ds, seq_kind):
    ds = file_backed_ds.with_seqs(seq_kind).with_tracks(False)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)

    batch_size = 2
    common = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        buffer_bytes=4 * 1024 * 1024,
    )

    buf_batches = list(ds.to_dataloader(mode="buffered", **common))
    db_batches = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))

    assert len(db_batches) == len(buf_batches), (
        f"batch count mismatch: buffered={len(buf_batches)} double_buffered={len(db_batches)}"
    )
    for i, (b, d) in enumerate(zip(buf_batches, db_batches)):
        if isinstance(b, tuple):
            assert isinstance(d, tuple) and len(b) == len(d), (
                f"batch {i}: tuple length mismatch"
            )
            for j, (x, y) in enumerate(zip(b, d)):
                np.testing.assert_array_equal(
                    np.asarray(x),
                    np.asarray(y),
                    err_msg=f"batch {i} element {j} mismatch",
                )
        else:
            # The seqs batch is a (possibly jagged) Ragged; densify offset-aware
            # with a shared pad rather than np.asarray, which raises on jagged
            # haplotypes ("cannot convert a jagged Ragged to a dense array").
            np.testing.assert_array_equal(
                to_padded(b, b"N"),
                to_padded(d, b"N"),
                err_msg=f"batch {i} mismatch",
            )


# ---------------------------------------------------------------------------
# T13: Failure paths + cleanup
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_producer_exception_reraised(file_backed_ds, monkeypatch):
    """Inject failure via the GVL_TEST_PRODUCER_RAISE env var the producer respects."""
    ds = file_backed_ds.with_seqs("reference").with_tracks(False)
    monkeypatch.setenv("GVL_TEST_PRODUCER_RAISE", "1")
    with pytest.raises(RuntimeError, match="ProducerError|ProducerDied"):
        for _ in ds.to_dataloader(
            mode="double_buffered",
            batch_size=2,
            shuffle=False,
            drop_last=True,
            buffer_bytes=1 << 20,
            heartbeat_seconds=5,
        ):
            pass


@pytest.mark.slow
@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes"])
def test_double_buffered_schema_settings_parity(file_backed_ds, seq_kind):
    """rc_neg and deterministic are replayed correctly in the producer subprocess."""
    ds = (
        file_backed_ds.with_seqs(seq_kind)
        .with_tracks(False)
        .with_settings(rc_neg=False, deterministic=True)
    )
    batch_size = 2
    common = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        buffer_bytes=4 * 1024 * 1024,
    )
    buf_batches = list(ds.to_dataloader(mode="buffered", **common))
    db_batches = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))

    assert len(db_batches) == len(buf_batches)
    for i, (b, d) in enumerate(zip(buf_batches, db_batches)):
        if isinstance(b, tuple):
            for j, (x, y) in enumerate(zip(b, d)):
                np.testing.assert_array_equal(
                    np.asarray(x),
                    np.asarray(y),
                    err_msg=f"batch {i} element {j} mismatch",
                )
        else:
            # The seqs batch is a (possibly jagged) Ragged; densify offset-aware
            # with a shared pad rather than np.asarray, which raises on jagged
            # haplotypes ("cannot convert a jagged Ragged to a dense array").
            np.testing.assert_array_equal(
                to_padded(b, b"N"),
                to_padded(d, b"N"),
                err_msg=f"batch {i} mismatch",
            )


@pytest.mark.slow
def test_double_buffered_producer_reaped_on_drop(file_backed_ds):
    """Dropping a loader must terminate its producer subprocess promptly.

    Regression: the loader registered ``atexit.register(self.close)``, which
    held a strong reference for the whole process lifetime, so per-loader
    producers (and their shm) accumulated until process exit. Creating many
    loaders in one process (e.g. the bench's per-cell loop) then exhausted RAM.
    """
    import gc
    import time

    ds = ds_ref = file_backed_ds.with_seqs("reference").with_tracks(False)
    loader = ds_ref.to_dataloader(
        mode="double_buffered",
        batch_size=2,
        shuffle=False,
        drop_last=True,
        buffer_bytes=1 << 20,
    )
    list(loader)
    impl = loader.dataset._impl
    proc = impl._producer
    assert proc is not None and proc.is_alive()

    del loader, impl, ds, ds_ref
    gc.collect()

    for _ in range(50):  # allow the finalizer's terminate()/join() to land
        if not proc.is_alive():
            break
        time.sleep(0.1)
    assert not proc.is_alive(), (
        "producer subprocess still alive after the loader was dropped — "
        "atexit must not retain a strong reference to the loader"
    )


@pytest.mark.slow
def test_double_buffered_annotated_matches_buffered(file_backed_ds):
    """Regression: double_buffered must support ``annotated`` output.

    Annotated output is a ``RaggedAnnotatedHaps`` (three ragged arrays:
    haps / var_idxs / ref_coords). ``write_chunk`` had no case for it and
    raised ``TypeError: unsupported array type RaggedAnnotatedHaps`` in the
    producer; this checks double_buffered reproduces buffered for annotated.
    """
    ds = (
        file_backed_ds.with_seqs("annotated")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    common = dict(
        batch_size=2,
        shuffle=False,
        drop_last=True,
        buffer_bytes=4 * 1024 * 1024,
    )
    buf_batches = list(ds.to_dataloader(mode="buffered", **common))
    db_batches = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))

    assert len(db_batches) == len(buf_batches)
    for i, (b, d) in enumerate(zip(buf_batches, db_batches)):
        b_pad, d_pad = b.to_padded(), d.to_padded()
        np.testing.assert_array_equal(
            b_pad.haps, d_pad.haps, err_msg=f"batch {i} haps mismatch"
        )
        np.testing.assert_array_equal(
            b_pad.var_idxs, d_pad.var_idxs, err_msg=f"batch {i} var_idxs mismatch"
        )
        np.testing.assert_array_equal(
            b_pad.ref_coords, d_pad.ref_coords, err_msg=f"batch {i} ref_coords mismatch"
        )


@pytest.mark.slow
def test_double_buffered_variants_offset_overflow_regression(
    data_dir, hg38_reference, tmp_path
):
    """Regression: double_buffered slots must be sized for the *serialized*
    footprint, not payload alone.

    A serialized ragged chunk also stores int64 offset/lengths arrays. For
    ``variants`` output the alt/ref inner offsets cost ~8 bytes per variant,
    which dwarfs the 1-byte SNV allele payload; on realistic data (1KG,
    ~287k variants) these overflowed the fixed slot slack and the producer
    raised ``ProducerError (ValueError): buffer is smaller than requested
    size``. ``buffered`` mode never serializes into a fixed slot, so it was
    unaffected — hence we compare the two.
    """
    import seqpro as sp

    svar = data_dir / "1kg" / "filtered.svar"
    regions = data_dir / "1kg" / "regions.bed"
    if not svar.is_dir() or not regions.exists():
        pytest.skip("missing 1kg filtered.svar / regions.bed; run pixi run -e dev gen")

    # A 32-region subset still has SNV-dense regions whose alt/ref inner
    # offsets (~8 B/variant) overflowed the slot pre-fix, but writes far less
    # data — keeps this slow-marked test to a fraction of a second of work.
    bed = sp.bed.read(regions).head(32)
    gvl_path = tmp_path / "ds_1kg.gvl"
    gvl.write(path=gvl_path, bed=bed, variants=svar, overwrite=True)

    ds = gvl.Dataset.open(gvl_path, reference=hg38_reference).with_seqs("variants")
    common = dict(
        batch_size=16,
        shuffle=False,
        drop_last=True,
        buffer_bytes=8 * 1024 * 1024,
    )
    buf_batches = list(ds.to_dataloader(mode="buffered", **common))
    # This line raised ProducerError before the slot-sizing fix.
    db_batches = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))

    assert len(db_batches) == len(buf_batches)
    for i, (b, d) in enumerate(zip(buf_batches, db_batches)):
        assert len(b) == len(d), f"batch {i} instance-count mismatch"


@pytest.mark.slow
def test_shm_cleanup_after_close(file_backed_ds):
    """No gvl-prefixed entries leak in /dev/shm after the loader is closed."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("Linux-only /dev/shm check")

    ds = file_backed_ds.with_seqs("reference").with_tracks(False)
    before = set(os.listdir("/dev/shm"))

    loader = ds.to_dataloader(
        mode="double_buffered",
        batch_size=2,
        shuffle=False,
        drop_last=True,
        buffer_bytes=1 << 20,
    )
    list(loader)

    # Reach into the IterableDataset wrapper to explicitly close.
    inner = loader.dataset
    if hasattr(inner, "_impl"):
        inner._impl.close()
    del loader, inner

    import gc

    gc.collect()

    after = set(os.listdir("/dev/shm"))
    leaked = {n for n in after - before if n.startswith("gvl-")}
    assert not leaked, f"leaked shm segments: {leaked}"


@pytest.mark.slow
@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes", "variants"])
def test_double_buffered_flat_matches_ragged(file_backed_ds, seq_kind):
    """double_buffered in flat output equals double_buffered in ragged output,
    batch for batch, after .to_ragged().

    Ragged slicing returns non-copying views over the full chunk buffer (offsets
    not rebased), while the flat path rebases+trims; so compare via offset-aware
    to_padded()/to_list(), not raw .data/.offsets.
    """
    base = file_backed_ds.with_seqs(seq_kind).with_tracks(False)
    if seq_kind == "haplotypes":
        base = base.with_settings(deterministic=True)

    common = dict(
        batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024
    )
    ragged_batches = list(
        base.to_dataloader(mode="double_buffered", copy=True, **common)
    )
    flat_batches = list(
        base.with_output_format("flat").to_dataloader(
            mode="double_buffered", copy=True, **common
        )
    )

    assert len(flat_batches) == len(ragged_batches)
    for i, (rb, fb) in enumerate(zip(ragged_batches, flat_batches)):
        got = fb.to_ragged()
        if seq_kind == "variants":
            _rv_eq(got, rb)
        else:
            np.testing.assert_array_equal(
                to_padded(got, b"N"), to_padded(rb, b"N"), err_msg=f"batch {i}"
            )


@pytest.mark.slow
def test_double_buffered_flat_annotated_matches_ragged(file_backed_ds):
    """double_buffered annotated flat output equals ragged output per batch.

    As in test_double_buffered_flat_matches_ragged, ragged slicing returns
    non-copying views over the full chunk buffer (offsets not rebased) while the
    flat path rebases+trims, so raw .data/.offsets differ even when logically
    equal. Compare each component offset-aware via to_padded() with a shared pad
    value per dtype.
    """
    base = (
        file_backed_ds.with_seqs("annotated")
        .with_tracks(False)
        .with_settings(deterministic=True)
    )
    common = dict(
        batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024
    )
    ragged_batches = list(
        base.to_dataloader(mode="double_buffered", copy=True, **common)
    )
    flat_batches = list(
        base.with_output_format("flat").to_dataloader(
            mode="double_buffered", copy=True, **common
        )
    )
    assert len(flat_batches) == len(ragged_batches)
    pads = {"haps": b"N", "var_idxs": -1, "ref_coords": np.iinfo(np.int32).max}
    for i, (rb, fb) in enumerate(zip(ragged_batches, flat_batches)):
        got = fb.to_ragged()
        for comp, pad in pads.items():
            np.testing.assert_array_equal(
                to_padded(getattr(got, comp), pad),
                to_padded(getattr(rb, comp), pad),
                err_msg=f"batch {i} comp {comp}",
            )


@pytest.mark.slow
def test_double_buffered_flat_consumer_avoids_awkward(file_backed_ds, monkeypatch):
    """In flat mode the consumer must reconstruct via the flat readers, never
    the awkward kind-2 reader."""
    import genvarloader._shm_layout as L

    def _boom(*a, **k):
        raise AssertionError("_read_rag_variants called in flat double_buffered mode")

    monkeypatch.setattr(L, "_read_rag_variants", _boom)

    base = (
        file_backed_ds.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
    )
    common = dict(
        batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024
    )
    # Draining must succeed without hitting the awkward reader.
    batches = list(base.to_dataloader(mode="double_buffered", copy=True, **common))
    assert batches  # produced something


def test_reshape_ragged_for_chunk_leaves_raggedvariants_untouched():
    """A RaggedVariants (record Ragged carrying ploidy) must pass through
    _reshape_ragged_for_chunk unchanged — it must NOT enter the generic Ragged
    ploidy-reshape branch (which assumes a single-field Ragged with .data/.offsets)."""
    import numpy as np
    from seqpro.rag import Ragged
    from genvarloader import RaggedVariants
    from genvarloader._double_buffered_loader import _reshape_ragged_for_chunk

    alt = Ragged.from_offsets(
        np.frombuffer(b"ACGT", dtype="S1").copy(),
        (2, 1, None),
        np.array([0, 1, 2], np.int64),
        str_offsets=np.array([0, 2, 4], np.int64),
    ).to_strings()
    start = Ragged.from_offsets(
        np.arange(2, dtype=np.int32), (2, 1, None), np.array([0, 1, 2], np.int64)
    )
    ilen = Ragged.from_offsets(
        np.zeros(2, np.int32), (2, 1, None), np.array([0, 1, 2], np.int64)
    )
    rv = RaggedVariants(alt=alt, start=start, ilen=ilen)  # shape (2, 1, ~v)
    out = _reshape_ragged_for_chunk([rv], n_instances=2)[0]
    assert out is rv  # untouched
    assert out.shape == (2, 1, None)


def test_reshape_ragged_for_chunk_passes_variant_windows():
    """A _FlatVariantWindows (window-mode variants output) must pass through
    _reshape_ragged_for_chunk unchanged -- the shm reader already builds correct
    (b, rs, None, None) per-slot shapes from regular_size, so the generic
    Ragged/_Flat ploidy-reshape branches must not touch it."""
    import numpy as np
    from genvarloader._flat import _Flat
    from genvarloader._dataset._flat_variants import _FlatWindow, _FlatVariantWindows
    from genvarloader._double_buffered_loader import _reshape_ragged_for_chunk

    start = _Flat(
        np.array([1, 2], np.int32), np.array([0, 1, 2], np.int64), (2, 1, None)
    )
    rw = _FlatWindow(
        np.arange(4, dtype=np.uint8),
        np.array([0, 2, 4], np.int64),
        np.array([0, 1, 2], np.int64),
        (2, 1, None, None),
    )
    fvw = _FlatVariantWindows({"start": start}, ref_window=rw)
    out = _reshape_ragged_for_chunk([fvw], n_instances=2)[0]
    assert isinstance(out, _FlatVariantWindows)
    assert out.ref_window is not None and out.ref_window.shape[0] == 2


# ---------------------------------------------------------------------------
# PR2: variant-windows and flank-token parity over double_buffered
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("ref,alt", [("window", "window"), ("window", "allele")])
def test_double_buffered_variant_windows_matches_buffered(file_backed_ds, ref, alt):
    ds = (
        file_backed_ds.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(
                flank_length=2,
                token_alphabet=b"ACGT",
                unknown_token=4,
                ref=ref,
                alt=alt,
            ),
        )
    )
    common = dict(
        batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024
    )
    buf = list(ds.to_dataloader(mode="buffered", **common))
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    assert len(db) == len(buf)
    for b, d in zip(buf, db):
        da, dbb = b.to_ragged(), d.to_ragged()
        assert set(da) == set(dbb)
        for k in da:
            assert da[k].to_ak().to_list() == dbb[k].to_ak().to_list()


@pytest.mark.slow
def test_double_buffered_variants_flank_tokens_matches_buffered(file_backed_ds):
    # unknown_token=0 collides with "A"'s own token id (0) in b"ACGT". This used
    # to make double_buffered raise (the producer schema recovered token_alphabet
    # by inverting Haps.token_lut, which is provably lossy for this collision) even
    # though mode="buffered"/mode=None handle it fine -- a parity break for a
    # legal, common config (0 as a natural pad/unknown id). Haps now stores
    # token_alphabet directly, so this must pass end-to-end.
    ds = (
        file_backed_ds.with_seqs("variants")
        .with_tracks(False)
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    common = dict(
        batch_size=2, shuffle=False, drop_last=True, buffer_bytes=4 * 1024 * 1024
    )
    buf = list(ds.to_dataloader(mode="buffered", **common))
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    assert len(db) == len(buf)
    for b, d in zip(buf, db):
        # Values are ragged (variable ploidy/length) across instances in this
        # file-backed dataset, so a plain nested-list comparison is used instead of
        # np.testing.assert_array_equal, which raises ValueError trying to build a
        # homogeneous ndarray from a jagged nested list rather than comparing it.
        assert (
            b.flank_tokens.to_ragged().to_ak().to_list()
            == d.flank_tokens.to_ragged().to_ak().to_list()
        )


# ---------------------------------------------------------------------------
# Task 2.6: slot-fit regression -- byte accounting must not undersize slots
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_double_buffered_variant_windows_slot_fits(file_backed_ds):
    """Content-parity + no-raise check: double_buffered variant-windows vs. buffered.

    Mirrors ``test_double_buffered_variants_offset_overflow_regression`` but for
    the newer ``variant-windows`` flat output. Confirms ``double_buffered``
    does not raise and its chunk count matches ``buffered`` under a
    moderately tight ``buffer_bytes``.

    This does NOT tightly pin the byte-accounting bound in
    ``_output_bytes_per_instance``: on this small (~10 region x 3 sample)
    fixture, the shm slot's fixed ~8KB slack (``HEADER_RESERVED`` + 4096, see
    ``_double_buffered_loader.py``) absorbs under-estimates far larger than
    anything this fixture can produce, so an under-count here would not
    reliably surface as ``ProducerError``. The direct, slack-free bound check
    lives in ``tests/unit/dataset/test_output_bytes_dummy_variant.py``
    (compares ``_output_bytes_per_instance(..., include_offsets=True)``
    straight against ``_shm_layout.write_chunk``'s real output, no subprocess
    or fixed slack involved) -- that file is what actually guards the
    byte-accounting fix.
    """
    ds = (
        file_backed_ds.with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    common = dict(batch_size=4, shuffle=False, drop_last=True, buffer_bytes=1 << 20)
    # Must not raise ProducerError (buffer too small) -- byte accounting must not undersize.
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    buf = list(ds.to_dataloader(mode="buffered", **common))
    assert len(db) == len(buf)


@pytest.mark.slow
def test_double_buffered_dummy_variant_windows_slot_fits(file_backed_ds):
    """Content-parity + no-raise check: double_buffered variant-windows + dummy_variant.

    When ``dummy_variant`` is set and a (region, sample, ploid) group has
    zero real variants, the flat builder inserts one dummy row (a full
    ``2*flank_length + len(dummy allele)`` token window per window slot) into
    that otherwise-empty group. This test confirms ``double_buffered``
    doesn't raise and matches ``buffered`` element-for-element on this
    fixture (``snap_dataset``-style fixtures built from ``synthetic_case``
    are known to contain empty groups in the first few (region, sample)
    pairs -- see ``test_b_dummy_fill_no_empty_groups`` in
    ``tests/dataset/test_flat_mode_equivalence.py`` -- so ``file_backed_ds``
    exercises the dummy-fill path without any extra fixture construction).

    Like ``test_double_buffered_variant_windows_slot_fits``, this does NOT
    tightly pin the byte-accounting bound -- the fixed shm slack absorbs
    under-estimates on a fixture this small. See
    ``tests/unit/dataset/test_output_bytes_dummy_variant.py`` for the direct,
    slack-free bound check (including the AF-filter + dummy_variant
    interaction, which this fixture's raw-empty-only groups don't exercise).
    """
    dv = gvl.DummyVariant(start=-1, alt=b"N", ref=b"N")
    ds = (
        file_backed_ds.with_tracks(False)
        .with_output_format("flat")
        .with_settings(dummy_variant=dv)
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4),
        )
    )
    common = dict(batch_size=4, shuffle=False, drop_last=True, buffer_bytes=1 << 20)
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    buf = list(ds.to_dataloader(mode="buffered", **common))
    assert len(db) == len(buf)
    for b, d in zip(buf, db):
        da, dbb = b.to_ragged(), d.to_ragged()
        assert set(da) == set(dbb)
        for k in da:
            assert da[k].to_ak().to_list() == dbb[k].to_ak().to_list()


@pytest.mark.slow
def test_double_buffered_dummy_variant_flank_tokens_slot_fits(file_backed_ds):
    """Content-parity + no-raise check: double_buffered variants + flank_tokens + dummy_variant.

    Same shape as ``test_double_buffered_dummy_variant_windows_slot_fits``,
    but for the sibling ``variants`` branch of ``_output_bytes_per_instance``
    (Config B: flat ``variants`` output with ride-along ``flank_tokens``). A
    dummy-filled empty group contributes a dummy allele's worth of alt/ref
    bytes plus a full ``2*flank_length`` token ``flank_tokens`` row.

    As with the other slot-fit tests in this section, this checks content
    parity + no-raise, not the byte-accounting bound itself (the fixed shm
    slack on this fixture absorbs under-estimates too small to trip
    ``ProducerError`` here) -- see
    ``tests/unit/dataset/test_output_bytes_dummy_variant.py`` for the direct
    bound check.
    """
    dv = gvl.DummyVariant(start=-1, alt=b"N", ref=b"N")
    ds = (
        file_backed_ds.with_seqs("variants")
        .with_tracks(False)
        .with_settings(
            dummy_variant=dv, flank_length=2, token_alphabet=b"ACGT", unknown_token=4
        )
        .with_output_format("flat")
    )
    common = dict(batch_size=4, shuffle=False, drop_last=True, buffer_bytes=1 << 20)
    db = list(ds.to_dataloader(mode="double_buffered", copy=True, **common))
    buf = list(ds.to_dataloader(mode="buffered", **common))
    assert len(db) == len(buf)
    for b, d in zip(buf, db):
        assert (
            b.flank_tokens.to_ragged().to_ak().to_list()
            == d.flank_tokens.to_ragged().to_ak().to_list()
        )
