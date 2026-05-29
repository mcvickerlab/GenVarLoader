"""End-to-end tests for mode='double_buffered'."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")

import genvarloader as gvl


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
            np.testing.assert_array_equal(
                np.asarray(b),
                np.asarray(d),
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
            np.testing.assert_array_equal(
                np.asarray(b),
                np.asarray(d),
                err_msg=f"batch {i} mismatch",
            )


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
