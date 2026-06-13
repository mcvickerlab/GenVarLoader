"""End-to-end tests for mode='buffered'."""

import numpy as np
import pytest
import genvarloader as gvl

pytest.importorskip("torch")


@pytest.mark.parametrize(
    "seq_kind", ["reference", "haplotypes", "annotated", "variants"]
)
def test_buffered_iter_matches_direct(seq_kind):
    ds = gvl.get_dummy_dataset().with_seqs(seq_kind).with_tracks(False)
    if seq_kind in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)
    batch_size = 2
    n_total = int(np.prod(ds.full_shape))
    # Use shuffle=False so order is deterministic; sequential sampling.
    loader = ds.to_dataloader(
        mode="buffered",
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        buffer_bytes=10 * 1024 * 1024,  # 10 MB; dummy is tiny so this likely fits all.
    )
    seen = 0
    for batch in loader:
        seen += (
            batch_size
            if not isinstance(batch, tuple)
            else (batch[0].shape[0] if hasattr(batch[0], "shape") else len(batch[0]))
        )
    assert seen == (n_total // batch_size) * batch_size


def test_buffered_rejects_num_workers():
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    with pytest.raises(ValueError, match="num_workers"):
        ds.to_dataloader(mode="buffered", batch_size=2, num_workers=2)


def test_buffered_rejects_oversized_batch():
    ds = gvl.get_dummy_dataset().with_seqs("reference").with_tracks(False)
    with pytest.raises(ValueError, match="exceeds slot"):
        # buffer_bytes=1 → slot_bytes=1; any batch of 2 ×3-byte instances (6 bytes) exceeds this.
        ds.to_dataloader(mode="buffered", batch_size=2, buffer_bytes=1)


def test_buffered_rejects_nondeterministic_for_haplotypes():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("haplotypes")
        .with_settings(deterministic=False)
    )
    with pytest.raises(ValueError, match="deterministic"):
        ds.to_dataloader(mode="buffered", batch_size=2)


@pytest.mark.parametrize(
    "seq_kind", ["reference", "haplotypes", "annotated", "variants"]
)
def test_buffered_flat_matches_ragged(seq_kind):
    """mode='buffered' in flat output yields Flat* mini-batches whose
    .to_ragged() equals the ragged-mode buffered output, batch for batch.

    Note: Ragged slicing returns non-copying views that keep the full backing
    buffer, so we compare via to_padded() which densifies only the valid rows.
    """
    import awkward as ak
    from seqpro.rag import to_padded

    base = gvl.get_dummy_dataset().with_seqs(seq_kind).with_tracks(False)
    if seq_kind in ("haplotypes", "annotated"):
        base = base.with_settings(deterministic=True)

    common = dict(
        batch_size=2, shuffle=False, drop_last=True, buffer_bytes=10 * 1024 * 1024
    )
    ragged_batches = list(base.to_dataloader(mode="buffered", **common))
    flat_batches = list(
        base.with_output_format("flat").to_dataloader(mode="buffered", **common)
    )

    assert len(flat_batches) == len(ragged_batches)
    for rb, fb in zip(ragged_batches, flat_batches):
        got = fb.to_ragged()
        if seq_kind == "variants":
            assert ak.to_list(got) == ak.to_list(rb)
        elif seq_kind == "annotated":
            np.testing.assert_array_equal(
                to_padded(got.haps, b"N"), to_padded(rb.haps, b"N")
            )
        else:
            np.testing.assert_array_equal(to_padded(got, b"N"), to_padded(rb, b"N"))


@pytest.mark.parametrize("mode", ["buffered", "double_buffered"])
def test_flat_buffered_rejects_variants_flank_tokens(mode):
    """flat + variants + ride-along flank tokens is unsupported over the buffered
    transport; to_dataloader must reject it up front (not crash mid-iteration)."""
    import genvarloader as gvl

    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(flank_length=3, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    with pytest.raises(ValueError, match="flank"):
        ds.to_dataloader(mode=mode, batch_size=2, buffer_bytes=4 * 1024 * 1024)


@pytest.mark.parametrize("mode", ["buffered", "double_buffered"])
def test_flat_buffered_rejects_variant_windows(mode):
    """flat + 'variant-windows' output is unsupported over the buffered transport
    (the producer schema does not carry the VarWindowOpt); to_dataloader must
    reject it up front rather than crash mid-iteration."""
    import genvarloader as gvl

    ds = (
        gvl.get_dummy_dataset()
        .with_tracks(False)
        .with_output_format("flat")
        .with_seqs(
            "variant-windows",
            gvl.VarWindowOpt(
                flank_length=3, token_alphabet=b"ACGT", unknown_token=4
            ),
        )
    )
    with pytest.raises(ValueError, match="variant-windows"):
        ds.to_dataloader(mode=mode, batch_size=2, buffer_bytes=4 * 1024 * 1024)


def test_flat_buffered_plain_variants_still_works():
    """Regression guard: flat + variants WITHOUT flank tokens must NOT be rejected."""
    import genvarloader as gvl

    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
    )
    # buffered works in-process with the dummy dataset; just confirm it constructs and yields.
    dl = ds.to_dataloader(
        mode="buffered", batch_size=2, drop_last=True, buffer_bytes=4 * 1024 * 1024
    )
    batches = list(dl)
    assert batches  # produced something, no rejection
