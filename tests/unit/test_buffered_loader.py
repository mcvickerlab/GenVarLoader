"""End-to-end tests for mode='buffered'."""

import numpy as np
import pytest
import genvarloader as gvl

pytest.importorskip("torch")


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
            _rv_eq(got, rb)
        elif seq_kind == "annotated":
            np.testing.assert_array_equal(
                to_padded(got.haps, b"N"), to_padded(rb.haps, b"N")
            )
        else:
            np.testing.assert_array_equal(to_padded(got, b"N"), to_padded(rb, b"N"))


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


def _win_eq(a, b):
    da, db = a.to_ragged(), b.to_ragged()
    assert set(da) == set(db), f"keys differ: {set(da)} vs {set(db)}"
    for k in da:
        assert da[k].to_ak().to_list() == db[k].to_ak().to_list(), f"{k} mismatch"


def _iter_mode_none(ds, batch_size):
    """Per-item oracle: same sequential batches mode=None would yield."""
    n = int(np.prod(ds.full_shape))
    n = (n // batch_size) * batch_size  # drop_last=True
    r, s = np.unravel_index(np.arange(n), ds.shape)
    for i in range(0, n, batch_size):
        yield ds[r[i : i + batch_size], s[i : i + batch_size]]


def _with_ref_alleles(ds):
    """Attach synthetic REF alleles to a dummy dataset's variants.

    ``get_dummy_dataset()`` ships with ``variants.ref=None``; ``VarWindowOpt(ref="allele")``
    needs REF allele bytes to be present to exercise the bare-tokenized-REF-allele
    code path. Bare REF-allele mode reads ``variants.ref`` directly and never
    consults the underlying (all-``"N"``) reference genome, so any valid
    nucleotide bytes of the right per-variant lengths are sufficient for a parity
    check; only the lengths need to stay consistent with each variant's ILEN.

    Args:
        ds: A dataset built from :func:`gvl.get_dummy_dataset`.

    Returns:
        A new dataset (input is left unmodified) whose ``variants.ref`` is set.
    """
    import dataclasses

    import seqpro as sp
    from einops import repeat

    from genvarloader._utils import lengths_to_offsets
    from genvarloader._variants._records import RaggedAlleles

    variants = ds._seqs.variants
    n_regions = 4
    n_samples = 4
    # Per genvarloader._dummy.get_dummy_dataset: alt allele lengths per region are
    # [1, 1, 1, 2] with ilens [-2, -1, 0, 1], so ref_len = alt_len - ilen.
    ref_lens = np.array([3, 2, 1, 1])
    ref = RaggedAlleles.from_offsets(
        data=repeat(sp.cast_seqs("ACGTACG"), "a -> (r a)", r=n_regions),
        shape=(n_regions * n_samples, None),
        offsets=lengths_to_offsets(repeat(ref_lens, "s -> (r s)", r=n_regions)),
    )
    new_vars = dataclasses.replace(variants, ref=ref)
    new_seqs = dataclasses.replace(ds._seqs, variants=new_vars)
    return dataclasses.replace(ds, _seqs=new_seqs)


@pytest.mark.parametrize(
    "ref,alt", [("window", "window"), ("window", "allele"), ("allele", "allele")]
)
def test_buffered_variant_windows_matches_per_item(ref, alt):
    ds = gvl.get_dummy_dataset().with_tracks(False).with_output_format("flat")
    if ref == "allele":
        # get_dummy_dataset() has no REF alleles by default; VarWindowOpt(ref="allele")
        # requires them, so attach synthetic ones for this parametrization only.
        ds = _with_ref_alleles(ds)
    ds = ds.with_seqs(
        "variant-windows",
        gvl.VarWindowOpt(
            flank_length=2, token_alphabet=b"ACGT", unknown_token=4, ref=ref, alt=alt
        ),
    )
    bs = 2
    got = list(
        ds.to_dataloader(
            mode="buffered",
            batch_size=bs,
            shuffle=False,
            drop_last=True,
            buffer_bytes=10 * 1024 * 1024,
        )
    )
    exp = list(_iter_mode_none(ds, bs))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        _win_eq(g, e)


def test_buffered_variants_flank_tokens_matches_per_item():
    ds = (
        gvl.get_dummy_dataset()
        .with_seqs("variants")
        .with_tracks(False)
        .with_settings(flank_length=2, token_alphabet=b"ACGT", unknown_token=0)
        .with_output_format("flat")
    )
    bs = 2
    got = list(
        ds.to_dataloader(
            mode="buffered",
            batch_size=bs,
            shuffle=False,
            drop_last=True,
            buffer_bytes=10 * 1024 * 1024,
        )
    )
    exp = list(_iter_mode_none(ds, bs))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        np.testing.assert_array_equal(
            g.flank_tokens.to_ragged().to_ak().to_list(),
            e.flank_tokens.to_ragged().to_ak().to_list(),
        )
