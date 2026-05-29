"""End-to-end tests for mode='buffered'."""
import numpy as np
import pytest
import genvarloader as gvl

pytest.importorskip("torch")


@pytest.mark.parametrize("seq_kind", ["reference", "haplotypes", "annotated", "variants"])
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
        seen += batch_size if not isinstance(batch, tuple) else (
            batch[0].shape[0] if hasattr(batch[0], "shape") else len(batch[0])
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
    ds = gvl.get_dummy_dataset().with_seqs("haplotypes").with_settings(deterministic=False)
    with pytest.raises(ValueError, match="deterministic"):
        ds.to_dataloader(mode="buffered", batch_size=2)
