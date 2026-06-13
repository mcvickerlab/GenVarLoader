"""flat-mode output, re-wrapped via .to_ragged(), must be byte-identical to ragged mode."""
from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from seqpro.rag import Ragged

from genvarloader import RaggedVariants
from genvarloader._ragged import RaggedAnnotatedHaps

IDX = [0, (np.array([0, 1, 2]),)]  # scalar-ish and a list index


def _to_plain(obj):
    """Normalize a ragged/annot/flat object into dict of ndarrays for comparison."""
    if isinstance(obj, RaggedAnnotatedHaps):
        return {
            "haps": np.asarray(obj.haps.data), "haps_off": np.asarray(obj.haps.offsets),
            "vidx": np.asarray(obj.var_idxs.data), "pos": np.asarray(obj.ref_coords.data),
        }
    if isinstance(obj, Ragged):
        return {"data": np.asarray(obj.data), "off": np.asarray(obj.offsets)}
    raise TypeError(type(obj))


@pytest.mark.parametrize("seqs", ["haplotypes", "reference", "annotated"])
@pytest.mark.parametrize("idx", IDX)
def test_a0_flat_to_ragged_matches_ragged(snap_dataset, seqs, idx):
    ds = snap_dataset.with_seqs(seqs).with_tracks(False)
    ragged = ds[idx]
    flat = ds.with_output_format("flat")[idx]
    rewrapped = flat.to_ragged()
    r, f = _to_plain(ragged), _to_plain(rewrapped)
    assert r.keys() == f.keys()
    for k in r:
        np.testing.assert_array_equal(r[k], f[k], err_msg=f"field {k}")


def _rv_to_lists(rv: RaggedVariants) -> dict:
    out = {"alt": ak.to_list(rv["alt"]), "start": ak.to_list(rv["start"])}
    for f in ("ref", "ilen", "dosage"):
        if f in rv.fields:
            out[f] = ak.to_list(rv[f])
    return out


@pytest.mark.parametrize("idx", IDX)
def test_a_flat_variants_to_ragged_matches_ragged(snap_dataset, idx):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    ragged = ds[idx]                                  # RaggedVariants (current path)
    flat = ds.with_output_format("flat")[idx]         # _FlatVariants
    rewrapped = flat.to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)


def test_a_flat_variants_scalar_scalar_index_matches_ragged(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    # both region AND sample scalar -> triggers squeeze(0) on the variant output
    ragged = ds[0, 0]
    rewrapped = ds.with_output_format("flat")[0, 0].to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)


def test_a_flat_variants_scalar_scalar_index_nonempty_matches_ragged(snap_dataset):
    # Second scalar-scalar case with NON-EMPTY alleles, so actual allele bytes are
    # exercised through the squeezed (b, p, None) -> (p, None) path, not just empty
    # lists. (region=0, sample=2) is known to carry variants in snap_dataset.
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    ragged = ds[0, 2]
    assert any(len(p) > 0 for p in ak.to_list(ragged["alt"])), "expected non-empty alt"
    rewrapped = ds.with_output_format("flat")[0, 2].to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)


def test_a_flat_variants_2d_index_matches_ragged(snap_dataset):
    # Exercise the MULTI-DIM path through _reshape_outer: a genuine 2-D fancy
    # index (region, sample) produces a multi-dim out_reshape, so _FlatAlleles
    # must re-append its ragged None just like its _Flat siblings.
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    n_r, n_s = snap_dataset.shape
    r = np.arange(min(3, n_r))
    s = np.arange(min(2, n_s))
    idx = (r[:, None], s[None, :])  # 2-D (region, sample) fancy index
    ragged = ds[idx]
    flat = ds.with_output_format("flat")[idx]
    # Structural invariant: every field (both _Flat and _FlatAlleles) must carry
    # the ragged None as its trailing axis after the multi-dim out_reshape. A
    # _FlatAlleles that stored out_reshape verbatim would be missing it.
    for name, f in flat.fields.items():
        assert f.shape[-1] is None, f"field {name} shape {f.shape} lost ragged axis"
        assert f.shape.count(None) == 1, f"field {name} shape {f.shape} not single-ragged"
    rewrapped = flat.to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)


def test_a_flat_variants_empty_region_and_ploidy(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    idx = (np.arange(min(6, snap_dataset.shape[0])),)
    ragged = ds[idx]
    rewrapped = ds.with_output_format("flat")[idx].to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)
