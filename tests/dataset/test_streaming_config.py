import shutil

import numpy as np
import polars as pl
import pytest
from genoray import SparseVar

import genvarloader as gvl
from genvarloader._dataset._streaming import StreamingDataset
from genvarloader._ragged import RaggedSeqs


def _tiny_sds() -> StreamingDataset:
    # Internal/test construction path (no real variant source): inject a
    # reconstruct callback so we exercise the config surface only.
    #
    # NOTE: `StreamingDataset.__init__` calls `sp.bed.read(regions)` when
    # `regions` isn't already a `pl.DataFrame`, and `sp.bed.read` only accepts
    # a str/Path -- a raw numpy array is not a valid bed input. The injected-
    # path tests elsewhere (`test_streaming_scheduler.py`) pass a `pl.DataFrame`
    # bed directly, so do the same here.
    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 10], "chromEnd": [10, 20]}
    )
    return StreamingDataset(
        bed,
        contigs=["chr1"],
        n_samples=2,
        ploidy=2,
        _reconstruct_window=lambda r, s: None,
    )


def test_defaults_preserve_current_behavior():
    sds = _tiny_sds()
    assert sds._seq_kind is RaggedSeqs
    assert sds._output_length == "ragged"
    assert sds._jitter == 0
    assert sds._rng is None
    assert sds._deterministic is True


def test_with_len_sets_output_length_and_copies():
    sds = _tiny_sds()
    out = sds.with_len(200)
    assert out is not sds
    assert out._output_length == 200
    assert sds._output_length == "ragged"  # original unchanged
    assert sds.with_len("ragged")._output_length == "ragged"


def test_with_len_rejects_variable_and_nonpositive():
    sds = _tiny_sds()
    with pytest.raises((ValueError, NotImplementedError)):
        sds.with_len("variable")  # no streaming analog
    with pytest.raises(ValueError):
        sds.with_len(0)


def test_with_settings_sets_jitter_rng_deterministic():
    sds = _tiny_sds().with_settings(jitter=4, rng=0, deterministic=False)
    assert sds._jitter == 4
    assert sds._rng == 0
    assert sds._deterministic is False


def test_with_settings_min_max_af_stored(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert sds._min_af is None and sds._max_af is None
    out = sds.with_settings(min_af=0.1, max_af=0.9)
    assert out._min_af == 0.1 and out._max_af == 0.9
    assert sds._min_af is None and sds._max_af is None  # immutable
    assert out._jitter == sds._jitter  # copy preserves others


def test_af_filter_rejects_non_variants_output(streaming_case):
    regions, reference, variants, _ = streaming_case("svar1")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_seqs("haplotypes")
        .with_settings(min_af=0.1)
    )
    with pytest.raises(NotImplementedError, match="af"):
        next(iter(sds.to_iter(batch_size=4)))


def _af_cached_svar(streaming_case, tmp_path):
    regions, reference, variants, _ = streaming_case("svar1")
    dst = tmp_path / "af.svar"
    shutil.copytree(variants, dst)
    sv = SparseVar(str(dst))
    sv.cache_afs()
    af = sv.index.sort("index")["AF"].to_numpy()
    return regions, reference, str(dst), af


def test_svar1_min_af_actually_filters(streaming_case, tmp_path):
    regions, reference, svar, af = _af_cached_svar(streaming_case, tmp_path)
    thr = float(np.median(af))
    base = gvl.StreamingDataset(regions, reference=reference, variants=svar).with_seqs(
        "variants"
    )
    filt = base.with_settings(min_af=thr)

    def total(ds):
        # Count VARIANTS per (cell, hap) via `start` -- one scalar per kept variant.
        # NB: do NOT count `alt[h]`: for a hap with zero kept variants `alt[h]` is an
        # empty bytestring `b''`, and `np.atleast_1d(np.asarray(b''))` has shape (1,),
        # so an alt-byte count saturates at a fixed structural value and never reflects
        # filtering. `start[h]` is one integer per variant, so `.size` is the true count.
        n = 0
        for data, r_idx, _s in ds.to_iter(batch_size=4):
            for k in range(len(r_idx)):
                for h in range(ds.ploidy):
                    n += np.asarray(data[k].start[h]).size
        return n

    n_base, n_filt = total(base), total(filt)
    # An AF threshold above every possible AF (1.0) must drop EVERY variant -- the
    # regression guard proving the Rust AF keep actually runs end-to-end (not a no-op).
    n_extreme = total(base.with_settings(min_af=2.0))
    assert n_base > 0, "unfiltered stream must yield variants"
    assert n_filt < n_base, "median-AF min_af must drop some variants"
    assert n_extreme == 0, "min_af above max AF must drop all variants"


def test_with_seqs_accepts_annotated_and_variants_rejects_variant_windows():
    sds = _tiny_sds()
    from genvarloader._dataset._rag_variants import RaggedVariants
    from genvarloader._ragged import RaggedAnnotatedHaps

    assert sds.with_seqs("annotated")._seq_kind is RaggedAnnotatedHaps
    assert sds.with_seqs("haplotypes")._seq_kind is RaggedSeqs
    # Wave B PR-B1 (#304): "variants" is now accepted at the config layer for all
    # backends. SVAR1/VCF/PGEN produce variants; only the `.svar2` backend still
    # raises NotImplementedError, and that raises later at iterate time
    # (`_iter_batches`), not here at the config layer.
    assert sds.with_seqs("variants")._seq_kind is RaggedVariants
    # "variant-windows"/"reference" remain later Wave B / follow-up work.
    with pytest.raises(NotImplementedError):
        sds.with_seqs("variant-windows")
