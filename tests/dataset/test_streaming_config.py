import shutil
import subprocess
from pathlib import Path

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


@pytest.mark.parametrize("backend", ["svar1", "pgen"])
def test_af_missing_guard_raises(streaming_case, backend):
    # svar1 here = the UN-cached fixture (streaming_case does NOT run cache_afs);
    # pgen has no AF path. Both must raise the same RuntimeError as the written path.
    regions, reference, variants, _ = streaming_case(backend)
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_seqs("variants")
        .with_settings(min_af=0.1)
    )
    with pytest.raises(RuntimeError, match="AFs cached"):
        next(iter(sds.to_iter(batch_size=4)))


def test_af_missing_guard_raises_vcf_without_info_af(streaming_case):
    # The committed streaming_case("vcf") fixture's VCF has NO INFO/AF header field
    # (verified), so has_cached_af is False -> the guard raises.
    regions, reference, vcf_no_af, _ = streaming_case("vcf")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=vcf_no_af)
        .with_seqs("variants")
        .with_settings(min_af=0.1)
    )
    with pytest.raises(RuntimeError, match="AFs cached"):
        next(iter(sds.to_iter(batch_size=4)))


def test_available_var_fields_svar1_includes_ref(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert "ref" in sds.available_var_fields
    assert {"alt", "ilen", "start"} <= set(sds.available_var_fields)


def test_active_var_fields_defaults_to_alt_ilen_start(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert sds.active_var_fields == ["alt", "ilen", "start"]


def test_unknown_var_field_raises(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    with pytest.raises(ValueError, match="not available"):
        sds.with_settings(var_fields=["alt", "start", "NOPE"])


def test_var_fields_with_haplotypes_output_raises(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=["alt", "start", "ref"])
        .with_seqs("haplotypes")
    )
    with pytest.raises(NotImplementedError, match="var_fields"):
        next(iter(sds.to_iter(batch_size=2)))


def test_pgen_var_fields_limited_to_ref(streaming_case):
    regions, reference, variants, _written = streaming_case("pgen")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert "ref" in sds.available_var_fields
    assert sds.available_var_fields == ["alt", "ilen", "start", "ref"]


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


# --- Wave B PR-B3a review (code-review fix pass) -----------------------------

_RESERVED_COLLISION_REF_SEQ = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"  # 40bp

# A VCF INFO field named `alt_offsets` -- the same name as the FFI dict's fixed
# `alt_offsets` key (`next_batch_variants`'s `dict.set_item("alt_offsets", ...)`
# in both `record_stream/engine.rs` and `ffi/stream_engine.rs`). Deliberately
# collides to regression-guard `_RESERVED_VAR_FIELD_NAMES` (Critical 1).
_VCF_RESERVED_NAME_COLLISION = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##INFO=<ID=alt_offsets,Number=1,Type=Integer,Description="Deliberately collides with the FFI dict's fixed alt_offsets key">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\talt_offsets=999\tGT\t1|0\t0|0
chr1\t16\t.\tT\tC\t.\t.\talt_offsets=777\tGT\t1|1\t0|1
"""


def _write_reserved_name_collision_vcf(tmp_path: Path) -> tuple[Path, Path]:
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_RESERVED_COLLISION_REF_SEQ}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)
    vcf = tmp_path / "in.vcf"
    vcf.write_text(_VCF_RESERVED_NAME_COLLISION)
    vcf_gz = tmp_path / "in.vcf.gz"
    subprocess.run(["bcftools", "view", "-Oz", "-o", str(vcf_gz), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(vcf_gz)], check=True)
    return vcf_gz, ref


def test_reserved_name_collision_not_advertised_and_rejected(tmp_path):
    """Regression for Critical 1 (Wave B PR-B3a review, #304): a VCF INFO field
    named `alt_offsets` collides with the FFI dict's fixed `alt_offsets` key --
    `PyDict::set_item` silently overwrites, so before the fix
    (`_RESERVED_VAR_FIELD_NAMES` omitted `alt_offsets`, guarding only
    `start`/`ilen`/`ref`/`offsets`/`ref_offsets`) this name was advertised via
    `available_var_fields`, accepted by `with_settings`, and would silently
    overwrite the ALT sequence offsets array with the INFO column's values at
    read time -- wrong ALT bytes, no exception (confirmed empirically in the
    review on a 2-variant fixture: ALT changed from `[b'C', b'AGG']` to
    `[b'G', b'']`).

    This test would FAIL against the unfixed frozenset in BOTH ways checked
    below: `alt_offsets` would appear in `available_var_fields` (assertion 1),
    and requesting it via `with_settings` would not raise (assertion 2) --
    verified by running this test before applying the fix (see the fix-pass
    report).
    """
    vcf_gz, ref = _write_reserved_name_collision_vcf(tmp_path)
    regions = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [len(_RESERVED_COLLISION_REF_SEQ)],
        }
    )
    sds = gvl.StreamingDataset(regions, reference=str(ref), variants=str(vcf_gz))

    # 1. The reserved (FFI-fixed-key-colliding) name must never be advertised as
    #    a requestable var_field.
    assert "alt_offsets" not in sds.available_var_fields

    # 2. Explicitly requesting it must fail loudly and clearly (ValueError, "not
    #    available"), never silently succeed and corrupt output.
    with pytest.raises(ValueError, match="not available"):
        sds.with_settings(var_fields=["alt", "start", "alt_offsets"])


def test_af_cached_svar1_af_field_unservable_raises_early(streaming_case, tmp_path):
    """Regression for Important 1 (Wave B PR-B3a review): an AF-cached SVAR1
    store advertises "AF" via `available_var_fields` (a real on-disk numeric
    index column) but the SVAR1 Rust engine has no general INFO/index-column
    gather yet -- unlike per-call FORMAT/dosage fields, which PR-B3b wired
    through (see `test_streaming_variants_parity.py`'s
    `test_streaming_svar1_dosage_matches_written`/
    `test_streaming_svar1_custom_format_field_matches_written`). Before the
    fix, this failure only surfaced from inside the per-batch packing loop
    after a full `build_engine` + producer thread + first window read;
    `with_settings` must now reject it immediately, at configuration time,
    naming the field and that this is still deferred follow-up work -- never
    silently building an engine that can't serve the request. (Minor 1, PR-B3b
    review, #304: the message no longer says "deferred to PR-B3b" -- PR-B3b IS
    this task, and it did NOT close this particular gap.)
    """
    regions, reference, svar, _af = _af_cached_svar(streaming_case, tmp_path)
    sds = gvl.StreamingDataset(regions, reference=reference, variants=svar)
    assert "AF" in sds.available_var_fields
    assert "AF" not in sds.servable_var_fields
    with pytest.raises(NotImplementedError, match="numeric INDEX column"):
        sds.with_settings(var_fields=["alt", "ilen", "start", "AF"])


def test_var_fields_ref_is_forwarded_svar1(streaming_case):
    """Smoke test (Important 3, Wave B PR-B3a review): pins that `want_ref` is
    actually forwarded from `var_fields` through `build_engine` to the Rust
    engine and packed into real output. All five of Task 5's original tests
    were config-surface only (never iterated a batch with `ref` requested), so
    the suite would still have passed if `_Svar1Backend.build_engine` never
    read `"ref" in _active_fields` at all -- `ref_rag` would then just stay
    `None` in the packing loop with nothing to catch it. This is a SMOKE test,
    not byte-identical parity against the written path (a later task's job):
    it only proves non-empty `ref` data actually arrives, with one entry per
    variant (same count as the already-covered `start`).

    NOTE: `RaggedVariants.ref[h]` (a single-hap opaque-string index) collapses
    ALL of that hap's REF alleles into ONE concatenated `bytes` object
    (verified empirically: `seqpro.rag.Ragged.to_strings()` merges the whole
    ragged group, not just the character dimension within each variant) -- so
    counting "entries" must go through the field's own string-boundary layout
    (`._rl.str_offsets`, the same primitive `RaggedVariants.ilen` itself uses
    to derive REF-allele lengths), not a naive `len(field[h])`.
    """
    regions, reference, variants, _written = streaming_case("svar1")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ilen", "start", "ref"])
    )
    total_variants = 0
    saw_nonempty_ref = False
    for data, r_idx, _s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            cell = data[k]
            n_variants = len(cell.start.data)  # one scalar start per variant, all haps
            n_ref_entries = len(np.diff(cell.ref._rl.str_offsets))
            assert n_ref_entries == n_variants, (
                "ref must have exactly one string entry per variant, same count as start"
            )
            total_variants += n_variants
            if cell.ref.data.size > 0:
                saw_nonempty_ref = True
    assert total_variants > 0, (
        "streaming_case('svar1')'s fixture must carry >=1 variant -- a vacuous "
        "all-empty pass would not prove ref data actually arrived"
    )
    assert saw_nonempty_ref, "expected at least one non-empty ref allele byte"
