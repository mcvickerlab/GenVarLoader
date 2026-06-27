# tests/parity/generate_goldens.py
"""Regenerate frozen golden fixtures for the parity suite.

RUN MANUALLY while numba is still installed (Stage A):
    pixi run -e dev python -m tests.parity.generate_goldens

For each kernel: draw N deterministic examples, compute the golden from RUST,
and assert the numba oracle agrees BEFORE saving. After numba deletion this
script still regenerates from rust (the numba cross-check is skipped if the
backend is gone).

Verified signatures / out_index values (ground-truthed against existing parity tests):

intervals_to_tracks (test_intervals_to_tracks_parity.py):
  Strategy yields 7-tuple: (offset_idxs, starts, itv_starts, itv_ends, itv_values,
    itv_offsets, out_offsets). out_index=6; out dtype float32; size=int(inp[6][-1]).
  Confirmed: assert_inplace_kernel_parity("intervals_to_tracks", inputs, ..., out_index=6).
  Brief placeholder (out_index=7) was wrong.

shift_and_realign_tracks_sparse (test_shift_and_realign_tracks_parity.py):
  Strategy yields (total_out, inputs_tuple); out=np.zeros(total_out, f32) at index 0.
  Registered rust= is _shift_and_realign_tracks_sparse_rust_wrapper (Python wrapper).

reconstruct_haplotypes_from_sparse (test_reconstruct_haplotypes_parity.py):
  Strategy yields (total_out, inputs_tuple); out=np.zeros(total_out, u8) at index 0.
  Registered rust= is _ext.reconstruct_haplotypes_from_sparse (bare FFI).

get_diffs_sparse, choose_exonic_variants, gather_rows_i32/f32:
  Require _as_starts_stops(offsets) normalisation; confirmed via test_flat_variants_parity.py
  and test_get_diffs_sparse_parity.py / test_choose_exonic_variants_parity.py.

gather_alleles: requires ascontiguousarray on all inputs.

fill_empty_scalar_i32/f32: fill arg must be Python int/float (not np.scalar).
fill_empty_fixed_i32/f32: inner and fill args must be Python int/float.
  Confirmed via _fill_empty_scalar / _fill_empty_fixed public wrapper source.

get_reference: registered rust= is _get_reference_rust wrapper (normalises dtypes,
  converts pad_char to int). RUST_KERNELS entry updated in _golden.py to match.
"""
from __future__ import annotations

import numpy as np

from genvarloader import _dispatch

# Import modules to trigger register() calls in _dispatch._REGISTRY before
# _have_numba() or any _dispatch.backends() call is made.
from genvarloader._dataset import _flat_variants  # noqa: F401
from genvarloader._dataset import _genotypes  # noqa: F401
from genvarloader._dataset import _intervals  # noqa: F401
from genvarloader._dataset import _reference  # noqa: F401
from genvarloader._dataset import _tracks  # noqa: F401
from genvarloader._dataset._genotypes import _as_starts_stops
from tests.parity import _golden, strategies

RETURN, TUPLE, INPLACE = "return", "tuple", "inplace"


# ---------------------------------------------------------------------------
# Input normalizers — mirror what the existing parity tests pass to kernels.
# Each function takes the raw strategy output and returns a normalised tuple.
# ---------------------------------------------------------------------------


def _pre_get_diffs_sparse(inp):
    """Normalise offsets to (2,n) int64 and ensure all arrays are contiguous."""
    goi, gvi, offsets, ilens, keep, keep_off, qs, qe, vs = inp
    return (
        np.ascontiguousarray(goi, np.int64),
        np.ascontiguousarray(gvi, np.int32),
        _as_starts_stops(offsets),
        np.ascontiguousarray(ilens, np.int32),
        None if keep is None else np.ascontiguousarray(keep, np.bool_),
        None if keep_off is None else np.ascontiguousarray(keep_off, np.int64),
        None if qs is None else np.ascontiguousarray(qs, np.int32),
        None if qe is None else np.ascontiguousarray(qe, np.int32),
        None if vs is None else np.ascontiguousarray(vs, np.int32),
    )


def _pre_choose_exonic(inp):
    qs, qe, goi, gvi, offsets, vs, ilens = inp
    return (
        np.ascontiguousarray(qs, np.int32),
        np.ascontiguousarray(qe, np.int32),
        np.ascontiguousarray(goi, np.int64),
        np.ascontiguousarray(gvi, np.int32),
        _as_starts_stops(offsets),
        np.ascontiguousarray(vs, np.int32),
        np.ascontiguousarray(ilens, np.int32),
    )


def _pre_gather_rows(inp):
    goi, off, data = inp
    return (
        np.ascontiguousarray(goi, np.int64),
        _as_starts_stops(off),
        np.ascontiguousarray(data),
    )


def _pre_gather_alleles(inp):
    v_idxs, allele_bytes, allele_offsets = inp
    return (
        np.ascontiguousarray(v_idxs, np.int32),
        np.ascontiguousarray(allele_bytes, np.uint8),
        np.ascontiguousarray(allele_offsets, np.int64),
    )


def _pre_fill_empty_scalar_i32(inp):
    data, offsets, fill = inp
    return (data, offsets, int(fill))


def _pre_fill_empty_scalar_f32(inp):
    data, offsets, fill = inp
    return (data, offsets, float(fill))


def _pre_fill_empty_fixed_i32(inp):
    data, offsets, inner, fill = inp
    return (data, offsets, int(inner), int(fill))


def _pre_fill_empty_fixed_f32(inp):
    data, offsets, inner, fill = inp
    return (data, offsets, int(inner), float(fill))


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

# SPEC: (name, strategy, shape, n, preprocess_fn)
#   shape   = RETURN | TUPLE — how the rust callable returns its result
#   preprocess_fn: callable(raw_inp) → normalised_inp, or None for no-op
SPEC: list[tuple] = [
    ("get_diffs_sparse",
     strategies.get_diffs_sparse_inputs(),       TUPLE,  200, _pre_get_diffs_sparse),
    ("choose_exonic_variants",
     strategies.choose_exonic_variants_inputs(),  TUPLE,  200, _pre_choose_exonic),
    ("gather_rows_i32",
     strategies.gather_rows_inputs(np.int32),     TUPLE,  100, _pre_gather_rows),
    ("gather_rows_f32",
     strategies.gather_rows_inputs(np.float32),   TUPLE,  100, _pre_gather_rows),
    ("gather_alleles",
     strategies.gather_alleles_inputs(),          TUPLE,  100, _pre_gather_alleles),
    ("compact_keep_i32",
     strategies.compact_keep_inputs(np.int32),    TUPLE,  100, None),
    ("compact_keep_f32",
     strategies.compact_keep_inputs(np.float32),  TUPLE,  100, None),
    ("fill_empty_scalar_i32",
     strategies.fill_empty_scalar_inputs(np.int32),   TUPLE, 100, _pre_fill_empty_scalar_i32),
    ("fill_empty_scalar_f32",
     strategies.fill_empty_scalar_inputs(np.float32), TUPLE, 100, _pre_fill_empty_scalar_f32),
    ("fill_empty_fixed_i32",
     strategies.fill_empty_fixed_inputs(np.int32),    TUPLE, 100, _pre_fill_empty_fixed_i32),
    ("fill_empty_fixed_f32",
     strategies.fill_empty_fixed_inputs(np.float32),  TUPLE, 100, _pre_fill_empty_fixed_f32),
    ("fill_empty_seq_u8",
     strategies.fill_empty_seq_inputs(np.uint8),  TUPLE,  100, None),
    ("fill_empty_seq_i32",
     strategies.fill_empty_seq_inputs(np.int32),  TUPLE,  100, None),
    ("tracks_to_intervals",
     strategies.tracks_to_intervals_inputs(),     TUPLE,  200, None),
    ("get_reference",
     strategies.get_reference_inputs(),           RETURN, 200, None),
]

# INPLACE_SPEC: (name, strategy, n, out_factory, out_index)
#   For shift_and_realign and reconstruct: strategy yields (total_out, inputs_tuple),
#     out_factory receives total_out (scalar), out inserted at index 0.
#   For intervals_to_tracks: strategy yields 7-tuple directly, out_factory receives
#     the inputs tuple, out inserted at index 6 (verified: assert_inplace_kernel_parity
#     in test_intervals_to_tracks_parity.py uses out_index=6, NOT 7).
INPLACE_SPEC: list[tuple] = [
    (
        "intervals_to_tracks",
        strategies.intervals_to_tracks_inputs(),
        200,
        # inp[6] = out_offsets; inp[6][-1] = total output length.
        # NaN sentinel: unwritten positions stay NaN and are caught by oracle.
        lambda inp: np.full(int(inp[6][-1]), np.nan, np.float32),
        6,  # out is inserted before out_offsets (the 7th element)
    ),
    (
        "shift_and_realign_tracks_sparse",
        strategies.shift_and_realign_tracks_inputs(),
        200,
        lambda total_out: np.zeros(total_out, np.float32),
        0,
    ),
    (
        "reconstruct_haplotypes_from_sparse",
        strategies.reconstruct_haplotypes_inputs(),
        200,
        lambda total_out: np.zeros(total_out, np.uint8),
        0,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(out):
    """Normalise kernel output to ndarray or tuple of ndarrays for comparison."""
    if isinstance(out, tuple):
        return tuple(np.asarray(x) for x in out)
    if isinstance(out, dict):
        return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in out.items()}
    return np.asarray(out)


def _assert_oracle(name: str, a, b) -> None:
    """Assert numba (a) == rust (b); both already normalised.

    If this fires it is a REAL numba/rust divergence — do NOT suppress it.
    See the numba-oracle-bug policy: determine whether numba is the buggy side,
    file a separate issue, and block this golden until the divergence is resolved.
    """
    if isinstance(a, tuple):
        assert len(a) == len(b), f"{name}: tuple len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            np.testing.assert_array_equal(
                x, y, err_msg=f"{name}[{i}] oracle mismatch"
            )
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{name}: dict keys mismatch {set(a)} vs {set(b)}"
        for k in a:
            np.testing.assert_array_equal(a[k][0], b[k][0])
            np.testing.assert_array_equal(
                np.asarray(a[k][1], np.int64), np.asarray(b[k][1], np.int64)
            )
    else:
        np.testing.assert_array_equal(a, b, err_msg=f"{name} oracle mismatch")


def _have_numba(name: str) -> bool:
    try:
        _dispatch.backends(name)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_value_kernels() -> None:
    for name, strat, shape, n, preprocess in SPEC:
        examples = _golden.collect_examples(strat, n)
        rust = _golden.RUST_KERNELS[name]
        nb_fn = _dispatch.backends(name)[0] if _have_numba(name) else None
        cases = []
        for raw_inp in examples:
            inp = preprocess(raw_inp) if preprocess is not None else raw_inp
            r = _normalize(rust(*inp))
            if nb_fn is not None:
                _assert_oracle(name, _normalize(nb_fn(*inp)), r)
            cases.append((inp, r))
        _golden.save_golden(name, cases)
        print(f"  {name}: {len(cases)} cases")


def gen_inplace_kernels() -> None:
    for name, strat, n, out_factory, out_index in INPLACE_SPEC:
        examples = _golden.collect_examples(strat, n)
        rust = _golden.RUST_KERNELS[name]
        nb_fn = _dispatch.backends(name)[0] if _have_numba(name) else None
        cases = []
        for ex in examples:
            # shift/reconstruct strategies yield (total_out, inputs_tuple);
            # intervals_to_tracks yields the 7-element inputs tuple directly.
            if isinstance(ex, tuple) and len(ex) == 2 and np.isscalar(ex[0]):
                total_out, inputs = ex
                of = lambda _inp, t=total_out: out_factory(t)
            else:
                inputs = ex
                of = out_factory
            # Run Rust kernel on a fresh out buffer
            out_r = of(inputs)
            args = list(inputs)
            args.insert(out_index, out_r)
            rust(*args)
            # Cross-check against numba oracle — STOP if mismatch (not suppressed)
            if nb_fn is not None:
                out_n = of(inputs)
                args_n = list(inputs)
                args_n.insert(out_index, out_n)
                nb_fn(*args_n)
                np.testing.assert_array_equal(
                    out_n, out_r, err_msg=f"{name} oracle mismatch"
                )
            cases.append((inputs, np.asarray(out_r)))
        _golden.save_golden(name, cases)
        print(f"  {name}: {len(cases)} cases")


if __name__ == "__main__":
    print("Generating value-kernel goldens...")
    gen_value_kernels()
    print("Generating in-place-kernel goldens...")
    gen_inplace_kernels()
    print("Done.")
