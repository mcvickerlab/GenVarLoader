# tests/parity/generate_goldens.py
"""Regenerate frozen golden fixtures for the parity suite.

RUN MANUALLY while numba is still installed (Stage A):
    pixi run -e dev python -m tests.parity.generate_goldens

For each kernel: draw N deterministic examples, compute the golden from RUST,
and assert the numba oracle agrees BEFORE saving.

*** DANGER (post-W5): numba was DELETED in W5. Re-running this script now freezes
rust == rust with NO oracle cross-check — a silent rust==rust freeze that defeats
the parity contract. Only regenerate on a numba-PRESENT checkout (a commit at or
before the Stage-A snapshot, with numba installed), or the goldens are meaningless. ***

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

try:
    from genvarloader import _dispatch
except ImportError:
    _dispatch = None

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
    (
        "get_diffs_sparse",
        strategies.get_diffs_sparse_inputs(),
        TUPLE,
        200,
        _pre_get_diffs_sparse,
    ),
    (
        "choose_exonic_variants",
        strategies.choose_exonic_variants_inputs(),
        TUPLE,
        200,
        _pre_choose_exonic,
    ),
    (
        "gather_rows_i32",
        strategies.gather_rows_inputs(np.int32),
        TUPLE,
        100,
        _pre_gather_rows,
    ),
    (
        "gather_rows_f32",
        strategies.gather_rows_inputs(np.float32),
        TUPLE,
        100,
        _pre_gather_rows,
    ),
    (
        "gather_alleles",
        strategies.gather_alleles_inputs(),
        TUPLE,
        100,
        _pre_gather_alleles,
    ),
    ("compact_keep_i32", strategies.compact_keep_inputs(np.int32), TUPLE, 100, None),
    ("compact_keep_f32", strategies.compact_keep_inputs(np.float32), TUPLE, 100, None),
    (
        "fill_empty_scalar_i32",
        strategies.fill_empty_scalar_inputs(np.int32),
        TUPLE,
        100,
        _pre_fill_empty_scalar_i32,
    ),
    (
        "fill_empty_scalar_f32",
        strategies.fill_empty_scalar_inputs(np.float32),
        TUPLE,
        100,
        _pre_fill_empty_scalar_f32,
    ),
    (
        "fill_empty_fixed_i32",
        strategies.fill_empty_fixed_inputs(np.int32),
        TUPLE,
        100,
        _pre_fill_empty_fixed_i32,
    ),
    (
        "fill_empty_fixed_f32",
        strategies.fill_empty_fixed_inputs(np.float32),
        TUPLE,
        100,
        _pre_fill_empty_fixed_f32,
    ),
    ("fill_empty_seq_u8", strategies.fill_empty_seq_inputs(np.uint8), TUPLE, 100, None),
    (
        "fill_empty_seq_i32",
        strategies.fill_empty_seq_inputs(np.int32),
        TUPLE,
        100,
        None,
    ),
    ("tracks_to_intervals", strategies.tracks_to_intervals_inputs(), TUPLE, 200, None),
    ("get_reference", strategies.get_reference_inputs(), RETURN, 200, None),
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
            np.testing.assert_array_equal(x, y, err_msg=f"{name}[{i}] oracle mismatch")
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
    if _dispatch is None:
        return False
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

                def of(_inp, t=total_out):
                    return out_factory(t)
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


# ---------------------------------------------------------------------------
# PRNG primitives (xorshift64 / hash4): deterministic scalar table
# ---------------------------------------------------------------------------

UINT64_MAX = 2**64 - 1


def gen_prng() -> None:
    """Freeze xorshift64 and hash4 golden tables.

    Deterministic inputs; no hypothesis required here — we pick a fixed list of
    representative uint64 values and cross-check rust vs numba at generation time.
    """
    from genvarloader._dataset._tracks import _hash4 as _hash4_numba
    from genvarloader._dataset._tracks import _xorshift64 as _xorshift64_numba
    from genvarloader.genvarloader import _debug_hash4 as _hash4_rust
    from genvarloader.genvarloader import _debug_xorshift64 as _xorshift64_rust

    # Representative uint64 inputs: 0, 1, small values, mid-range, near-max.
    xs_inputs: list[int] = [
        0,
        1,
        2,
        42,
        255,
        256,
        65535,
        65536,
        0xDEAD,
        0xBEEF,
        0xDEADBEEF,
        0xCAFEBABEDEAD,
        2**32 - 1,
        2**32,
        2**48,
        2**63 - 1,
        2**63,
        UINT64_MAX - 1,
        UINT64_MAX,
    ] + list(range(1000, 1100))  # 100 sequential values for sequential patterns

    xs_cases = []
    for x in xs_inputs:
        rust_out = int(_xorshift64_rust(x))
        numba_out = int(_xorshift64_numba(np.uint64(x)))
        if rust_out != numba_out:
            raise AssertionError(
                f"xorshift64({x:#x}): rust={rust_out:#x} numba={numba_out:#x}"
            )
        xs_cases.append(((x,), np.uint64(rust_out)))
    _golden.save_golden("prng_xorshift64", xs_cases)
    print(f"  prng_xorshift64: {len(xs_cases)} cases")

    # hash4: representative (a, b, c, d) quadruples.
    h4_quads: list[tuple[int, int, int, int]] = [
        (0, 0, 0, 0),
        (1, 2, 3, 4),
        (0xDEADBEEF, 0xCAFE, 0xBABE, 1),
        (UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX),
        (2**63, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (42, 43, 44, 45),
        (2**32, 2**32 + 1, 2**32 + 2, 2**32 + 3),
    ] + [(i, i + 1, i + 2, i + 3) for i in range(100, 150)]

    h4_cases = []
    for a, b, c, d in h4_quads:
        rust_out = int(_hash4_rust(a, b, c, d))
        numba_out = int(
            _hash4_numba(np.uint64(a), np.uint64(b), np.uint64(c), np.uint64(d))
        )
        if rust_out != numba_out:
            raise AssertionError(
                f"hash4({a:#x},{b:#x},{c:#x},{d:#x}): rust={rust_out:#x} numba={numba_out:#x}"
            )
        h4_cases.append(((a, b, c, d), np.uint64(rust_out)))
    _golden.save_golden("prng_hash4", h4_cases)
    print(f"  prng_hash4: {len(h4_cases)} cases")


# ---------------------------------------------------------------------------
# rc_alleles: freeze in-place RC golden
# ---------------------------------------------------------------------------


def _rc_alleles_batch_strategy():
    """Composite strategy mirroring the test_rc_alleles_parity._allele_batch."""
    from hypothesis import strategies as st

    _ACGTN = np.frombuffer(b"ACGTN", np.uint8)

    @st.composite
    def _allele_batch(draw):
        n_rows = draw(st.integers(1, 4))
        alleles_per_row = [draw(st.integers(0, 3)) for _ in range(n_rows)]
        var_offsets = np.concatenate([[0], np.cumsum(alleles_per_row)]).astype(np.int64)
        n_alleles = int(var_offsets[-1])
        lens = [draw(st.integers(0, 5)) for _ in range(n_alleles)]
        seq_offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        total = int(seq_offsets[-1])
        data = (
            _ACGTN[draw(st.lists(st.integers(0, 4), min_size=total, max_size=total))]
            if total
            else np.zeros(0, np.uint8)
        )
        data = np.ascontiguousarray(data, np.uint8)
        mask = np.array([draw(st.booleans()) for _ in range(n_rows)], np.bool_)
        return data, seq_offsets, var_offsets, mask

    return _allele_batch()


def gen_rc_alleles() -> None:
    """Freeze rc_alleles golden: store (initial_byte_data, seq_off, var_off, mask) → result."""
    nb_fn = _dispatch.backends("rc_alleles")[0] if _have_numba("rc_alleles") else None
    rust_fn = _golden.RUST_KERNELS["rc_alleles"]
    strat = _rc_alleles_batch_strategy()
    examples = _golden.collect_examples(strat, 200)
    cases = []
    for raw in examples:
        data, seq_offsets, var_offsets, mask = raw
        # Normalise inputs (mirrors _rc_alleles_rust wrapper requirements)
        data = np.ascontiguousarray(data, np.uint8)
        seq_offsets = np.ascontiguousarray(seq_offsets, np.int64)
        var_offsets = np.ascontiguousarray(var_offsets, np.int64)
        mask = np.ascontiguousarray(mask, np.bool_)

        # Run Rust on a copy (in-place mutation)
        buf_r = data.copy()
        rust_fn(buf_r, seq_offsets, var_offsets, mask)

        # Cross-check against numba oracle
        if nb_fn is not None:
            buf_n = data.copy()
            nb_fn(buf_n, seq_offsets, var_offsets, mask)
            np.testing.assert_array_equal(
                buf_n, buf_r, err_msg="rc_alleles oracle mismatch"
            )

        # Store: inputs include initial data so replay can copy it
        cases.append(((data, seq_offsets, var_offsets, mask), buf_r))

    _golden.save_golden("rc_alleles", cases)
    print(f"  rc_alleles: {len(cases)} cases")


# ---------------------------------------------------------------------------
# assemble_variant_buffers: freeze fixed parametrised cases
# ---------------------------------------------------------------------------


def gen_assemble_variant_buffers() -> None:
    """Freeze all parametrised assemble_variant_buffers cases.

    Mirrors the exact inputs from test_assemble_variant_buffers_parity.py so the
    golden covers the same mode matrix without re-running numba at test time.
    """
    nb_fn = (
        _dispatch.backends("assemble_variant_buffers")[0]
        if _have_numba("assemble_variant_buffers")
        else None
    )
    rust_fn = _golden.RUST_KERNELS["assemble_variant_buffers"]

    def _reference():
        bases = np.frombuffer(b"ACGT", np.uint8)
        ref = np.tile(bases, 10).astype(np.uint8)
        ref_offsets = np.array([0, ref.size], np.int64)
        return ref, ref_offsets

    def _lut(dtype):
        lut = np.full(256, 4, dtype)
        for i, b in enumerate(b"ACGT"):
            lut[b] = i
        return lut

    def _globals():
        alt_data = np.frombuffer(b"ACGT", np.uint8)
        alt_off = np.array([0, 1, 3, 4], np.int64)
        ref_data = np.frombuffer(b"CGAA", np.uint8)
        ref_off = np.array([0, 1, 2, 4], np.int64)
        v_starts = np.array([5, 12, 20], np.int32)
        ilens = np.array([0, -1, 1], np.int32)
        return alt_data, alt_off, ref_data, ref_off, v_starts, ilens

    cases = []

    ref, ref_offsets = _reference()
    alt_data, alt_off, ref_data, ref_off, v_starts, ilens = _globals()

    # test_windows_mode_matrix: tok_dtype × (ref_mode, alt_mode)
    for tok_dtype in [np.uint8, np.int32]:
        for ref_mode, alt_mode in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            lut = _lut(tok_dtype)
            v_idxs = np.array([0, 1, 2], np.int32)
            row_offsets = np.array([0, 3], np.int64)
            v_contigs = np.zeros(3, np.int32)
            inp = (
                1,
                v_idxs,
                row_offsets,
                alt_data,
                alt_off,
                ref_data,
                ref_off,
                False,
                False,
                ref_mode,
                alt_mode,
                2,
                lut,
                v_contigs,
                v_starts,
                ilens,
                ref,
                ref_offsets,
                ord("N"),
            )
            r = _normalize(rust_fn(*inp))
            if nb_fn is not None:
                _assert_oracle(
                    "assemble_variant_buffers/windows", _normalize(nb_fn(*inp)), r
                )
            cases.append((inp, r))

    # test_variants_mode_matrix: tok_dtype × (want_ref, want_flank)
    for tok_dtype in [np.uint8, np.int32]:
        for want_ref, want_flank in [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]:
            lut = _lut(tok_dtype) if want_flank else None
            v_idxs = np.array([2, 0, 1], np.int32)
            row_offsets = np.array([0, 1, 3], np.int64)
            v_contigs = np.zeros(3, np.int32)
            inp = (
                0,
                v_idxs,
                row_offsets,
                alt_data,
                alt_off,
                ref_data,
                ref_off,
                want_ref,
                want_flank,
                0,
                0,
                2,
                lut,
                v_contigs,
                v_starts,
                ilens,
                ref,
                ref_offsets,
                ord("N"),
            )
            r = _normalize(rust_fn(*inp))
            if nb_fn is not None:
                _assert_oracle(
                    "assemble_variant_buffers/variants", _normalize(nb_fn(*inp)), r
                )
            cases.append((inp, r))

    # test_empty_selection: (mode, ref_mode, alt_mode)
    for mode, ref_mode, alt_mode in [(0, 0, 0), (1, 1, 1)]:
        lut = _lut(np.uint8)
        v_idxs = np.array([], np.int32)
        row_offsets = np.array([0, 0], np.int64)
        v_contigs = np.array([], np.int32)
        inp = (
            mode,
            v_idxs,
            row_offsets,
            alt_data,
            alt_off,
            ref_data,
            ref_off,
            False,
            (mode == 0),
            ref_mode,
            alt_mode,
            2,
            lut,
            v_contigs,
            v_starts,
            ilens,
            ref,
            ref_offsets,
            ord("N"),
        )
        r = _normalize(rust_fn(*inp))
        if nb_fn is not None:
            _assert_oracle("assemble_variant_buffers/empty", _normalize(nb_fn(*inp)), r)
        cases.append((inp, r))

    _golden.save_golden("assemble_variant_buffers", cases)
    print(f"  assemble_variant_buffers: {len(cases)} cases")


if __name__ == "__main__":
    print("Generating value-kernel goldens...")
    gen_value_kernels()
    print("Generating in-place-kernel goldens...")
    gen_inplace_kernels()
    print("Generating PRNG goldens...")
    gen_prng()
    print("Generating rc_alleles golden...")
    gen_rc_alleles()
    print("Generating assemble_variant_buffers golden...")
    gen_assemble_variant_buffers()
    print("Done.")
