# tests/parity/_golden.py
"""Frozen-golden snapshot + replay for the parity suite.

Goldens are generated from the RUST implementation and cross-checked against
the numba oracle at generation time (see generate_goldens.py). Replay imports
rust callables DIRECTLY — never via _dispatch — so these tests survive the
numba/dispatch deletion in Stage B.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, Phase, given, settings

GOLDEN_DIR = Path(__file__).parent / "golden"


def collect_examples(strategy, n: int) -> list:
    """Deterministically draw ``n`` examples from a hypothesis strategy.

    Derandomized + no database + generate-only phase ⇒ stable across runs for a
    fixed hypothesis version. Inputs are frozen INTO the golden, so the replay
    test never re-runs hypothesis.
    """
    out: list = []

    @settings(
        max_examples=n,
        derandomize=True,
        database=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
        deadline=None,
    )
    @given(strategy)
    def _collect(ex):
        if len(out) < n:
            out.append(ex)

    _collect()
    return out


def save_golden(name: str, cases: list) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_DIR / f"{name}.npz", cases=np.array(cases, dtype=object))


def load_golden(name: str) -> list:
    data = np.load(GOLDEN_DIR / f"{name}.npz", allow_pickle=True)
    return list(data["cases"])


# --- direct rust-callable table -------------------------------------------------
# Each entry MUST equal the `rust=` argument of the matching register(...) call in
# production. Verify each against the dispatch map before trusting it.
def _build_rust_kernels() -> dict[str, Callable]:
    from genvarloader import genvarloader as _ext  # compiled extension

    # Kernels whose registered rust= is a Python wrapper (not a bare FFI function):
    # import the same wrapper the register() call used.
    from genvarloader._dataset._reference import (
        _get_reference_rust,  # wraps _ext.get_reference; normalises dtypes + int(pad_char)
    )
    from genvarloader._dataset._tracks import (
        _shift_and_realign_tracks_sparse_rust_wrapper,  # wraps _ext.shift_and_realign_tracks_sparse
    )

    table: dict[str, Callable] = {
        "intervals_to_tracks": _ext.intervals_to_tracks,
        "tracks_to_intervals": _ext.tracks_to_intervals,
        "get_diffs_sparse": _ext.get_diffs_sparse,
        "choose_exonic_variants": _ext.choose_exonic_variants,
        "gather_alleles": _ext.gather_alleles,
        "gather_rows_i32": _ext.gather_rows_i32,
        "gather_rows_f32": _ext.gather_rows_f32,
        "compact_keep_i32": _ext.compact_keep_i32,
        "compact_keep_f32": _ext.compact_keep_f32,
        "fill_empty_scalar_i32": _ext.fill_empty_scalar_i32,
        "fill_empty_scalar_f32": _ext.fill_empty_scalar_f32,
        "fill_empty_fixed_i32": _ext.fill_empty_fixed_i32,
        "fill_empty_fixed_f32": _ext.fill_empty_fixed_f32,
        "fill_empty_seq_u8": _ext.fill_empty_seq_u8,
        "fill_empty_seq_i32": _ext.fill_empty_seq_i32,
        # These two registered rust= is a Python wrapper, NOT the bare FFI function.
        # Using the wrapper ensures correct input normalisation (dtypes, int casts, etc.)
        # and keeps RUST_KERNELS in sync with the dispatch table (per the note above).
        "get_reference": _get_reference_rust,
        "shift_and_realign_tracks_sparse": _shift_and_realign_tracks_sparse_rust_wrapper,
        "reconstruct_haplotypes_from_sparse": _ext.reconstruct_haplotypes_from_sparse,
        "rc_alleles": _ext.rc_alleles,
    }
    # NOTE: kernels whose `rust=` is a PYTHON WRAPPER (not a bare extension fn) —
    # e.g. assemble_variant_buffers (u8/i32 dtype dispatch). Add those by importing
    # the SAME wrapper the registration used; ground-truth against the register() call.
    return table


RUST_KERNELS: dict[str, Callable] = _build_rust_kernels()


def _eq(name: str, i: int, got, exp) -> None:
    got = np.asarray(got)
    exp = np.asarray(exp)
    assert got.dtype == exp.dtype, f"{name}[{i}]: dtype {got.dtype} != {exp.dtype}"
    assert got.shape == exp.shape, f"{name}[{i}]: shape {got.shape} != {exp.shape}"
    np.testing.assert_array_equal(got, exp, err_msg=f"{name}[{i}] value mismatch")


def replay_return(name: str, cases: list) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        _eq(f"{name}#{ci}", 0, fn(*inputs), golden)


def replay_tuple(name: str, cases: list) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        got = fn(*inputs)
        got = got if isinstance(got, tuple) else (got,)
        gold = golden if isinstance(golden, tuple) else (golden,)
        assert len(got) == len(gold), f"{name}#{ci}: tuple len {len(got)} != {len(gold)}"
        for j, (a, b) in enumerate(zip(got, gold)):
            _eq(f"{name}#{ci}", j, a, b)


def replay_inplace(name: str, cases: list, out_factory: Callable, out_index: int) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        out = out_factory(inputs)
        args = list(inputs)
        args.insert(out_index, out)
        fn(*args)
        _eq(f"{name}#{ci}", 0, out, golden)


def replay_dict(name: str, cases: list) -> None:
    fn = RUST_KERNELS[name]
    for ci, (inputs, golden) in enumerate(cases):
        got = fn(*inputs)
        assert set(got) == set(golden), f"{name}#{ci}: keys {set(got)} != {set(golden)}"
        for k in sorted(golden):
            _eq(f"{name}#{ci}:{k}.data", 0, np.asarray(got[k][0]), np.asarray(golden[k][0]))
            _eq(f"{name}#{ci}:{k}.off", 1,
                np.asarray(got[k][1], np.int64), np.asarray(golden[k][1], np.int64))
