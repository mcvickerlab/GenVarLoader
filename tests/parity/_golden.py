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

    from genvarloader._dataset._flat_variants import (
        _assemble_variant_buffers_rust,  # Python wrapper: routes to u8/i32 by lut dtype
        _rc_alleles_rust,  # Python wrapper: asserts contiguous uint8 then calls ext
    )

    # Shim for reconstruct_haplotypes_from_sparse: the FFI now requires `parallel`
    # but existing replay_inplace callers don't pass it. Default to False (serial)
    # so existing golden replays are byte-identical to the pre-C1 implementation.
    # The rayon-equivalence test explicitly passes parallel=True to exercise the
    # parallel branch.
    _rhfs_raw = _ext.reconstruct_haplotypes_from_sparse

    def _reconstruct_haplotypes_from_sparse_shim(*args, parallel: bool = False, **kwargs):
        return _rhfs_raw(*args, parallel=parallel, **kwargs)

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
        # These registered rust= callables are Python wrappers, NOT bare FFI functions.
        # Using the wrapper ensures correct input normalisation (dtypes, int casts, etc.)
        # and keeps RUST_KERNELS in sync with the dispatch table.
        "get_reference": _get_reference_rust,
        "shift_and_realign_tracks_sparse": _shift_and_realign_tracks_sparse_rust_wrapper,
        # Shim adds `parallel=False` default so existing replay_inplace callers
        # (which don't pass parallel) continue to work unchanged.
        "reconstruct_haplotypes_from_sparse": _reconstruct_haplotypes_from_sparse_shim,
        # rc_alleles: registered rust= is _rc_alleles_rust (wrapper); use wrapper here.
        "rc_alleles": _rc_alleles_rust,
        # assemble_variant_buffers: registered rust= is _assemble_variant_buffers_rust
        # (dtype-selecting shim: routes to u8/i32 monomorphization by lut dtype).
        "assemble_variant_buffers": _assemble_variant_buffers_rust,
    }
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


# ---------------------------------------------------------------------------
# Dataset-level output serialization (flatten + compare)
# ---------------------------------------------------------------------------


def flatten_output(out):
    """Serialize a Dataset.__getitem__ result to a dict of arrays for golden storage.

    Handles:
      - seqpro.rag.Ragged         → {"kind":"ragged", "data":..., "offsets":...}
      - RaggedAnnotatedHaps        → {"kind":"annot", "haps_data":..., ...}
      - RaggedVariants             → {"kind":"ragged_variants", "field_names":[...], "fields":{...}}
      - _FlatVariantWindows        → {"kind":"flat_variant_windows", "windows":{...}}
      - plain ndarray              → {"kind":"array", "data":...}
      - tuple thereof              → {"kind":"tuple", "items":[...]}
    """
    from seqpro.rag import Ragged
    from genvarloader._ragged import RaggedAnnotatedHaps

    # Lazily import to avoid circular imports at module level
    try:
        from genvarloader._dataset._rag_variants import RaggedVariants as _RaggedVariants
    except Exception:
        _RaggedVariants = None

    try:
        from genvarloader._dataset._flat_variants import _FlatVariantWindows as _FVW
    except Exception:
        _FVW = None

    # RaggedAnnotatedHaps must come before Ragged (it's a subclass of Ragged)
    if isinstance(out, RaggedAnnotatedHaps):
        return {
            "kind": "annot",
            "haps_data": np.asarray(out.haps.data),
            "haps_offsets": np.asarray(out.haps.offsets, np.int64),
            "var_idxs_data": np.asarray(out.var_idxs.data),
            "var_idxs_offsets": np.asarray(out.var_idxs.offsets, np.int64),
            "ref_coords_data": np.asarray(out.ref_coords.data),
            "ref_coords_offsets": np.asarray(out.ref_coords.offsets, np.int64),
        }

    # RaggedVariants must come before Ragged (it's a subclass)
    if _RaggedVariants is not None and isinstance(out, _RaggedVariants):
        flat_fields: dict = {}
        for fname in out.fields:
            f = out[fname]
            is_str = bool(getattr(f, "is_string", False))
            flat_fields[fname] = {
                "is_string": is_str,
                "data": np.asarray(f.data, dtype="S1") if is_str else np.asarray(f.data),
                "offsets": np.asarray(f.offsets, np.int64),
            }
        return {
            "kind": "ragged_variants",
            "field_names": list(out.fields),
            "fields": flat_fields,
        }

    if _FVW is not None and isinstance(out, _FVW):
        flat_wins: dict = {}
        for wname in ("ref_window", "alt_window", "ref", "alt"):
            w = getattr(out, wname, None)
            if w is not None:
                flat_wins[wname] = {
                    "data": np.asarray(w.data),
                    "seq_offsets": np.asarray(w.seq_offsets, np.int64),
                    "var_offsets": np.asarray(w.var_offsets, np.int64),
                }
        return {"kind": "flat_variant_windows", "windows": flat_wins}

    if isinstance(out, Ragged):
        return {
            "kind": "ragged",
            "data": np.asarray(out.data),
            "offsets": np.asarray(out.offsets, np.int64),
        }

    if isinstance(out, tuple):
        return {"kind": "tuple", "items": [flatten_output(o) for o in out]}

    return {"kind": "array", "data": np.asarray(out)}


def _assert_flat_eq(got_flat, exp_flat, name: str) -> None:
    """Recursively assert two flattened dicts are byte-identical."""
    got_kind = got_flat["kind"] if isinstance(got_flat, dict) else type(got_flat).__name__
    exp_kind = exp_flat["kind"] if isinstance(exp_flat, dict) else type(exp_flat).__name__
    assert got_kind == exp_kind, f"{name}: kind {got_kind!r} != {exp_kind!r}"
    kind = got_flat["kind"]

    if kind == "ragged":
        _eq(name + ".data", 0, got_flat["data"], exp_flat["data"])
        _eq(name + ".offsets", 0, got_flat["offsets"], exp_flat["offsets"])

    elif kind == "annot":
        for key in ("haps_data", "haps_offsets", "var_idxs_data", "var_idxs_offsets",
                    "ref_coords_data", "ref_coords_offsets"):
            _eq(f"{name}.{key}", 0, got_flat[key], exp_flat[key])

    elif kind == "array":
        _eq(name + ".data", 0, got_flat["data"], exp_flat["data"])

    elif kind == "tuple":
        gi, ei = got_flat["items"], exp_flat["items"]
        assert len(gi) == len(ei), f"{name}: tuple len {len(gi)} != {len(ei)}"
        for i, (g, e) in enumerate(zip(gi, ei)):
            _assert_flat_eq(g, e, f"{name}[{i}]")

    elif kind == "ragged_variants":
        gf, ef = got_flat["fields"], exp_flat["fields"]
        assert set(gf) == set(ef), f"{name}: field names {set(gf)} != {set(ef)}"
        for fname in ef:
            g, e = gf[fname], ef[fname]
            assert g["is_string"] == e["is_string"], f"{name}.{fname}: is_string mismatch"
            _eq(f"{name}.{fname}.data", 0, g["data"], e["data"])
            _eq(f"{name}.{fname}.offsets", 0, g["offsets"], e["offsets"])

    elif kind == "flat_variant_windows":
        gw, ew = got_flat["windows"], exp_flat["windows"]
        assert set(gw) == set(ew), f"{name}: windows {set(gw)} != {set(ew)}"
        for wname in ew:
            g, e = gw[wname], ew[wname]
            _eq(f"{name}.{wname}.data", 0, g["data"], e["data"])
            _eq(f"{name}.{wname}.seq_offsets", 0, g["seq_offsets"], e["seq_offsets"])
            _eq(f"{name}.{wname}.var_offsets", 0, g["var_offsets"], e["var_offsets"])

    else:
        raise ValueError(f"Unknown kind {kind!r}")


def assert_output_matches_golden(out, golden) -> None:
    """Assert a fresh Dataset output equals a frozen golden (byte-identical)."""
    got_flat = flatten_output(out)
    _assert_flat_eq(got_flat, golden, "output")


def save_flat_golden(name: str, out) -> None:
    """Flatten ``out`` and save as a single-item golden for dataset-level replay."""
    save_golden(name, [flatten_output(out)])


def load_flat_golden(name: str):
    """Load a single flattened dataset golden saved via ``save_flat_golden``."""
    return load_golden(name)[0]


def make_kernel_spy(kernel_name: str):
    """Install a counting spy on the direct rust callable at its production call site.

    Returns ``(spy_fn, calls_dict, restore_fn)``. Call ``restore_fn()`` to undo.
    """
    import importlib

    # Each entry is (primary_module, attr_name, [extra_modules_to_also_patch]).
    # Extra modules have the same attr bound via a direct import; we must patch
    # each alias so the spy intercepts all call sites.
    _KERNEL_SITES: dict[str, tuple[str, str, list[str]]] = {
        "get_reference": ("genvarloader._dataset._reference", "_get_reference_rust", []),
        "assemble_variant_buffers": ("genvarloader._dataset._flat_variants", "_assemble_variant_buffers_rust", []),
        "gather_rows_i32": ("genvarloader._dataset._flat_variants", "_gather_rows_i32_rust", []),
        "compact_keep_i32": ("genvarloader._dataset._flat_variants", "_compact_keep_i32_rust", []),
        "rc_alleles": ("genvarloader._dataset._flat_variants", "_rc_alleles_rust", ["genvarloader._dataset._rag_variants"]),
    }

    if kernel_name not in _KERNEL_SITES:
        raise KeyError(f"make_kernel_spy: no site registered for {kernel_name!r}; known: {sorted(_KERNEL_SITES)}")

    mod_name, attr_name, extra_mod_names = _KERNEL_SITES[kernel_name]
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr_name)
    calls: dict = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    setattr(mod, attr_name, spy)
    extra_mods = [importlib.import_module(m) for m in extra_mod_names]
    for em in extra_mods:
        setattr(em, attr_name, spy)

    def restore():
        setattr(mod, attr_name, orig)
        for em in extra_mods:
            setattr(em, attr_name, orig)

    return spy, calls, restore
