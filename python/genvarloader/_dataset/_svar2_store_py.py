"""Read-bound SVAR2 haplotype reconstruction: one all-Rust FFI call gathering off a
query-only genoray ``Svar2Store`` (``genoray_core::query::gather_haps_readbound``), with
NO interval-search-tree rebuild and NO dense-union rebuild.

Byte-identical to the existing union-path oracle (``SparseVar2Source.reconstruct``,
``_svar2_source.py``), which calls ``reconstruct_haplotypes_from_svar2`` over
``SparseVar2.overlap_batch``'s eagerly-unioned dense channel. This module instead
marshals ``SparseVar2.find_ranges``'s per-class-split ranges through
``genoray_core::query::gather_haps_readbound`` -> ``svar2::split_to_flat`` (Rust side)
and reuses that same validated kernel — see ``reconstruct_haplotypes_from_svar2_readbound``
in ``src/ffi/mod.rs``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from .._flat import _Flat
from ..genvarloader import (
    Svar2Store,
    reconstruct_haplotypes_from_svar2_readbound,
    shift_and_realign_tracks_from_svar2_readbound,
)

if TYPE_CHECKING:
    from genoray import SparseVar2
    from numpy.typing import NDArray
    from seqpro.rag import Ragged


def build_readbound_haps(
    svar2: "SparseVar2",
    contig: str,
    regions,  # iterable of (start, end), length R
    ref_: "NDArray[np.uint8]",  # the contig reference bytes
    ref_offsets: "NDArray[np.int64]",  # e.g. np.array([0, len(ref_)])
    pad_char: int,
    shifts: "NDArray[np.int32] | None" = None,  # (R*S, P); None -> zeros
    output_length: int = -1,
    parallel: bool = False,
) -> "Ragged[np.bytes_]":
    """Reconstruct the full-cohort haplotypes over ``regions`` via the read-bound kernel.

    Mirrors ``SparseVar2Source.reconstruct``'s signature/return shape exactly (query
    order region-major, sample-minor: ``q = r*S + s``), but drives
    ``SparseVar2.find_ranges`` (search-only, no dense union) + one Rust FFI call
    instead of ``overlap_batch``'s eager per-region dense union.
    """
    reg = [(int(s), int(e)) for s, e in regions]
    R = len(reg)
    S = svar2.n_samples
    P = svar2.ploidy

    d = svar2.find_ranges(
        contig, [s for s, _ in reg], [e for _, e in reg], samples=None
    )

    region_starts_r = np.asarray(d["region_starts"], np.int64)  # (R,)
    sample_cols = np.asarray(d["sample_cols"], np.int64)  # (S,)
    # vk_*_range rows are already (R, S, P) row-major == query-major (q = r*S+s,
    # row = q*P + p), so they pass through unchanged.
    vk_snp_range = np.ascontiguousarray(d["vk_snp_range"], np.int64)  # (R*S*P, 2)
    vk_indel_range = np.ascontiguousarray(d["vk_indel_range"], np.int64)
    dense_snp_range_r = np.asarray(d["dense_snp_range"], np.int64)  # (R, 2)
    dense_indel_range_r = np.asarray(d["dense_indel_range"], np.int64)  # (R, 2)

    n_q = R * S
    region_starts = np.repeat(region_starts_r, S).astype(np.uint32)  # (n_q,)
    orig_samples = np.tile(sample_cols, R)  # (n_q,)
    dense_snp_range = np.ascontiguousarray(
        np.repeat(dense_snp_range_r, S, axis=0), np.int64
    )  # (n_q, 2)
    dense_indel_range = np.ascontiguousarray(
        np.repeat(dense_indel_range_r, S, axis=0), np.int64
    )  # (n_q, 2)

    reg_arr = np.asarray(reg, np.int32).reshape(R, 2)
    region_bounds = np.ascontiguousarray(
        np.repeat(reg_arr, S, axis=0), np.int32
    )  # (n_q, 2)

    if shifts is None:
        shifts_a = np.zeros((n_q, P), dtype=np.int32)
    else:
        shifts_a = np.ascontiguousarray(shifts, np.int32).reshape(n_q, P)

    store = Svar2Store(str(svar2.path), svar2.contigs, svar2.n_samples, svar2.ploidy)

    out_data, out_offsets = reconstruct_haplotypes_from_svar2_readbound(
        store,
        contig,
        region_starts,
        orig_samples,
        vk_snp_range,
        vk_indel_range,
        dense_snp_range,
        dense_indel_range,
        region_bounds,
        shifts_a,
        np.ascontiguousarray(ref_, np.uint8),
        np.ascontiguousarray(ref_offsets, np.int64),
        np.uint8(pad_char),
        np.int64(output_length),
        parallel,
    )

    shape = (R, S, P, None)
    return cast(
        "Ragged[np.bytes_]", _Flat.from_offsets(out_data, shape, out_offsets).view("S1")
    )


def build_readbound_tracks(
    svar2: "SparseVar2",
    contig: str,
    regions,  # iterable of (start, end), length R
    tracks: "NDArray[np.float32]",  # flat per-REGION track buffer
    track_offsets: "NDArray[np.int64]",  # (R+1) offsets into tracks
    params: "NDArray[np.float64]",
    strategy_id: int,
    base_seed: int,
    shifts: "NDArray[np.int32] | None" = None,  # (R*S, P); None -> zeros
    parallel: bool = False,
) -> "Ragged[np.float32]":
    """Realign the full-cohort tracks over ``regions`` via the read-bound kernel.

    Mirrors ``SparseVar2Source.realign_tracks``'s signature/return shape exactly
    (query order region-major, sample-minor: ``q = r*S + s``), but drives
    ``SparseVar2.find_ranges`` (search-only, no dense union) + one Rust FFI call
    instead of ``overlap_batch``'s eager per-region dense union.
    """
    reg = [(int(s), int(e)) for s, e in regions]
    R = len(reg)
    S = svar2.n_samples
    P = svar2.ploidy

    d = svar2.find_ranges(
        contig, [s for s, _ in reg], [e for _, e in reg], samples=None
    )

    region_starts_r = np.asarray(d["region_starts"], np.int64)  # (R,)
    sample_cols = np.asarray(d["sample_cols"], np.int64)  # (S,)
    # vk_*_range rows are already (R, S, P) row-major == query-major (q = r*S+s,
    # row = q*P + p), so they pass through unchanged.
    vk_snp_range = np.ascontiguousarray(d["vk_snp_range"], np.int64)  # (R*S*P, 2)
    vk_indel_range = np.ascontiguousarray(d["vk_indel_range"], np.int64)
    dense_snp_range_r = np.asarray(d["dense_snp_range"], np.int64)  # (R, 2)
    dense_indel_range_r = np.asarray(d["dense_indel_range"], np.int64)  # (R, 2)

    n_q = R * S
    region_starts = np.repeat(region_starts_r, S).astype(np.uint32)  # (n_q,)
    orig_samples = np.tile(sample_cols, R)  # (n_q,)
    dense_snp_range = np.ascontiguousarray(
        np.repeat(dense_snp_range_r, S, axis=0), np.int64
    )  # (n_q, 2)
    dense_indel_range = np.ascontiguousarray(
        np.repeat(dense_indel_range_r, S, axis=0), np.int64
    )  # (n_q, 2)

    reg_arr = np.asarray(reg, np.int32).reshape(R, 2)
    region_bounds = np.ascontiguousarray(
        np.repeat(reg_arr, S, axis=0), np.int32
    )  # (n_q, 2)

    if shifts is None:
        shifts_a = np.zeros((n_q, P), dtype=np.int32)
    else:
        shifts_a = np.ascontiguousarray(shifts, np.int32).reshape(n_q, P)

    # `tracks`/`track_offsets` are per-REGION (R of them), but the kernel reads
    # `track_offsets` by `query` (= r*S+s) — expand the R track windows to R*S
    # by repeating each region's window S times (mirrors
    # `SparseVar2Source.realign_tracks`, `_svar2_source.py:108-114`).
    t = np.asarray(tracks, np.float32)
    toff = np.asarray(track_offsets, np.int64)
    tracks_rs = (
        np.concatenate([t[toff[r] : toff[r + 1]] for r in range(R) for _ in range(S)])
        if R
        else t
    )
    lengths = np.repeat(np.diff(toff), S)
    track_offsets_rs = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    store = Svar2Store(str(svar2.path), svar2.contigs, svar2.n_samples, svar2.ploidy)

    out_data, out_offsets = shift_and_realign_tracks_from_svar2_readbound(
        store,
        contig,
        region_starts,
        orig_samples,
        vk_snp_range,
        vk_indel_range,
        dense_snp_range,
        dense_indel_range,
        region_bounds,
        shifts_a,
        np.ascontiguousarray(tracks_rs, np.float32),
        track_offsets_rs,
        np.ascontiguousarray(params, np.float64),
        np.int64(strategy_id),
        np.uint64(base_seed),
        parallel,
    )

    shape = (R, S, P, None)
    return cast("Ragged[np.float32]", _Flat.from_offsets(out_data, shape, out_offsets))
