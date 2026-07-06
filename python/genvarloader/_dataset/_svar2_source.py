"""SVAR2 two-source reconstruction adapter — parity oracle only (not a live read path).

Bridges genoray ``SparseVar2.overlap_batch``'s raw two-channel dict to gvl's SVAR2 kernels
(``reconstruct_haplotypes_from_svar2`` / ``shift_and_realign_tracks_from_svar2``), decoding
``var_key ⋈ dense`` inline with no intermediate variant table. This is the *union* path
(genoray ``overlap_batch``, whole-cohort).

Live dataset dispatch is NOT wired through here. ``Dataset`` reconstruction for ``.svar2``-backed
datasets is handled by the read-bound path in ``Svar2Haps`` (``_svar2_haps.py``), which gathers off
the write-time ranges cache and calls the ``*_from_svar2_readbound`` kernels — no interval-search-tree
rebuild and no dense-union rebuild per read. This ``SparseVar2Source`` adapter is retained solely as
the byte-identical *parity oracle* the read-bound kernels are tested against (see
``tests/dataset/test_svar2_readbound_*.py``); it is not imported on any live read path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from seqpro.rag import Ragged

from .._flat import _Flat
from ..genvarloader import (
    reconstruct_haplotypes_from_svar2,
    shift_and_realign_tracks_from_svar2,
)

if TYPE_CHECKING:
    from genoray import SparseVar2
    from numpy.typing import NDArray


class SparseVar2Source:
    """Reconstruct haplotypes / realign tracks from a genoray ``SparseVar2`` via the two-source path."""

    def __init__(self, svar2: "SparseVar2") -> None:
        self.svar2 = svar2

    def _query(self, contig, regions):
        d = self.svar2.overlap_batch(contig, [(int(s), int(e)) for s, e in regions])
        R = int(d["n_regions"])
        S = int(d["n_samples"])
        P = int(d["ploidy"])
        reg = np.asarray(regions, dtype=np.int32).reshape(R, 2)
        # (R*S, 3): contig_idx=0, start, end — repeat each query region S times.
        reg_rs = np.repeat(reg, S, axis=0)  # (R*S, 2)
        regions_gvl = np.zeros((R * S, 3), dtype=np.int32)
        regions_gvl[:, 1:] = reg_rs
        dense_range_gvl = np.ascontiguousarray(
            np.repeat(np.asarray(d["dense_range"], np.int32), S, axis=0), np.int32
        )  # (R*S, 2)
        return d, R, S, P, regions_gvl, dense_range_gvl

    def reconstruct(
        self,
        contig: str,
        regions,  # iterable of (start, end), length R
        ref_: "NDArray[np.uint8]",  # the contig reference bytes
        ref_offsets: "NDArray[np.int64]",  # e.g. np.array([0, len(ref_)])
        pad_char: int,
        shifts: "NDArray[np.int32] | None" = None,  # (R*S, P); None -> zeros
        output_length: int = -1,
        parallel: bool = False,
    ) -> "Ragged[np.bytes_]":
        d, R, S, P, regions_gvl, dense_range_gvl = self._query(contig, regions)
        n_q = R * S
        if shifts is None:
            shifts_a = np.zeros((n_q, P), dtype=np.int32)
        else:
            shifts_a = np.ascontiguousarray(shifts, np.int32).reshape(n_q, P)
        out_data, out_offsets = reconstruct_haplotypes_from_svar2(
            np.ascontiguousarray(regions_gvl, np.int32),
            shifts_a,
            np.ascontiguousarray(d["vk_pos"], np.int32),
            np.ascontiguousarray(d["vk_key"], np.int32),
            np.ascontiguousarray(d["vk_off"], np.int64),
            np.ascontiguousarray(d["dense_pos"], np.int32),
            np.ascontiguousarray(d["dense_key"], np.int32),
            dense_range_gvl,
            np.ascontiguousarray(d["dense_present"], np.uint8),
            np.ascontiguousarray(d["dense_present_off"], np.int64),
            np.ascontiguousarray(d["lut_bytes"], np.uint8),
            np.ascontiguousarray(d["lut_off"], np.int64),
            np.ascontiguousarray(ref_, np.uint8),
            np.ascontiguousarray(ref_offsets, np.int64),
            np.uint8(pad_char),
            np.int64(output_length),
            parallel,
        )
        shape = (R, S, P, None)
        return cast(
            "Ragged[np.bytes_]",
            _Flat.from_offsets(out_data, shape, out_offsets).view("S1"),
        )

    def realign_tracks(
        self,
        contig: str,
        regions,
        tracks: "NDArray[np.float32]",  # flat per-query track buffer
        track_offsets: "NDArray[np.int64]",  # (R+1) offsets into tracks
        params: "NDArray[np.float64]",
        strategy_id: int,
        base_seed: int,
        shifts: "NDArray[np.int32] | None" = None,
        parallel: bool = False,
    ) -> "Ragged[np.float32]":
        d, R, S, P, regions_gvl, dense_range_gvl = self._query(contig, regions)
        n_q = R * S
        if shifts is None:
            shifts_a = np.zeros((n_q, P), dtype=np.int32)
        else:
            shifts_a = np.ascontiguousarray(shifts, np.int32).reshape(n_q, P)
        # tracks are per query REGION (R of them); the driver reads track_offsets by `query`
        # (= r*S+s), so expand the R track windows to R*S by repeating each S times.
        t = np.asarray(tracks, np.float32)
        toff = np.asarray(track_offsets, np.int64)
        tracks_rs = (
            np.concatenate(
                [t[toff[r] : toff[r + 1]] for r in range(R) for _ in range(S)]
            )
            if R
            else t
        )
        lengths = np.repeat(np.diff(toff), S)
        track_offsets_rs = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        out_data, out_offsets = shift_and_realign_tracks_from_svar2(
            np.ascontiguousarray(regions_gvl, np.int32),
            shifts_a,
            np.ascontiguousarray(d["vk_pos"], np.int32),
            np.ascontiguousarray(d["vk_key"], np.int32),
            np.ascontiguousarray(d["vk_off"], np.int64),
            np.ascontiguousarray(d["dense_pos"], np.int32),
            np.ascontiguousarray(d["dense_key"], np.int32),
            dense_range_gvl,
            np.ascontiguousarray(d["dense_present"], np.uint8),
            np.ascontiguousarray(d["dense_present_off"], np.int64),
            np.ascontiguousarray(d["lut_bytes"], np.uint8),
            np.ascontiguousarray(d["lut_off"], np.int64),
            np.ascontiguousarray(tracks_rs, np.float32),
            track_offsets_rs,
            np.ascontiguousarray(params, np.float64),
            np.int64(strategy_id),
            np.uint64(base_seed),
            parallel,
        )
        shape = (R, S, P, None)
        return cast(
            "Ragged[np.float32]", _Flat.from_offsets(out_data, shape, out_offsets)
        )
