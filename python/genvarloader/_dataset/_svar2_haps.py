"""SVAR2-backed haplotype/variant reconstructor (dataset read dispatch).

``Svar2Haps`` is a *separate* reconstructor from the SVAR1 :class:`Haps` — the
SVAR1 path is left byte-unchanged. It subclasses :class:`Haps` only so the many
``isinstance(_, Haps)`` / ``case Haps()`` checks throughout the dataset
machinery keep working; every read method is overridden.

For a query block of ``n_q`` rows, each row ``q = (region r_q, sample slot si_q)``
with post-jitter bounds ``[start_q, end_q)``, the cache (written by
``_write._write_from_svar2`` under ``genotypes/svar2_ranges/``) is sliced by
fancy-indexing — NO per-read interval search, NO dense-union rebuild — and fed to
the read-bound Rust kernels (``reconstruct_haplotypes_from_svar2_readbound`` /
``decode_variants_from_svar2_readbound`` / ``hap_diffs_from_svar2_readbound``).

The FFI-input shaping + output wrapping mirror
``_svar2_store_py.build_readbound_*`` exactly; the only difference is the source
of the per-query ranges (this module slices the on-disk cache for the specific
``(r_q, si_q)`` block, whereas the helpers call ``SparseVar2.find_ranges`` over
the full cohort).

Out of scope for this plan (guarded with ``NotImplementedError``): spliced
output, ``filter == "exonic"`` (keep mask), ``min_af``/``max_af``, annotated
haps, in-kernel reverse-complement, and ``unphased_union``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from genoray._types import POS_TYPE, V_IDX_TYPE
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._flat import _Flat
from .._ragged import RaggedAnnotatedHaps
from .._threads import should_parallelize
from .._utils import lengths_to_offsets
from .._variants._records import RaggedAlleles
from ..genvarloader import (
    Svar2Store,
    decode_variants_from_svar2_readbound,
    hap_diffs_from_svar2_readbound,
    reconstruct_haplotypes_from_svar2_readbound,
    shift_and_realign_tracks_from_svar2_readbound,
)
from ._flat_variants import _FlatVariantWindows
from ._intervals import intervals_to_tracks
from ._haps import _H, Haps, _Variants
from ._rag_variants import RaggedVariants
from ._reference import Reference
from ._svar2_link import Svar2Link, _resolve_svar2, _verify_svar2_fingerprint

if TYPE_CHECKING:
    from ._splice import SplicePlan


@dataclass(slots=True)
class _Svar2Cache:
    """The six memmapped ``svar2_ranges/`` arrays (all int64), sliced per query.

    ``vk_*_range`` are ``(R, S, P, 2)`` (per region/sample/ploid byte windows into
    the store's var_key tables); ``dense_*_range`` are ``(R, 2)`` (per-region,
    sample-independent); ``region_starts`` is ``(R,)`` (write-time starts; kept for
    parity/debug, NOT fed to the FFI — the read path uses post-jitter starts);
    ``sample_cols`` is ``(S,)`` (selected slot -> original store sample index).
    """

    vk_snp_range: NDArray[np.int64]
    vk_indel_range: NDArray[np.int64]
    dense_snp_range: NDArray[np.int64]
    dense_indel_range: NDArray[np.int64]
    region_starts: NDArray[np.int64]
    sample_cols: NDArray[np.int64]


def _ragged_arange_src(
    offsets: NDArray[np.integer], perm: NDArray[np.integer]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Source-row index + new offsets for a 1-level ragged reorder by ``perm``.

    ``new_data == data[src]``; ``src`` and ``new_off`` depend only on
    ``(offsets, perm)`` — so callers reordering several parallel data arrays by
    the same key compute this ONCE and index each array.
    """
    offsets = np.asarray(offsets, np.int64)
    lens = np.diff(offsets)
    new_lens = lens[perm]
    new_off = lengths_to_offsets(new_lens, np.int64)
    n = int(new_off[-1])
    if n == 0:
        return np.zeros(0, np.int64), new_off
    within = np.arange(n, dtype=np.int64) - np.repeat(new_off[:-1], new_lens)
    src = np.repeat(offsets[perm], new_lens) + within
    return src, new_off


def _ragged_arange_gather(
    data: NDArray, offsets: NDArray[np.integer], perm: NDArray[np.integer]
) -> tuple[NDArray, NDArray[np.int64]]:
    """Reorder the rows of a 1-level ragged array ``(data, offsets)`` by ``perm``."""
    src, new_off = _ragged_arange_src(offsets, perm)
    if src.size == 0:
        return data[:0].copy(), new_off
    return data[src], new_off


def _ragged_arange_gather_2level(
    data: NDArray,
    var_off: NDArray[np.integer],
    str_off: NDArray[np.integer],
    perm: NDArray[np.integer],
) -> tuple[NDArray, NDArray[np.int64], NDArray[np.int64]]:
    """Reorder the rows of a 2-level ragged (opaque-string) array by ``perm``.

    ``var_off`` (len ``n_rows + 1``) bounds variants per row; ``str_off`` (len
    ``n_variants + 1``) bounds bytes per variant. Returns
    ``(new_data, new_var_off, new_str_off)``. Fully vectorized.
    """
    var_off = np.asarray(var_off, np.int64)
    str_off = np.asarray(str_off, np.int64)
    var_lens = np.diff(var_off)
    new_var_lens = var_lens[perm]
    new_var_off = lengths_to_offsets(new_var_lens, np.int64)
    total_vars = int(new_var_off[-1])
    if total_vars == 0:
        return data[:0].copy(), new_var_off, np.zeros(1, np.int64)
    within_var = np.arange(total_vars, dtype=np.int64) - np.repeat(
        new_var_off[:-1], new_var_lens
    )
    old_var_idx = np.repeat(var_off[perm], new_var_lens) + within_var
    var_byte_len = np.diff(str_off)
    new_byte_len = var_byte_len[old_var_idx]
    new_str_off = lengths_to_offsets(new_byte_len, np.int64)
    nbytes = int(new_str_off[-1])
    if nbytes == 0:
        return data[:0].copy(), new_var_off, new_str_off
    within_b = np.arange(nbytes, dtype=np.int64) - np.repeat(
        new_str_off[:-1], new_byte_len
    )
    src = np.repeat(str_off[old_var_idx], new_byte_len) + within_b
    return data[src], new_var_off, new_str_off


@dataclass(slots=True)
class Svar2Haps(Haps[_H]):
    """Read-bound SVAR2 reconstructor. See module docstring."""

    # New fields must default (they follow base Haps' defaulted fields).
    store: "Svar2Store | None" = None
    cache: "_Svar2Cache | None" = None
    store_contigs: list[str] = field(default_factory=list)
    """The .svar2 store's contig names (used to open the store's ContigReaders)."""
    ds_contigs: list[str] = field(default_factory=list)
    """The dataset's contig names (``regions[:, 0]`` indexes into this)."""
    max_jitter: int = 0
    """The dataset's write-time max_jitter. When > 0 the cache's per-query ranges
    were computed over a max_jitter-padded window, which over-includes variants past
    the (unpadded) read window in variants mode (the decode kernel has no right-clip);
    guarded below."""

    def __post_init__(self):
        # Deliberately does NOT call Haps.__post_init__ (that reads an SVAR1
        # variants table / AF cache which svar2 has no analogue for). Set only
        # the init=False fields the base machinery reads.
        self.n_variants = self.genotypes.lengths
        self.available_var_fields = ["alt", "ilen", "start"]

    # ---- construction ----

    @classmethod
    def from_path(  # type: ignore[override]  # separate svar2 signature; base returns Haps[RaggedVariants]
        cls,
        path: Path,
        reference: Reference | None,
        samples: list[str],
        ploidy: int,
        svar2_link: Svar2Link | None,
        svar2_override: Path | str | None,
        contigs: list[str],
        kind: type = RaggedVariants,
        min_af: float | None = None,
        max_af: float | None = None,
        max_jitter: int = 0,
    ) -> "Svar2Haps":
        ranges_dir = path / "genotypes" / "svar2_ranges"
        with open(ranges_dir / "svar2_meta.json") as f:
            meta = json.load(f)

        def _mm(name: str, shape: list[int]) -> NDArray[np.int64]:
            return np.memmap(
                ranges_dir / name, dtype=np.int64, mode="r", shape=tuple(shape)
            )

        R = int(meta["region_starts"]["shape"][0])
        S = int(meta["vk_snp_range"]["shape"][1])
        P = int(meta["ploidy"])
        if P != ploidy:
            raise ValueError(f"svar2 cache ploidy ({P}) != dataset ploidy ({ploidy}).")

        cache = _Svar2Cache(
            vk_snp_range=_mm("vk_snp_range.npy", meta["vk_snp_range"]["shape"]),
            vk_indel_range=_mm("vk_indel_range.npy", meta["vk_indel_range"]["shape"]),
            dense_snp_range=_mm(
                "dense_snp_range.npy", meta["dense_snp_range"]["shape"]
            ),
            dense_indel_range=_mm(
                "dense_indel_range.npy", meta["dense_indel_range"]["shape"]
            ),
            region_starts=_mm("region_starts.npy", meta["region_starts"]["shape"]),
            sample_cols=np.load(ranges_dir / "sample_cols.npy"),
        )

        svar2_path = _resolve_svar2(path, svar2_link, svar2_override)
        _verify_svar2_fingerprint(svar2_path, svar2_link)

        # Open the query-only store. n_samples must be the store's FULL sample
        # count (orig_samples / sample_cols index into it), not len(samples).
        from genoray import SparseVar2

        sv = SparseVar2(str(svar2_path))
        store = Svar2Store(str(svar2_path), sv.contigs, sv.n_samples, sv.ploidy)

        # Minimal base-Haps fields. genotypes carries only the (R, S, P, None)
        # shape (so ploidy = shape[-2] and n_variants.shape are available); its
        # data is empty (svar2 has no per-region sparse genotype store).
        empty_geno = Ragged.from_offsets(
            np.empty(0, V_IDX_TYPE),
            (R, S, P, None),
            np.zeros(R * S * P + 1, np.int64),
        )
        empty_alt = RaggedAlleles.from_offsets(
            np.empty(0, np.uint8).view("S1"), (0, None), np.zeros(1, np.int64)
        )
        dummy_variants = _Variants(
            path=svar2_path,
            start=np.empty(0, POS_TYPE),
            ilen=np.empty(0, np.int32),
            ref=None,
            alt=empty_alt,
            info={},
        )

        return cls(
            path=path,
            reference=reference,
            variants=dummy_variants,
            genotypes=empty_geno,
            dosages=None,
            kind=cast("type[_H]", kind),
            filter=None,
            min_af=min_af,
            max_af=max_af,
            store=store,
            cache=cache,
            store_contigs=list(sv.contigs),
            ds_contigs=list(contigs),
            max_jitter=max_jitter,
        )

    # ---- reconstructor entry ----

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: "SplicePlan | None" = None,
        flat: bool = False,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> _H:
        self._guard_unsupported(splice_plan)

        if issubclass(self.kind, (RaggedVariants, _FlatVariantWindows)):
            if issubclass(self.kind, _FlatVariantWindows):
                raise NotImplementedError(
                    "svar2 datasets do not support with_seqs('variant-windows') yet."
                )
            # ``decode_variants_from_svar2_readbound`` has NO right-clip: it emits
            # every gathered variant that passes the left-clip. The cache's per-query
            # ranges cover the read window ONLY when it equals the write window --
            # i.e. no jitter anywhere. max_jitter>0 pads the cache at WRITE (so even
            # a jitter=0 read over-includes variants in (end, end+max_jitter]); a
            # jitter>0 read narrows/slides the window at READ. Guard on BOTH.
            # (Haplotypes/tracks are unaffected: their kernel right-clips to q_end.)
            if self.max_jitter > 0 or jitter > 0:
                raise NotImplementedError(
                    "variants output for svar2 datasets written with max_jitter>0"
                    f" (here max_jitter={self.max_jitter}) or read with jitter>0"
                    f" (here jitter={jitter}) is not yet supported: the read-bound"
                    " variants decode does not right-clip to the post-jitter window."
                    " Use max_jitter=0 at write and jitter=0 at read, or use"
                    " haplotypes/tracks modes."
                )
            # RaggedVariants: RC is applied by the caller (_getitem_unspliced),
            # so to_rc is intentionally ignored here (mirrors SVAR1 Haps).
            return cast(_H, self._reconstruct_variants(idx, regions))

        if issubclass(self.kind, RaggedAnnotatedHaps):
            raise NotImplementedError(
                "svar2 datasets do not support with_seqs('annotated') yet."
            )

        # Haplotypes: RC would need to be folded in-kernel; the read-bound haps
        # kernel has no to_rc param, so any real RC is unsupported here.
        if to_rc is not None and bool(np.asarray(to_rc).any()):
            raise NotImplementedError(
                "In-kernel reverse-complement is not supported for svar2 haplotypes."
            )

        haps, *_ = self.get_haps_and_shifts(
            idx=idx,
            regions=regions,
            output_length=output_length,
            rng=rng,
            deterministic=deterministic,
            splice_plan=splice_plan,
            to_rc=to_rc,
            need_hap_lengths=False,
        )
        return cast(_H, haps)

    def get_haps_and_shifts(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        output_length: Literal["ragged", "variable"] | int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: "SplicePlan | None" = None,
        to_rc: "NDArray[np.bool_] | None" = None,
        need_hap_lengths: bool = True,
    ) -> tuple[
        Ragged[np.bytes_],
        NDArray[np.intp],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.bool_] | None,
        NDArray[np.int64] | None,
    ]:
        """Reconstruct haplotypes + return the SVAR1-shaped 7-tuple.

        The tracks follow-up (7c) reuses this for the shared shifts/diffs/
        hap_lengths; ``geno_offset_idx`` is a placeholder for svar2 (the cache is
        re-sliced from ``idx`` there), and ``keep``/``keep_offsets`` are None.
        """
        self._guard_unsupported(splice_plan)
        regions = np.asarray(regions, np.int32)
        P = int(self.genotypes.shape[-2])
        b = len(idx)
        R_all, S_all = int(self.genotypes.shape[0]), int(self.genotypes.shape[1])
        r_q, si_q = np.unravel_index(np.asarray(idx), (R_all, S_all))
        contig_ids = regions[:, 0].astype(np.int64)
        lengths = (regions[:, 2] - regions[:, 1]).astype(np.int64)

        groups = self._contig_groups(contig_ids)

        # diffs are needed pre-reconstruct ONLY to (a) bound randomized jitter
        # shifts, or (b) return hap_lengths/diffs to a caller that uses them
        # (the tracks path). A deterministic/ragged haplotypes read needs
        # neither: reconstruct sizes itself internally. Avoid the redundant
        # gather+split+diffs in that (common warm-read) case.
        randomized = not (deterministic or isinstance(output_length, str))
        need_diffs = randomized or need_hap_lengths

        # --- diffs (per contig group, stitched back to (b, P) query order) ---
        if need_diffs:
            diffs = np.empty((b, P), np.int32)
            for ci, qsel in groups:
                gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)
                d = hap_diffs_from_svar2_readbound(
                    self.store,
                    self.ds_contigs[ci],
                    gi[0],
                    gi[1],
                    gi[2],
                    gi[3],
                    gi[4],
                    gi[5],
                    gi[6],
                    P,
                )
                diffs[qsel] = np.asarray(d, np.int32).reshape(len(qsel), P)

            hap_lengths = (lengths[:, None] + diffs).astype(np.int32)
        else:
            diffs = np.zeros(
                (b, P), np.int32
            )  # diffs discarded by __call__ (only caller reaching this branch)
            # hap_lengths below still feeds g_total -> should_parallelize (a perf
            # heuristic only; never affects output bytes), so the ref-length
            # placeholder is byte-identical-safe.
            hap_lengths = np.broadcast_to(
                lengths[:, None].astype(np.int32), (b, P)
            ).copy()

        # --- shifts (single rng draw; mirrors Haps._prepare_request) ---
        if randomized:
            max_shift = diffs.clip(min=0)
            max_shift = max_shift + (lengths - output_length).clip(min=0)[:, None]
            shifts = rng.integers(0, max_shift + 1, dtype=np.int32)
        else:
            shifts = np.zeros((b, P), np.int32)

        ffi_out_len = (
            np.int64(-1) if isinstance(output_length, str) else np.int64(output_length)
        )

        # --- reconstruct (per contig group), collect in grouped query order ---
        cat_data: list[NDArray[np.uint8]] = []
        cat_hap_lens: list[NDArray[np.int64]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        for ci, qsel in groups:
            gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)
            ref_, ref_offsets = self._ref_for_contig(ci)
            g_shifts = np.ascontiguousarray(shifts[qsel], np.int32)
            if isinstance(output_length, int):
                g_total = int(output_length) * len(qsel) * P
            else:
                g_total = int(hap_lengths[qsel].sum())
            g_data, g_off = reconstruct_haplotypes_from_svar2_readbound(
                self.store,
                self.ds_contigs[ci],
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                g_shifts,
                ref_,
                ref_offsets,
                np.uint8(self.reference.pad_char),  # type: ignore[union-attr]  # reference guaranteed for haplotypes
                ffi_out_len,
                should_parallelize(g_total),
            )
            cat_data.append(np.asarray(g_data, np.uint8))
            cat_hap_lens.append(np.diff(np.asarray(g_off, np.int64)))
            cat_query_order.append(qsel)

        out = self._assemble_haps(cat_data, cat_hap_lens, cat_query_order, b, P)

        geno_offset_idx = np.repeat(
            np.asarray(idx, np.intp)[:, None], P, axis=1
        )  # svar2 placeholder; 7c re-slices the cache from idx.
        return out, geno_offset_idx, shifts, diffs, hap_lengths, None, None

    # ---- tracks (7c) ----

    def realign_track_block(
        self,
        idx: NDArray[np.integer],
        o_idx: NDArray[np.integer],
        regions: NDArray[np.integer],
        shifts: NDArray[np.int32],
        track_lengths: NDArray[np.integer],
        intervals,
        params: NDArray[np.float64],
        strategy_id: int,
        base_seed: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Haplotype-realign ONE track for a query block, returning a flat f32
        buffer + offsets in global ``(b, P)`` C-order (row = ``q * P + p``).

        The two-step SVAR2 track path (there is no fused interval→realign kernel):

        1. **Materialize** the per-query reference-space track window from
           ``intervals`` via the standalone :func:`intervals_to_tracks` FFI. Each
           window starts at ``regions[q, 1]`` and spans ``track_lengths[q]`` ref
           bases (= region length + max deletion across ploidy), exactly the
           ``tracks``/``track_offsets`` input the realign kernel expects (one
           window per query; all P haps of a query share it).
        2. **Realign** to haplotype coordinates via
           :func:`shift_and_realign_tracks_from_svar2_readbound`, cache-sliced per
           contig group EXACTLY like :meth:`get_haps_and_shifts` slices for haps.

        ``o_idx`` selects the interval row per query (``idx`` for SAMPLE tracks,
        ``r_idx`` otherwise). Per-hap output length is ``ref_len + diff`` (the
        kernel's native sizing), so the stitched offsets equal the haps'
        ``hap_lengths`` in the same order.
        """
        assert self.store is not None
        regions = np.asarray(regions, np.int32)
        P = int(self.genotypes.shape[-2])
        b = len(idx)
        R_all, S_all = int(self.genotypes.shape[0]), int(self.genotypes.shape[1])
        r_q, si_q = np.unravel_index(np.asarray(idx), (R_all, S_all))
        contig_ids = regions[:, 0].astype(np.int64)
        groups = self._contig_groups(contig_ids)

        params_c = np.ascontiguousarray(params, np.float64)
        o_idx = np.asarray(o_idx)
        track_lengths = np.asarray(track_lengths, np.int64)

        cat_data: list[NDArray[np.float32]] = []
        cat_lens: list[NDArray[np.int64]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        for ci, qsel in groups:
            gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)

            # (1) materialize ref-space track windows for this group's queries.
            tl_g = track_lengths[qsel]
            track_ofsts_g = lengths_to_offsets(tl_g, np.int64)
            tracks_buf = np.empty(int(track_ofsts_g[-1]), np.float32)
            intervals_to_tracks(
                offset_idxs=o_idx[qsel],
                starts=regions[qsel, 1],
                itv_starts=intervals.starts.data,
                itv_ends=intervals.ends.data,
                itv_values=intervals.values.data,
                itv_offsets=intervals.starts.offsets,
                out=tracks_buf,
                out_offsets=track_ofsts_g,
            )

            # (2) realign to haplotype coordinates (cache-sliced, per contig).
            g_shifts = np.ascontiguousarray(shifts[qsel], np.int32)
            g_total = int(track_ofsts_g[-1])
            out_data, out_off = shift_and_realign_tracks_from_svar2_readbound(
                self.store,
                self.ds_contigs[ci],
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                g_shifts,
                np.ascontiguousarray(tracks_buf, np.float32),
                track_ofsts_g,
                params_c,
                np.int64(strategy_id),
                np.uint64(base_seed),
                should_parallelize(g_total * 4),
            )
            cat_data.append(np.asarray(out_data, np.float32))
            cat_lens.append(np.diff(np.asarray(out_off, np.int64)))
            cat_query_order.append(qsel)

        data = np.concatenate(cat_data) if cat_data else np.zeros(0, np.float32)
        lens = np.concatenate(cat_lens) if cat_lens else np.zeros(0, np.int64)
        grouped_off = lengths_to_offsets(lens, np.int64)
        perm = self._inverse_row_perm(cat_query_order, b, P)
        return _ragged_arange_gather(data, grouped_off, perm)

    # ---- variants ----

    def _reconstruct_variants(
        self, idx: NDArray[np.integer], regions: NDArray[np.integer]
    ) -> RaggedVariants:
        regions = np.asarray(regions, np.int32)
        P = int(self.genotypes.shape[-2])
        b = len(idx)
        R_all, S_all = int(self.genotypes.shape[0]), int(self.genotypes.shape[1])
        r_q, si_q = np.unravel_index(np.asarray(idx), (R_all, S_all))
        contig_ids = regions[:, 0].astype(np.int64)
        groups = self._contig_groups(contig_ids)

        cat_var_lens: list[NDArray[np.int64]] = []
        cat_pos: list[NDArray[np.int32]] = []
        cat_ilen: list[NDArray[np.int32]] = []
        cat_alt: list[NDArray[np.uint8]] = []
        cat_var_bytelen: list[NDArray[np.int64]] = []
        cat_query_order: list[NDArray[np.intp]] = []
        for ci, qsel in groups:
            gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)
            pos, ilen, alt_bytes, str_off, var_off = (
                decode_variants_from_svar2_readbound(
                    self.store,
                    self.ds_contigs[ci],
                    gi[0],
                    gi[1],
                    gi[2],
                    gi[3],
                    gi[4],
                    gi[5],
                    P,
                )
            )
            var_off = np.asarray(var_off, np.int64)
            str_off = np.asarray(str_off, np.int64)
            cat_var_lens.append(np.diff(var_off))
            cat_pos.append(np.asarray(pos, np.int32))
            cat_ilen.append(np.asarray(ilen, np.int32))
            cat_alt.append(np.asarray(alt_bytes, np.uint8))
            cat_var_bytelen.append(np.diff(str_off))
            cat_query_order.append(qsel)

        # Concatenate grouped outputs, then permute hap-rows back to global order.
        var_lens = (
            np.concatenate(cat_var_lens) if cat_var_lens else np.zeros(0, np.int64)
        )
        grouped_var_off = lengths_to_offsets(var_lens, np.int64)
        pos_c = np.concatenate(cat_pos) if cat_pos else np.zeros(0, np.int32)
        ilen_c = np.concatenate(cat_ilen) if cat_ilen else np.zeros(0, np.int32)
        alt_c = np.concatenate(cat_alt) if cat_alt else np.zeros(0, np.uint8)
        var_bytelen = (
            np.concatenate(cat_var_bytelen)
            if cat_var_bytelen
            else np.zeros(0, np.int64)
        )
        grouped_str_off = lengths_to_offsets(var_bytelen, np.int64)

        perm = self._inverse_row_perm(cat_query_order, b, P)

        src, var_off_g = _ragged_arange_src(grouped_var_off, perm)
        if src.size == 0:
            pos_g = pos_c[:0].copy()
            ilen_g = ilen_c[:0].copy()
        else:
            pos_g = pos_c[src]
            ilen_g = ilen_c[src]
        alt_g, alt_var_off_g, alt_str_off_g = _ragged_arange_gather_2level(
            alt_c, grouped_var_off, grouped_str_off, perm
        )

        shape = (b, P, None)
        pos_r = Ragged.from_offsets(pos_g, shape, var_off_g)
        ilen_r = Ragged.from_offsets(ilen_g, shape, var_off_g)
        alt_r = Ragged.from_offsets(
            alt_g.view("S1"), shape, alt_var_off_g, str_offsets=alt_str_off_g
        )
        return RaggedVariants(alt=alt_r, start=pos_r, ilen=ilen_r)

    # ---- helpers ----

    def _guard_unsupported(self, splice_plan: "SplicePlan | None") -> None:
        if splice_plan is not None:
            raise NotImplementedError(
                "Spliced output is not supported for svar2 datasets yet."
            )
        if self.filter == "exonic":
            raise NotImplementedError(
                "var_filter='exonic' (keep mask) is not supported for svar2 yet."
            )
        if self.min_af is not None or self.max_af is not None:
            raise NotImplementedError(
                "min_af/max_af filtering is not supported for svar2 datasets yet."
            )
        if self.unphased_union:
            raise NotImplementedError(
                "unphased_union is not supported for svar2 datasets yet."
            )

    def _contig_groups(
        self, contig_ids: NDArray[np.int64]
    ) -> list[tuple[int, NDArray[np.intp]]]:
        """Group query positions by contig id (store readers are per-contig).

        Preserves original order within each contig group.
        """
        groups: list[tuple[int, NDArray[np.intp]]] = []
        for ci in np.unique(contig_ids):
            qsel = np.nonzero(contig_ids == ci)[0].astype(np.intp)
            groups.append((int(ci), qsel))
        return groups

    def _gather_inputs(
        self,
        r_q: NDArray[np.integer],
        si_q: NDArray[np.integer],
        regions_grp: NDArray[np.int32],
        P: int,
    ) -> tuple[
        NDArray[np.uint32],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.int32],
    ]:
        """Cache-slice a per-contig query block into the read-bound FFI inputs.

        Fancy-indexes the memmapped cache (sub-linear; no per-read search). The
        vk_* rows come out ``(n, P, 2)`` -> reshaped ``(n*P, 2)`` in row = q*P+p
        order, which is exactly what the kernel expects.
        """
        assert self.cache is not None
        c = self.cache
        region_starts = np.ascontiguousarray(regions_grp[:, 1], np.uint32)
        orig_samples = np.ascontiguousarray(c.sample_cols[si_q], np.int64)
        vk_snp = np.ascontiguousarray(
            np.asarray(c.vk_snp_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        vk_indel = np.ascontiguousarray(
            np.asarray(c.vk_indel_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        dense_snp = np.ascontiguousarray(np.asarray(c.dense_snp_range[r_q]), np.int64)
        dense_indel = np.ascontiguousarray(
            np.asarray(c.dense_indel_range[r_q]), np.int64
        )
        region_bounds = np.ascontiguousarray(regions_grp[:, 1:3], np.int32)
        return (
            region_starts,
            orig_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            region_bounds,
        )

    def _ref_for_contig(
        self, contig_idx: int
    ) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
        """The single-contig reference slice + ``[0, len]`` offsets the kernel wants.

        ``reference.offsets`` is built in ``ds_contigs`` order (Reference.from_path
        was called with the dataset's contigs), so ``contig_idx`` indexes it
        directly.
        """
        ref = self.reference
        assert ref is not None
        o_s = int(ref.offsets[contig_idx])
        o_e = int(ref.offsets[contig_idx + 1])
        ref_ = np.ascontiguousarray(ref.reference[o_s:o_e], np.uint8)
        ref_offsets = np.array([0, o_e - o_s], np.int64)
        return ref_, ref_offsets

    @staticmethod
    def _inverse_row_perm(
        cat_query_order: list[NDArray[np.intp]], b: int, P: int
    ) -> NDArray[np.intp]:
        """Permutation mapping grouped hap-row order back to global (b, P) order.

        ``final_row[k] == grouped_row[perm[k]]``.
        """
        if cat_query_order:
            grouped_queries = np.concatenate(cat_query_order)
        else:
            grouped_queries = np.zeros(0, np.intp)
        grouped_rows = (
            grouped_queries[:, None] * P + np.arange(P, dtype=np.intp)
        ).ravel()
        perm = np.empty(b * P, np.intp)
        perm[grouped_rows] = np.arange(b * P, dtype=np.intp)
        return perm

    def _assemble_haps(
        self,
        cat_data: list[NDArray[np.uint8]],
        cat_hap_lens: list[NDArray[np.int64]],
        cat_query_order: list[NDArray[np.intp]],
        b: int,
        P: int,
    ) -> Ragged[np.bytes_]:
        data = np.concatenate(cat_data) if cat_data else np.zeros(0, np.uint8)
        hap_lens = (
            np.concatenate(cat_hap_lens) if cat_hap_lens else np.zeros(0, np.int64)
        )
        grouped_off = lengths_to_offsets(hap_lens, np.int64)
        perm = self._inverse_row_perm(cat_query_order, b, P)
        out_data, out_off = _ragged_arange_gather(data, grouped_off, perm)
        return cast(
            "Ragged[np.bytes_]",
            _Flat.from_offsets(out_data, (b, P, None), out_off).view("S1"),
        )
