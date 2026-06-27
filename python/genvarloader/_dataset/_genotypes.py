import numpy as np
from numpy.typing import NDArray
from seqpro.rag import OFFSET_TYPE

from ..genvarloader import choose_exonic_variants as _choose_exonic_variants_rust
from ..genvarloader import get_diffs_sparse as _get_diffs_sparse_rust
from ..genvarloader import (
    reconstruct_haplotypes_from_sparse as _reconstruct_haplotypes_from_sparse_rust,
)


def _as_starts_stops(offsets: NDArray[np.integer]) -> NDArray[np.int64]:
    """Normalize 1-D (n+1,) or 2-D (2, n) offsets to a contiguous (2, n) int64
    starts/stops array. Both backends consume this single form."""
    o = np.asarray(offsets)
    if o.ndim == 1:
        return np.ascontiguousarray(np.stack([o[:-1], o[1:]]), dtype=np.int64)
    return np.ascontiguousarray(o, dtype=np.int64)


def get_diffs_sparse(
    geno_offset_idx: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    ilens: NDArray[np.integer],
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
    q_starts: NDArray[np.integer] | None = None,
    q_ends: NDArray[np.integer] | None = None,
    v_starts: NDArray[np.integer] | None = None,
) -> NDArray[np.int32]:
    """Per-(query, hap) reference-length diffs; dispatches to Rust."""
    return _get_diffs_sparse_rust(
        np.ascontiguousarray(geno_offset_idx, np.int64),
        np.ascontiguousarray(geno_v_idxs, np.int32),
        _as_starts_stops(geno_offsets),
        np.ascontiguousarray(ilens, np.int32),
        None if keep is None else np.ascontiguousarray(keep, np.bool_),
        None if keep_offsets is None else np.ascontiguousarray(keep_offsets, np.int64),
        None if q_starts is None else np.ascontiguousarray(q_starts, np.int32),
        None if q_ends is None else np.ascontiguousarray(q_ends, np.int32),
        None if v_starts is None else np.ascontiguousarray(v_starts, np.int32),
    )


def reconstruct_haplotypes_from_sparse(
    out: NDArray[np.uint8],
    out_offsets: NDArray[np.integer],
    regions: NDArray[np.integer],
    shifts: NDArray[np.integer],
    geno_offset_idx: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    alt_alleles: NDArray[np.uint8],
    alt_offsets: NDArray[np.integer],
    ref: NDArray[np.uint8],
    ref_offsets: NDArray[np.integer],
    pad_char: int,
    keep: NDArray[np.bool_] | None = None,
    keep_offsets: NDArray[np.integer] | None = None,
    annot_v_idxs: NDArray[np.integer] | None = None,
    annot_ref_pos: NDArray[np.integer] | None = None,
):
    """Reconstruct haplotypes from reference sequence and variants (dispatch wrapper).

    Dispatches to the Rust backend. Normalizes array dtypes and layouts before dispatch.
    """
    _reconstruct_haplotypes_from_sparse_rust(
        out,
        np.ascontiguousarray(out_offsets, np.int64),
        np.ascontiguousarray(regions, np.int32),
        np.ascontiguousarray(shifts, np.int32),
        np.ascontiguousarray(geno_offset_idx, np.int64),
        _as_starts_stops(geno_offsets),
        np.ascontiguousarray(geno_v_idxs, np.int32),
        np.ascontiguousarray(v_starts, np.int32),
        np.ascontiguousarray(ilens, np.int32),
        np.ascontiguousarray(alt_alleles, np.uint8),
        np.ascontiguousarray(alt_offsets, np.int64),
        np.ascontiguousarray(ref, np.uint8),
        np.ascontiguousarray(ref_offsets, np.int64),
        np.uint8(pad_char),
        None if keep is None else np.ascontiguousarray(keep, np.bool_),
        None if keep_offsets is None else np.ascontiguousarray(keep_offsets, np.int64),
        annot_v_idxs,
        annot_ref_pos,
    )


def choose_exonic_variants(
    starts: NDArray[np.integer],
    ends: NDArray[np.integer],
    geno_offset_idx: NDArray[np.integer],
    geno_v_idxs: NDArray[np.integer],
    geno_offsets: NDArray[np.integer],
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
) -> tuple[NDArray[np.bool_], NDArray[OFFSET_TYPE]]:
    """Exonic keep-mask; dispatches to Rust. keep_offsets dtype == OFFSET_TYPE."""
    keep, keep_offsets = _choose_exonic_variants_rust(
        np.ascontiguousarray(starts, np.int32),
        np.ascontiguousarray(ends, np.int32),
        np.ascontiguousarray(geno_offset_idx, np.int64),
        np.ascontiguousarray(geno_v_idxs, np.int32),
        _as_starts_stops(geno_offsets),
        np.ascontiguousarray(v_starts, np.int32),
        np.ascontiguousarray(ilens, np.int32),
    )
    return keep, keep_offsets.astype(OFFSET_TYPE, copy=False)


def reconstruct_haplotype_from_sparse(
    v_idxs,
    v_starts,
    ilens,
    shift: int,
    alt_alleles,
    alt_offsets,
    ref,
    ref_start: int,
    out,
    pad_char: int,
    keep=None,
    annot_v_idxs=None,
    annot_ref_pos=None,
):
    """Reconstruct a single haplotype from reference sequence and variants.

    Pure Python fallback (no numba). Used directly by parity/unit tests.
    Use :func:`reconstruct_haplotypes_from_sparse` (plural) to reconstruct a batch.
    """
    import numpy as np

    length = len(out)
    n_variants = len(v_idxs)
    ref_idx = ref_start
    out_idx = 0
    shifted = 0

    if ref_idx < 0:
        pad_len = -ref_idx
        shifted = min(shift, pad_len)
        pad_len -= shifted
        out[out_idx : out_idx + pad_len] = pad_char
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx : out_idx + pad_len] = -1
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx : out_idx + pad_len] = -1
        out_idx += pad_len
        ref_idx = 0

    for v in range(n_variants):
        if keep is not None and not keep[v]:
            continue

        variant = int(v_idxs[v])
        v_pos = int(v_starts[variant])
        v_diff = int(ilens[variant])
        allele = alt_alleles[int(alt_offsets[variant]) : int(alt_offsets[variant + 1])]
        v_len = len(allele)
        v_ref_end = v_pos - min(0, v_diff) + 1

        if v_pos < ref_start and v_diff < 0 and v_ref_end >= ref_start:
            ref_idx = v_ref_end
            continue

        if v_pos < ref_idx:
            continue

        if shifted < shift:
            ref_shift_dist = v_pos - ref_idx
            if shifted + ref_shift_dist + v_len < shift:
                continue
            elif shifted + ref_shift_dist >= shift:
                ref_idx += shift - shifted
                shifted = shift
            else:
                allele_start_idx = shift - shifted - ref_shift_dist
                shifted = shift
                if allele_start_idx == v_len:
                    ref_idx = v_ref_end
                    continue
                ref_idx = v_pos
                allele = allele[allele_start_idx:]
                v_len = len(allele)

        ref_len = v_pos - ref_idx
        if out_idx + ref_len >= length:
            break
        out[out_idx : out_idx + ref_len] = ref[ref_idx : ref_idx + ref_len]
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx : out_idx + ref_len] = -1
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx : out_idx + ref_len] = np.arange(
                ref_idx, ref_idx + ref_len
            )
        out_idx += ref_len

        writable_length = min(v_len, length - out_idx)
        out[out_idx : out_idx + writable_length] = allele[:writable_length]
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx : out_idx + writable_length] = variant
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx : out_idx + writable_length] = v_pos
        out_idx += writable_length

        ref_idx = v_ref_end

        if out_idx >= length:
            break

    if shifted < shift:
        ref_idx += shift - shifted
        ref_idx = min(ref_idx, len(ref))
        shifted = shift

    unfilled_length = length - out_idx
    if unfilled_length > 0:
        writable_ref = max(0, min(unfilled_length, len(ref) - ref_idx))
        out_end_idx = out_idx + writable_ref
        ref_end_idx = ref_idx + writable_ref
        out[out_idx:out_end_idx] = ref[ref_idx:ref_end_idx]
        if annot_v_idxs is not None:
            annot_v_idxs[out_idx:out_end_idx] = -1
        if annot_ref_pos is not None:
            annot_ref_pos[out_idx:out_end_idx] = np.arange(ref_idx, ref_end_idx)

        if out_end_idx < length:
            out[out_end_idx:] = pad_char
            if annot_v_idxs is not None:
                annot_v_idxs[out_end_idx:] = -1
            if annot_ref_pos is not None:
                import numpy as np

                annot_ref_pos[out_end_idx:] = np.iinfo(np.int32).max
