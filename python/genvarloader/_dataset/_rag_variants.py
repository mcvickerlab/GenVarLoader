from __future__ import annotations

import awkward as ak
import numba as nb
import numpy as np
from attrs import define
from awkward.contents import RegularArray
from genoray._svar import DOSAGE_TYPE, POS_TYPE, V_IDX_TYPE
from numpy.typing import NDArray
from seqpro._ragged import OFFSET_TYPE, Ragged
from typing_extensions import Self


@define
class RaggedVariants:
    """Typically contains ragged arrays with shape (batch, ploidy, ~variants)"""

    alts: ak.Array  # (batch, ploidy, ~variants, ~length)
    """Alternate alleles, note extra ragged dimension for length yielding a shape of
    (..., ploidy, ~variants, ~length)"""
    v_starts: Ragged[POS_TYPE]
    """0-based start positions"""
    ilens: Ragged[np.int32]
    """Indel lengths"""
    dosages: Ragged[DOSAGE_TYPE] | None
    """Dosages, potentially interpreted as CCFs depending on how the dosages were defined."""

    @property
    def shape(self) -> tuple[int, ...]:
        return self.v_starts.shape

    def reshape(self, shape: tuple[int, ...]) -> Self:
        # bpvl -> pvl -> vl
        layout = self.alts.layout.content
        for len_ in reversed(shape[1:]):
            layout = RegularArray(layout, len_)

        return type(self)(
            ak.Array(layout),
            self.v_starts.reshape(shape),
            self.ilens.reshape(shape),
            None if self.dosages is None else self.dosages.reshape(shape),
        )

    def squeeze(self, axis: int = 0) -> Self:
        return type(self)(
            ak.flatten(self.alts, axis + 1),
            self.v_starts.squeeze(axis),
            self.ilens.squeeze(axis),
            None if self.dosages is None else self.dosages.squeeze(axis),
        )

    def infer_germline_ccfs_(self, max_ccf: float = 1.0) -> Self:
        """Treat dosages as cancer cell fractions and infer germline CCFs in-place.

        Germline variants are identified by having missing CCFs i.e. they have a variant
        index but missing CCFs. Missing CCFs are inferred to be :code:`max_ccf` - sum(overlapping CCFs).

        Parameters
        ----------
        max_ccf
            Maximum CCF value.
        """
        if self.dosages is None:
            raise ValueError("Cannot infer germline CCFs without dosages.")
        _infer_germline_ccfs(
            self.dosages.data,
            self.v_starts.offsets,
            self.v_starts.data,
            self.ilens.data,
            max_ccf=max_ccf,
        )
        return self


@nb.njit(parallel=True, nogil=True, cache=True)
def _infer_germline_ccfs(
    ccfs: NDArray[DOSAGE_TYPE],
    v_offsets: NDArray[OFFSET_TYPE],
    v_starts: NDArray[POS_TYPE],
    ilens: NDArray[np.int32],
    max_ccf: float = 1.0,
):
    """Infer germline CCFs from the variant indices and variant starts. Updates CCFs in-place.

    Germline variants are identified by having missing CCFs.
    i.e. they have a variant index but missing CCFs. Germline CCFs are inferred
    to be 1 - sum(overlapping somatic CCFs).

    Parameters
    ----------
    ccfs
        Shape: (alts) raveled view of ragged cancer cell fractions.
    v_offsets
        Shape: (alts + 1) offsets into :code:`ccfs`.
    v_starts
        Shape: (alts) 0-based start positions.
    ilens
        Shape: (alts) indel lengths.
    max_ccf
        Maximum cancer cell fraction.
    """
    n_sp = len(v_offsets) - 1
    for o_idx in nb.prange(n_sp):
        o_s, o_e = v_offsets[o_idx], v_offsets[o_idx + 1]
        n_variants: int = o_e - o_s
        if n_variants == 0:
            continue

        ccf = ccfs[o_s:o_e]
        if not np.isnan(ccf).any():
            continue
        v_start = v_starts[o_s:o_e]
        ilen = ilens[o_s:o_e]

        v_end = (
            v_start - np.minimum(0, ilen) + 1
        )  # +1 for atomic variants, +shared_len for non-atomic
        v_end_sorter = np.argsort(v_end)
        v_end = v_end[v_end_sorter]

        # sorted merge by starts then ends
        # ends are marked by being negative
        starts_ends = np.empty(n_variants * 2, POS_TYPE)
        se_local_idx = np.empty(n_variants * 2, V_IDX_TYPE)
        start_idx = 0
        end_idx = 0
        for i in range(n_variants * 2):
            end = v_end[end_idx]
            if start_idx < n_variants and v_start[start_idx] < end:
                starts_ends[i] = v_start[start_idx]
                se_local_idx[i] = start_idx
                start_idx += 1
            else:
                starts_ends[i] = -end
                se_local_idx[i] = v_end_sorter[end_idx]
                end_idx += 1

        running_ccf = DOSAGE_TYPE(0)
        for i in range(n_variants * 2):
            pos: POS_TYPE = starts_ends[i]
            local_idx: V_IDX_TYPE = se_local_idx[i]
            pos_ccf: DOSAGE_TYPE = ccf[local_idx]
            is_germ = np.isnan(pos_ccf)

            if is_germ:
                if pos < 0:
                    ccf[local_idx] = max_ccf - running_ccf
                continue

            # sign of pos, with 0 being positive
            running_ccf += (2 * (pos >= 0) - 1) * pos_ccf

        np.nan_to_num(ccf, copy=False, nan=max_ccf)
