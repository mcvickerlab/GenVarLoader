from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, TypedDict, cast

import awkward as ak
import numba as nb
import numpy as np
import seqpro as sp
from attrs import define
from awkward.contents import (
    Content,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from awkward.index import Index
from genoray._svar import DOSAGE_TYPE, POS_TYPE, V_IDX_TYPE
from numpy.typing import NDArray
from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets
from typing_extensions import Self

from .._torch import TORCH_AVAILABLE, requires_torch

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    from torch.nested import nested_tensor_from_jagged as nt_jag


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
    def shape(self) -> tuple[int | None, ...]:
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

    def to_packed(self) -> Self:
        """Apply :func:`ak.to_packed` to all arrays."""
        alts = ak.to_packed(self.alts)
        v_starts = ak.to_packed(self.v_starts)
        ilens = ak.to_packed(self.ilens)
        dosages = None if self.dosages is None else ak.to_packed(self.dosages)

        return type(self)(alts, v_starts, ilens, dosages)

    def rc_(self, to_rc: NDArray[np.bool_] | None = None) -> Self:
        """Reverse complement the alternative alleles. This is an in-place operation if the data is already packed.

        Parameters
        ----------
        to_rc
            A boolean mask of the same shape as the variant dimension. If :code:`True`, the alternative allele will be reverse complemented.
            If :code:`None`, will reverse complement all alternative alleles.

        Returns
        -------
            The RaggedVariants object with the alternative alleles reverse complemented.
        """
        ragv = self.to_packed()

        alts = ragv.alts.layout
        while not isinstance(alts.content, NumpyArray):
            alts = alts.content
        alts = ak.Array(alts)

        if to_rc is None:
            to_rc = np.ones(ragv.shape[:-2], np.bool_)  # type: ignore

        # (batch) -> (batch * ploidy * n_variants)
        # batch * ploidy * n_variants = n_alts
        _to_rc, _ = ak.broadcast_arrays(to_rc, ragv.ilens)
        _to_rc = _to_rc.layout
        while not isinstance(_to_rc, NumpyArray):
            _to_rc = _to_rc.content
        _to_rc = cast(NDArray[np.bool_], _to_rc.data)  # type: ignore

        rc_helper(alts, _to_rc)

        return ragv

    @requires_torch
    def to_nested_tensor_batch(
        self,
        device: str | torch.device = "cpu",
        tokenizer: Literal["seqpro"]
        | Callable[[NDArray[np.bytes_]], NDArray[np.integer]]
        | None = None,
    ) -> RagVarBatch:
        """Convert a RaggedVariants object to a tuple of nested tensors. Will flatten across
        the ploidy dimension for attributes ILEN, starts, and dosages such that their shapes are (batch * ploidy, ~variants).
        For the alternative alleles, will flatten across both the ploidy and variant dimensions such that the
        shape is (batch * ploidy * ~variants, ~alt_len).

        .. important::
            This function assumes all variant data is packed (see :func:`ak.to_packed`).

        Parameters
        ----------
        device
            The device to move the tensors to.
        tokenizer
            The tokenizer to use for the alternative alleles.

            - If :code:`"seqpro"`, will use :func:`seqpro.tokenize` to convert :code:`ACGTN -> 0 1 2 3 4`.
            - If :code:`None`, will use the integer ASCII value of each character i.e. :code:`ACGTN -> 65 67 71 84 78`.
            - Otherwise, will use the provided callable to convert the alternative alleles to a tensor of integers.

        Returns
        -------
            Dictionary of `nested tensors <https://docs.pytorch.org/docs/stable/nested.html>`_ and integers with the following keys:

            - :code:`"alts"` with shape :code:`(batch * ploidy * ~variants, ~alt_len)`
            - :code:`"ilens"` with shape :code:`(batch * ploidy, ~variants)`
            - :code:`"starts"` with shape :code:`(batch * ploidy, ~variants)`
            - :code:`"dosages"` with shape :code:`(batch * ploidy, ~variants)`
            - :code:`"max_seqlen"`: int, maximum number of variants
            - :code:`"max_alt_len"`: int, maximum length of an alternative allele

        """
        alts = cast(Content, self.alts.layout)
        while not isinstance(alts, NumpyArray):
            if isinstance(alts, (ListArray, ListOffsetArray)):
                offsets = alts
            alts = cast(Content, alts.content)
        alts = cast(NDArray[np.bytes_], alts.data)  # type: ignore

        if tokenizer == "seqpro":
            alts = sp.tokenize(alts, dict(zip(sp.DNA.alphabet, range(4))), 4)
        elif tokenizer is not None:
            alts = tokenizer(alts)
        else:
            alts = alts.view(np.uint8)

        alts = torch.from_numpy(alts).to(device)

        offsets = cast(ListArray | ListOffsetArray, offsets)  # type: ignore
        # (N ~V ~L) -> (N ~V) -> (N*~V)
        if isinstance(offsets, ListArray):
            lengths = cast(NDArray, offsets.stops.data - offsets.starts.data)  # type: ignore
            offsets = lengths_to_offsets(lengths, np.int32)
        else:
            offsets = offsets.offsets.data.astype(np.int32)  # type: ignore
            lengths = np.diff(offsets)

        max_alen = lengths.max().item()
        offsets = torch.from_numpy(offsets).to(device)
        # ((N, ~V), ~L)
        alts = nt_jag(alts, offsets)

        max_vlen = np.diff(self.v_starts.offsets).max().item()
        v_offsets = torch.from_numpy(self.v_starts.offsets.astype(np.int32)).to(device)
        ilens = torch.from_numpy(self.ilens.data).to(device)
        ilens = nt_jag(ilens, v_offsets)
        starts = torch.from_numpy(self.v_starts.data).to(device)
        starts = nt_jag(starts, v_offsets)

        if self.dosages is not None:
            dosages = torch.from_numpy(self.dosages.data).to(device)
            dosages = nt_jag(dosages, v_offsets)
        else:
            dosages = None

        return RagVarBatch(
            alts=alts,
            ilens=ilens,
            starts=starts,
            dosages=dosages,
            max_seqlen=max_vlen,
            max_alt_len=max_alen,
        )

    def prepend_pad_var(
        self, alt_char: str = "N", ilen: int = 0, start: int = -1, dosage: float = 0.0
    ) -> Self:
        """Prepend a pad variant so that every group is guaranteed to have at least 1 variant.

        Parameters
        ----------
        alt_char
            The character to use for the pad variant's ALT
        ilen
            The ILEN to use for the pad variant
        start
            The start position to use for the pad variant
        dosage
            The dosage to use for the pad variant

        Returns
        -------
            The RaggedVariants object with the pad variant prepended to each group.
        """
        b, p, _ = self.ilens.shape
        b = cast(int, b)
        p = cast(int, p)

        # (b p 1 1)
        node = NumpyArray(
            np.full((b, p), ord(alt_char), np.uint8).ravel(),  # type: ignore
            parameters={"__array__": "char"},
        )
        node = ListOffsetArray(Index(np.arange(len(node) + 1)), node)
        node = RegularArray(node, 1)
        node = RegularArray(node, p)
        pad_alt = ak.Array(node)
        # (b p ~v ~l)
        new_alts = ak.concatenate([pad_alt, ak.to_packed(self.alts)], axis=2)

        # (b p 1)
        pad_ilen = ak.from_numpy(np.full((b, p, 1), ilen, np.int32), regulararray=True)
        # (b p ~v)
        new_ilens = ak.concatenate([pad_ilen, self.ilens], axis=2)

        pad_start = ak.from_numpy(
            np.full((b, p, 1), start, np.int32), regulararray=True
        )
        # (b p ~v)
        new_starts = ak.concatenate([pad_start, self.v_starts], axis=2)

        if self.dosages is not None:
            pad_dosage = ak.from_numpy(
                np.full((b, p, 1), dosage, np.float32), regulararray=True
            )
            # (b p ~v)
            new_dosages = ak.concatenate([pad_dosage, self.dosages], axis=2)
        else:
            new_dosages = None

        return type(self)(
            alts=new_alts,
            ilens=Ragged(new_ilens),
            v_starts=Ragged(new_starts),
            dosages=Ragged(new_dosages),
        )


class RagVarBatch(TypedDict):
    alts: torch.Tensor
    ilens: torch.Tensor
    starts: torch.Tensor
    dosages: torch.Tensor | None
    max_seqlen: int
    max_alt_len: int


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
        # use -1 to mark that we are not currently within a germline variant
        g_idx = V_IDX_TYPE(-1)
        # set g_end to maximum possible value
        g_end = np.iinfo(POS_TYPE).max
        for i in range(n_variants * 2):
            pos: POS_TYPE = starts_ends[i]
            local_idx: V_IDX_TYPE = se_local_idx[i]
            pos_ccf: DOSAGE_TYPE = ccf[local_idx]
            is_germ = np.isnan(pos_ccf)

            # end of variant overlaps with end of current germline variant
            #! without this we will decrement the running CCF before setting the germline CCF
            # this is because tied ends are sorted by start, but the ends are 0-based exclusive
            # so we need to set the germline CCF before we start any decrementing
            if -pos >= g_end:
                ccf[g_idx] = max_ccf - running_ccf
                g_idx = -1
                g_end = np.iinfo(POS_TYPE).max

            # start of a germline variant
            if is_germ and pos > 0:
                # for now: check for overlapping variants and set to zero
                # to correspond to behavior of haplotype reconstruction
                # which only keeps first variant out of an overlapping set
                # TODO: handle overlapping germline vars without excessive memory
                # iterate over all g_ends, matching running ccf for each?
                if g_idx != -1 and np.isnan(ccf[g_idx]):
                    ccf[local_idx] = 0
                    continue
                g_idx = local_idx
                # have to recompute the end because we sorted them above so the local idx points
                # to the wrong place
                g_end = pos - min(0, ilen[local_idx]) + 1
            else:
                # sign of pos, with 0 being positive
                running_ccf += (2 * (pos >= 0) - 1) * pos_ccf

        np.nan_to_num(ccf, copy=False, nan=max_ccf)


@nb.njit(nogil=True, cache=True)
def rc_helper(alts: ak.Array, to_rc: NDArray[np.bool_]):
    for alt, rc in zip(alts, to_rc):
        if rc:
            alt = np.asarray(alt)
            rc_alt = np.empty_like(alt)
            rc_alt[alt == ord("A")] = ord("T")
            rc_alt[alt == ord("C")] = ord("G")
            rc_alt[alt == ord("G")] = ord("C")
            rc_alt[alt == ord("T")] = ord("A")
            alt[:] = rc_alt[::-1]
