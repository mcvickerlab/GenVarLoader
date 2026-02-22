from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import awkward as ak
import numba as nb
import numpy as np
import seqpro as sp
from awkward.contents import (
    Content,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from genoray._types import DOSAGE_TYPE, POS_TYPE, V_IDX_TYPE
from numpy.typing import NDArray
from seqpro.rag import OFFSET_TYPE, Ragged, is_rag_dtype, lengths_to_offsets
from typing_extensions import Self

from .._ragged import reverse_complement
from .._torch import TORCH_AVAILABLE, requires_torch

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    from torch.nested import nested_tensor_from_jagged as nt_jag
    from torch.nested._internal.nested_tensor import NestedTensor


class RaggedVariant(ak.Record):
    pass


class RaggedVariants(ak.Array):
    """An awkward record array with shape :code:`(batch, ploidy, ~variants, [~length])`.
    Guaranteed to at least have the field :code:`"alt"` and :code:`"start"` and one of :code:`"ref"` or :code:`"ilen"`.
    """

    def __init__(
        self,
        alt: ak.Array,
        start: Ragged[POS_TYPE],
        ref: ak.Array | None = None,
        ilen: Ragged[np.int32] | None = None,
        dosage: Ragged[DOSAGE_TYPE] | None = None,
        **kwargs: Ragged[np.number],
    ):
        if ref is None and ilen is None:
            raise ValueError("Must provide one of refs or ilens.")

        to_zip = {"alt": alt, "start": start}
        if ref is not None:
            to_zip["ref"] = ref
        if ilen is not None:
            to_zip["ilen"] = ilen
        if dosage is not None:
            to_zip["dosage"] = dosage

        arr = ak.zip(
            to_zip | kwargs, 1, parameters={"__record__": RaggedVariants.__name__}
        )

        super().__init__(arr)

    @classmethod
    def from_ak(cls, arr: ak.Array) -> RaggedVariants:
        """Create a RaggedVariants object from an awkward array.

        Parameters
        ----------
        arr
            The awkward array to create a RaggedVariants object from.
        """
        fields = set(arr.fields)

        if missing := {"alt", "start"} - fields:
            raise ValueError(f"Missing required fields: {missing}")

        if {"ref", "ilen"}.isdisjoint(fields):
            raise ValueError("Must have one of ref or ilen.")

        def find_and_convert_to_ragged(content: Content, depth_context: dict, **kwargs):
            if isinstance(content, (ListArray, ListOffsetArray)):
                depth_context["n_varlen"] += 1

            if (
                # is a varlen leaf
                isinstance(content, (ListArray, ListOffsetArray))
                and isinstance(content.content, NumpyArray)
                # is the only varlen leaf in this branch
                and depth_context["n_varlen"] == 1
                # has no parameters that might conflict with Ragged
                and len(content.parameters) == 0
            ):
                return ak.with_parameter(content, "__list__", "Ragged", highlevel=False)

        arr = ak.transform(  # type: ignore
            find_and_convert_to_ragged, arr, depth_context={"n_varlen": 0}
        )

        return ak.with_parameter(arr, "__record__", RaggedVariants.__name__)

    @property
    def alt(self) -> ak.Array:
        """Alternative alleles."""
        return cast(ak.Array, super().__getitem__("alt"))

    @property
    def start(self) -> Ragged[POS_TYPE]:
        """0-based start positions."""
        return cast(Ragged[POS_TYPE], super().__getitem__("start"))

    @property
    def ilen(self) -> Ragged[np.int32]:
        """Indel lengths. Infallible."""
        if "ilen" not in self.fields:
            ilen = ak.str.length(self.alt) - ak.str.length(self.ref)  # type: ignore
            ilen = Ragged(ilen)
            return ilen

        return cast(Ragged[np.int32], super().__getitem__("ilen"))

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.start.shape

    @property
    def end(self) -> Ragged[POS_TYPE]:
        """0-based, exclusive end positions."""
        if hasattr(self, "ref"):
            ref = cast(Ragged[np.bytes_], self.ref)
            return self.start + ak.num(ref, -1)
        else:
            ilen = cast(Ragged[np.int32], self.ilen)
            return self.start - np.clip(ilen, None, 0) + 1

    def reshape(self, shape: tuple[int | None, ...]) -> Self:
        """Reshape leading, regular axes. Assumes no trailing regular axes."""
        reshaped = {}

        for field in self.fields:
            arr = cast(Ragged | ak.Array, self[field])
            if isinstance(arr, Ragged):
                arr = arr.reshape(shape)
            else:
                # strip regular axes
                node = arr.layout
                while isinstance(node, RegularArray):
                    node = node.content

                # create new regular axes
                for len_ in reversed(shape[1:]):
                    if len_ is None:
                        continue
                    node = RegularArray(node, len_)
                arr = ak.Array(node)

            reshaped[field] = arr

        return type(self)(**reshaped)

    def squeeze(self, **kwargs) -> Self:
        """Squeeze first axis."""
        return self[0]  # type: ignore

    def infer_germline_ccfs_(
        self, ccf_field: str = "dosages", max_ccf: float = 1.0
    ) -> Self:
        """Infer germline CCFs in-place.

        Germline variants are identified by having missing CCFs i.e. they have a variant
        index but missing CCFs. Missing CCFs are inferred to be :code:`max_ccf` - sum(overlapping CCFs).

        Parameters
        ----------
        max_ccf
            Maximum CCF value.
        """
        if not hasattr(self, ccf_field):
            raise ValueError(f"Cannot infer germline CCFs without {ccf_field}.")

        ccfs = self[ccf_field]
        if not isinstance(ccfs, Ragged) or not is_rag_dtype(ccfs, DOSAGE_TYPE):
            raise ValueError(f"{ccf_field} must be a Ragged array of {DOSAGE_TYPE}.")

        _infer_germline_ccfs(
            ccfs.data,
            self.start.offsets,
            self.start.data,
            self.ilen.data,
            max_ccf=max_ccf,
        )
        return self

    def to_packed(self) -> Self:
        """Apply :func:`ak.to_packed` to all arrays."""
        return ak.to_packed(self)

    def rc_(self, to_rc: NDArray[np.bool_] | None = None) -> Self:
        """Reverse complement the alleles. This is an in-place operation.

        Parameters
        ----------
        to_rc
            A boolean mask of the same shape as the variant dimension. If :code:`True`, the alternative allele will be reverse complemented.
            If :code:`None`, will reverse complement all alternative alleles.

        Returns
        -------
            The RaggedVariants object with the alleles reverse complemented.
        """
        if to_rc is None:
            to_rc = np.ones(self.shape[0], np.bool_)  # type: ignore
        elif not to_rc.any():
            return self

        self["alt"] = ak.where(
            to_rc,
            reverse_complement(self["alt"]),  # type: ignore
            self["alt"],
        )

        if "ref" in self.fields:
            self["ref"] = ak.where(
                to_rc,
                reverse_complement(self["ref"]),  # type: ignore
                self["ref"],
            )

        return self

    @requires_torch
    def to_nested_tensor_batch(
        self,
        device: str | torch.device = "cpu",
        tokenizer: Literal["seqpro"]
        | Callable[[NDArray[np.bytes_]], NDArray[np.integer]]
        | None = None,
    ) -> dict[str, NestedTensor | int]:
        """Convert a RaggedVariants object to a dictionary of nested tensors. Will flatten across
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
            - :code:`"max_n_vars"`: int, maximum number of variants
            - :code:`"max_alt_len"`: int, maximum length of an alternative allele
            - :code:`"max_ref_len"`: int, maximum length of a reference allele

        """
        batch = {}
        variant_offsets = None
        for field in self.fields:
            arr = cast(Ragged | ak.Array, self[field])
            if isinstance(arr, Ragged):
                data = torch.from_numpy(arr.data).to(device)
                if variant_offsets is None:
                    variant_offsets = torch.from_numpy(arr.offsets.astype(np.int32)).to(
                        device
                    )
                    batch["max_n_vars"] = int(np.diff(arr.offsets).max())
                batch[field] = nt_jag(data, variant_offsets)
            elif field in {"ref", "alt"}:
                data, offsets, max_alen = _alleles_to_nested_tensor(arr, tokenizer)
                data = data.to(device)
                batch[f"max_{field}_len"] = max_alen
                batch[field] = nt_jag(data, offsets)

        return batch

    def pad(
        self,
        allele: str | bytes = b"N",
        ilen: int = 0,
        start: int = -1,
        dosage: float = 0.0,
        **pad_values: Any,
    ) -> Self:
        """Append a pad variant so that every group is guaranteed to have at least 1 variant. If the group has variants,
        no variant is appended.

        Parameters
        ----------
        allele
            The allele to use for ALTs and REFs
        ilen
        start
            The start position to use for the pad variant
        dosage
            The dosage to use for the pad variant
        **pad_values
            Additional values to use for each field. Raises a ValueError if any field does not have a pad value.

        Returns
        -------
            The RaggedVariants object with the pad variant appended to each group that has no variants.
        """
        if isinstance(allele, str):
            allele = allele.encode()

        pad_values |= {
            "alt": allele,
            "ref": allele,
            "ilen": ilen,
            "start": start,
            "dosage": dosage,
        }

        if missing_fields := set(self.fields) - set(pad_values.keys()):
            raise ValueError(f"Missing pad values for fields: {missing_fields}")

        arr = ak.pad_none(self, 1, -1)
        for field in self.fields:
            value = pad_values[field]
            arr = ak.with_field(arr, ak.fill_none(arr[field], value, -1), field)
        return arr


def _alleles_to_nested_tensor(
    alleles: ak.Array,
    tokenizer: Literal["seqpro"]
    | Callable[[NDArray[np.bytes_]], NDArray[np.integer]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _alleles = cast(Content, alleles.layout)
    while not isinstance(_alleles, NumpyArray):
        if isinstance(_alleles, (ListArray, ListOffsetArray)):
            offsets = _alleles
        _alleles = cast(Content, _alleles.content)
    _alleles = cast(NDArray[np.bytes_], _alleles.data)  # type: ignore

    if tokenizer == "seqpro":
        _alleles = sp.tokenize(_alleles, dict(zip(sp.DNA.alphabet, range(4))), 4)
    elif tokenizer is not None:
        _alleles = tokenizer(_alleles)
    else:
        _alleles = _alleles.view(np.uint8)

    _alleles = torch.from_numpy(_alleles)

    offsets = cast(ListArray | ListOffsetArray, offsets)  # type: ignore
    # (N ~V ~L) -> (N ~V) -> (N*~V)
    if isinstance(offsets, ListArray):
        lengths = cast(NDArray, offsets.stops.data - offsets.starts.data)  # type: ignore
        offsets = lengths_to_offsets(lengths, np.int32)
    else:
        offsets = offsets.offsets.data.astype(np.int32)  # type: ignore
        lengths = np.diff(offsets)

    if len(lengths) == 0:
        max_alen = 0
    else:
        max_alen = lengths.max().item()

    offsets = torch.from_numpy(offsets)
    return _alleles, offsets, max_alen


ak.behavior["*", RaggedVariants.__name__] = RaggedVariants


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


def _rc_helper(
    ragv: RaggedVariants, field: str, to_rc: NDArray[np.bool_] | None = None
):
    # flatten all but last two dimensions & strip params for numba
    alleles = ragv[field].layout  # type: ignore
    while not isinstance(alleles.content, NumpyArray):  # type: ignore
        alleles = alleles.content  # type: ignore
    alleles = ak.without_parameters(alleles)

    if to_rc is None:
        to_rc = np.ones(ragv.shape[:-1], np.bool_)  # type: ignore

    # broadcast to same shape as variants, and flatten
    # (batch) -> (batch * ploidy * n_variants)
    # batch * ploidy * n_variants = n_alts
    _to_rc, _ = ak.broadcast_arrays(to_rc, ragv.start)
    _to_rc = _to_rc.layout
    while not isinstance(_to_rc, NumpyArray):
        _to_rc = _to_rc.content
    _to_rc = cast(NDArray[np.bool_], _to_rc.data)  # type: ignore

    _rc_numba_helper(alleles, _to_rc)


@nb.njit(nogil=True, cache=True)
def _rc_numba_helper(alts: ak.Array, to_rc: NDArray[np.bool_]):
    for alt, rc in zip(alts, to_rc):
        if rc:
            alt = np.asarray(alt)
            rc_alt = np.empty_like(alt)
            rc_alt[alt == ord("A")] = ord("T")
            rc_alt[alt == ord("C")] = ord("G")
            rc_alt[alt == ord("G")] = ord("C")
            rc_alt[alt == ord("T")] = ord("A")
            alt[:] = rc_alt[::-1]
