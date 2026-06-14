from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import awkward as ak
import numba as nb
import numpy as np
import seqpro as sp
from awkward.contents import (
    Content,
    IndexedArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from genoray._types import DOSAGE_TYPE, POS_TYPE
from numpy.typing import NDArray
from seqpro.rag import Ragged, lengths_to_offsets
from typing_extensions import Self

from .._ragged import reverse_complement_masked
from .._torch import TORCH_AVAILABLE, requires_torch

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    from torch.nested import nested_tensor_from_jagged as nt_jag
    from torch.nested._internal.nested_tensor import NestedTensor


def _is_canonical_alleles(layout: Content) -> bool:
    """True if an alt/ref layout is the canonical, directly-extractable chain
    ``RegularArray -> ListOffsetArray -> ListOffsetArray -> NumpyArray`` (possibly
    sliced, i.e. non-zero-based offsets — handled by the existing fast path). Any
    ``IndexedArray``/``ListArray`` wrapping (from fancy-index/reverse) returns False."""
    return (
        isinstance(layout, RegularArray)
        and isinstance(layout.content, ListOffsetArray)
        and isinstance(layout.content.content, ListOffsetArray)
        and isinstance(layout.content.content.content, NumpyArray)
    )


def _decompose_alleles(
    arr: ak.Array,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.uint8],
    int,
]:
    """Decompose a (possibly non-canonical) (b, p, ~v, ~l) allele array into raw
    primitives for :func:`_pack_alleles`. Reads ``.starts``/``.stops`` (present on
    both ``ListArray`` and ``ListOffsetArray``) and the optional outer index.

    Returns ``(row_src, var_starts, var_stops, allele_starts, allele_stops, leaf, ploidy)``
    where ``row_src[b*p + h] = index[b]*p + h`` indexes the variant-list rows.
    """
    lay = arr.layout
    if isinstance(lay, (IndexedArray, IndexedOptionArray)):
        index = np.asarray(lay.index, np.int64)
        reg = lay.project() if isinstance(lay, IndexedOptionArray) else lay.content
        # For IndexedArray, content is the (un-indexed) RegularArray; for the option
        # case we project (gvl variants never contain None, but be safe).
        if isinstance(lay, IndexedOptionArray):
            index = None  # project() already applied the gather
    else:
        index = None
        reg = lay

    if not isinstance(reg, RegularArray):
        raise ValueError(f"Unsupported allele layout for packing: {arr.layout.form}")
    ploidy = int(reg.size)

    var_node = reg.content
    var_starts = np.asarray(var_node.starts, np.int64)
    var_stops = np.asarray(var_node.stops, np.int64)

    allele_node = var_node.content
    allele_starts = np.asarray(allele_node.starts, np.int64)
    allele_stops = np.asarray(allele_node.stops, np.int64)
    leaf = np.asarray(allele_node.content.data).view(np.uint8)

    if index is None:
        n_out_rows = len(reg) * ploidy
        row_src = np.arange(n_out_rows, dtype=np.int64)
    else:
        row_src = (index[:, None] * ploidy + np.arange(ploidy, dtype=np.int64)).reshape(
            -1
        )
    return row_src, var_starts, var_stops, allele_starts, allele_stops, leaf, ploidy


@nb.njit(nogil=True, cache=True)
def _pack_alleles(row_src, var_starts, var_stops, allele_starts, allele_stops, leaf):
    """Gather doubly-nested alleles into contiguous, zero-based byte buffers in
    canonical ``(b, p, ~v, ~l)`` row-major order. Sequential (offset accumulation);
    only invoked off the hot path for non-canonical layouts."""
    n_rows = row_src.shape[0]
    n_alleles = 0
    n_bytes = 0
    for i in range(n_rows):
        src = row_src[i]
        for a in range(var_starts[src], var_stops[src]):
            n_alleles += 1
            n_bytes += allele_stops[a] - allele_starts[a]

    packed = np.empty(n_bytes, np.uint8)
    allele_off = np.empty(n_alleles + 1, np.int64)
    group_off = np.empty(n_rows + 1, np.int64)
    allele_off[0] = 0
    group_off[0] = 0

    ai = 0
    bi = 0
    for i in range(n_rows):
        src = row_src[i]
        for a in range(var_starts[src], var_stops[src]):
            s = allele_starts[a]
            e = allele_stops[a]
            for k in range(s, e):
                packed[bi] = leaf[k]
                bi += 1
            ai += 1
            allele_off[ai] = bi
        group_off[i + 1] = ai
    return packed, allele_off, group_off


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

        arr = ak.transform(  # type: ignore[bad-assignment]  # ak.transform stub returns Array|tuple|None; we know it's Array here
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
            ilen = ak.str.length(self.alt) - ak.str.length(self.ref)  # type: ignore[missing-attribute]  # ak.str submodule isn't exposed in awkward's top-level type stubs
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

    def squeeze(self, axis: int | None = None, **kwargs) -> Self:
        """Squeeze first axis."""
        return self[0]

    def to_packed(self) -> Self:
        """Pack all fields into contiguous, zero-based arrays.

        Replaces the previous :func:`ak.to_packed` call with field-wise packing:
        seqpro :meth:`~seqpro.rag.Ragged.to_packed` for numeric :class:`~seqpro.rag.Ragged`
        fields, and an allele-level seqpro pack + group-offset rebase +
        :func:`~._haps._build_allele_layout` rebuild for the doubly-nested ``alt``/``ref``
        fields.
        """
        from seqpro.rag import Ragged

        # local import to avoid circular dependency (_haps imports RaggedVariants)
        from ._haps import _alt_layout_parts, _build_allele_layout

        packed: dict = {}
        for field in self.fields:
            arr = self[field]
            if field in ("alt", "ref"):
                if _is_canonical_alleles(arr.layout):
                    # fast path (unchanged): canonical (possibly sliced) layout
                    leaf, allele_off, group_off, ploidy = _alt_layout_parts(arr)
                    # _alt_layout_parts returns the FULL (un-sliced) leaf and allele_off even
                    # for a sliced view — only group_off carries the slice's offset.  We must
                    # use group_off[0] to locate where this view's allele groups begin in the
                    # full allele_off, then slice and zero-base both allele_off and leaf to
                    # match so that _build_allele_layout sees a clean, contiguous layout.
                    g0 = int(group_off[0])
                    rebased_group = np.asarray(group_off, np.int64) - g0
                    # slice allele_off to only the alleles in this view and zero-base
                    a0 = int(allele_off[g0])
                    sliced_allele_off = np.asarray(allele_off[g0:], np.int64) - a0
                    sliced_leaf = leaf[a0:]
                    # pack the allele (byte) level: contiguates bytes
                    allele_lvl = Ragged.from_offsets(
                        sliced_leaf.view("S1"),
                        (sliced_allele_off.size - 1, None),
                        sliced_allele_off,
                    ).to_packed()
                    packed[field] = _build_allele_layout(
                        np.asarray(allele_lvl.data).view(np.uint8),
                        np.asarray(allele_lvl.offsets),
                        rebased_group,
                        ploidy,
                    )
                else:
                    # non-canonical (IndexedArray/ListArray from slicing/reorder):
                    # numba gather, no ak.to_packed / awkward gather primitives.
                    (
                        row_src,
                        var_starts,
                        var_stops,
                        allele_starts,
                        allele_stops,
                        leaf,
                        ploidy,
                    ) = _decompose_alleles(arr)
                    packed_bytes, allele_off, group_off = _pack_alleles(
                        row_src,
                        var_starts,
                        var_stops,
                        allele_starts,
                        allele_stops,
                        leaf,
                    )
                    packed[field] = _build_allele_layout(
                        packed_bytes, allele_off, group_off, ploidy
                    )
            else:
                packed[field] = (
                    arr.to_packed()
                    if isinstance(arr, Ragged)
                    else Ragged(arr).to_packed()
                )
        return type(self)(**packed)

    def rc_(self, to_rc: NDArray[np.bool_] | None = None) -> Self:
        """Reverse complement the alleles. This is an in-place operation for
        canonical (contiguous) layouts. For non-canonical (sliced/reordered)
        views, the data is materialized into a new contiguous object first, so a
        NEW object is returned and ``self`` is left unmutated — callers should
        use the return value.

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
            to_rc = np.ones(self.shape[0], np.bool_)  # type: ignore[no-matching-overload]  # ak.Array shape may contain None; np.ones overload expects int|Sequence[int]
        elif not to_rc.any():
            return self

        # Non-canonical (sliced/reordered) views can't be reverse-complemented in
        # place safely. Materialize a contiguous canonical copy, then recurse — the
        # recursion hits the in-place fast path below. Returns a new object; the sole
        # caller (reverse_complement_ragged) uses the return value.
        if any(
            not _is_canonical_alleles(self[f].layout)
            for f in ("alt", "ref")
            if f in self.fields
        ):
            return self.to_packed().rc_(to_rc)

        # local import: _haps imports RaggedVariants (avoid circular import)
        from ._haps import _alt_layout_parts

        for field in ("alt", "ref"):
            if field not in self.fields:
                continue
            arr = self[field]
            leaf, allele_off, group_off, ploidy = _alt_layout_parts(arr)
            # per-allele mask: to_rc is per-batch; broadcast across ploidy then variants
            per_bp = np.repeat(np.ascontiguousarray(to_rc, np.bool_), ploidy)
            per_allele = np.repeat(per_bp, np.diff(group_off))
            view = Ragged.from_offsets(
                leaf.view("S1"), (per_allele.size, None), allele_off
            )
            # in-place: mutates `leaf`, which shares memory with `arr`'s buffer
            reverse_complement_masked(view, per_allele)

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
    _alleles = cast(NDArray[np.bytes_], _alleles.data)

    if tokenizer == "seqpro":
        _alleles = sp.tokenize(_alleles, dict(zip(sp.DNA.alphabet, range(4))), 4)
    elif tokenizer is not None:
        _alleles = tokenizer(_alleles)
    else:
        _alleles = _alleles.view(np.uint8)

    _alleles = torch.from_numpy(_alleles)

    offsets = cast(ListArray | ListOffsetArray, offsets)  # type: ignore[redundant-cast]  # cast is documentation here; pyrefly narrows but readers benefit
    # (N ~V ~L) -> (N ~V) -> (N*~V)
    if isinstance(offsets, ListArray):
        lengths = cast(NDArray, offsets.stops.data - offsets.starts.data)
        offsets = lengths_to_offsets(lengths, np.int32)
    else:
        offsets = offsets.offsets.data.astype(np.int32)  # type: ignore[missing-attribute]  # awkward Index.data typed as ArrayLike; numpy ndarray method missing on stub
        lengths = np.diff(offsets)

    if len(lengths) == 0:
        max_alen = 0
    else:
        max_alen = lengths.max().item()

    offsets = torch.from_numpy(offsets)
    return _alleles, offsets, max_alen


ak.behavior["*", RaggedVariants.__name__] = RaggedVariants
