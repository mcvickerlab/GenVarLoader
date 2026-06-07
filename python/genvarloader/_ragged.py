from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, cast

import awkward as ak
import awkward.operations.str as ak_str
import numba as nb
import numpy as np
from awkward.contents import NumpyArray
from numpy.typing import NDArray
from phantom import Phantom
from seqpro.rag import Ragged, is_rag_dtype
from seqpro.rag import RDTYPE_co as RDTYPE
from seqpro.rag import reverse_complement as _sp_reverse_complement
from seqpro.rag import to_padded as _sp_to_padded

from ._torch import TORCH_AVAILABLE
from ._types import AnnotatedHaps

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    from torch.nested import nested_tensor_from_jagged as nt_jag

__all__ = ["Ragged", "RaggedIntervals", "RaggedTracks"]
INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)


@dataclass(slots=True)
class RaggedIntervals:
    starts: Ragged[np.int32]
    ends: Ragged[np.int32]
    values: Ragged[np.float32]

    def __getitem__(self, idx) -> RaggedIntervals:
        out = RaggedIntervals(self.starts[idx], self.ends[idx], self.values[idx])  # type: ignore[bad-argument-type]  # Ragged.__getitem__ widens to Array per awkward stubs
        return out

    @property
    def shape(self):
        """Shape of the haplotypes and all annotations."""
        return self.values.shape

    def to_padded(
        self, start: int, end: int, value: float
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
        """Convert this RaggedIntervals to a tuple of rectilinear arrays by right-padding each entry with appropriate values.
        The final axis will have the maximum length across all entries."""
        starts = to_padded(self.starts, start)
        ends = to_padded(self.ends, end)
        values = to_padded(self.values, value)
        return starts, ends, values

    def reshape(self, shape: int | tuple[int, ...]) -> RaggedIntervals:
        """Reshape the haplotypes and all annotations.

        Parameters
        ----------
        shape
            New shape for the haplotypes and all annotations. The total number of elements
            must remain the same.
        """
        return RaggedIntervals(
            self.starts.reshape(shape),
            self.ends.reshape(shape),
            self.values.reshape(shape),
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> RaggedIntervals:
        """Squeeze the haplotypes and all annotations along the specified axis.

        Parameters
        ----------
        axis
            Axis or axes to squeeze. If None, all axes of length 1 are squeezed.
        """
        return RaggedIntervals(
            self.starts.squeeze(axis),  # type: ignore[bad-argument-type]  # seqpro Ragged.squeeze stub returns broader union than Ragged[T]
            self.ends.squeeze(axis),  # type: ignore[bad-argument-type]  # see above
            self.values.squeeze(axis),  # type: ignore[bad-argument-type]  # see above
        )

    def to_fixed_shape(
        self, shape: tuple[int, ...]
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
        """If all entries in the ragged array have the same shape, convert to a rectilinear shape.

        Parameters
        ----------
        shape
            Shape to convert to, including the length axis. The total number of elements must remain the same.
        """
        starts = self.starts.data.reshape(shape)
        ends = self.ends.data.reshape(shape)
        values = self.values.data.reshape(shape)
        return starts, ends, values

    def to_packed(self) -> RaggedIntervals:
        """Apply :func:`ak.to_packed` to all arrays."""
        starts = ak.to_packed(self.starts)
        ends = ak.to_packed(self.ends)
        values = ak.to_packed(self.values)
        return RaggedIntervals(starts, ends, values)

    def to_nested_tensor_batch(
        self, device: str | torch.device = "cpu"
    ) -> list[RagItvBatch]:
        out = []
        n_tracks = cast(int, self.values.shape[1])
        for t in range(n_tracks):
            # (batch tracks ... ~itv) -> (batch ... ~itv)
            starts = ak.to_packed(self.starts[:, t])
            ends = ak.to_packed(self.ends[:, t])
            values = ak.to_packed(self.values[:, t])

            offsets = torch.from_numpy(values.offsets.astype(np.int32)).to(device)
            max_len = int(values.lengths.max())

            starts = torch.from_numpy(starts.data).to(device)
            starts = nt_jag(starts, offsets)
            ends = torch.from_numpy(ends.data).to(device)
            ends = nt_jag(ends, offsets)
            values = torch.from_numpy(values.data.astype(np.float32)).to(device)
            values = nt_jag(values, offsets)

            out.append(
                RagItvBatch(starts=starts, ends=ends, values=values, max_seqlen=max_len)
            )

        return out

    def prepend_pad_itv(
        self, start: int = -1, end: int = -1, value: float = 0.0
    ) -> RaggedIntervals:
        """Prepend a pad interval so that every group is guaranteed to have at least 1 interval.

        Parameters
        ----------
        start
            The start position to use for the pad interval
        end
            The end position to use for the pad interval
        value
            The value to use for the pad interval
        """
        b, t, *_ = self.values.shape
        b = cast(int, b)
        t = cast(int, t)

        pad_start = ak.from_numpy(
            np.full((b, t, 1), start, np.int32), regulararray=True
        )
        # (b t ~v)
        new_starts = ak.concatenate([pad_start, self.starts], axis=2)
        pad_end = ak.from_numpy(np.full((b, t, 1), end, np.int32), regulararray=True)
        # (b t ~v)
        new_ends = ak.concatenate([pad_end, self.ends], axis=2)
        pad_value = ak.from_numpy(
            np.full((b, t, 1), value, np.float32), regulararray=True
        )
        # (b t ~v)
        new_values = ak.concatenate([pad_value, self.values], axis=2)

        return RaggedIntervals(Ragged(new_starts), Ragged(new_ends), Ragged(new_values))


class RagItvBatch(TypedDict):
    """Dictionary of nested tensors."""

    starts: torch.Tensor
    ends: torch.Tensor
    values: torch.Tensor
    max_seqlen: int


class RaggedSeqs(
    Ragged[np.bytes_], Phantom, predicate=partial(is_rag_dtype, dtype=np.bytes_)
): ...


class RaggedTracks(
    Ragged[np.float32], Phantom, predicate=partial(is_rag_dtype, dtype=np.float32)
): ...


@dataclass(slots=True)
class RaggedAnnotatedHaps:
    """Ragged version of :class:`AnnotatedHaps`."""

    haps: Ragged[np.bytes_]
    """Haplotypes with dtype S1."""
    var_idxs: Ragged[np.int32]
    """Variant indices for each position in the haplotypes. A value of -1 indicates no variant was applied at the position."""
    ref_coords: Ragged[np.int32]
    """Reference coordinates for each position in haplotypes."""

    @property
    def shape(self):
        """Shape of the haplotypes and all annotations."""
        return self.haps.shape

    def to_padded(self) -> AnnotatedHaps:
        """Convert this Ragged array to a rectilinear array by right-padding each entry with appropriate values.
        The final axis will have the maximum length across all entries."""
        haps = to_padded(self.haps, b"N")
        var_idxs = to_padded(self.var_idxs, -1)
        ref_coords = to_padded(self.ref_coords, np.iinfo(self.ref_coords.dtype).max)
        return AnnotatedHaps(haps, var_idxs, ref_coords)

    def reshape(self, shape: int | tuple[int, ...]) -> RaggedAnnotatedHaps:
        """Reshape the haplotypes and all annotations.

        Parameters
        ----------
        shape
            New shape for the haplotypes and all annotations. The total number of elements
            must remain the same.
        """
        return RaggedAnnotatedHaps(
            self.haps.reshape(shape),
            self.var_idxs.reshape(shape),
            self.ref_coords.reshape(shape),
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> RaggedAnnotatedHaps:
        """Squeeze the haplotypes and all annotations along the specified axis.

        Parameters
        ----------
        axis
            Axis or axes to squeeze. If None, all axes of length 1 are squeezed.
        """
        return RaggedAnnotatedHaps(
            self.haps.squeeze(axis),
            self.var_idxs.squeeze(axis),
            self.ref_coords.squeeze(axis),
        )

    def to_numpy(self) -> AnnotatedHaps:
        """If all entries in the ragged array have the same shape, convert to a rectilinear shape.

        Parameters
        ----------
        shape
            Shape to convert to, including the length axis. The total number of elements must remain the same.
        """
        haps = self.haps.to_numpy()
        var_idxs = self.var_idxs.to_numpy()
        ref_coords = self.ref_coords.to_numpy()
        return AnnotatedHaps(haps, var_idxs, ref_coords)


def to_padded(rag: Ragged[RDTYPE], pad_value: Any) -> NDArray[RDTYPE]:
    """Convert this Ragged array to a rectilinear array by right-padding each entry with a value.
    The ragged axis will be padded to have the maximum length across all entries.

    Thin pass-through to :func:`seqpro.rag.to_padded` (seqpro 0.13+), a single-pass,
    parallel flat-buffer densify-and-pad kernel that replaced the old awkward
    ``ak_str.rpad`` / ``ak.pad_none`` + ``fill_none`` + ``to_numpy`` idiom. Output is
    byte-identical for every dtype/pad gvl uses (S1, int32, float32); seqpro pads to
    the batch maximum ``rag.lengths.max()`` when no explicit length is given.

    Parameters
    ----------
    rag
        Ragged array to pad.
    pad_value
        Value to pad the entries with.

    Returns
    -------
        Padded array.
    """
    return _sp_to_padded(rag, pad_value)


_COMP = np.frombuffer(bytes.maketrans(b"ACGT", b"TGCA"), np.uint8)


@nb.vectorize(["u1(u1)"], nopython=True)
def ufunc_comp_dna(seq: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return _COMP[seq]


def _ak_comp_dna_helper(layout, **kwargs):
    if layout.is_numpy:
        return NumpyArray(
            ufunc_comp_dna(layout.data),
            parameters=layout.parameters,
        )


T = TypeVar("T", bound=ak.Array)


def reverse_complement(arr: T) -> T:
    og_type = type(arr)
    arr = ak.to_packed(arr)
    arr = ak_str.reverse(ak.transform(_ak_comp_dna_helper, arr))
    return og_type(arr)


def reverse_complement_masked(
    rag: Ragged[np.bytes_], mask: NDArray[np.bool_]
) -> Ragged[np.bytes_]:
    """Masked reverse-complement of an S1 ragged batch, in place.

    Flat-buffer replacement for the awkward idiom
    ``Ragged(ak.to_packed(ak.where(mask, reverse_complement(rag), rag)))``: seqpro's
    kernel touches only the ``mask``-selected rows, runs a single in-place pass per row,
    and reuses ``rag``'s offsets. Reuses :data:`_COMP` (the same A<->T, C<->G table the
    awkward path uses) so output is byte-identical.

    Mutates ``rag`` in place (``copy=False``); only call on a freshly reconstructed batch
    the caller owns.

    ``mask`` is one entry per outer query (e.g. per region); awkward's ``ak.where``
    used to broadcast it across any inner fixed axes (e.g. ploidy) left-aligned. seqpro's
    flat kernel wants one entry per flattened ragged row, so replicate the mask across the
    inner axes in C order to match.
    """
    mask = np.ascontiguousarray(mask, dtype=np.bool_).reshape(-1)
    n_rows = int(np.prod(rag.shape[: rag.rag_dim], dtype=np.int64))
    if mask.size != n_rows:
        inner_factor, rem = divmod(n_rows, mask.size)
        if rem != 0:
            raise ValueError(
                f"mask has {mask.size} entries but ragged array has {n_rows} rows, "
                "which is not an integer multiple."
            )
        mask = np.repeat(mask, inner_factor)
    return _sp_reverse_complement(rag, _COMP, mask=mask, copy=False)
