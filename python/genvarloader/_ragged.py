from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, cast

import awkward as ak
import numba as nb
import numpy as np
from attrs import define
from einops import repeat
from numpy.typing import NDArray
from phantom import Phantom
from seqpro.rag import RDTYPE, Ragged, is_rag_dtype

from ._torch import TORCH_AVAILABLE
from ._types import AnnotatedHaps

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    from torch.nested import nested_tensor_from_jagged as nt_jag

__all__ = ["Ragged", "RaggedIntervals", "RaggedTracks"]
INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)


@define
class RaggedIntervals:
    starts: Ragged[np.int32]
    ends: Ragged[np.int32]
    values: Ragged[np.float32]

    def __getitem__(self, idx) -> RaggedIntervals:
        out = RaggedIntervals(self.starts[idx], self.ends[idx], self.values[idx])  # type: ignore
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
            self.starts.squeeze(axis),
            self.ends.squeeze(axis),
            self.values.squeeze(axis),
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


@define
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

    def to_fixed_shape(self, shape: tuple[int, ...]) -> AnnotatedHaps:
        """If all entries in the ragged array have the same shape, convert to a rectilinear shape.

        Parameters
        ----------
        shape
            Shape to convert to, including the length axis. The total number of elements must remain the same.
        """
        haps = self.haps.data.reshape(shape)
        var_idxs = self.var_idxs.data.reshape(shape)
        ref_coords = self.ref_coords.data.reshape(shape)
        return AnnotatedHaps(haps, var_idxs, ref_coords)


def to_padded(rag: Ragged[RDTYPE], pad_value: Any) -> NDArray[RDTYPE]:
    """Convert this Ragged array to a rectilinear array by right-padding each entry with a value.
    The ragged axis will be padded to have the maximum length across all entries.

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
    length = rag.lengths.max()
    rag = ak.pad_none(rag, length, clip=True)
    rag = ak.fill_none(rag, pad_value)
    arr = ak.to_numpy(rag)
    return arr


NUCLEOTIDES = b"ACGT"
COMPLEMENTS = b"TGCA"


#! for whatever reason, this causes data corruption with parallel=True?!
#! assumes offsets are 1D
@nb.njit(nogil=True, cache=True)
def _rc_helper(
    data: NDArray[np.uint8], offsets: NDArray[np.int64], mask: NDArray[np.bool_]
) -> NDArray[np.uint8]:
    out = data.copy()
    for i in nb.prange(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        _data = data[start:end]
        _out = out[start:end]
        if mask[i]:
            for nuc, comp in zip(NUCLEOTIDES, COMPLEMENTS):
                _out[_data == nuc] = comp
            _out[:] = _out[::-1]
    return out


def reverse_complement(
    seqs: Ragged[np.bytes_], mask: NDArray[np.bool_]
) -> Ragged[np.bytes_]:
    # (b [p] ~l), (b)
    if seqs.ndim == 3:
        ploidy = seqs.shape[1]
        mask = repeat(mask, "b -> (b p)", p=ploidy)
    rc_seqs = _rc_helper(seqs.data.view(np.uint8), seqs.offsets, mask)
    return Ragged.from_offsets(rc_seqs.view("S1"), seqs.shape, seqs.offsets)


#! for whatever reason, this causes data corruption with parallel=True?!
#! assumes offsets are 1D
@nb.njit(nogil=True, cache=True)
def _reverse_helper(data: NDArray, offsets: NDArray[np.int64], mask: NDArray[np.bool_]):
    for i in nb.prange(len(offsets) - 1):
        if mask[i]:
            start, end = offsets[i], offsets[i + 1]
            data[start:end] = np.flip(data[start:end])


def reverse(tracks: Ragged, mask: NDArray[np.bool_]):
    """Reverses data along the ragged axis in-place."""
    # (b t [p] ~l), (b)
    if tracks.ndim == 3:
        n_tracks = tracks.shape[1]
        mask = repeat(mask, "b -> (b t)", t=n_tracks)
    elif tracks.ndim == 4:
        n_tracks, ploidy = tracks.shape[1:-1]
        mask = repeat(mask, "b -> (b t p)", t=n_tracks, p=ploidy)
    _reverse_helper(tracks.data, tracks.offsets, mask)
