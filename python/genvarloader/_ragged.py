from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from phantom import Phantom
import seqpro.rag as spr
from seqpro.rag import Ragged, is_rag_dtype
from seqpro.rag import RDTYPE_co as RDTYPE
from seqpro.rag import reverse_complement as _sp_reverse_complement
from .genvarloader import ragged_to_padded

from ._flat import _Flat
from ._torch import TORCH_AVAILABLE
from ._types import AnnotatedHaps

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    from torch.nested import nested_tensor_from_jagged as nt_jag

__all__ = ["FlatIntervals", "Ragged", "RaggedIntervals", "RaggedTracks"]
INTERVAL_DTYPE = np.dtype(
    [("start", np.int32), ("end", np.int32), ("value", np.float32)], align=True
)


@dataclass(slots=True)
class RaggedIntervals:
    starts: Ragged[np.int32]
    ends: Ragged[np.int32]
    values: Ragged[np.float32]

    def __getitem__(self, idx) -> RaggedIntervals:
        out = RaggedIntervals(self.starts[idx], self.ends[idx], self.values[idx])  # type: ignore[bad-argument-type]  # _core.Ragged.__getitem__ return type widens to Array in stubs
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
        """Pack all arrays into contiguous buffers."""
        starts = self.starts.to_packed()
        ends = self.ends.to_packed()
        values = self.values.to_packed()
        return RaggedIntervals(starts, ends, values)

    def to_nested_tensor_batch(
        self, device: str | torch.device = "cpu"
    ) -> list[RagItvBatch]:
        out = []
        n_tracks = cast(int, self.values.shape[1])
        for t in range(n_tracks):
            # (batch tracks ... ~itv) -> (batch ... ~itv)
            starts = self.starts[:, t].to_packed()
            ends = self.ends[:, t].to_packed()
            values = self.values[:, t].to_packed()

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
        n = b * t

        def _pad(val, dtype):
            return Ragged.from_offsets(
                np.full(n, val, dtype),
                (b, t, None),
                np.arange(n + 1, dtype=np.int64),
            )

        # (b t ~v): prepend one pad element per group
        new_starts = spr.concatenate([_pad(start, np.int32), self.starts], axis=-1)
        new_ends = spr.concatenate([_pad(end, np.int32), self.ends], axis=-1)
        new_values = spr.concatenate([_pad(value, np.float32), self.values], axis=-1)

        return RaggedIntervals(new_starts, new_ends, new_values)


@dataclass(slots=True)
class FlatIntervals:
    """Flat-buffer analog of :class:`RaggedIntervals` over three :class:`_Flat` s.

    Pure-numpy ``(data, offsets, shape)`` per field; converts to the
    :class:`RaggedIntervals` only via :meth:`to_ragged`. Returned by eager indexing
    when ``with_tracks(kind="intervals")`` is combined with
    ``with_output_format("flat")``.
    """

    starts: _Flat
    ends: _Flat
    values: _Flat

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.values.shape

    def to_ragged(self) -> RaggedIntervals:
        return RaggedIntervals(
            self.starts.to_ragged(),
            self.ends.to_ragged(),
            self.values.to_ragged(),
        )

    def reshape(self, shape: int | tuple[int, ...]) -> "FlatIntervals":
        return FlatIntervals(
            self.starts.reshape(shape),
            self.ends.reshape(shape),
            self.values.reshape(shape),
        )

    def squeeze(self, axis: int | None = None) -> "FlatIntervals":
        return FlatIntervals(
            self.starts.squeeze(axis),
            self.ends.squeeze(axis),
            self.values.squeeze(axis),
        )


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
    """Densify a Ragged into a right-padded array via GVL's seqpro-core Rust bridge.

    Byte-identical to :func:`seqpro.rag.to_padded`; the inner row-copy runs in
    the shared seqpro-core kernel (Rust->Rust, no Python-seqpro round-trip).
    """
    if rag._is_record:
        raise NotImplementedError(
            "to_padded is not defined on record-layout Ragged arrays."
        )
    rag_dim = rag.rag_dim
    if any(d is not None for d in rag.shape[rag_dim + 1 :]):
        raise ValueError(
            f"to_padded requires the ragged axis to be last, got shape {rag.shape}."
        )
    if not rag.is_contiguous:
        rag = spr.to_packed(rag)

    offsets = np.ascontiguousarray(rag.offsets, dtype=np.int64)
    n_rows = offsets.shape[0] - 1
    out_len = int(rag.lengths.max()) if n_rows else 0

    rag_data: NDArray[Any] = rag.data  # record layout rejected above
    dtype = rag_data.dtype
    out = np.full((n_rows, out_len), pad_value, dtype=dtype)
    if n_rows and out_len:
        data_u1 = np.ascontiguousarray(rag_data).reshape(-1).view(np.uint8)
        out_u1 = out.reshape(-1).view(np.uint8)
        ragged_to_padded(data_u1, offsets, out_u1, dtype.itemsize, out_len)

    leading = rag.shape[:rag_dim]
    if leading:
        out = out.reshape((*leading, out_len))  # pyrefly: ignore[no-matching-overload]
    return out


_COMP = np.frombuffer(bytes.maketrans(b"ACGT", b"TGCA"), np.uint8)


def ufunc_comp_dna(seq: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return _COMP[seq]


def reverse_complement_masked(
    rag: Ragged[np.bytes_], mask: NDArray[np.bool_]
) -> Ragged[np.bytes_]:
    """Masked reverse-complement of an S1 ragged batch, in place.

    seqpro's flat kernel touches only the ``mask``-selected rows, runs a single in-place
    pass per row, and reuses ``rag``'s offsets. Uses :data:`_COMP` (the A<->T, C<->G
    lookup table) so output is byte-identical to a naive per-row reverse-complement.

    Mutates ``rag`` in place (``copy=False``); only call on a freshly reconstructed batch
    the caller owns.

    ``mask`` is one entry per outer query (e.g. per region); seqpro's flat kernel wants
    one entry per flattened ragged row, so replicate the mask across any inner fixed axes
    (e.g. ploidy) in C order to match.
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
