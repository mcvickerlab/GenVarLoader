"""Reference-only reconstructor.

Produces reference-sequence bytes for the given regions, with optional
splice-plan permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import Ragged

from .._flat import _Flat
from .._utils import lengths_to_offsets
from ._protocol import Reconstructor
from ._reference import Reference, _fetch_spliced_ref, get_reference
from ._splice import SplicePlan


@dataclass(slots=True)
class Ref(Reconstructor[Ragged[np.bytes_]]):
    reference: Reference
    """The reference genome. This is kept in memory."""

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
        to_rc: "NDArray[np.bool_] | None" = None,
    ) -> Ragged[np.bytes_]:
        batch_size = len(idx)

        if isinstance(output_length, int):
            # (b)
            out_lengths = np.full(batch_size, output_length, dtype=np.int32)
            regions = regions.copy()
            regions[:, 2] = regions[:, 1] + out_lengths
        else:
            lengths = regions[:, 2] - regions[:, 1]
            out_lengths = lengths

        if splice_plan is None:
            # (b+1)
            out_offsets = lengths_to_offsets(out_lengths)

            # ragged (b ~l) — on Rust backend, RC is folded into the kernel.
            ref = get_reference(
                regions=regions,
                out_offsets=out_offsets,
                reference=self.reference.reference,
                ref_offsets=self.reference.offsets,
                pad_char=self.reference.pad_char,
                to_rc=to_rc,
            )  # uint8 flat buffer

            return cast(
                "Ragged[np.bytes_]",
                _Flat.from_offsets(ref, (batch_size, None), out_offsets).view("S1"),
            )

        # Spliced path: delegate to the shared kernel-dispatch helper.
        # to_rc is the permuted per-element mask from _getitem_spliced.
        return _fetch_spliced_ref(
            regions=regions,
            plan=splice_plan,
            reference=self.reference.reference,
            ref_offsets=self.reference.offsets,
            pad_char=self.reference.pad_char,
            to_rc=to_rc,
        )
