"""Emit the pre-0.43.0 dense `svar2_ranges` layout, for backward-compat tests.

`gvl.write` emits only the sparse layout (#357), so without this helper the dense
reader and the dense-input concat path would ship with no coverage at all.

Deliberately a test helper and NOT a `layout=` kwarg on `gvl.write`: a deprecated
on-disk format should not be reachable from the public API. Deliberately not a
checked-in binary fixture either: that would rot silently as the meta evolves.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def rewrite_as_dense(dataset: Path, out: Path) -> Path:
    """Copy a sparse `.gvl` dataset, re-emitting its range cache as dense.

    Args:
        dataset: A dataset written by `gvl.write` (sparse layout).
        out: Destination directory; must not exist.

    Returns:
        `out`, a byte-for-byte copy except that `genotypes/svar2_ranges/` holds
        the legacy dense layout.
    """
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    shutil.copytree(dataset, out)
    rd = out / "genotypes" / "svar2_ranges"
    reader = _ranges_reader(rd)
    R, S, P = reader.n_regions, reader.n_samples, reader.ploidy

    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    snp, indel = reader.lookup(r_q, si_q, P)
    snp.reshape(R, S, P, 2).tofile(rd / "vk_snp_range.npy")
    indel.reshape(R, S, P, 2).tofile(rd / "vk_indel_range.npy")

    for name in ("region_ptr.npy", "cell_id.npy", "cell_vk.npy"):
        (rd / name).unlink()

    meta = json.loads((rd / "svar2_meta.json").read_text())
    for k in (
        "layout",
        "n_regions",
        "n_samples",
        "n_entries",
        "fill",
        "region_ptr",
        "cell_id",
        "cell_vk",
    ):
        meta.pop(k, None)
    meta["vk_snp_range"] = {"shape": [R, S, P, 2], "dtype": "<i8"}
    meta["vk_indel_range"] = {"shape": [R, S, P, 2], "dtype": "<i8"}
    (rd / "svar2_meta.json").write_text(json.dumps(meta))
    return out
