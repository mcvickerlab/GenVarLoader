"""Format-version gate and structural/size integrity checks for a gvl dataset.

A dataset cannot auto-rebuild (it retains no source), so every failure raises an
actionable `ValueError` instructing the user to regenerate with `gvl.write`.
Only total byte/length sizes are checked — full-content hashing of multi-GB
datasets is intentionally avoided (same tradeoff as the FASTA cache).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._write import DATASET_FORMAT_VERSION, Metadata

__all__ = ["validate_dataset"]


def validate_dataset(metadata: Metadata, path: Path) -> None:
    """Validate an on-disk dataset before constructing readers.

    Raises:
        ValueError: On an incompatible format version or a structural/size
            integrity failure.
    """
    _check_format_version(metadata, path)
    _check_integrity(metadata, path)


def _check_format_version(metadata: Metadata, path: Path) -> None:
    fmt = metadata.format_version
    if fmt is None:
        return  # legacy dataset: treat as the current layout, load best-effort
    if fmt.major != DATASET_FORMAT_VERSION.major:
        raise ValueError(
            f"Dataset at {path} has format version {fmt}, incompatible with this "
            f"genvarloader's format version {DATASET_FORMAT_VERSION}. Regenerate "
            f"the dataset with `gvl.write`"
            + (
                " or upgrade genvarloader."
                if fmt.major > DATASET_FORMAT_VERSION.major
                else "."
            )
        )


def _check_integrity(metadata: Metadata, path: Path) -> None:
    for required in ("metadata.json", "input_regions.arrow", "regions.npy"):
        if not (path / required).exists():
            raise ValueError(
                f"Dataset at {path} is missing required file '{required}'. "
                "Regenerate the dataset with `gvl.write`."
            )

    regions = np.load(path / "regions.npy", mmap_mode="r")
    if regions.shape != (metadata.n_regions, 4):
        raise ValueError(
            f"Dataset at {path}: regions.npy has shape {regions.shape}, expected "
            f"({metadata.n_regions}, 4). The dataset is corrupt or truncated; "
            "regenerate with `gvl.write`."
        )

    geno_offsets = path / "genotypes" / "offsets.npy"
    if geno_offsets.exists() and not (path / "genotypes" / "svar_meta.json").exists():
        # offsets.npy for VCF/PGEN genotypes is a raw int64 memmap (no numpy
        # header), so we check its size in bytes rather than calling np.load.
        if metadata.ploidy is None:
            raise ValueError(
                f"Dataset at {path} has genotypes but no ploidy in metadata; "
                "regenerate with `gvl.write`."
            )
        expected = metadata.n_regions * metadata.ploidy * metadata.n_samples + 1
        actual_bytes = geno_offsets.stat().st_size
        expected_bytes = expected * np.dtype(np.int64).itemsize
        if actual_bytes != expected_bytes:
            actual_len = actual_bytes // np.dtype(np.int64).itemsize
            raise ValueError(
                f"Dataset at {path}: genotypes/offsets.npy has length "
                f"{actual_len}, expected {expected} "
                f"(n_regions * ploidy * n_samples + 1). The dataset is corrupt or "
                "truncated; regenerate with `gvl.write`."
            )
