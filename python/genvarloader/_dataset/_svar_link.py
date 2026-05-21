"""Resolution and integrity for the GVL dataset → SVAR back-reference."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class SvarFingerprint(BaseModel):
    n_variants: int
    variant_idxs_bytes: int


class SvarLink(BaseModel):
    relative_path: str
    absolute_path: str
    fingerprint: SvarFingerprint


def _resolve_svar(
    gvl_path: Path,
    link: SvarLink | None,
    override: Path | str | None,
) -> Path:
    """Resolve the SVAR directory referenced by a GVL dataset.

    Order: override → link.relative_path → link.absolute_path → sibling *.svar.
    Raises FileNotFoundError if none resolve to a directory.
    """
    if override is not None:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(
                f"svar override path does not exist or is not a directory: {p}"
            )
        return p

    if link is not None:
        rel = (gvl_path / link.relative_path).resolve()
        if rel.is_dir():
            return rel
        absp = Path(link.absolute_path)
        if absp.is_dir():
            return absp

    siblings = sorted(gvl_path.parent.glob("*.svar"))
    if len(siblings) == 1:
        return siblings[0]

    expected = Path(link.absolute_path).name if link is not None else "<unknown>.svar"
    raise FileNotFoundError(
        f"Could not locate svar '{expected}' for GVL dataset at {gvl_path}. "
        f"Tried: stored relative path, stored absolute path, sibling *.svar. "
        f"Pass `svar=` to `Dataset.open(...)` to override."
    )


def _verify_fingerprint(svar_path: Path, link: SvarLink | None) -> None:
    """Compare the recorded fingerprint against the resolved svar.

    No-op when ``link`` is None (legacy dataset).
    Raises ValueError on mismatch, FileNotFoundError on missing variant_idxs.npy.
    """
    if link is None:
        return

    variant_idxs = svar_path / "variant_idxs.npy"
    if not variant_idxs.exists():
        raise FileNotFoundError(
            f"Expected variant_idxs.npy at {variant_idxs}; "
            f"resolved svar is malformed."
        )

    observed_bytes = variant_idxs.stat().st_size

    import polars as pl

    n_variants_observed = (
        pl.scan_ipc(svar_path / "index.arrow").select(pl.len()).collect().item()
    )

    exp = link.fingerprint
    mismatches: list[str] = []
    if n_variants_observed != exp.n_variants:
        mismatches.append(
            f"n_variants: expected {exp.n_variants}, observed {n_variants_observed}"
        )
    if observed_bytes != exp.variant_idxs_bytes:
        mismatches.append(
            f"variant_idxs_bytes: expected {exp.variant_idxs_bytes}, "
            f"observed {observed_bytes}"
        )
    if mismatches:
        raise ValueError(
            f"svar fingerprint mismatch at {svar_path}: " + "; ".join(mismatches)
        )
