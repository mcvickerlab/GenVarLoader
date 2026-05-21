"""Resolution and integrity for the GVL dataset → SVAR back-reference."""

from __future__ import annotations

from pydantic import BaseModel


class SvarFingerprint(BaseModel):
    n_variants: int
    variant_idxs_bytes: int


class SvarLink(BaseModel):
    relative_path: str
    absolute_path: str
    fingerprint: SvarFingerprint
