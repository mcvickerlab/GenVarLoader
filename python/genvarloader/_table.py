"""Tabular interval track source for :func:`gvl.write()`.

Mirrors the :class:`BigWigs` reader API surface so that
:func:`genvarloader._dataset._write._write_track` can dispatch to either.

polars-bio (v0.20.1) API findings used by Tasks 8/9:
- `on_cols` is NOT yet supported — AssertionError. Filter by sample manually
  and loop+concat across samples.
- `pb.overlap` suffixes all columns: `<col>_1` for left (queries),
  `<col>_2` for right (table). Output column order: df1 coords, df2 coords,
  df2 extras, df1 extras.
- `pb.count_overlaps` zero-fills queries with no matches; output has a
  `count: Int64` column and no suffixing on the left side.
- Coordinate system metadata: set per-DataFrame via
  `df.config_meta.set(coordinate_system_zero_based=True)` (BED is 0-based
  half-open) OR globally via
  `pb.set_option("datafusion.bio.coordinate_system_check", "false")`.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ._ragged import RaggedIntervals


CANONICAL_COLS = ("sample_id", "chrom", "start", "end", "value")


class Table:
    """Long-form interval track keyed by ``(sample_id, chrom, start, end, value)``."""

    name: str
    samples: list[str]
    contigs: dict[str, int]

    def __init__(
        self,
        name: str,
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        df = self._normalize_input(data, column_map)
        df = df.cast({
            "sample_id": pl.Utf8,
            "chrom": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "value": pl.Float32,
        }).sort("chrom", "sample_id", "start")
        self._df = df
        self.samples = sorted(df["sample_id"].unique().to_list())
        self.contigs = {
            row["chrom"]: int(row["max_end"])
            for row in df.group_by("chrom")
            .agg(pl.col("end").max().alias("max_end"))
            .iter_rows(named=True)
        }

    @classmethod
    def from_path(
        cls,
        name: str,
        path: str | Path | Mapping[str, str | Path],
        column_map: Mapping[str, str] | None = None,
    ) -> "Table":
        if isinstance(path, Mapping):
            data: dict[str, pl.DataFrame] = {
                sid: cls._read_path(Path(p)) for sid, p in path.items()
            }
            return cls(name, data, column_map)
        return cls(name, cls._read_path(Path(path)), column_map)

    @staticmethod
    def _read_path(p: Path) -> pl.DataFrame:
        suf = p.suffix.lower()
        if suf == ".csv":
            return pl.read_csv(p)
        if suf in (".tsv", ".txt"):
            return pl.read_csv(p, separator="\t")
        if suf == ".parquet":
            return pl.read_parquet(p)
        if suf in (".arrow", ".ipc"):
            return pl.read_ipc(p)
        raise ValueError(
            f"Unsupported file extension {suf!r}. "
            "Expected one of .csv, .tsv, .txt, .parquet, .arrow, .ipc."
        )

    @staticmethod
    def _normalize_input(
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None,
    ) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            df = Table._apply_column_map(data, column_map, expect_sample_id=True)
        else:
            # dict[sample_id, df] without sample_id col
            frames: list[pl.DataFrame] = []
            for sid, sub in data.items():
                renamed = Table._apply_column_map(sub, column_map, expect_sample_id=False)
                frames.append(renamed.with_columns(sample_id=pl.lit(sid)))
            if not frames:
                raise ValueError("Empty mapping passed to Table.")
            df = pl.concat(frames, how="vertical_relaxed")
        missing = [c for c in CANONICAL_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) {missing}. "
                f"Use `column_map` to rename if your columns differ from {CANONICAL_COLS}."
            )
        return df.select(*CANONICAL_COLS)

    @staticmethod
    def _apply_column_map(
        df: pl.DataFrame,
        column_map: Mapping[str, str] | None,
        expect_sample_id: bool,
    ) -> pl.DataFrame:
        if not column_map:
            return df
        # column_map is canonical -> actual; invert to actual -> canonical for rename
        rename = {actual: canonical for canonical, actual in column_map.items() if actual in df.columns}
        if not expect_sample_id:
            rename.pop("sample_id", None)
        return df.rename(rename)
