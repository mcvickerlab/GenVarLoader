"""Tabular interval track source for :func:`gvl.write()`.

Mirrors the :class:`BigWigs` reader API surface so that
:func:`genvarloader._dataset._write._write_track` can dispatch to either.
Overlap queries are served by the Rust ``RustTable`` (COITrees) backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from genoray._contigs import ContigNormalizer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._ragged import RaggedIntervals


CANONICAL_COLS = ("sample_id", "chrom", "start", "end", "value")


class Table:
    """Long-form interval track keyed by ``(sample_id, chrom, start, end, value)``.

    Overlap queries are served by a Rust COITrees backend. Coordinates are
    zero-based, half-open ``[start, end)``.
    """

    name: str
    samples: list[str]
    contigs: Mapping[str, int]

    def __init__(
        self,
        name: str,
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None = None,
    ) -> None:
        from .genvarloader import RustTable

        self.name = name
        df = self._normalize_input(data, column_map)
        df = df.cast(
            {
                "sample_id": pl.Utf8,
                "chrom": pl.Utf8,
                "start": pl.Int32,
                "end": pl.Int32,
                "value": pl.Float32,
            }
        ).sort("chrom", "sample_id", "start")
        self._df = df
        self.samples = sorted(df["sample_id"].unique().to_list())
        self.contigs = {
            row["chrom"]: int(row["max_end"])
            for row in df.group_by("chrom")
            .agg(pl.col("end").max().alias("max_end"))
            .iter_rows(named=True)
        }

        # Factor-encode for the Rust store. Contig order is the dict insertion
        # order of `self.contigs`; sample order is `self.samples`.
        self._contig_list = list(self.contigs.keys())
        self._cnorm = ContigNormalizer(self._contig_list)
        sample_to_code = {s: i for i, s in enumerate(self.samples)}
        sample_codes = (
            df.select(
                pl.col("sample_id").replace_strict(
                    sample_to_code, return_dtype=pl.Int32
                )
            )
            .to_series()
            .to_numpy()
        )
        chrom_to_code = {c: i for i, c in enumerate(self._contig_list)}
        chrom_codes = (
            df.select(
                pl.col("chrom").replace_strict(chrom_to_code, return_dtype=pl.Int32)
            )
            .to_series()
            .to_numpy()
        )
        self._rust = RustTable(
            np.ascontiguousarray(sample_codes, dtype=np.int32),
            np.ascontiguousarray(chrom_codes, dtype=np.int32),
            np.ascontiguousarray(df["start"].to_numpy(), dtype=np.int32),
            np.ascontiguousarray(df["end"].to_numpy(), dtype=np.int32),
            np.ascontiguousarray(df["value"].to_numpy(), dtype=np.float32),
            len(self.samples),
            len(self._contig_list),
        )

    @classmethod
    def from_path(
        cls,
        name: str,
        path: str | Path | Mapping[str, str | Path],
        column_map: Mapping[str, str] | None = None,
    ) -> Table:
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
            frames: list[pl.DataFrame] = []
            for sid, sub in data.items():
                renamed = Table._apply_column_map(
                    sub, column_map, expect_sample_id=False
                )
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
        rename = {
            actual: canonical
            for canonical, actual in column_map.items()
            if actual in df.columns
        }
        if not expect_sample_id:
            rename.pop("sample_id", None)
        return df.rename(rename)

    def _resolve_samples(self, sample: str | list[str] | None) -> list[str]:
        if sample is None:
            return list(self.samples)
        if isinstance(sample, str):
            samples = [sample]
        else:
            samples = list(sample)
        if missing := set(samples) - set(self.samples):
            raise ValueError(f"Sample(s) {missing} not found in Table.")
        return samples

    def _chrom_code(self, contig: str) -> int:
        """Resolve `contig` to its code in this Table, or -1 if absent."""
        norm = self._cnorm.norm(contig)
        if norm is None:
            return -1
        return self._contig_list.index(norm)

    def _sample_codes(self, samples: list[str]) -> NDArray[np.int32]:
        s2c = {s: i for i, s in enumerate(self.samples)}
        return np.array([s2c[s] for s in samples], dtype=np.int32)

    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        samples = self._resolve_samples(sample)
        starts_arr = np.ascontiguousarray(np.atleast_1d(starts), dtype=np.int32)
        ends_arr = np.ascontiguousarray(np.atleast_1d(ends), dtype=np.int32)
        return self._rust.count(
            self._chrom_code(contig),
            starts_arr,
            ends_arr,
            self._sample_codes(samples),
        )

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> RaggedIntervals:
        from seqpro.rag import Ragged

        from ._ragged import RaggedIntervals

        samples = self._resolve_samples(sample)
        starts_arr = np.ascontiguousarray(np.atleast_1d(starts), dtype=np.int32)
        ends_arr = np.ascontiguousarray(np.atleast_1d(ends), dtype=np.int32)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        n_regions = len(starts_arr)
        n_samples = len(samples)
        shape = (n_regions, n_samples, None)

        coords, values = self._rust.intervals(
            self._chrom_code(contig),
            starts_arr,
            ends_arr,
            self._sample_codes(samples),
            offsets,
        )
        flat_starts = np.ascontiguousarray(coords[:, 0], dtype=np.int32)
        flat_ends = np.ascontiguousarray(coords[:, 1], dtype=np.int32)
        return RaggedIntervals(
            Ragged.from_offsets(flat_starts, shape, offsets),
            Ragged.from_offsets(flat_ends, shape, offsets),
            Ragged.from_offsets(values, shape, offsets),
        )


def annot_overlap(regions: pl.DataFrame, annot: pl.DataFrame) -> "RaggedIntervals":
    """Sample-less interval overlap of `regions` (chrom/chromStart/chromEnd) against a
    BED-like `annot` (chrom/chromStart/chromEnd/score). Returns a RaggedIntervals of
    shape (n_regions, None) ordered by (region, start), via the Rust COITrees backend."""
    from seqpro.rag import Ragged

    from ._ragged import RaggedIntervals
    from ._utils import lengths_to_offsets

    if annot.height == 0:
        n_regions = regions.height
        offsets = np.zeros(n_regions + 1, dtype=np.int64)
        shape = (n_regions, None)
        return RaggedIntervals(
            Ragged.from_offsets(np.empty(0, dtype=np.int32), shape, offsets),
            Ragged.from_offsets(np.empty(0, dtype=np.int32), shape, offsets),
            Ragged.from_offsets(np.empty(0, dtype=np.float32), shape, offsets),
        )

    annot_long = annot.select(
        pl.lit("__annot__").alias("sample_id"),
        "chrom",
        pl.col("chromStart").alias("start"),
        pl.col("chromEnd").alias("end"),
        pl.col("score").alias("value"),
    )
    table = Table("__annot__", annot_long)

    n_regions = regions.height
    # Per-region interval arrays indexed by input region position.
    per_start: list[NDArray[np.int32]] = [np.empty(0, np.int32)] * n_regions
    per_end: list[NDArray[np.int32]] = [np.empty(0, np.int32)] * n_regions
    per_val: list[NDArray[np.float32]] = [np.empty(0, np.float32)] * n_regions

    reg = regions.with_row_index("_q")
    for (contig,), part in reg.partition_by(
        "chrom", as_dict=True, include_key=False, maintain_order=True
    ).items():
        contig = str(contig)
        q_idx = part["_q"].to_numpy()
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()
        counts = table.count_intervals(contig, starts, ends, sample=["__annot__"])
        offsets = lengths_to_offsets(counts.ravel())
        itvs = table._intervals_from_offsets(
            contig, starts, ends, offsets, sample=["__annot__"]
        ).squeeze(1)
        for j, qi in enumerate(q_idx):
            per_start[qi] = np.asarray(itvs.starts[j], dtype=np.int32)
            per_end[qi] = np.asarray(itvs.ends[j], dtype=np.int32)
            per_val[qi] = np.asarray(itvs.values[j], dtype=np.float32)

    lengths = np.array([len(s) for s in per_start], dtype=np.int32)
    offsets = lengths_to_offsets(lengths)
    flat_starts = np.concatenate(per_start) if n_regions else np.empty(0, np.int32)
    flat_ends = np.concatenate(per_end) if n_regions else np.empty(0, np.int32)
    flat_values = np.concatenate(per_val) if n_regions else np.empty(0, np.float32)
    shape = (n_regions, None)
    return RaggedIntervals(
        Ragged.from_offsets(flat_starts, shape, offsets),
        Ragged.from_offsets(flat_ends, shape, offsets),
        Ragged.from_offsets(flat_values, shape, offsets),
    )
