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

from ._utils import normalize_contig_name

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

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

    def count_intervals(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        sample: "str | list[str] | None" = None,
        **kwargs,
    ) -> "NDArray[np.int32]":
        import numpy as np
        import polars_bio as pb

        # BED data is always 0-based half-open; configure polars-bio accordingly.
        # Both calls are idempotent and safe to repeat.
        pb.set_option("datafusion.bio.coordinate_system_check", "false")
        pb.set_option("datafusion.bio.coordinate_system_zero_based", True)

        samples = self._resolve_samples(sample)
        starts_arr = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends_arr = np.atleast_1d(np.asarray(ends, dtype=np.int64))
        n_regions = len(starts_arr)
        n_samples = len(samples)

        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is None:
            return np.zeros((n_regions, n_samples), dtype=np.int32)
        contig = _contig

        contig_subset = self._df.filter(pl.col("chrom") == contig)
        if contig_subset.height == 0:
            return np.zeros((n_regions, n_samples), dtype=np.int32)

        queries = pl.DataFrame({
            "chrom": np.repeat(np.array([contig], dtype=object), n_regions),
            "start": starts_arr,
            "end": ends_arr,
            "_q": np.arange(n_regions, dtype=np.int64),
        })

        # polars-bio v0.20.1 does not yet support on_cols, so loop per sample.
        out = np.zeros((n_regions, n_samples), dtype=np.int32)
        for si, s in enumerate(samples):
            sub_s = contig_subset.filter(pl.col("sample_id") == s).select("chrom", "start", "end")
            if sub_s.height == 0:
                continue
            counts_df = pb.count_overlaps(
                queries,
                sub_s,
                cols1=["chrom", "start", "end"],
                cols2=["chrom", "start", "end"],
                output_type="polars.DataFrame",
            )
            # Schema: (chrom, start, end, _q, count). Order matches queries (zero-filled).
            # Sort by _q to ensure alignment with output array, in case order differs.
            sorted_counts = counts_df.sort("_q")["count"].to_numpy().astype(np.int32, copy=False)
            if len(sorted_counts) < n_regions:
                # Safety fallback: use left-join in case zero-fill wasn't complete.
                idx_df = pl.DataFrame({"_q": np.arange(n_regions, dtype=np.int64)})
                filled = idx_df.join(counts_df.select("_q", "count"), on="_q", how="left").fill_null(0)
                sorted_counts = filled["count"].to_numpy().astype(np.int32, copy=False)
            out[:, si] = sorted_counts
        return out

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: "ArrayLike",
        ends: "ArrayLike",
        offsets: "NDArray[np.int64]",
        sample: "str | list[str] | None" = None,
        **kwargs,
    ) -> "RaggedIntervals":
        import numpy as np
        import polars_bio as pb
        from seqpro.rag import Ragged

        from ._ragged import RaggedIntervals

        pb.set_option("datafusion.bio.coordinate_system_check", "false")

        samples = self._resolve_samples(sample)
        starts_arr = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends_arr = np.atleast_1d(np.asarray(ends, dtype=np.int64))
        n_regions = len(starts_arr)
        n_samples = len(samples)
        shape = (n_regions, n_samples, None)

        total = int(offsets[-1])
        flat_starts = np.empty(total, dtype=np.int32)
        flat_ends = np.empty(total, dtype=np.int32)
        flat_values = np.empty(total, dtype=np.float32)

        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is not None and total > 0:
            contig = _contig
            contig_subset = self._df.filter(pl.col("chrom") == contig)
            if contig_subset.height > 0:
                queries = pl.DataFrame({
                    "chrom": np.repeat(np.array([contig], dtype=object), n_regions),
                    "start": starts_arr,
                    "end": ends_arr,
                    "_q": np.arange(n_regions, dtype=np.int64),
                })
                # Loop per sample: polars-bio v0.20.1 doesn't support on_cols.
                for si, s in enumerate(samples):
                    sub_s = (
                        contig_subset.filter(pl.col("sample_id") == s)
                        .select("chrom", "start", "end", "value")
                    )
                    if sub_s.height == 0:
                        continue
                    joined = pb.overlap(
                        queries,
                        sub_s,
                        cols1=["chrom", "start", "end"],
                        cols2=["chrom", "start", "end"],
                        output_type="polars.DataFrame",
                    )
                    if joined.height == 0:
                        continue
                    # Sort by query index, then by table-side start to match BigWigs order.
                    joined = joined.sort("_q_1", "start_2")
                    q_idx = joined["_q_1"].to_numpy()
                    j_starts = joined["start_2"].to_numpy().astype(np.int32, copy=False)
                    j_ends   = joined["end_2"].to_numpy().astype(np.int32, copy=False)
                    j_values = joined["value_2"].to_numpy().astype(np.float32, copy=False)

                    # Place each row at offsets[r*n_samples + si] + intra_cell_idx.
                    if len(q_idx) > 0:
                        cell_idx = q_idx * n_samples + si
                        # Build intra-cell running index: reset per cell boundary.
                        boundaries = np.concatenate(([0], np.where(np.diff(cell_idx) != 0)[0] + 1))
                        counts_per_cell = np.diff(np.concatenate((boundaries, [len(cell_idx)])))
                        intra = np.concatenate([np.arange(c) for c in counts_per_cell])
                        write_pos = offsets[cell_idx] + intra
                        flat_starts[write_pos] = j_starts
                        flat_ends[write_pos] = j_ends
                        flat_values[write_pos] = j_values

        return RaggedIntervals(
            Ragged.from_offsets(flat_starts, shape, offsets),
            Ragged.from_offsets(flat_ends, shape, offsets),
            Ragged.from_offsets(flat_values, shape, offsets),
        )

    def _resolve_samples(self, sample: "str | list[str] | None") -> "list[str]":
        if sample is None:
            return list(self.samples)
        if isinstance(sample, str):
            samples = [sample]
        else:
            samples = list(sample)
        if missing := set(samples) - set(self.samples):
            raise ValueError(f"Sample(s) {missing} not found in Table.")
        return samples
