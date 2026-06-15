"""Tabular interval track source for :func:`gvl.write()`.

Mirrors the :class:`BigWigs` reader API surface so that
:func:`genvarloader._dataset._write._write_track` can dispatch to either.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ._utils import normalize_contig_name

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from ._ragged import RaggedIntervals


CANONICAL_COLS = ("sample_id", "chrom", "start", "end", "value")


class ExperimentalWarning(UserWarning):
    """Warning emitted when using an experimental, unsupported feature."""


#: ``gvl.Table`` is **experimental** and not exercised in CI. Its overlap backend,
#: ``polars-bio``, has intermittently segfaulted the interpreter during overlap
#: queries (a non-deterministic native-runtime issue observed on CPython 3.12 and
#: 3.13; upstream https://github.com/biodatageeks/polars-bio/issues/395). It has
#: been stable enough in production use to ship as an opt-in experimental feature.
#: ``polars-bio`` is not a core dependency of genvarloader; install it via the
#: ``table`` extra (``pip install genvarloader[table]``) to use ``gvl.Table``.
_TABLE_EXPERIMENTAL_MSG = (
    "gvl.Table is an experimental feature and is not tested in CI. Its overlap "
    "backend, polars-bio, has intermittently segfaulted during overlap queries "
    "(upstream https://github.com/biodatageeks/polars-bio/issues/395). It is "
    "considered stable enough for opt-in production use, but prefer gvl.BigWigs "
    "when possible."
)

_POLARS_BIO_MISSING_MSG = (
    "gvl.Table requires the optional 'polars-bio' package, which is not "
    "installed. Install it via the 'table' extra: "
    "`pip install genvarloader[table]`."
)


def _import_polars_bio():
    try:
        import polars_bio as pb
    except ImportError as e:  # pragma: no cover - exercised only without polars-bio
        raise ImportError(_POLARS_BIO_MISSING_MSG) from e
    return pb


class Table:
    """Long-form interval track keyed by ``(sample_id, chrom, start, end, value)``.

    .. warning::
        **Experimental** and not tested in CI. The overlap backend
        (``polars-bio``) has intermittently segfaulted the interpreter
        (`#395 <https://github.com/biodatageeks/polars-bio/issues/395>`_). It is
        stable enough for opt-in production use, but prefer :class:`BigWigs` when
        possible. Install the overlap backend via the ``table`` extra:
        ``pip install genvarloader[table]``.
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
        warnings.warn(_TABLE_EXPERIMENTAL_MSG, ExperimentalWarning, stacklevel=2)
        self.name = name
        df = self._normalize_input(data, column_map)
        df = df.cast(
            {
                "sample_id": pl.Utf8,
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
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
            # dict[sample_id, df] without sample_id col
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
        # column_map is canonical -> actual; invert to actual -> canonical for rename
        rename = {
            actual: canonical
            for canonical, actual in column_map.items()
            if actual in df.columns
        }
        if not expect_sample_id:
            rename.pop("sample_id", None)
        return df.rename(rename)

    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        pb = _import_polars_bio()

        # pb.set_option is idempotent; called per-method to avoid relying on import order.
        pb.set_option("datafusion.bio.coordinate_system_check", "false")
        pb.set_option("datafusion.bio.coordinate_system_zero_based", True)

        samples = self._resolve_samples(sample)
        sample_to_si = {s: i for i, s in enumerate(samples)}
        starts_arr = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends_arr = np.atleast_1d(np.asarray(ends, dtype=np.int64))
        n_regions = len(starts_arr)
        n_samples = len(samples)

        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is None:
            return np.zeros((n_regions, n_samples), dtype=np.int32)
        contig = _contig

        contig_subset = self._df.filter(
            (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
        )
        if contig_subset.height == 0:
            return np.zeros((n_regions, n_samples), dtype=np.int32)

        queries = pl.DataFrame(
            {
                "chrom": [contig] * n_regions,
                "start": starts_arr,
                "end": ends_arr,
                "_q": np.arange(n_regions, dtype=np.int64),
            }
        )
        result = pb.overlap(
            queries,
            contig_subset.select("chrom", "start", "end", "sample_id"),
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            output_type="polars.DataFrame",
        )
        out = np.zeros((n_regions, n_samples), dtype=np.int32)
        if result.height == 0:
            return out
        q_idx = result["_q_1"].to_numpy()
        si_idx = (
            result.select(
                pl.col("sample_id_2").replace_strict(
                    sample_to_si, return_dtype=pl.Int64
                )
            )
            .to_series()
            .to_numpy()
        )
        np.add.at(out, (q_idx, si_idx), 1)
        return out

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> RaggedIntervals:
        pb = _import_polars_bio()
        from seqpro.rag import Ragged

        from ._ragged import RaggedIntervals

        # pb.set_option is idempotent; called per-method to avoid relying on import order.
        pb.set_option("datafusion.bio.coordinate_system_check", "false")
        pb.set_option("datafusion.bio.coordinate_system_zero_based", True)

        samples = self._resolve_samples(sample)
        sample_to_si = {s: i for i, s in enumerate(samples)}
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
            contig_subset = self._df.filter(
                (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
            )
            if contig_subset.height > 0:
                queries = pl.DataFrame(
                    {
                        "chrom": [contig] * n_regions,
                        "start": starts_arr,
                        "end": ends_arr,
                        "_q": np.arange(n_regions, dtype=np.int64),
                    }
                )
                joined = pb.overlap(
                    queries,
                    contig_subset.select("chrom", "start", "end", "sample_id", "value"),
                    cols1=["chrom", "start", "end"],
                    cols2=["chrom", "start", "end"],
                    output_type="polars.DataFrame",
                )
                if joined.height > 0:
                    # Sort by query index, sample index, then table start (matches BigWigs order).
                    si_idx = (
                        joined.select(
                            pl.col("sample_id_2").replace_strict(
                                sample_to_si, return_dtype=pl.Int64
                            )
                        )
                        .to_series()
                        .to_numpy()
                    )
                    q_idx = joined["_q_1"].to_numpy()
                    j_starts_raw = joined["start_2"].to_numpy()
                    order = np.lexsort(
                        (
                            j_starts_raw,
                            si_idx,
                            q_idx,
                        )
                    )  # last key = primary
                    q_idx = q_idx[order]
                    si_idx = si_idx[order]
                    j_starts = j_starts_raw[order].astype(np.int32, copy=False)
                    j_ends = (
                        joined["end_2"].to_numpy()[order].astype(np.int32, copy=False)
                    )
                    j_values = (
                        joined["value_2"]
                        .to_numpy()[order]
                        .astype(np.float32, copy=False)
                    )

                    cell_idx = q_idx * n_samples + si_idx
                    boundaries = np.concatenate(
                        (
                            [0],
                            np.where(np.diff(cell_idx) != 0)[0] + 1,
                        )
                    )
                    counts_per_cell = np.diff(
                        np.concatenate((boundaries, [len(cell_idx)]))
                    )
                    intra = np.arange(len(cell_idx)) - np.repeat(
                        boundaries, counts_per_cell
                    )
                    write_pos = offsets[cell_idx] + intra
                    flat_starts[write_pos] = j_starts
                    flat_ends[write_pos] = j_ends
                    flat_values[write_pos] = j_values

        return RaggedIntervals(
            Ragged.from_offsets(flat_starts, shape, offsets),
            Ragged.from_offsets(flat_ends, shape, offsets),
            Ragged.from_offsets(flat_values, shape, offsets),
        )

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
