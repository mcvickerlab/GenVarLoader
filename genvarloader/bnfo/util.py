from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandera as pa
import pandera.typing as pat
import polars as pl


def _set_fixed_length_around_center(bed: pl.DataFrame, length: int):
    if "peak" in bed.columns:
        center = pl.col("chromStart") + pl.col("peak")
    else:
        center = (pl.col("chromStart") + pl.col("chromEnd")) / 2
    bed = bed.with_columns(
        chromStart=(center - length / 2).round(0).cast(pl.Int64),
        chromEnd=(center + length / 2).round(0).cast(pl.Int64),
    )
    return bed


def read_bedlike(path: Union[str, Path]) -> pl.DataFrame:
    """Reads a bed-like (BED3+) file as a pandas DataFrame. The file type is inferred
    from the file extension.

    Parameters
    ----------
    path : Union[str, Path]

    Returns
    -------
    polars.DataFrame
    """
    path = Path(path)
    if ".bed" in path.suffixes:
        return _read_bed(path)
    elif ".narrowPeak" in path.suffixes:
        return _read_narrowpeak(path)
    elif ".broadPeak" in path.suffixes:
        return _read_broadpeak(path)
    else:
        try:
            return _read_bed_table(path)
        except ValueError:
            raise ValueError(
                f"""Unrecognized file extension: {''.join(path.suffixes)}. Expected one 
                of .bed, .narrowPeak, .broadPeak, or a table file (e.g. .csv, .tsv)"""
            )


class BEDSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: Optional[pat.Series[str]] = pa.Field(nullable=True)
    score: Optional[pat.Series[float]] = pa.Field(nullable=True)
    strand: Optional[pat.Series[str]] = pa.Field(isin=["+", "-", "."], nullable=True)
    thickStart: Optional[pat.Series[int]] = pa.Field(nullable=True)
    thickEnd: Optional[pat.Series[int]] = pa.Field(nullable=True)
    itemRgb: Optional[pat.Series[str]] = pa.Field(nullable=True)
    blockCount: Optional[pat.Series[pa.UInt]] = pa.Field(nullable=True)
    blockSizes: Optional[pat.Series[str]] = pa.Field(nullable=True)
    blockStarts: Optional[pat.Series[str]] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_bed(bed_path: Union[str, Path]):
    with open(bed_path) as f:
        skip_rows = 0
        while (line := f.readline()).startswith(("track", "browser")):
            skip_rows += 1
    n_cols = line.count("\t") + 1
    bed_cols = [
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "score",
        "strand",
        "thickStart",
        "thickEnd",
        "itemRgb",
        "blockCount",
        "blockSizes",
        "blockStarts",
    ]
    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=bed_cols[:n_cols],
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    bed = BEDSchema.to_schema()(bed)
    return pl.from_pandas(bed)


def _read_bed_table(table: Union[str, Path], **table_reader_kwargs):
    table = Path(table)
    suffixes = set(table.suffixes)
    reader_kwargs: Dict[str, Any] = {}
    reader_kwargs.update(table_reader_kwargs)
    if ".csv" in suffixes:
        reader_kwargs["separator"] = ","
        reader_kwargs["dtypes"] = {"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8}
        reader = pl.scan_csv
    elif {".txt", ".tsv"} & suffixes:
        reader_kwargs["separator"] = "\t"
        reader_kwargs["dtypes"] = {"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8}
        reader = pl.scan_csv
    elif {".fth", ".feather", ".ipc", ".arrow"} & suffixes:
        reader = pl.scan_ipc
    else:
        raise ValueError(f"Table has unrecognized file extension: {table.name}")
    bed = reader(table, **reader_kwargs).collect().to_pandas()
    bed = BEDSchema.to_schema()(bed)
    return pl.from_pandas(bed)


class NarrowPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str] = pa.Field(nullable=True)
    score: pat.Series[float] = pa.Field(nullable=True)
    strand: pat.Series[str] = pa.Field(isin=["+", "-", "."], nullable=True)
    signalValue: pat.Series[float] = pa.Field(nullable=True)
    pValue: pat.Series[float] = pa.Field(nullable=True)
    qValue: pat.Series[float] = pa.Field(nullable=True)
    peak: pat.Series[int] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_narrowpeak(narrowpeak_path: Union[str, Path]):
    with open(narrowpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    narrowpeaks = pl.read_csv(
        narrowpeak_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
            "peak",
        ],
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    narrowpeaks = NarrowPeakSchema.to_schema()(narrowpeaks)
    return pl.from_pandas(narrowpeaks)


class BroadPeakSchema(pa.DataFrameModel):
    chrom: pat.Series[str]
    chromStart: pat.Series[int]
    chromEnd: pat.Series[int]
    name: pat.Series[str] = pa.Field(nullable=True)
    score: pat.Series[float] = pa.Field(nullable=True)
    strand: pat.Series[str] = pa.Field(isin=["+", "-", "."], nullable=True)
    signalValue: pat.Series[float] = pa.Field(nullable=True)
    pValue: pat.Series[float] = pa.Field(nullable=True)
    qValue: pat.Series[float] = pa.Field(nullable=True)

    class Config:
        coerce = True


def _read_broadpeak(broadpeak_path: Union[str, Path]):
    with open(broadpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    broadpeaks = pl.read_csv(
        broadpeak_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=[
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "signalValue",
            "pValue",
            "qValue",
        ],
        dtypes={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).to_pandas()
    broadpeaks = BroadPeakSchema.to_schema()(broadpeaks)
    return pl.from_pandas(broadpeaks)
