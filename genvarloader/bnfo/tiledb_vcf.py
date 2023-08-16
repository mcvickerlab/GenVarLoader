from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import polars as pl
import tiledbvcf
from numpy.typing import NDArray

from .types import Variants


class TileDB_VCF(Variants):
    def __init__(
        self, path: Union[str, Path], ploidy: int, samples: Optional[List[str]] = None
    ) -> None:
        self.path = path
        self.ds = tiledbvcf.Dataset(str(path))
        self.ploidy = ploidy
        if samples is None:
            self.samples = self.ds.samples()
        else:
            self.samples = samples
        self.n_samples = len(self.samples)

    def read(
        self, contig: str, start: int, end: int, **kwargs
    ) -> Optional[Tuple[NDArray[np.uint32], NDArray[np.int32], NDArray[np.bytes_]]]:
        region = f"{contig}:{start+1}-{end}"

        samples: Iterable[str]
        samples = kwargs.get("samples", self.samples)

        ploid: Iterable[int]
        ploid = kwargs.get("ploid", range(self.ploidy))

        df = self.ds.read_arrow(
            ["pos_start", "alleles", "fmt_GT", "sample_name"],
            regions=[region],
            samples=samples,
        )
        df = cast(pl.DataFrame, pl.from_arrow(df))

        if df.height == 0:
            return None

        df = (
            df.filter(~pl.col("fmt_GT").list.contains(-1))
            .with_row_count()
            .explode("fmt_GT")
            .groupby("row_nr")
            .agg(
                pl.exclude("alleles", "fmt_GT").first(),
                pl.col("alleles").list.get(pl.col("fmt_GT")),
            )
            .drop("row_nr")
        )

        alleles = (
            pl.col("alleles").list.get(int(p)).alias(f"allele_{p}") for p in ploid
        )
        with pl.StringCache():
            pl.Series(samples, dtype=pl.Categorical)
            df = (
                df.sort(pl.col("sample_name").cast(pl.Categorical), "pos_start")
                .with_columns(*alleles)
                .drop("alleles")
            )
        counts = (
            df.select(pl.col("sample_name").rle())
            .unnest("sample_name")["lengths"]
            .cast(pl.UInt32)
            .to_numpy()
        )
        counts = cast(NDArray[np.uint32], counts)
        offsets = np.zeros(len(counts) + 1, dtype=counts.dtype)
        counts.cumsum(out=offsets[1:])
        positions = df["pos_start"].to_numpy() - 1  # convert to 0-based
        alleles = [f"allele_{p}" for p in ploid]
        alleles = df.select(alleles).to_numpy().astype("S1").T
        # offsets (samples + 1)
        # positions (variants)
        # alleles (ploid, variants)
        return offsets, positions, alleles
