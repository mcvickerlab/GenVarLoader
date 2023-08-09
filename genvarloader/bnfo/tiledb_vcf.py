from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

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
        self, contig: str, start: int, end: int
    ) -> Optional[Tuple[NDArray[np.uint32], NDArray[np.int32], NDArray[np.bytes_]]]:
        region = f"{contig}:{start+1}-{end}"
        df = self.ds.read_arrow(
            ["pos_start", "alleles", "sample_name"],
            regions=[region],
            samples=self.samples,
        )
        df = cast(pl.DataFrame, pl.from_arrow(df))

        if df.height == 0:
            return None

        alleles = (
            pl.col("alleles").list.get(p).alias(f"allele_{p}")
            for p in range(self.ploidy)
        )
        with pl.StringCache():
            pl.Series(self.samples, dtype=pl.Categorical)
            df = (
                df.sort(pl.col("sample_name").cast(pl.Categorical), "pos_start")
                .with_columns(*alleles)
                .drop("alleles")
            )
        counts = (
            df.groupby("sample_name", maintain_order=True).count()["count"].to_numpy()
        )
        counts = cast(NDArray[np.uint32], counts)
        offsets = np.zeros(len(counts) + 1, dtype=counts.dtype)
        counts.cumsum(out=offsets[1:])
        positions = df["pos_start"].to_numpy()
        alleles = df.select("^allele_.*$").to_numpy().astype("S1").T
        return offsets, positions, alleles
