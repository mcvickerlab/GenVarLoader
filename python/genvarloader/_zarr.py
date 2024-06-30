from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from more_itertools import batched
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from ._types import Reader
from ._utils import _normalize_contig_name

__all__ = []
try:
    import tensorstore as ts
    import zarr

    ZARR_TENSORSTORE_INSTALLED = True
except ImportError:
    ZARR_TENSORSTORE_INSTALLED = False


"""
Implementation note regarding using tensorstore with num_workers > 0
https://github.com/google/tensorstore/issues/61
TL;DR TensorStore is not Send.
"""


class ZarrTracks(Reader):
    chunked = True

    def __init__(self, name: str, path: Union[str, Path]):
        if not ZARR_TENSORSTORE_INSTALLED:
            raise ImportError(
                "ZarrTracks support requires zarr and tensorstore to be installed."
            )

        self.name = name
        self.path = Path(path)
        z = zarr.open_group(self.path, mode="r")  # pyright: ignore
        self.contigs = cast(Dict[str, int], z.attrs["contigs"])
        self.sizes = cast(Dict[str, int], z.attrs["sizes"])
        self.coords = {d: np.asarray(z.attrs[d]) for d in self.sizes}
        self.samples = cast(Optional[List[str]], z.attrs.get("sample", None))
        self.ploidy = cast(Optional[int], z.attrs.get("ploid", None))
        self.dtype = z[next(iter(self.contigs))].dtype
        # each is (s? p? l)
        self.tstores: Optional[Dict[str, Any]] = None

    def rev_strand_fn(self, x):
        return x[..., ::-1]

    @classmethod
    def from_reader(
        cls,
        reader: Reader,
        path: Union[str, Path],
        chunk_shape: Optional[Tuple[int, ...]] = None,
    ):
        if isinstance(path, str):
            path = Path(path)
        z = zarr.open_group(path)  # pyright: ignore[reportPossiblyUnboundVariable]
        z.attrs["contigs"] = reader.contigs
        z.attrs["sizes"] = reader.sizes
        n_chunks = len(reader.contigs)

        if "sample" in reader.sizes:
            samples = reader.coords["sample"].tolist()
            n_chunks *= len(samples) // 10 + 1
            z.attrs["sample"] = samples
        if "ploid" in reader.sizes:
            z.attrs["ploid"] = reader.sizes["ploidy"]

        if chunk_shape is None:
            if "sample" in reader.sizes and "ploid" in reader.sizes:
                chunk_shape = (10, reader.sizes["ploid"], int(5e5))
            elif "sample" in reader.sizes:
                chunk_shape = (20, int(5e5))
            elif "ploidy" in reader.sizes:
                chunk_shape = (2, int(5e5))

        chunk_layout = ts.ChunkLayout(  # pyright: ignore
            chunk_shape=chunk_shape
        )

        with tqdm(total=n_chunks) as pbar:
            for contig, e in reader.contigs.items():
                ts_open = partial(
                    ts.open,
                    {
                        "driver": "zarr",
                        "kvstore": {"driver": "file", "path": str(path / contig)},
                        "metadata": {
                            "compressor": {
                                "id": "blosc",
                                "shuffle": -1,
                                "clevel": 5,
                                "cname": "lz4",
                            }
                        },
                    },
                    read=True,
                    write=True,
                    create=True,
                    delete_existing=True,
                    chunk_layout=chunk_layout,
                )
                tstore = None
                if "sample" in reader.sizes:
                    for b_idx, samples in enumerate(
                        batched(reader.coords["sample"], 10)
                    ):
                        pbar.set_description(f"Reading {contig}")
                        data = reader.read(contig, 0, e, sample=samples)

                        if tstore is None:
                            tstore = ts_open(
                                dtype=data.dtype,
                                shape=(reader.sizes["sample"], *data.shape[1:]),
                            ).result()

                        pbar.set_description(f"Writing {contig}")
                        start = b_idx * 10
                        stop = b_idx * 10 + len(samples)
                        tstore[start:stop].write(data).result()

                        pbar.update()
                else:
                    pbar.set_description(f"Reading {contig}")
                    data = reader.read(contig, 0, e)

                    if tstore is None:
                        tstore = ts_open(dtype=data.dtype, shape=data.shape).result()

                    pbar.set_description(f"Writing {contig}")
                    tstore.write(data).result()

                    pbar.update()
        zarr.consolidate_metadata(str(path))  # pyright: ignore

        return cls(reader.name, path)

    def _tstore(self, contig: str):
        tstore = ts.open(  # pyright: ignore
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(self.path / contig)},
            },
            read=True,
        ).result()
        return tstore

    def read(
        self, contig: str, starts: ArrayLike, ends: ArrayLike, **kwargs
    ) -> NDArray:
        if self.tstores is None:
            self.tstores = {contig: self._tstore(contig) for contig in self.contigs}

        contig = _normalize_contig_name(contig, self.contigs)  # pyright: ignore[reportAssignmentType]
        if contig is None:
            raise ValueError(f"Contig {contig} not found")

        tstore = self.tstores[contig]
        samples = cast(Optional[List[str]], kwargs.get("sample", self.samples))
        if samples is not None:
            if self.samples is None:
                samples = None
            elif missing := set(samples).difference(self.samples):
                raise ValueError(f"Samples {missing} were not found")
            else:
                key_idx, query_idx = np.intersect1d(
                    self.samples, samples, return_indices=True
                )[1:]
                sample_idx = key_idx[query_idx]
                # (s p? l)
                tstore = tstore[sample_idx]

        ploidy = cast(Optional[ArrayLike], kwargs.get("ploid", None))
        if ploidy is not None:
            if self.ploidy is None:
                ploidy = None
            else:
                haplotype_idx = np.asarray(ploidy, dtype=int)
                if (haplotype_idx >= self.ploidy).any():
                    raise ValueError("Ploidies requested exceed maximum ploidy")
                # (s? p l)
                if self.samples is None:
                    # (p l)
                    tstore = tstore[haplotype_idx]
                else:
                    # (s p l)
                    tstore = tstore[:, haplotype_idx]

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = np.atleast_1d(np.asarray(ends, dtype=int))

        sub_values = [None] * len(starts)
        for i, (s, e) in enumerate(zip(starts, ends)):
            # no variants in query regions
            if s == e:
                continue
            # (s? p? l)
            sub_values[i] = tstore[..., s:e]

        values = ts.concat(sub_values, axis=-1)[  # pyright: ignore  # pyright: ignore
            ts.d[0].translate_to[0]  # pyright: ignore
        ]

        values = cast(NDArray, values.read().result())

        return values

    def vidx(
        self,
        contigs: ArrayLike,
        starts: ArrayLike,
        length: int,
        samples: Optional[ArrayLike] = None,
        ploidy: Optional[ArrayLike] = None,
        **kwargs,
    ) -> NDArray:
        if self.tstores is None:
            self.tstores = {contig: self._tstore(contig) for contig in self.contigs}

        contigs = np.array(
            [
                _normalize_contig_name(c, self.contigs)
                for c in np.atleast_1d(np.asarray(contigs))
            ]
        )
        if (contigs == None).any():  # noqa: E711
            raise ValueError("Some contigs not found")

        if samples is not None:
            if self.samples is None:
                raise ValueError("No sample information available")

            unique_samples, inverse = np.unique(samples, return_inverse=True)
            if missing := set(unique_samples).difference(self.samples):
                raise ValueError(f"Samples {missing} were not found")

            key_idx, query_idx = np.intersect1d(
                self.samples, unique_samples, return_indices=True, assume_unique=True
            )[1:]
            sample_idx = key_idx[query_idx[inverse]]
        else:
            if self.samples is None:
                sample_idx = [None]
            else:
                sample_idx = [slice(None)]

        if ploidy is not None:
            if self.ploidy is None:
                raise ValueError("No ploidy information available")

            haplotype_idx = np.asarray(ploidy, dtype=int)
            if (haplotype_idx >= self.ploidy).any():
                raise ValueError("Ploidies requested exceed maximum ploidy")
        else:
            if self.ploidy is None:
                haplotype_idx = [None]
            else:
                haplotype_idx = [slice(None)]

        starts = np.atleast_1d(np.asarray(starts, dtype=int))
        ends = starts + length

        sub_values = [None] * len(starts)
        for i, (c, s, e, sp, h) in enumerate(
            zip_longest(contigs, starts, ends, sample_idx, haplotype_idx)
        ):
            # no variants in query regions
            if s == e:
                continue
            tstore = self.tstores[c]
            # (s? p? l)
            if sp is None and h is None:
                # (l)
                sub_values[i] = tstore[s:e]
            elif sp is None:
                # (p l)
                sub_values[i] = tstore[h, s:e]
            elif h is None:
                # (s p l)
                sub_values[i] = tstore[sp, s:e]
            else:
                sub_values[i] = tstore[sp, h, s:e]

        values = ts.concat(sub_values, axis=-1)[  # pyright: ignore
            ts.d[0].translate_to[0]  # pyright: ignore
        ]

        values = cast(NDArray, values.read().result())

        return values
