try:
    import torch
except ImportError:
    raise ImportError("The `torch` submodule requires PyTorch.")


from typing import TYPE_CHECKING, Dict, List, cast

import pandas as pd

from genome_loader.loaders.types import Queries, QueriesSchema
from genome_loader.utils import PathType

if TYPE_CHECKING:
    from genome_loader.loaders import GenVarLoader


def parse_queries(queries_path: PathType) -> Queries:
    queries = cast(
        Queries,
        QueriesSchema.to_schema().validate(
            pd.read_csv(
                queries_path, dtype={"contig": str, "strand": str, "sample": str}
            )
        ),
    )
    return queries


class TorchCollator:
    def __init__(
        self, genvarloader: "GenVarLoader", queries_path: PathType, length: int
    ) -> None:
        self.gvl = genvarloader
        self.queries = parse_queries(queries_path)
        self.length = length

        if "index" in self.gvl.loaders:
            raise RuntimeError(
                """
                GenVarLoader has as loader named 'index' which causes a naming
                conflict since the collator needs to use a key called 'index'
                to store batch indices. Create a new GenVarLoader that doesn't
                have any loaders named 'index'.
                """
            )

    def __call__(self, batch_indices: List[int]) -> Dict[str, torch.Tensor]:
        batch = cast(Queries, self.queries[batch_indices])
        out = {
            k: torch.as_tensor(v) for k, v in self.gvl.sel(batch, self.length).items()
        }
        out["index"] = torch.as_tensor(batch_indices)
        return out
