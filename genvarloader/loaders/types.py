from typing import Optional

import numpy as np
import pandas as pd
import pandera as pa
from natsort import natsorted
from pandera.engines import pandas_engine
from pandera.typing import DataFrame, Series


# Register pandera dtype that is guaranteed to have naturally ordered categories
# Important for contig columns
@pandas_engine.Engine.register_dtype  # type: ignore[arg-type]
@pa.dtypes.immutable  # type: ignore
class NatOrderCategory(pandas_engine.Category):
    """Naturally ordered categorical data. This means, for example,
    that '1' < '2' < '15' < 'X' rather than '1' < '15' < '2' < 'X'.
    """

    def coerce(self, series: pd.Series):
        data = series.values
        return pd.Series(
            pd.Categorical(data, categories=natsorted(np.unique(data)), ordered=True)  # type: ignore
        )


class QueriesSchema(pa.SchemaModel):
    contig: Series[NatOrderCategory] = pa.Field(coerce=True)  # type: ignore
    start: Series[pa.Int]
    strand: Optional[Series[pa.Category]] = pa.Field(coerce=True, isin=["+", "-"])
    sample: Optional[Series[pa.Category]] = pa.Field(coerce=True)
    ploid_idx: Optional[Series[pa.Int]] = pa.Field(ge=0)


Queries = DataFrame[QueriesSchema]
