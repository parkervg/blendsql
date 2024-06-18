from typing import Dict, Union
import pandas as pd

from ._duckdb import DuckDB


def Pandas(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame], tablename: str = "w"
) -> DuckDB:
    """This is just a wrapper over the `DuckDB.from_pandas` class method.
    Makes it more intuitive to do a `from blendsql.db import Pandas`, for those
    developers who might not know DuckDB supports pandas dataframes.

    Examples:
        ```python
        from blendsql.db import Pandas
        db = Pandas(
            pd.DataFrame(
                {
                    "name": ["John", "Parker"],
                    "age": [12, 26]
                },
            )
        )
        # Or, load multiple dataframes
        db = Pandas(
            {
                "students": pd.DataFrame(
                    {
                        "name": ["John", "Parker"],
                        "age": [12, 26]
                    },
                ),
                "classes": pd.DataFrame(
                    {
                        "class": ["Physics 101", "Chemistry"],
                        "size": [50, 32]
                    },
                ),
            }
        )
        ```
    """
    return DuckDB.from_pandas(data, tablename)
