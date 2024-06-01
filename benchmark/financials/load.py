from typing import Tuple, Type, Set

from tests.utils import (
    starts_with,
    get_length,
    select_first_sorted,
    do_join,
    return_aapl,
    get_table_size,
)
from blendsql.utils import fetch_from_hub
from blendsql.ingredients import Ingredient
from blendsql.db import Database, SQLite


def load_benchmark() -> Tuple[Database, Set[Type[Ingredient]]]:
    return (
        SQLite(fetch_from_hub("multi_table.db")),
        {
            starts_with,
            get_length,
            select_first_sorted,
            do_join,
            return_aapl,
            get_table_size,
        },
    )
