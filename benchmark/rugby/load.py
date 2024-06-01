from typing import Tuple, Type, Set

from blendsql import LLMQA, LLMMap, LLMJoin
from blendsql.utils import fetch_from_hub
from blendsql.ingredients import Ingredient
from blendsql.db import Database, SQLite


def load_benchmark() -> Tuple[Database, Set[Type[Ingredient]]]:
    return (
        SQLite(
            fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
        ),
        {LLMQA, LLMMap, LLMJoin},
    )
