from typing import Tuple, Type, Set

from blendsql.common.utils import fetch_from_hub
from blendsql.ingredients import Ingredient, LLMQA, LLMMap, LLMJoin, ImageCaption
from blendsql.db import Database, SQLite
from blendsql.models import TransformersVisionModel


def load_benchmark() -> Tuple[Database, Set[Type[Ingredient]]]:
    vision_model = TransformersVisionModel("Mozilla/distilvit", caching=False)
    return (
        SQLite(fetch_from_hub("national_parks.db")),
        {LLMQA, LLMMap, LLMJoin, ImageCaption.from_args(model=vision_model)},
    )
