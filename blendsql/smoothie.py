from dataclasses import dataclass, field
import typing as t
import pandas as pd

from blendsql.ingredients import Ingredient
from blendsql.common.utils import tabulate
from blendsql.db.utils import truncate_df_content


class PrettyDataFrame(pd.DataFrame):
    def __str__(self):
        return tabulate(truncate_df_content(self, 50))

    def __repr__(self):
        return tabulate(truncate_df_content(self, 50))


@dataclass
class SmoothieMeta:
    num_values_passed: int = (
        field()
    )  # Number of values passed to a Map/Join/QA ingredient
    num_generation_calls: int = field()  # Number of generation calls made to the model
    prompt_tokens: int = field()
    completion_tokens: int = field()
    prompts: t.List[dict] = field()  # Log of prompts submitted to model
    raw_prompts: t.List[str] = field()
    ingredients: t.Iterable[t.Type[Ingredient]] = field()
    query: str = field()
    db_url: str = field()
    contains_ingredient: bool = field(default=True)
    process_time_seconds: float = field(default="N.A.")


@dataclass
class Smoothie:
    df: pd.DataFrame = field()
    meta: SmoothieMeta = field()

    def __post_init__(self):
        self.df = PrettyDataFrame(self.df)

    def summary(self):
        s = "-------------------------------- SUMMARY --------------------------------\n"
        s += self.meta.query + "\n"
        s += tabulate(
            pd.DataFrame(
                {
                    "Time (s)": self.meta.process_time_seconds
                    if hasattr(self.meta, "process_time_seconds")
                    else "N.A.",
                    "# Generation Calls": self.meta.num_generation_calls,
                    "Prompt Tokens": self.meta.prompt_tokens,
                    "Completion Tokens": self.meta.completion_tokens,
                },
                index=[0],
            )
        )
        return s

    def __str__(self):
        return self.summary()
