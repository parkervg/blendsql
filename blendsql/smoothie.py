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
    num_values_passed: int  # Number of values passed to a Map/Join/QA ingredient
    num_generation_calls: int  # Number of generation calls made to the model
    prompt_tokens: int
    completion_tokens: int
    prompts: t.List[dict]  # Log of prompts submitted to model
    raw_prompts: t.List[str]
    ingredients: t.Iterable[t.Type[Ingredient]]
    query: str
    db_url: str
    contains_ingredient: bool = True
    process_time_seconds: float = field(init=False)


@dataclass
class Smoothie:
    df: pd.DataFrame
    meta: SmoothieMeta

    def __post_init__(self):
        self.df = PrettyDataFrame(self.df)

    def summary(self):
        s = "-------------------------------- SUMMARY --------------------------------\n"
        s += self.meta.query + "\n"
        s += tabulate(
            pd.DataFrame(
                {
                    "Time (s)": self.meta.process_time_seconds,
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
