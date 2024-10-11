from dataclasses import dataclass, field
from typing import List, Iterable, Type
import pandas as pd

from .ingredients import Ingredient
from .utils import tabulate
from .db.utils import truncate_df_content


class PrettyDataFrame(pd.DataFrame):
    def __str__(self):
        return tabulate(truncate_df_content(self, 50))

    def __repr__(self):
        return tabulate(truncate_df_content(self, 50))


@dataclass
class SmoothieMeta:
    num_values_passed: int  # Number of values passed to a Map/Join/QA ingredient
    prompt_tokens: int
    completion_tokens: int
    prompts: List[dict]  # Log of prompts submitted to model
    raw_prompts: List[str]
    ingredients: Iterable[Type[Ingredient]]
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
        s = "---------------- SUMMARY ----------------\n"
        s += self.meta.query + "\n"
        s += tabulate(
            pd.DataFrame(
                {
                    "Time (s)": self.meta.process_time_seconds,
                    "Values Passed to Ingredients": self.meta.num_values_passed,
                    "Prompt Tokens": self.meta.prompt_tokens,
                    "Completion Tokens": self.meta.completion_tokens,
                },
                index=[0],
            )
        )
        return s
