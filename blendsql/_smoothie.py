from dataclasses import dataclass
from typing import List, Collection
import pandas as pd

from .ingredients import Ingredient

"""
Defines output of an executed BlendSQL script
"""


@dataclass
class SmoothieMeta:
    num_values_passed: int  # Number of values passed to a Map/Join/QA ingredient
    prompt_tokens: int
    completion_tokens: int
    prompts: List[str]  # Log of prompts submitted to model
    ingredients: Collection[Ingredient]
    query: str
    db_url: str
    contains_ingredient: bool = True
    process_time_seconds: float = None


@dataclass
class Smoothie:
    df: pd.DataFrame
    meta: SmoothieMeta
