from dataclasses import dataclass
from typing import List, Any, Collection
import pandas as pd

from .ingredients import Ingredient

"""
Defines output of an executed BlendSQL script
"""


@dataclass
class SmoothieMeta:
    process_time_seconds: float
    num_values_passed: int  # Number of values passed to a Map/Join/QA ingredient
    num_prompt_tokens: int  # Number of prompt tokens (counting user and assistant, i.e. input/output)
    prompts: List[str]  # Log of prompts submitted to model
    example_map_outputs: List[Any]  # outputs from a Map ingredient, for debugging
    ingredients: Collection[Ingredient]
    query: str
    db_path: str
    contains_ingredient: bool = True


@dataclass
class Smoothie:
    df: pd.DataFrame
    meta: SmoothieMeta
