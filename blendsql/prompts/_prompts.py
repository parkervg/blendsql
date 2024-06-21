from dataclasses import dataclass
from pathlib import Path
from attr import attrs, attrib
from typing import List, Iterable, Type, Set

from ..ingredients import Ingredient
from ..grammars._peg_grammar import grammar as peg_grammar


@attrs
class Examples:
    """Class for holding few-shot examples.

    Examples:
        ```python
        from blendsql.prompts import FewShot, Examples
        fewshot_prompts: Examples = FewShot.hybridqa
        print(fewshot_prompts[:2])
        ```
        ```text
        Examples:

        This is the first example

        ---

        This is the second example
        ```
    """

    data: str = attrib()

    split_data: List[str] = attrib(init=False)

    def __attrs_post_init__(self):
        self.data = self.data.strip()
        self.split_data: list = self.data.split("---")

    def __getitem__(self, subscript):
        newline = (
            "\n\n"
            if (isinstance(subscript, int) and subscript == 0)
            or (isinstance(subscript, slice) and subscript.start in {0, None})
            else ""
        )
        return "Examples:" + newline + "---".join(self.split_data[subscript])

    def __repr__(self):
        return "Examples:\n\n" + self.data

    def __str__(self):
        return "Examples:\n\n" + self.data

    def __len__(self):
        return len(self.split_data)

    def is_valid_query(self, query: str, ingredient_names: Set[str]) -> bool:
        """Checks if a given query is valid given the ingredient_names passed.
        A query is invalid if it includes an ingredient that is not specified in ingredient_names.
        """
        stack = [query]
        while len(stack) > 0:
            for res, _start, _end in peg_grammar.scanString(stack.pop()):
                if res.get("function").upper() not in ingredient_names:
                    return False
                for arg in res.get("args"):
                    stack.append(arg)
        return True

    def filter(self, ingredients: Iterable[Type[Ingredient]]) -> "Examples":
        """Retrieve only those prompts which do not include any ingredient not specified in `ingredients`."""
        ingredient_names: Set[str] = {
            ingredient.__name__.upper() for ingredient in ingredients
        }
        filtered_split_data = []
        for d in self.split_data:
            if self.is_valid_query(d, ingredient_names=ingredient_names):
                filtered_split_data.append(d)
        return Examples("---".join(filtered_split_data))


@dataclass
class FewShot:
    """A collection of few-shot examples, with some utility functions for easy manipulation.

    Examples:
        ```python
        from blendsql import LLMMap, LLMQA
        from blendsql.prompts import FewShot, Examples
        # Fetch the examples for HybridQA
        fewshot_prompts: Examples = FewShot.hybridqa
        print(f"We have {len(fewshot_prompts)} examples")
        # We can select a subset by indexing
        first_three_examples = fewshot_prompts[:3]
        # Additionally, we can filter to keep only those examples using specified ingredients
        filtered_fewshot = fewshot_prompts.filter({LLMQA, LLMMap})
        ```
    """

    hybridqa = Examples(open(Path(__file__).parent / "./few_shot/hybridqa.txt").read())
