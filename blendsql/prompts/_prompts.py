from dataclasses import dataclass
from pathlib import Path
from attr import attrs, attrib
from typing import List, Collection, Type, Set

from ..ingredients import Ingredient
from ..grammars._peg_grammar import grammar as peg_grammar


@attrs
class Examples:
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

    def filter(self, ingredients: Collection[Type[Ingredient]]) -> "Examples":
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
    hybridqa = Examples(open(Path(__file__).parent / "./few_shot/hybridqa.txt").read())
