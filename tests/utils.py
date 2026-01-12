from collections.abc import Collection
from typing import Iterable, Union
import pandas as pd
from typing import Callable

from blendsql.db.utils import single_quote_escape
from blendsql.ingredients import (
    AliasIngredient,
    Ingredient,
    JoinIngredient,
    MapIngredient,
    QAIngredient,
)


class test_starts_with(MapIngredient):
    def run(
        self,
        question: str,
        values: list[str],
        exit_condition: Callable | None = None,
        **kwargs,
    ) -> list[bool]:
        """Simple test function, equivalent to the following in SQL:
            `LIKE '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        d = {}
        for value in values:
            d[value] = bool(value.startswith(question))
            self.num_values_passed += 1
            if exit_condition is not None and exit_condition(d):
                print("Early exit applied!")
                break
        mapped_values = [d.get(value) for value in values]
        return mapped_values


class get_length(MapIngredient):
    def __call__(
        self,
        values: list[str] = None,
        *args,
        **kwargs,
    ) -> tuple:
        return super().__call__(question="length", values=values, *args, **kwargs)

    def run(
        self, values: list[str], exit_condition: Callable | None = None, **kwargs
    ) -> Iterable[int]:
        """Simple test function, equivalent to the following in SQL:
            `LENGTH '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        d = {}
        for value in values:
            d[value] = len(value)
            self.num_values_passed += 1
            if exit_condition is not None and exit_condition(d):
                break
        mapped_values = [d.get(value) for value in values]
        return mapped_values


class select_first_sorted(QAIngredient):
    def run(self, options: set, **kwargs) -> str:
        """Simple test function, equivalent to the following in SQL:
        `ORDER BY {colname} LIMIT 1`
        """
        chosen_value = sorted(options)[0]
        return f"'{chosen_value}'"


class return_aapl(QAIngredient):
    def run(self, **kwargs) -> str:
        """Executes to return the string 'AAPL'"""
        return "'AAPL'"


class return_true(QAIngredient):
    def run(self, **kwargs) -> bool:
        return True


class return_true_map(MapIngredient):
    def run(
        self, values: list[str], exit_condition: Callable | None = None, **kwargs
    ) -> bool:
        d = {}
        for value in values:
            d[value] = True
            self.num_values_passed += 1
            if exit_condition is not None and exit_condition(d):
                break
        mapped_values = [d.get(value) for value in values]
        return mapped_values


class get_table_size(QAIngredient):
    def __call__(
        self,
        context: list[pd.DataFrame] = None,
        **kwargs,
    ) -> tuple:
        return super().__call__(
            question="size", context=context, options=None, **kwargs
        )

    def run(self, context: list[pd.DataFrame], **kwargs) -> int:
        """Returns the length of the context subtable passed to it."""
        return sum([len(c) for c in context])


class select_first_option(QAIngredient):
    def run(
        self, question: str, context: list[pd.DataFrame], options: set, **kwargs
    ) -> str:
        """Returns the first item in the (ordered) options set"""
        return f"'{single_quote_escape(sorted(list(filter(lambda x: x, options)))[0])}'"


class return_aapl_alias(AliasIngredient):
    def run(self, *args, **kwargs) -> tuple[str, Collection[Ingredient]]:
        return (
            "{{select_first_option(options='AAPL;AMZN;TYL')}}",
            {select_first_option},
        )


class return_stocks_tuple(QAIngredient):
    def run(
        self, question: str, context: list[pd.DataFrame], options: set, **kwargs
    ) -> Union[str, int, float, tuple]:
        return tuple(["AAPL", "AMZN", "TYL"])


class return_stocks_tuple_alias(AliasIngredient):
    def run(self, *args, **kwargs) -> tuple[str, Collection[Ingredient]]:
        return ("{{return_stocks_tuple()}}", {return_stocks_tuple})


class do_join(JoinIngredient):
    """A very silly, overcomplicated way to do a traditional SQL join.
    But useful for testing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_skrub_joiner = False

    def run(self, left_values: list[str], right_values: list[str], **kwargs) -> dict:
        return {left_value: left_value for left_value in left_values}
