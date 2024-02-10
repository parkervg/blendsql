import pandas as pd
from typing import Iterable, Any, List
from blendsql.ingredients import MapIngredient, QAIngredient, JoinIngredient


class starts_with(MapIngredient):
    def run(self, question: str, values: List[str], **kwargs) -> Iterable[bool]:
        """Simple test function, equivalent to the following in SQL:
            `LIKE '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        mapped_values = [int(i.startswith(question)) for i in values]
        return mapped_values


class get_length(MapIngredient):
    def run(self, question: str, values: List[str], **kwargs) -> Iterable[int]:
        """Simple test function, equivalent to the following in SQL:
            `LENGTH '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        mapped_values = [len(i) for i in values]
        return mapped_values


class select_first_sorted(QAIngredient):
    def run(self, question: str, options: List[str], **kwargs) -> Iterable[Any]:
        """Simple test function, equivalent to the following in SQL:
        `ORDER BY {colname} LIMIT 1`
        """
        chosen_value = sorted(options)[0]
        return f"'{chosen_value}'"


class return_aapl(QAIngredient):
    def run(self, question: str, options: str = None, **kwargs) -> Iterable[Any]:
        """Executes to return the string 'AAPL'"""
        return "'AAPL'"


class do_join(JoinIngredient):
    """A very silly, overcomplicated way to do a join.
    But useful for testing.
    """

    def run(self, left_values: List[str], right_values: List[str], **kwargs) -> dict:
        return {left_value: left_value for left_value in left_values}


def assert_equality(smoothie, sql_df: pd.DataFrame, args: List[str] = None):
    blendsql_df = smoothie.df
    if args is not None:
        arg_overlap = blendsql_df.columns.intersection(args).tolist()
        if len(arg_overlap) > 0:
            blendsql_df = blendsql_df.drop(arg_overlap, axis=1)
    pd.testing.assert_frame_equal(blendsql_df, sql_df)
