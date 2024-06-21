import pandas as pd
from typing import Iterable, List, Union
from blendsql.ingredients import MapIngredient, QAIngredient, JoinIngredient
from blendsql.db.utils import single_quote_escape


class starts_with(MapIngredient):
    def run(self, question: str, values: List[str], **kwargs) -> List[bool]:
        """Simple test function, equivalent to the following in SQL:
            `LIKE '{arg}%`
        This allows us to compare the output of a BlendSQL script with a SQL script easily.
        """
        mapped_values = [bool(i.startswith(question)) for i in values]
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
    def run(self, options: set, **kwargs) -> Union[str, int, float]:
        """Simple test function, equivalent to the following in SQL:
        `ORDER BY {colname} LIMIT 1`
        """
        chosen_value = sorted(options)[0]
        return f"'{chosen_value}'"


class return_aapl(QAIngredient):
    def run(self, **kwargs) -> Union[str, int, float]:
        """Executes to return the string 'AAPL'"""
        return "'AAPL'"


class get_table_size(QAIngredient):
    def run(self, context: pd.DataFrame, **kwargs) -> Union[str, int, float]:
        """Returns the length of the context subtable passed to it."""
        return len(context)


class select_first_option(QAIngredient):
    def run(
        self, question: str, context: pd.DataFrame, options: set, **kwargs
    ) -> Union[str, int, float]:
        """Returns the first item in the (ordered) options set"""
        return f"'{single_quote_escape(sorted(list(filter(lambda x: x, options)))[0])}'"


class do_join(JoinIngredient):
    """A very silly, overcomplicated way to do a traditional SQL join.
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
    # Make column names abstract
    blendsql_df.columns = [i for i in range(len(blendsql_df.columns))]
    sql_df.columns = [i for i in range(len(sql_df.columns))]
    pd.testing.assert_frame_equal(
        blendsql_df, sql_df, check_like=True, check_dtype=False
    )
