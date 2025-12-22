import pytest
import time
import pandas as pd

from blendsql import BlendSQL


def assert_equality(smoothie, sql_df: pd.DataFrame, args: list[str] | None = None):
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


def assert_blendsql_equals_sql(
    bsql: BlendSQL,
    blendsql_query: str,
    sql_query: str,
    timing_collector,
    args: list | None = None,
    expected_num_values_passed: int | None = None,
    allow_lt_num_values_compare: bool = False,
    skip_assert_equality: bool = False,
    **bsql_kwargs,
):
    start = time.perf_counter()
    smoothie = bsql.execute(blendsql_query, **bsql_kwargs)
    blendsql_time = time.perf_counter() - start

    start = time.perf_counter()
    sql_df = bsql.db.execute_to_df(sql_query, lazy=False).to_pandas()
    sql_time = time.perf_counter() - start

    timing_collector.add(
        timing_collector._current_test, type(bsql.db).__name__, blendsql_time, sql_time
    )

    if not skip_assert_equality:
        assert_equality(smoothie=smoothie, sql_df=sql_df, args=args)

    if expected_num_values_passed is not None:
        if allow_lt_num_values_compare:
            assert (
                smoothie.meta.num_values_passed <= expected_num_values_passed
            ), f"{smoothie.meta.num_values_passed} !<= {expected_num_values_passed}"
        else:
            assert (
                smoothie.meta.num_values_passed == expected_num_values_passed
            ), f"{smoothie.meta.num_values_passed} != {expected_num_values_passed}"
    return smoothie


class TimedTestBase:
    """Base class that provides timing collection to all test methods."""

    @pytest.fixture(autouse=True)
    def setup_timing(self, timing_collector, request):
        self.timing_collector = timing_collector

    def assert_blendsql_equals_sql(
        self,
        bsql: BlendSQL,
        blendsql_query: str,
        sql_query: str,
        args: list | None = None,
        expected_num_values_passed: int | None = None,
        allow_lt_num_values_compare: bool = False,
        skip_assert_equality: bool = False,
        **bsql_kwargs,
    ):
        """Wrapper that automatically includes timing_collector."""
        return assert_blendsql_equals_sql(
            bsql=bsql,
            blendsql_query=blendsql_query,
            sql_query=sql_query,
            args=args,
            expected_num_values_passed=expected_num_values_passed,
            allow_lt_num_values_compare=allow_lt_num_values_compare,
            skip_assert_equality=skip_assert_equality,
            timing_collector=self.timing_collector,
            **bsql_kwargs,
        )
