import os
import re
from dataclasses import dataclass, field
from abc import abstractmethod
import pandas as pd
from sqlglot import exp
import json
from typing import Type, Callable, Any
from collections.abc import Collection, Iterable
import uuid
from typeguard import check_type
import polars as pl

from blendsql.common.exceptions import LMFunctionException
from blendsql.common.logger import logger, Color
from blendsql.common import utils
from blendsql.common.typing import (
    IngredientType,
    ColumnRef,
    AdditionalMapArg,
    StringConcatenation,
)
from blendsql.db import Database
from blendsql.db.utils import format_tuple, double_quote_escape, LazyTable
from blendsql.common.utils import get_tablename_colname
from blendsql.search.searcher import Searcher
from blendsql.ingredients.few_shot import Example
from blendsql.configure import DEFAULT_DETERMINISTIC, DETERMINISTIC_KEY


def unpack_default_kwargs(**kwargs):
    return (
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


@dataclass
class Ingredient:
    name: str = field()
    # Below gets passed via `Kitchen.extend()`
    db: Database = field()
    session_uuid: str = field()

    deterministic: bool = field(default=False)
    few_shot_retriever: Callable[[str], list[Example]] = field(default=None)
    list_options_in_prompt: bool = field(default=True)
    context_searcher: Searcher | None = field(default=None)
    options_searcher: Searcher | None = field(default=None)

    ingredient_type: str = field(init=False)
    allowed_output_types: tuple[Type] = field(init=False)

    num_values_passed: int = 0

    def __repr__(self):
        return f"{self.ingredient_type} {self.name}"

    def __str__(self):
        return f"{self.ingredient_type} {self.name}"

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        ...

    def _run(self, *args, **kwargs):
        return check_type(self.run(*args, **kwargs), self.allowed_output_types)

    @staticmethod
    def _maybe_set_name_to_var_name(partial_cls):
        """Allows us to do a poor-man's replacement scan
        https://duckdb.org/docs/stable/clients/c/replacement_scans.html
        """
        # Get the name from the caller's frame
        import inspect

        frame = None
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back.f_back:
                calling_frame = frame.f_back.f_back
                context = inspect.getframeinfo(calling_frame).code_context
                if context:
                    for line in context:
                        variable_declaration = re.search(
                            r"(.*?)\s*=\s*"
                            + re.escape(f"{partial_cls.__name__}.from_args"),
                            line.strip(),
                            flags=re.DOTALL,
                        )
                        if variable_declaration:
                            variable_name = (
                                variable_declaration.group().split("=")[0].strip()
                            )
                            logger.debug(
                                Color.model_or_data_update(
                                    f"Loading custom {partial_cls.__name__} with name '{variable_name}'..."
                                )
                            )
                            # Store the name in a class attribute
                            partial_cls.__name__ = variable_name
                            break
        except Exception as e:
            logger.debug(f"Failed to determine class name: {e}")
        finally:
            # Clean up references to frames to prevent reference cycles
            del frame
        return partial_cls

    def unpack_column_ref(
        self,
        v: ColumnRef,
        aliases_to_tablenames: dict[str, str],
        deterministic: bool = False,
    ) -> list[str]:
        tablename, colname = get_tablename_colname(v)
        tablename = aliases_to_tablenames.get(tablename, tablename)
        # IMPORTANT: Below was commented out, since it would cause:
        #   `SELECT {{select_first_sorted(options=w.Symbol)}} FROM w LIMIT 1`
        #   ...to always select the result of the `LIMIT 1`.
        # Check for previously created temporary tables
        # value_source_tablename, _ = self.maybe_get_temp_table(
        #     temp_table_func=get_temp_subquery_table, tablename=tablename
        # )
        # Optionally materialize a CTE
        if tablename in self.db.lazy_tables:
            materialized_smoothie = self.db.lazy_tables.pop(tablename).collect()
            self.num_values_passed += materialized_smoothie.meta.num_values_passed
            unpacked_values = (
                materialized_smoothie.pl.get_column(colname)
                .unique(maintain_order=deterministic)
                .cast(pl.Utf8)
                .to_list()
            )
        else:
            unpacked_values: list = [
                str(i)
                for i in self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{tablename}"'
                    + (" ORDER BY rowid" if deterministic else "")
                )
            ]
        return unpacked_values

    def maybe_get_temp_table(
        self, temp_table_func: Callable, tablename: str
    ) -> tuple[str, bool]:
        temp_tablename = temp_table_func(tablename)
        if self.db.has_temp_table(temp_tablename):
            # We've already applied some operation to this table
            # We want to use this as our base
            return (temp_tablename, True)
        return (tablename, False)

    def unpack_options(
        self,
        options: ColumnRef | list,
        aliases_to_tablenames: dict[str, str],
        deterministic: bool = False,
    ) -> list[str] | None:
        if isinstance(options, ColumnRef):
            unpacked_options = self.unpack_column_ref(
                options, aliases_to_tablenames, deterministic
            )
        else:
            unpacked_options = options

        if not unpacked_options:
            logger.debug(
                Color.error(
                    f"Tried to unpack options '{options}', but got an empty list\nThis may be a bug. Please report it."
                )
            )
            return None
        return list(unpacked_options)


@dataclass
class AliasIngredient(Ingredient):
    '''This ingredient performs no other function than to act as a stand-in for
    complex chainings of other ingredients. This allows us (or our lms) to write less verbose
    BlendSQL queries, while maximizing the information we embed.

    The `run()` function should return a tuple containing both the query text that should get subbed in,
    and any ingredient classes which are dependencies for executing the aliased query.

    Examples:
        ```python
        from textwrap import dedent
        from typing import Tuple, Collection

        from blendsql.ingredients import AliasIngredient, LLMQA

        class FetchDefinition(AliasIngredient):
            def run(self, term: str, *args, **kwargs) -> Tuple[str, Collection[Ingredient]]:
                new_query = dedent(
                f"""
                {{{{
                    LLMQA(
                        "What does {term} mean?"
                    )
                }}}}
                """)
                ingredient_dependencies = {LLMQA}
                return (new_query, ingredient_dependencies)

        # Now, we can use the ingredient like below
        blendsql_query = """
        SELECT {{FetchDefinition('delve')}} AS "Definition"
        """
        ```
    '''

    ingredient_type: str = IngredientType.ALIAS.value
    allowed_output_types: tuple[Type] = (tuple[str, Collection[Ingredient]],)

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


@dataclass
class MapIngredient(Ingredient):
    '''For a given table/column pair, maps an external function
    to each of the given values, creating a new column.

    Examples:
        ```python
        from typing import List
        from blendsql.ingredients import MapIngredient
        import requests


        class GetQRCode(MapIngredient):
            """Calls API to generate QR code for a given URL.
            Saves bytes to file in qr_codes/ and returns list of paths.
            https://goqr.me/api/doc/create-qr-code/"""


            def run(self, values: List[str], **kwargs) -> List[str]:
                imgs_as_bytes = []
                for value in values:
                    qr_code_bytes = requests.get(
                        "https://api.qrserver.com/v1/create-qr-code/?data=https://{}/&size=100x100".format(value)
                    ).content
                    imgs_as_bytes.append(qr_code_bytes)
                return imgs_as_bytes


            if __name__ == "__main__":
                from blendsql import BlendSQL
                from blendsql.db import SQLite
                from blendsql.utils import fetch_from_hub

                bsql = BlendSQL(fetch_from_hub('urls.db'), ingredients={GetQRCode})

                smoothie = bsql.execute("SELECT genre, url, {{GetQRCode('QR Code as Bytes:', 'w::url')}} FROM w WHERE genre = 'social'")

                smoothie.df
                # | genre  | url           | QR Code as Bytes:      |
                # |--------|---------------|-----------------------|
                # | social | facebook.com  | b'...'                |
        ```
    '''

    ingredient_type: str = IngredientType.MAP.value
    allowed_output_types: tuple[Type] = (Iterable[Any],)

    def unpack_default_kwargs(self, **kwargs):
        return unpack_default_kwargs(**kwargs)

    def __call__(
        self,
        question: str | None = None,
        values: ColumnRef | None = None,
        *additional_args: ColumnRef,
        context: str | pd.DataFrame | None = None,
        options: ColumnRef | list | None = None,
        **kwargs,
    ) -> tuple[str, str, str, pl.LazyFrame]:
        """Returns tuple with format (arg, tablename, colname, new_table)"""
        in_deterministic_mode = bool(
            int(os.getenv(DETERMINISTIC_KEY, DEFAULT_DETERMINISTIC))
        )
        # Extract single `additional_args` from kwargs if provided
        if "additional_args" in kwargs:
            additional_args_kwarg = kwargs.pop("additional_args")
            # Combine positional and keyword context
            if isinstance(additional_args_kwarg, (list, tuple)):
                additional_args = additional_args + tuple(additional_args_kwarg)
            else:
                additional_args = additional_args + (additional_args_kwarg,)

        additional_args_passed = bool(additional_args)
        aliases_to_tablenames: dict[str, str] = kwargs["aliases_to_tablenames"]
        get_temp_subquery_table: Callable = kwargs["get_temp_subquery_table"]
        get_temp_session_table: Callable = kwargs["get_temp_session_table"]
        prev_subquery_map_columns: set[str] = kwargs["prev_subquery_map_columns"]
        cascade_filter: LazyTable | None = kwargs["cascade_filter"]

        if isinstance(values, StringConcatenation):
            # original_tablenames could be aliases
            _original_tablenames, _ = zip(
                *[utils.get_tablename_colname(c) for c in values]
            )
            _original_tablenames = set(_original_tablenames)
            if len(_original_tablenames) > 1:
                raise LMFunctionException(
                    "Can only concatenate two columns from the same table for now!"
                )

            original_tablename = _original_tablenames.pop()
            tablename = aliases_to_tablenames.get(
                original_tablename, original_tablename
            )
            colname = "__concat__"
            value_source_tablename, _ = self.maybe_get_temp_table(
                temp_table_func=get_temp_subquery_table, tablename=tablename
            )
            concat_expr = re.sub(
                rf"{re.escape(original_tablename)}\.", "", values.raw_expr
            )

            logger.debug(
                Color.update("Prepping string concatenations for Map ingredient...")
            )
            # Add a '__concat__' column to our existing temp subquery table
            if value_source_tablename in self.db.lazy_tables:
                materialized_smoothie = self.db.lazy_tables.pop(
                    value_source_tablename
                ).collect()
                self.num_values_passed += materialized_smoothie.meta.num_values_passed

            _query = f"""SELECT *, {concat_expr} AS __concat__ FROM "{double_quote_escape(value_source_tablename)}" """
            logger.debug(
                Color.quiet_update("Executing ")
                + Color.quiet_sql(_query, ignore_prefix=True)
                + Color.quiet_update(
                    f" and setting to {value_source_tablename}...", ignore_prefix=True
                )
            )
            table_with_concat_column = self.db.execute_to_df(_query)
            self.db.to_temp_table(
                table_with_concat_column,
                value_source_tablename,
            )

            # Also add a placeholder column to the main table
            # TODO: we don't really need to concat over ALL columns, since we only need the subset
            #   we processed above. But, in order for the final map aggregation to work,
            #   we join on the '__concat__' column.
            (
                temp_session_tablename,
                temp_session_table_exists,
            ) = self.maybe_get_temp_table(
                temp_table_func=get_temp_session_table, tablename=tablename
            )
            original_table = self.db.execute_to_df(
                f"""
               SELECT *, {concat_expr} AS __concat__ FROM "{double_quote_escape(tablename)}"
               """
            )
            self.db.to_temp_table(original_table, temp_session_tablename)

        else:
            original_table = None
            # TODO: make sure we support all types of ValueArray references here
            tablename_or_aliasname, colname = utils.get_tablename_colname(values)
            tablename = aliases_to_tablenames.get(
                tablename_or_aliasname, tablename_or_aliasname
            )

        # Check for previously created temporary tables
        value_source_tablename, _ = self.maybe_get_temp_table(
            temp_table_func=get_temp_subquery_table, tablename=tablename
        )
        (
            temp_session_tablename,
            temp_session_table_exists,
        ) = self.maybe_get_temp_table(
            temp_table_func=get_temp_session_table, tablename=tablename
        )

        cascade_filter_colnames = set()
        if cascade_filter is not None:
            cascade_filter: pl.LazyFrame | None = cascade_filter.collect()
            if cascade_filter is not None:
                cascade_filter_colnames = set(cascade_filter.collect_schema().names())

        # Construct a `SELECT DISTINCT` function to get all unique combinations of values we need to apply the `Map` to
        # In the most basic case, this is the single column name that was passed
        # But, we also need to consider distinct pairs of values if `f(column1, column2)` was passed
        resolved_additional_args: list[AdditionalMapArg] = []
        if additional_args_passed:
            for additional_arg in additional_args:
                if isinstance(additional_arg, ColumnRef):
                    (
                        additional_arg_tablename_or_alias,
                        additional_arg_columnname,
                    ) = utils.get_tablename_colname(additional_arg)
                    resolved_additional_args.append(
                        AdditionalMapArg(
                            columnname=additional_arg_columnname,
                            tablename=aliases_to_tablenames.get(
                                additional_arg_tablename_or_alias,
                                additional_arg_tablename_or_alias,
                            ),
                        )
                    )
                else:
                    raise ValueError(
                        f"`Map` ingredients can only receive `ColumnRef` objects (e.g. `{{tablename}}.{{columnname}}`) as additional args\nDid you try to pass a subquery instead?"
                    )
            select_distinct_arg = ", ".join(
                f'"{double_quote_escape(c)}"'
                for c in set(
                    set([colname])
                    | set([i.columnname for i in resolved_additional_args])
                    | cascade_filter_colnames
                )
            )
            select_distinct_fn = lambda q: self.db.execute_to_df(q)
        else:
            if cascade_filter_colnames:
                select_distinct_arg = ", ".join(
                    f'"{double_quote_escape(c)}"'
                    for c in set(set([colname]) | cascade_filter_colnames)
                )
                select_distinct_fn = lambda q: self.db.execute_to_df(q)
            else:
                # Simplest base case - just a single column's values were passed
                select_distinct_arg = f'"{double_quote_escape(colname)}"'
                select_distinct_fn = lambda q: self.db.execute_to_list(q)

        # i.e, if we didn't create a string concatenation table
        # Optionally materialize a CTE
        if original_table is None:
            if tablename in self.db.lazy_tables:
                materialized_smoothie = self.db.lazy_tables.pop(tablename).collect()
                self.num_values_passed += materialized_smoothie.meta.num_values_passed
                original_table = materialized_smoothie.pl.lazy()
            else:
                original_table = self.db.execute_to_df(
                    f"""SELECT {select_distinct_arg} FROM "{tablename}" ORDER BY rowid"""
                )
        # Need to be sure the new column doesn't already exist here
        new_arg_column = question or uuid.uuid4().hex[:4]
        while (
            new_arg_column in set(self.db.iter_columns(tablename))
            # new_arg_column in set(self.db.iter_columns(value_source_tablename))
            or new_arg_column in prev_subquery_map_columns
        ):
            new_arg_column = "_" + new_arg_column

        suffix = ""
        if in_deterministic_mode:
            suffix = " ORDER BY rowid"
        # Get a list of values to map
        # First, check if we've already dumped some `MapIngredient` output to the main session table
        if temp_session_table_exists:
            temp_session_table = self.db.execute_to_df(
                f'SELECT * FROM "{double_quote_escape(temp_session_tablename)}" LIMIT 1'
            )
            # We don't need to run this function on everything,
            #   if a previous subquery already got to certain values
            if new_arg_column in temp_session_table.collect_schema().names():
                distinct_values = select_distinct_fn(
                    f'SELECT DISTINCT {select_distinct_arg} FROM "{temp_session_tablename}" WHERE "{new_arg_column}" IS NULL'
                    + suffix
                )
            # Base case: this is the first time we've used this particular ingredient
            # BUT, temp_session_tablename still exists
            else:
                distinct_values = select_distinct_fn(
                    f'SELECT DISTINCT {select_distinct_arg} FROM "{temp_session_tablename}"'
                    + suffix
                )
        else:
            distinct_values = select_distinct_fn(
                f'SELECT DISTINCT {select_distinct_arg} FROM "{value_source_tablename}"'
                + suffix
            )

        if cascade_filter is not None:
            # cascade_filters is a pl.LazyFrame containing some additional filters to apply to our distinct values
            # For example:
            # cascade_filters.collect() ==
            # ┌───────────────────┬─────────────────────────────────┐
            # │ Name              ┆ Known_For                       │
            # │ ---               ┆ ---                             │
            # │ str               ┆ str                             │
            # ╞═══════════════════╪═════════════════════════════════╡
            # │ Sabrina Carpenter ┆ Nonsense, Emails I Cant Send, … │
            # │ Charli XCX        ┆ Crash, How Im Feeling Now, Boo… │
            # │ Elvis Presley     ┆ 14 Grammys, King of Rock n Rol… │
            # └───────────────────┴─────────────────────────────────┘
            # ...means we only take values where `Name`, `Known_For` columns are present in the above.
            logger.debug(
                Color.optimization(
                    f"[ 🌊 ] Applying cascade filter from previous LM function..."
                )
            )
            distinct_values = distinct_values.join(
                cascade_filter, on=cascade_filter_colnames, how="semi"
            )
            # Remove columns, if they were only needed for the cascade_filter
            distinct_values = distinct_values.select(
                set([colname]) | set([i.columnname for i in resolved_additional_args])
            )

        if additional_args_passed or cascade_filter is not None:
            # We have a dataframe object we need to disentangle
            df = distinct_values.collect()
            if cascade_filter is not None:
                unpacked_values = (
                    df[colname].unique(maintain_order=in_deterministic_mode).to_list()
                )
            else:
                unpacked_values = df[colname].to_list()
            for additional_arg in resolved_additional_args:
                additional_arg.values = df.get_column(
                    additional_arg.columnname
                ).to_list()
        else:
            # Base case: a simple list of unique values from a column
            unpacked_values: list = distinct_values

        # No need to run ingredient if we have no values to map onto
        if not unpacked_values:
            original_table = original_table.with_columns(
                pl.lit(None).alias(new_arg_column)
            )
            return (new_arg_column, tablename, colname, original_table)

        unpacked_options = None
        if options is not None:
            unpacked_options = self.unpack_options(
                options=options,
                aliases_to_tablenames=aliases_to_tablenames,
                deterministic=in_deterministic_mode,
            )

        global_subtable_context = None
        if context is not None:
            if isinstance(context, ColumnRef):
                tablename, colname = utils.get_tablename_colname(additional_arg)
                tablename = aliases_to_tablenames.get(tablename, tablename)
                # Optionally materialize a CTE
                if tablename in self.db.lazy_tables:
                    materialized_smoothie = self.db.lazy_tables.pop(tablename).collect()
                    self.num_values_passed += (
                        materialized_smoothie.meta.num_values_passed
                    )
                    global_subtable_context = materialized_smoothie.pl.select([colname])
                    if isinstance(global_subtable_context, pl.LazyFrame):
                        global_subtable_context = global_subtable_context.collect()
                else:
                    global_subtable_context: pl.DataFrame = self.db.execute_to_df(
                        f'SELECT "{colname}" FROM "{tablename}"', lazy=False
                    )
            elif isinstance(context, pl.DataFrame):
                global_subtable_context: pl.DataFrame = context
            else:
                global_subtable_context = pl.DataFrame({"_col": context})
            self.num_values_passed += len(global_subtable_context)

        # Unpack questions, to later pass to a `context_searcher` or `options_searcher`
        unpacked_questions = None
        if question is not None and "{}" in question:
            unpacked_questions = [question.format(value) for value in unpacked_values]

            logger.debug(
                Color.quiet_update(f"Unpacked question to '{unpacked_questions[:10]}'")
            )

        mapped_values = self._run(
            question=question,
            unpacked_questions=unpacked_questions,
            values=unpacked_values,
            additional_args=resolved_additional_args,
            global_subtable_context=global_subtable_context,
            options=unpacked_options,
            tablename=tablename,
            colname=colname,
            **self.__dict__ | kwargs,
        )
        df_as_dict = {
            colname: list(unpacked_values),
            new_arg_column: list(mapped_values),
        }
        mapped_subtable = pl.LazyFrame(
            df_as_dict, strict=False
        )  # strict=False allows mixed types

        # Add new_table to original table
        if additional_args_passed:
            _mapped_subtable = pl.concat(
                [distinct_values, mapped_subtable.select(new_arg_column)],
                how="horizontal",
            )
            new_table = original_table.join(
                _mapped_subtable,
                how="left",
                # We DON'T need to join on cascade_filter_colnames, since these weren't neccesarily operated on in the map call.
                on=set([colname])
                | set([i.columnname for i in resolved_additional_args]),
            )
        else:
            new_table = original_table.join(mapped_subtable, how="left", on=colname)
        # Now, new table has original columns + column with the name of the question we answered
        return (new_arg_column, tablename, colname, new_table)

    @abstractmethod
    def run(self, *args, **kwargs) -> Iterable[Any]:
        ...


@dataclass
class JoinIngredient(Ingredient):
    '''Executes an `INNER JOIN` using dict mapping.
    'Join on color of food'
    {"tomato": "red", "broccoli": "green", "lemon": "yellow"}

    Examples:
        ```python
        from blendsql.ingredients import JoinIngredient

        class do_join(JoinIngredient):
            """A very silly, overcomplicated way to do a traditional SQL join.
            But useful for testing.
            """

            def run(self, left_values: List[str], right_values: List[str], **kwargs) -> dict:
                return {left_value: left_value for left_value in left_values}

        blendsql_query = """
        SELECT Account, Quantity FROM returns r
        JOIN account_history ah ON {{
            do_join(
                left_on=ah.Symbol,
                right_on=r.Symbol
            )
        }}
        """
        ```
    '''

    use_skrub_joiner: bool = field(default=True)

    ingredient_type: str = IngredientType.JOIN.value
    allowed_output_types: tuple[Type] = (dict,)

    def __call__(
        self,
        left_on: str | None = None,
        right_on: str | None = None,
        join_criteria: str | None = None,
        *args,
        **kwargs,
    ) -> tuple:
        # Unpack kwargs
        aliases_to_tablenames: dict[str, str] = kwargs["aliases_to_tablenames"]
        get_temp_subquery_table: Callable = kwargs["get_temp_subquery_table"]
        get_temp_session_table: Callable = kwargs["get_temp_session_table"]
        # Depending on the size of the underlying data, it may be optimal to swap
        #   the order of 'left_on' and 'right_on' columns during processing
        swapped = False
        values = []
        original_lr_identifiers = []
        modified_lr_identifiers = []
        mapping: dict[str, str] = {}
        for on_arg in [left_on, right_on]:
            # Since LLMJoin is unique, in that we need to inject the referenced tablenames back to the query,
            #   make sure we keep the `referenced_tablename` variable.
            # So the below works:
            #     SELECT f.name, colors.name FROM fruits f
            #     JOIN colors c ON {{LLMJoin(f.name, c.name, join_criteria='Align the fruit to its color')}}
            referenced_tablename, colname = utils.get_tablename_colname(on_arg)
            tablename = aliases_to_tablenames.get(
                referenced_tablename, referenced_tablename
            )
            original_lr_identifiers.append((referenced_tablename, colname))
            tablename, _ = self.maybe_get_temp_table(
                temp_table_func=get_temp_subquery_table,
                tablename=tablename,
            )
            values.append(
                self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{tablename}"', to_type=str
                )
            )
            modified_lr_identifiers.append((tablename, colname))

        sorted_values = values
        swapped = False
        if join_criteria is None:
            # Only do order optimization if we haven't passed a custom `join_criteria`
            sorted_values = sorted(values, key=len)
            # check swapping only once, at the beginning
            if sorted_values != values:
                swapped = True
            # First, check which values we actually need to call Model on
            # We don't want to join when there's already an intuitive alignment
            # First, make sure outer loop is shorter of the two lists
            outer, inner = sorted_values
            _outer = []
            inner = set(inner)
            mapping = {}
            for l in outer:
                if l in inner:
                    # Define this mapping, and remove from Model inference call
                    mapping[l] = l
                    inner.remove(l)
                else:
                    _outer.append(l)
                if len(inner) == 0:
                    break
            # Remained _outer and inner lists preserved the sorting order in length:
            # len(_outer) = len(outer) - #matched <= len(inner original) - matched = len(inner)
            if self.use_skrub_joiner and all(len(x) > 1 for x in [inner, _outer]):
                from skrub import Joiner

                # Create the main_table DataFrame
                main_table = pd.DataFrame(_outer, columns=["out"])
                # Create the aux_table DataFrame
                aux_table = pd.DataFrame(inner, columns=["in"])
                joiner = Joiner(
                    aux_table,
                    main_key="out",
                    aux_key="in",
                    max_dist=0.9,
                    add_match_info=False,
                )
                res = joiner.fit_transform(main_table)
                # Below is essentially set.difference on aux_table and those paired in res
                inner = aux_table.loc[~aux_table["in"].isin(res["in"]), "in"].tolist()
                # length(new inner) = length(inner) - #matched by fuzzy join
                _outer = res["out"][res["in"].isnull()].to_list()
                # length(new _outer) = length(_outer) - #matched by fuzzy join
                _skrub_mapping = (
                    res.dropna(subset=["in"]).set_index("out")["in"].to_dict()
                )
                logger.debug(
                    Color.warning("Made the following alignment with `skrub.Joiner`:")
                )
                logger.debug(Color.warning(json.dumps(_skrub_mapping, indent=4)))
                mapping = mapping | _skrub_mapping
            # order by length is still preserved regardless of using fuzzy join, so after initial matching and possible fuzzy join matching
            # This is because the lengths of each list will decrease at the same rate, so whichever list was larger at the beginning,
            # will be larger here at the end.
            # len(_outer) <= len(inner)
            sorted_values = [_outer, inner]

        # Now, we have our final values to process.
        left_values, right_values = sorted_values
        # right_values, left_values = sorted_values

        (left_tablename, left_colname), (
            right_tablename,
            right_colname,
        ) = original_lr_identifiers
        (_left_tablename, _left_colname), (
            _right_tablename,
            _right_colname,
        ) = modified_lr_identifiers

        if all(len(x) > 0 for x in [left_values, right_values]):
            # Some alignment still left to do
            self.num_values_passed += len(left_values) + len(right_values)

            _predicted_mapping: dict[str, str] = self._run(
                left_values=left_values,
                right_values=right_values,
                join_criteria=join_criteria,
                *args,
                **self.__dict__ | kwargs,
            )
            mapping = mapping | _predicted_mapping
        # Using mapped left/right values, create intermediary mapping table
        temp_join_tablename = get_temp_session_table(uuid.uuid4().hex[:4])
        # Below, we check to see if 'swapped' is True
        # If so, we need to inverse what is 'left', and what is 'right'
        joined_values_df = pl.DataFrame(
            data={
                "left" if not swapped else "right": mapping.keys(),
                "right" if not swapped else "left": mapping.values(),
            }
        )
        self.db.to_temp_table(df=joined_values_df, tablename=temp_join_tablename)

        if right_tablename in aliases_to_tablenames:
            right_table_ref = (
                f'"{aliases_to_tablenames[right_tablename]}" AS "{right_tablename}"'
            )
        else:
            right_table_ref = f'"{right_tablename}"'

        join_clause = (
            f'JOIN "{temp_join_tablename}" ON "{left_tablename}"."{left_colname}" = "{temp_join_tablename}".left\n'
            f'JOIN {right_table_ref} ON "{right_tablename}"."{right_colname}" = "{temp_join_tablename}".right'
        )

        return (left_tablename, right_tablename, join_clause, temp_join_tablename)

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        ...


@dataclass
class QAIngredient(Ingredient):
    """
    Given a table subset in the form of a pd.DataFrame 'context',
    returns a scalar or array of scalars (in the form of a tuple).

    Useful for end-to-end question answering tasks.
    """

    ingredient_type: str = IngredientType.QA.value
    allowed_output_types: tuple[Type] = (str | int | float | tuple | bool,)

    def __call__(
        self,
        question: str | None = None,
        *context: str | pl.DataFrame,
        options: list | str | None = None,
        **kwargs,
    ) -> tuple[str | int | float | tuple | exp.Expression | None]:
        in_deterministic_mode = bool(
            int(os.getenv(DETERMINISTIC_KEY, DEFAULT_DETERMINISTIC))
        )
        # Unpack kwargs
        # Extract single `context` from kwargs if provided
        if "context" in kwargs:
            context_kwarg = kwargs.pop("context")
            # Combine positional and keyword context
            if isinstance(context_kwarg, (list, tuple)):
                context = context + tuple(context_kwarg)
            else:
                context = context + (context_kwarg,)
        aliases_to_tablenames: dict[str, str] = kwargs["aliases_to_tablenames"]

        subtables: list[pl.DataFrame] = []
        for _context in context:
            if isinstance(_context, ColumnRef):
                tablename, colname = utils.get_tablename_colname(_context)
                tablename = aliases_to_tablenames.get(tablename, tablename)
                # Optionally materialize a CTE
                if tablename in self.db.lazy_tables:
                    materialized_smoothie = self.db.lazy_tables.pop(tablename).collect()
                    self.num_values_passed += (
                        materialized_smoothie.meta.num_values_passed
                    )
                    subtable = materialized_smoothie.pl.select([colname])
                    if isinstance(subtable, pl.LazyFrame):
                        subtable = subtable.collect()
                else:
                    subtable: pl.DataFrame = self.db.execute_to_df(
                        f'SELECT "{colname}" FROM "{tablename}"', lazy=False
                    )
            elif isinstance(_context, pl.DataFrame):
                subtable: pl.DataFrame = _context
            else:
                subtable = pl.DataFrame({"_col": _context})
            if subtable.is_empty():
                raise LMFunctionException("Empty subtable passed to QAIngredient!")
            self.num_values_passed += len(subtable)
            subtables.append(subtable)

        if options is not None:
            options = self.unpack_options(
                options=options,
                aliases_to_tablenames=aliases_to_tablenames,
                deterministic=in_deterministic_mode,
            )

        if question is not None and "{}" in question:
            if len(subtables) == 0:
                raise LMFunctionException(
                    f"Passed question with string template '{question}', but no context was passed to fill!"
                )
            unpacked_values = []
            for subtable in subtables:
                curr_values = (
                    subtable.to_series().to_list()
                    if subtable.width == 1
                    else list(subtable.to_pandas().values.flat)
                )
                if len(curr_values) > 1:
                    logger.debug(
                        Color.error(
                            f"More than 1 value found in {question}: {curr_values[:10]}\nThis could be a sign of a malformed query."
                        )
                    )
                unpacked_values.append(curr_values[0])
            question = question.format(*unpacked_values)
            logger.debug(Color.quiet_update(f"Unpacked question to '{question}'"))
            # This will now override whatever context we passed
            subtables = []

        response: [str | int | float | tuple] = self._run(
            question=question,
            context=subtables if subtables else None,
            options=options,
            **self.__dict__ | kwargs,
        )
        if isinstance(response, tuple):
            response = format_tuple(
                response, kwargs.get("wrap_tuple_in_parentheses", True)
            )
        return response

    @abstractmethod
    def run(self, *args, **kwargs) -> str | int | float | tuple:
        ...


@dataclass
class StringIngredient(Ingredient):
    """Outputs a string to be placed directly into the SQL query."""

    ingredient_type: str = IngredientType.STRING.value
    allowed_output_types: tuple[Type] = (str,)

    def unpack_default_kwargs(self, **kwargs):
        return unpack_default_kwargs(**kwargs)

    def __call__(self, identifier: str, *args, **kwargs) -> str:
        tablename, colname = utils.get_tablename_colname(identifier)
        kwargs["tablename"] = tablename
        kwargs["colname"] = colname
        # Don't pass identifier arg, we don't need it anymore
        args = tuple()
        new_str = self._run(*args, **kwargs)
        if not isinstance(new_str, str):
            raise LMFunctionException(
                f"{self.name}.run() should return str\nGot{type(new_str)}"
            )
        return new_str

    @abstractmethod
    def run(self, *args, **kwargs) -> str:
        ...
