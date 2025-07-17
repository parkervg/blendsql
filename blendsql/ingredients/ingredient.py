import re
from attr import attrs, attrib
from abc import abstractmethod
import pandas as pd
from sqlglot import exp
import json
import typing as t
from collections.abc import Collection, Iterable
import uuid
from colorama import Fore
from typeguard import check_type

from blendsql.common.exceptions import IngredientException
from blendsql.common.logger import logger
from blendsql.common import utils
from blendsql.common.constants import (
    IngredientType,
)
from blendsql.db import Database
from blendsql.db.utils import select_all_from_table_query, format_tuple
from blendsql.common.utils import get_tablename_colname
from blendsql.search.searcher import Searcher
from blendsql.ingredients.few_shot import Example
from blendsql.common.constants import ColumnRef


def unpack_default_kwargs(**kwargs):
    return (
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


@attrs
class Ingredient:
    name: str = attrib()
    # Below gets passed via `Kitchen.extend()`
    db: Database = attrib()
    session_uuid: str = attrib()

    few_shot_retriever: t.Callable[[str], t.List[Example]] = attrib(default=None)
    list_options_in_prompt: bool = attrib(default=True)
    searcher: t.Optional[Searcher] = attrib(default=None)
    enable_constrained_decoding: bool = attrib(default=True)

    ingredient_type: str = attrib(init=False)
    allowed_output_types: t.Tuple[t.Type] = attrib(init=False)

    num_values_passed: int = 0

    def __repr__(self):
        return f"{self.ingredient_type} {self.name}"

    def __str__(self):
        return f"{self.ingredient_type} {self.name}"

    @abstractmethod
    def run(self, *args, **kwargs) -> t.Any:
        ...

    @abstractmethod
    def __call__(self, *args, **kwargs) -> t.Any:
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
                                Fore.MAGENTA
                                + f"Loading custom {partial_cls.__name__} with name '{variable_name}'..."
                                + Fore.RESET
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
        aliases_to_tablenames: t.Dict[str, str],
    ) -> t.List[str]:
        tablename, colname = get_tablename_colname(v)
        tablename = aliases_to_tablenames.get(tablename, tablename)
        # IMPORTANT: Below was commented out, since it would cause:
        #   `SELECT {{select_first_sorted(options='w::Symbol')}} FROM w LIMIT 1`
        #   ...to always select the result of the `LIMIT 1`.
        # Check for previously created temporary tables
        # value_source_tablename, _ = self.maybe_get_temp_table(
        #     temp_table_func=get_temp_subquery_table, tablename=tablename
        # )
        # Optionally materialize a CTE
        if tablename in self.db.lazy_tables:
            materialized_smoothie = self.db.lazy_tables.pop(tablename).collect()
            self.num_values_passed += materialized_smoothie.meta.num_values_passed
            unpacked_values = [
                str(i) for i in materialized_smoothie.df[colname].unique()
            ]
        else:
            unpacked_values: list = [
                str(i)
                for i in self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{tablename}"'
                )
            ]
        return list(set(unpacked_values))

    def maybe_get_temp_table(
        self, temp_table_func: t.Callable, tablename: str
    ) -> t.Tuple[str, bool]:
        temp_tablename = temp_table_func(tablename)
        if self.db.has_temp_table(temp_tablename):
            # We've already applied some operation to this table
            # We want to use this as our base
            return (temp_tablename, True)
        return (tablename, False)

    def unpack_options(
        self,
        options: t.Union[ColumnRef, list],
        aliases_to_tablenames: t.Dict[str, str],
    ) -> t.Union[t.List[str], None]:
        unpacked_options = options
        if isinstance(options, ColumnRef):
            unpacked_options = self.unpack_column_ref(options, aliases_to_tablenames)
        if len(unpacked_options) == 0:
            logger.debug(
                Fore.LIGHTRED_EX
                + f"Tried to unpack options '{options}', but got an empty list\nThis may be a bug. Please report it."
                + Fore.RESET
            )
        return list(unpacked_options) if len(unpacked_options) > 0 else None

    def unpack_question(
        self,
        question: str,
        values: t.List[str],
    ) -> t.List[str]:
        """Unpack any f-string ValueArray references in question.

        Example:
            question='Where is {} located?', values=["Sydney", "San Jose"]
        """
        if "{}" in question:
            return [question.format(value) for value in values]
        return [question for _ in range(len(values))]


@attrs
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
    allowed_output_types: t.Tuple[t.Type] = (t.Tuple[str, Collection[Ingredient]],)

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


@attrs
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
    allowed_output_types: t.Tuple[t.Type] = (t.Iterable[t.Any],)

    def unpack_default_kwargs(self, **kwargs):
        return unpack_default_kwargs(**kwargs)

    def __call__(
        self,
        question: t.Optional[str] = None,
        values: t.Optional[t.Union[ColumnRef]] = None,
        options: t.Optional[t.Union[ColumnRef, list]] = None,
        *args,
        **kwargs,
    ) -> tuple:
        """Returns tuple with format (arg, tablename, colname, new_table)"""
        # Unpack kwargs
        aliases_to_tablenames: t.Dict[str, str] = kwargs["aliases_to_tablenames"]
        get_temp_subquery_table: t.Callable = kwargs["get_temp_subquery_table"]
        get_temp_session_table: t.Callable = kwargs["get_temp_session_table"]
        prev_subquery_map_columns: t.Set[str] = kwargs["prev_subquery_map_columns"]

        # TODO: make sure we support all types of ValueArray references here
        tablename, colname = utils.get_tablename_colname(values)
        tablename = aliases_to_tablenames.get(tablename, tablename)

        # Check for previously created temporary tables
        value_source_tablename, _ = self.maybe_get_temp_table(
            temp_table_func=get_temp_subquery_table, tablename=tablename
        )
        temp_session_tablename, temp_session_table_exists = self.maybe_get_temp_table(
            temp_table_func=get_temp_session_table, tablename=tablename
        )

        # Optionally materialize a CTE
        if tablename in self.db.lazy_tables:
            materialized_smoothie = self.db.lazy_tables.pop(tablename).collect()
            self.num_values_passed += materialized_smoothie.meta.num_values_passed
            original_table = materialized_smoothie.df
        else:
            original_table = self.db.execute_to_df(
                select_all_from_table_query(tablename)
            )

        # Need to be sure the new column doesn't already exist here
        new_arg_column = question or str(uuid.uuid4())[:4]
        while (
            new_arg_column in set(self.db.iter_columns(tablename))
            or new_arg_column in prev_subquery_map_columns
        ):
            new_arg_column = "_" + new_arg_column

        # Get a list of values to map
        # First, check if we've already dumped some `MapIngredient` output to the main session table
        if temp_session_table_exists:
            temp_session_table = self.db.execute_to_df(
                select_all_from_table_query(temp_session_tablename)
            )
            # We don't need to run this function on everything,
            #   if a previous subquery already got to certain values
            if new_arg_column in temp_session_table.columns:
                unpacked_values = self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}" WHERE "{new_arg_column}" IS NULL',
                )
            # Base case: this is the first time we've used this particular ingredient
            # BUT, temp_session_tablename still exists
            else:
                unpacked_values = self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}"',
                )
        else:
            unpacked_values = self.db.execute_to_list(
                f'SELECT DISTINCT "{colname}" FROM "{value_source_tablename}"',
            )

        # No need to run ingredient if we have no values to map onto
        if len(unpacked_values) == 0:
            original_table[new_arg_column] = None
            return (new_arg_column, tablename, colname, original_table)

        if options is not None:
            # Override any pattern with our new unpacked options
            options = self.unpack_options(
                options=options,
                aliases_to_tablenames=aliases_to_tablenames,
            )

        unpacked_questions = None
        if question is not None:
            if "{}" in question:
                unpacked_questions = [
                    question.format(value) for value in unpacked_values
                ]
                logger.debug(
                    Fore.LIGHTBLACK_EX
                    + f"Unpacked question to '{unpacked_questions[:10]}'"
                    + Fore.RESET
                )

        mapped_values: Collection[t.Any] = self._run(
            question=question,
            unpacked_questions=unpacked_questions,
            values=unpacked_values,
            options=options,
            tablename=tablename,
            colname=colname,
            *args,
            **self.__dict__ | kwargs,
        )
        self.num_values_passed += len(mapped_values)
        df_as_dict: t.Dict[str, list] = {colname: [], new_arg_column: []}
        for value, mapped_value in zip(unpacked_values, mapped_values):
            df_as_dict[colname].append(value)
            df_as_dict[new_arg_column].append(mapped_value)
        subtable = pd.DataFrame(df_as_dict)
        if all(
            isinstance(x, (int, type(None))) and not isinstance(x, bool)
            for x in mapped_values
        ):
            subtable[new_arg_column] = subtable[new_arg_column].astype("Int64")
        # Add new_table to original table
        new_table = original_table.merge(subtable, how="left", on=colname)
        if new_table.shape[0] != original_table.shape[0]:
            raise IngredientException(
                f"subtable from MapIngredient.run() needs same length as # rows from original\nOriginal has {original_table.shape[0]}, new_table has {new_table.shape[0]}"
            )
        # Now, new table has original columns + column with the name of the question we answered
        return (new_arg_column, tablename, colname, new_table)

    @abstractmethod
    def run(self, *args, **kwargs) -> Iterable[t.Any]:
        ...


@attrs
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

    use_skrub_joiner: bool = attrib(default=True)

    ingredient_type: str = IngredientType.JOIN.value
    allowed_output_types: t.Tuple[t.Type] = (dict,)

    def __call__(
        self,
        left_on: t.Optional[str] = None,
        right_on: t.Optional[str] = None,
        join_criteria: t.Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple:
        # Unpack kwargs
        aliases_to_tablenames: t.Dict[str, str] = kwargs["aliases_to_tablenames"]
        get_temp_subquery_table: t.Callable = kwargs["get_temp_subquery_table"]
        get_temp_session_table: t.Callable = kwargs["get_temp_session_table"]
        # Depending on the size of the underlying data, it may be optimal to swap
        #   the order of 'left_on' and 'right_on' columns during processing
        swapped = False
        values = []
        original_lr_identifiers = []
        modified_lr_identifiers = []
        mapping: t.Dict[str, str] = {}
        for on_arg in [left_on, right_on]:
            # Since LLMJoin is unique, in that we need to inject the referenced tablenames back to the query,
            #   make sure we keep the `referenced_tablename` variable.
            # So the below works:
            #     SELECT f.name, colors.name FROM fruits f
            #     JOIN {{LLMJoin('f::name', 'colors::name', join_criteria='Align the fruit to its color')}}
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
        sorted_values = sorted(values, key=len)
        # check swapping only once, at the beginning
        if sorted_values != values:
            swapped = True
        if join_criteria is None:
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
                    Fore.YELLOW
                    + "Made the following alignment with `skrub.Joiner`:"
                    + Fore.RESET
                )
                logger.debug(
                    Fore.YELLOW + json.dumps(_skrub_mapping, indent=4) + Fore.RESET
                )
                mapping = mapping | _skrub_mapping
            # order by length is still preserved regardless of using fuzzy join, so after initial matching and possible fuzzy join matching
            # This is because the lengths of each list will decrease at the same rate, so whichever list was larger at the beginning,
            # will be larger here at the end.
            # len(_outer) <= len(inner)
            sorted_values = [_outer, inner]

        # Now, we have our final values to process.
        left_values, right_values = sorted_values

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

            _predicted_mapping: t.Dict[str, str] = self._run(
                left_values=left_values,
                right_values=right_values,
                join_criteria=join_criteria,
                *args,
                **self.__dict__ | kwargs,
            )
            mapping = mapping | _predicted_mapping
        # Using mapped left/right values, create intermediary mapping table
        temp_join_tablename = get_temp_session_table(str(uuid.uuid4())[:4])
        # Below, we check to see if 'swapped' is True
        # If so, we need to inverse what is 'left', and what is 'right'
        joined_values_df = pd.DataFrame(
            data={
                "left" if not swapped else "right": mapping.keys(),
                "right" if not swapped else "left": mapping.values(),
            }
        )
        self.db.to_temp_table(df=joined_values_df, tablename=temp_join_tablename)
        if right_tablename in aliases_to_tablenames:
            join_right_clause = f"""JOIN "{aliases_to_tablenames[right_tablename]}" AS "{right_tablename}" ON "{right_tablename}"."{right_colname}" = "{temp_join_tablename}".right"""
        else:
            join_right_clause = f"""JOIN "{right_tablename}" ON "{right_tablename}"."{right_colname}" = "{temp_join_tablename}".right"""
        return (
            left_tablename,
            right_tablename,
            f"""JOIN "{temp_join_tablename}" ON "{left_tablename}"."{left_colname}" = "{temp_join_tablename}".left\n{join_right_clause}""",
            temp_join_tablename,
        )

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        ...


@attrs
class QAIngredient(Ingredient):
    """
    Given a table subset in the form of a pd.DataFrame 'context',
    returns a scalar or array of scalars (in the form of a tuple).

    Useful for end-to-end question answering tasks.
    """

    ingredient_type: str = IngredientType.QA.value
    allowed_output_types: t.Tuple[t.Type] = (t.Union[str, int, float, tuple],)

    def __call__(
        self,
        question: t.Optional[str] = None,
        *context: t.Union[str, pd.DataFrame],
        options: t.Optional[t.Union[list, str]] = None,
        **kwargs,
    ) -> t.Tuple[t.Union[str, int, float, tuple], t.Optional[exp.Expression]]:
        # Unpack kwargs
        # Extract single `context` from kwargs if provided
        if "context" in kwargs:
            context_kwarg = kwargs.pop("context")
            # Combine positional and keyword context
            if isinstance(context_kwarg, (list, tuple)):
                context = context + tuple(context_kwarg)
            else:
                context = context + (context_kwarg,)
        aliases_to_tablenames: t.Dict[str, str] = kwargs["aliases_to_tablenames"]

        subtables: t.List[pd.DataFrame] = []
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
                    subtable: pd.DataFrame = pd.DataFrame(
                        materialized_smoothie.df[colname]
                    )
                else:
                    subtable: pd.DataFrame = self.db.execute_to_df(
                        f'SELECT "{colname}" FROM "{tablename}"'
                    )
            elif isinstance(_context, pd.DataFrame):
                subtable: pd.DataFrame = _context
            else:
                subtable = pd.DataFrame([{"_col": _context}])
            if subtable.empty:
                raise IngredientException("Empty subtable passed to QAIngredient!")
            self.num_values_passed += len(subtable)
            subtables.append(subtable)

        if options is not None:
            options = self.unpack_options(
                options=options,
                aliases_to_tablenames=aliases_to_tablenames,
            )

        if question is not None:
            if "{}" in question:
                if len(subtables) == 0:
                    raise IngredientException(
                        f"Passed question with string template '{question}', but no context was passed to fill!"
                    )
                unpacked_values = []
                for subtable in subtables:
                    curr_values = list(subtable.values.flat)
                    if len(curr_values) > 1:
                        logger.debug(
                            Fore.RED
                            + f"More than 1 value found in {question}: {curr_values[:10]}\nThis could be a sign of a malformed query."
                            + Fore.RESET
                        )
                    unpacked_values.append(curr_values[0])
                question = question.format(*unpacked_values)
                logger.debug(
                    Fore.LIGHTBLACK_EX
                    + f"Unpacked question to '{question}'"
                    + Fore.RESET
                )
                # This will now override whatever context we passed
                subtables = []

        response: t.Union[str, int, float, tuple] = self._run(
            question=question,
            context=subtables if len(subtables) > 0 else None,
            options=options,
            **self.__dict__ | kwargs,
        )
        if isinstance(response, tuple):
            response = format_tuple(
                response, kwargs.get("wrap_tuple_in_parentheses", True)
            )
        return response

    @abstractmethod
    def run(self, *args, **kwargs) -> t.Union[str, int, float, tuple]:
        ...


@attrs
class StringIngredient(Ingredient):
    """Outputs a string to be placed directly into the SQL query."""

    ingredient_type: str = IngredientType.STRING.value
    allowed_output_types: t.Tuple[t.Type] = (str,)

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
            raise IngredientException(
                f"{self.name}.run() should return str\nGot{type(new_str)}"
            )
        return new_str

    @abstractmethod
    def run(self, *args, **kwargs) -> str:
        ...
