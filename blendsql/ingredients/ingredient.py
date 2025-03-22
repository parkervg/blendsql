from attr import attrs, attrib
from abc import abstractmethod
import pandas as pd
from sqlglot import exp
import json
from typing import Any, Union, Dict, Tuple, Callable, Set, Optional, Type, List
from collections.abc import Collection, Iterable
import uuid
from colorama import Fore
from typeguard import check_type

from .._exceptions import IngredientException
from .._logger import logger
from .. import utils
from .._constants import (
    IngredientKwarg,
    IngredientType,
)
from ..db import Database
from ..db.utils import select_all_from_table_query, format_tuple
from .utils import unpack_options
from .few_shot import Example
from .utils import partialclass


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

    few_shot_retriever: Callable[[str], List[Example]] = attrib(default=None)
    list_options_in_prompt: bool = attrib(default=True)

    ingredient_type: str = attrib(init=False)
    allowed_output_types: Tuple[Type] = attrib(init=False)

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

    def maybe_get_temp_table(
        self, temp_table_func: Callable, tablename: str
    ) -> Tuple[str, bool]:
        temp_tablename = temp_table_func(tablename)
        _tablename = tablename
        if self.db.has_temp_table(temp_tablename):
            # We've already applied some operation to this table
            # We want to use this as our base
            _tablename = temp_tablename
        return (_tablename, True) if _tablename != tablename else (_tablename, False)


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
    allowed_output_types: Tuple[Type] = Tuple[str, Collection[Ingredient]]

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
                from blendsql import blend
                from blendsql.db import SQLite
                from blendsql.utils import fetch_from_hub

                blendsql = "SELECT genre, url, {{GetQRCode('QR Code as Bytes:', 'w::url')}} FROM w WHERE genre = 'social'"

                smoothie = blend(
                    query=blendsql,
                    default_model=None,
                    db=SQLite(fetch_from_hub("urls.db")),
                    ingredients={GetQRCode}
                )
                # | genre  | url           | QR Code as Bytes:      |
                # |--------|---------------|-----------------------|
                # | social | facebook.com  | b'...'                |
        ```
    '''

    ingredient_type: str = IngredientType.MAP.value
    allowed_output_types: Tuple[Type] = (Iterable[Any],)

    def unpack_default_kwargs(self, **kwargs):
        return unpack_default_kwargs(**kwargs)

    def __call__(
        self,
        question: Optional[str] = None,
        context: Optional[str] = None,
        # regex: Optional[Callable] = None,
        options: Optional[Union[list, str]] = None,
        *args,
        **kwargs,
    ) -> tuple:
        """Returns tuple with format (arg, tablename, colname, new_table)"""
        # Unpack kwargs
        aliases_to_tablenames: Dict[str, str] = kwargs["aliases_to_tablenames"]
        get_temp_subquery_table: Callable = kwargs["get_temp_subquery_table"]
        get_temp_session_table: Callable = kwargs["get_temp_session_table"]
        prev_subquery_map_columns: Set[str] = kwargs["prev_subquery_map_columns"]

        tablename, colname = utils.get_tablename_colname(context)
        tablename = aliases_to_tablenames.get(tablename, tablename)
        kwargs["tablename"] = tablename
        kwargs["colname"] = colname
        # Check for previously created temporary tables
        value_source_tablename, _ = self.maybe_get_temp_table(
            temp_table_func=get_temp_subquery_table, tablename=tablename
        )
        temp_session_tablename, temp_session_table_exists = self.maybe_get_temp_table(
            temp_table_func=get_temp_session_table, tablename=tablename
        )

        # Optionally materialize a CTE
        if tablename in self.db.lazy_tables:
            original_table = self.db.lazy_tables.pop(tablename).collect()
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
                values = self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}" WHERE "{new_arg_column}" IS NULL',
                )
            # Base case: this is the first time we've used this particular ingredient
            # BUT, temp_session_tablename still exists
            else:
                values = self.db.execute_to_list(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}"',
                )
        else:
            values = self.db.execute_to_list(
                f'SELECT DISTINCT "{colname}" FROM "{value_source_tablename}"',
            )

        # No need to run ingredient if we have no values to map onto
        if len(values) == 0:
            original_table[new_arg_column] = None
            return (new_arg_column, tablename, colname, original_table)

        unpacked_options = None
        if options is not None:
            # Override any pattern with our new unpacked options
            unpacked_options = list(
                unpack_options(
                    options=options,
                    aliases_to_tablenames=aliases_to_tablenames,
                    db=self.db,
                )
            )
        # else:
        #     kwargs[IngredientKwarg.REGEX] = regex
        kwargs[IngredientKwarg.VALUES] = values
        kwargs[IngredientKwarg.QUESTION] = question
        kwargs[IngredientKwarg.OPTIONS] = unpacked_options
        mapped_values: Collection[Any] = self._run(*args, **self.__dict__ | kwargs)
        self.num_values_passed += len(mapped_values)
        df_as_dict: Dict[str, list] = {colname: [], new_arg_column: []}
        for value, mapped_value in zip(values, mapped_values):
            df_as_dict[colname].append(value)
            df_as_dict[new_arg_column].append(mapped_value)
        subtable = pd.DataFrame(df_as_dict)
        # if kwargs.get("output_type") == "boolean":
        #     subtable[new_arg_column] = subtable[new_arg_column].astype(bool)
        # else:
        if all(
            isinstance(x, (int, type(None))) and not isinstance(x, bool)
            for x in mapped_values
        ):
            subtable[new_arg_column] = subtable[new_arg_column].astype("Int64")
        # Add new_table to original table
        new_table = original_table.merge(subtable, how="left", on=colname)
        if new_table.shape[0] != original_table.shape[0]:
            raise IngredientException(
                f"subtable from run() needs same length as # rows from original\nOriginal has {original_table.shape[0]}, new_table has {new_table.shape[0]}"
            )
        # Now, new table has original columns + column with the name of the question we answered
        return (new_arg_column, tablename, colname, new_table)

    @abstractmethod
    def run(self, *args, **kwargs) -> Iterable[Any]:
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
        SELECT Account, Quantity FROM returns
        JOIN {{
            do_join(
                left_on='account_history::Symbol',
                right_on='returns::Symbol'
            )
        }}
        """
        ```
    '''

    use_skrub_joiner: bool = attrib(default=True)

    ingredient_type: str = IngredientType.JOIN.value
    allowed_output_types: Tuple[Type] = (dict,)

    @classmethod
    def from_args(cls, use_skrub_joiner: bool = True):
        return partialclass(cls, use_skrub_joiner=use_skrub_joiner)

    def __call__(
        self,
        question: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        *args,
        **kwargs,
    ) -> tuple:
        # Unpack kwargs
        aliases_to_tablenames: Dict[str, str] = kwargs["aliases_to_tablenames"]
        get_temp_subquery_table: Callable = kwargs["get_temp_subquery_table"]
        get_temp_session_table: Callable = kwargs["get_temp_session_table"]
        # Depending on the size of the underlying data, it may be optimal to swap
        #   the order of 'left_on' and 'right_on' columns during processing
        swapped = False
        values = []
        original_lr_identifiers = []
        modified_lr_identifiers = []
        mapping: Dict[str, str] = {}
        for on_arg in [left_on, right_on]:
            tablename, colname = utils.get_tablename_colname(on_arg)
            tablename = aliases_to_tablenames.get(tablename, tablename)
            original_lr_identifiers.append((tablename, colname))
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
        if question is None:
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

        left_values, right_values = sorted_values
        kwargs["left_values"] = left_values
        kwargs["right_values"] = right_values

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
            self.num_values_passed += len(kwargs["left_values"]) + len(
                kwargs["right_values"]
            )

            kwargs[IngredientKwarg.QUESTION] = question
            _predicted_mapping: Dict[str, str] = self._run(
                *args, **self.__dict__ | kwargs
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
        return (
            left_tablename,
            right_tablename,
            f"""JOIN "{temp_join_tablename}" ON "{right_tablename}"."{right_colname}" = "{temp_join_tablename}".right
               JOIN "{left_tablename}" ON "{left_tablename}"."{left_colname}" = "{temp_join_tablename}".left
               """,
            temp_join_tablename,
        )

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        ...


@attrs
class QAIngredient(Ingredient):
    '''
    Given a table subset in the form of a pd.DataFrame 'context',
    returns a scalar or array of scalars (in the form of a tuple).

    Useful for end-to-end question answering tasks.

    Examples:
        ```python
        import pandas as pd
        import guidance

        from blendsql.models import Model, LocalModel, RemoteModel
        from blendsql.ingredients import QAIngredient
        from blendsql._program import Program


        class SummaryProgram(Program):
            """Program to call Model and return summary of the passed table.
            """

            def __call__(self, model: Model, serialized_db: str):
                prompt = f"Summarize the table. {serialized_db}"
                if isinstance(model, LocalModel):
                    # Below we follow the guidance pattern for unconstrained text generation
                    # https://github.com/guidance-ai/guidance
                    response = (model.model_obj + guidance.gen(max_tokens=20, name="response"))._variables["response"]
                else:
                    response = model.generate(
                        messages_list=[[{"role": "user", "content": prompt}]],
                        max_tokens=20
                    )[0]
                # Finally, return (response, prompt) tuple
                # Returning the prompt here allows the underlying BlendSQL classes to track token usage
                return (response, prompt)


            class TableSummary(QAIngredient):
                def run(self, model: Model, context: pd.DataFrame, **kwargs) -> str:
                    result = model.predict(program=SummaryProgram, serialized_db=context.to_string())
                    return f"'{result}'"


            if __name__ == "__main__":
                from blendsql import blend
                from blendsql.db import SQLite
                from blendsql.utils import fetch_from_hub
                from blendsql.models import LiteLLM

                blendsql = """
                SELECT {{
                    TableSummary(
                        context=(SELECT * FROM transactions LIMIT 10)
                    )
                }} AS "Summary"
                """

                smoothie = blend(
                    query=blendsql,
                    default_model=LiteLLM("openai/gpt-4o-mini"),
                    db=SQLite(fetch_from_hub("single_table.db")),
                    ingredients={TableSummary}
                )
                # Now, we can get results
                print(smoothie.df)
                # 'The table summarizes a series of cash flow transactions made through Zelle'
                # ...and token usage
                print(smoothie.meta.prompt_tokens)
                print(smoothie.meta.completion_tokens)
        ```
    '''

    ingredient_type: str = IngredientType.QA.value
    allowed_output_types: Tuple[Type] = (Union[str, int, float, tuple],)

    def __call__(
        self,
        question: Optional[str] = None,
        context: Optional[Union[str, pd.DataFrame]] = None,
        options: Optional[Union[list, str]] = None,
        *args,
        **kwargs,
    ) -> Tuple[Union[str, int, float], Optional[exp.Expression]]:
        # Unpack kwargs
        aliases_to_tablenames: Dict[str, str] = kwargs["aliases_to_tablenames"]

        subtable: Union[pd.DataFrame, None] = None
        if context is not None:
            if isinstance(context, str):
                tablename, colname = utils.get_tablename_colname(context)
                # Optionally materialize a CTE
                if tablename in self.db.lazy_tables:
                    subtable: pd.DataFrame = self.db.lazy_tables.pop(
                        tablename
                    ).collect()[colname]
                else:
                    subtable: pd.DataFrame = self.db.execute_to_df(
                        f'SELECT "{colname}" FROM "{tablename}"'
                    )
            elif isinstance(context, pd.DataFrame):
                subtable: pd.DataFrame = context
            else:
                raise ValueError(
                    f"Unknown type for `identifier` arg in QAIngredient: {type(context)}"
                )
            if subtable.empty:
                raise IngredientException("Empty subtable passed to QAIngredient!")

        if options is not None:
            kwargs[IngredientKwarg.OPTIONS] = unpack_options(
                options=options, aliases_to_tablenames=aliases_to_tablenames, db=self.db
            )
        else:
            kwargs[IngredientKwarg.OPTIONS] = None

        self.num_values_passed += len(subtable) if subtable is not None else 0
        kwargs[IngredientKwarg.CONTEXT] = subtable
        kwargs[IngredientKwarg.QUESTION] = question
        response: Union[str, int, float, tuple] = self._run(
            *args, **self.__dict__ | kwargs
        )
        if isinstance(response, tuple):
            response = format_tuple(
                response, kwargs.get("wrap_tuple_in_parentheses", True)
            )
        return response

    @abstractmethod
    def run(self, *args, **kwargs) -> Union[str, int, float]:
        ...


@attrs
class StringIngredient(Ingredient):
    """Outputs a string to be placed directly into the SQL query."""

    ingredient_type: str = IngredientType.STRING.value
    allowed_output_types: Tuple[Type] = (str,)

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
