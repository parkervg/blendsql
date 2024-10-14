from attr import attrs, attrib
from abc import abstractmethod
import pandas as pd
from sqlglot import exp
import json
from skrub import Joiner
from typing import Any, Union, Dict, Tuple, Callable, Set, Optional, Type, List
from collections.abc import Collection, Iterable
import uuid
from colorama import Fore
from typeguard import check_type
from functools import partialmethod, partial

from .._exceptions import IngredientException
from .._logger import logger
from .. import utils
from .._constants import (
    IngredientKwarg,
    IngredientType,
)
from ..db import Database
from ..db.utils import select_all_from_table_query
from ..models import Model
from .utils import unpack_options
from .few_shot import (
    Example,
    AnnotatedQAExample,
    AnnotatedMapExample,
    AnnotatedJoinExample,
)


def unpack_default_kwargs(**kwargs):
    return (
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


def initialize_retriever(
    examples: Example, k: int = None
) -> Callable[[str], List[Example]]:
    if k is None:
        # Just return all the examples everytime this is called
        return lambda *_: examples
    assert k < len(
        examples
    ), f"The `k` argument to an ingredient must be less than `len(few_shot_examples)`!\n`k` is {k}, `len(few_shot_examples)` is {len(examples)}"
    from .retriever import Retriever

    logger.debug(Fore.YELLOW + "Processing documents with haystack..." + Fore.RESET)
    retriever = Retriever(
        documents=[example.to_string() for example in examples], return_objs=examples
    )
    return partial(retriever.retrieve_top_k, k=k)


def partialclass(cls, *args, **kwds):
    # https://stackoverflow.com/a/38911383
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    NewCls.__name__ = cls.__name__
    return NewCls


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
class MapIngredient(Ingredient):
    """For a given table/column pair, maps an external function
    to each of the given values, creating a new column."""

    batch_size: int = attrib(default=None)

    ingredient_type: str = IngredientType.MAP.value
    allowed_output_types: Tuple[Type] = (Iterable[Any],)

    model: Model = attrib(default=None)

    @classmethod
    def from_args(
        cls,
        model: Optional[Model] = None,
        few_shot_examples: Optional[List[dict]] = None,
        list_options_in_prompt: bool = True,
        batch_size: Optional[int] = None,
        k: Optional[int] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            model (Optional[Model]): The model to be used. Defaults to None.
            few_shot_examples (Optional[List["MapExample"]]): A list of dictionary MapExample few-shot examples.
                If not specified, will use the below as default:
                    ```python
                    DEFAULT_MAP_FEW_SHOT: List[AnnotatedMapExample] = [
                        {
                            "question": "Total penalty count?",
                            "column_name": "Penalties (P+P+S+S)",
                            "table_name": "Biathlon World Championships 2013",
                            "output_type": "integer",
                            "example_outputs": ["12", "3"],
                            "examples": {
                                "1 (0+0+0+1)": "1",
                                "10 (5+3+2+0)": "10",
                                "6 (2+2+2+0)": "6",
                            },
                        },
                        {
                            "question": "Is the time less than a week?",
                            "column_name": "Length of use",
                            "table_name": "Crest Whitestrips",
                            "output_type": "boolean",
                            "example_outputs": ["t", "f"],
                            "examples": {"14 days": "f", "10 days": "f", "daily": "t", "2 hours": "t"},
                        },
                    ]
                    ```
            list_options_in_prompt (bool): Whether to list options in the prompt. Defaults to True.
            batch_size (Optional[int]): The batch size for processing. Defaults to None.
            k (Optional[int]): Determines number of few-shot examples to use for each ingredient call.
                Default is None, which will use all few-shot examples on all calls.
                If specified, will initialize a haystack-based DPR retriever to filter examples.

        Returns:
            Type[MapIngredient]: A partial class of MapIngredient with predefined arguments.
        """
        if few_shot_examples:
            few_shot_examples = [AnnotatedMapExample(**d) for d in few_shot_examples]
        few_shot_retriever = initialize_retriever(examples=few_shot_examples, k=k)
        return partialclass(
            cls,
            model=model,
            few_shot_retriever=few_shot_retriever,
            list_options_in_prompt=list_options_in_prompt,
            batch_size=batch_size,
        )

    def unpack_default_kwargs(self, **kwargs):
        return unpack_default_kwargs(**kwargs)

    def __call__(
        self,
        question: Optional[str] = None,
        context: Optional[str] = None,
        regex: Optional[Callable] = None,
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
        else:
            kwargs[IngredientKwarg.REGEX] = regex
        kwargs[IngredientKwarg.VALUES] = values
        kwargs[IngredientKwarg.QUESTION] = question
        mapped_values: Collection[Any] = self._run(
            *args,
            **kwargs,
            options=unpacked_options,
            few_shot_retriever=self.few_shot_retriever,
            list_options_in_prompt=self.list_options_in_prompt,
            batch_size=self.batch_size,
        )
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
    """Executes an `INNER JOIN` using dict mapping.
    Example:
        'Join on color of food'
        {"tomato": "red", "broccoli": "green", "lemon": "yellow"}
    """

    use_skrub_joiner: bool = attrib(default=True)

    ingredient_type: str = IngredientType.JOIN.value
    allowed_output_types: Tuple[Type] = (dict,)

    @classmethod
    def from_args(
        cls,
        few_shot_examples: List[dict] = None,
        use_skrub_joiner: bool = True,
        k: Optional[int] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            few_shot_examples (List[AnnotatedJoinExamples]): A list of AnnotatedJoinExamples dictionaries for few-shot learning.
                If not specified, will use the below as default:
                    ```python
                    DEFAULT_JOIN_FEW_SHOT: List[AnnotatedJoinExample] = [
                        {
                            "join_criteria": "Join to same topics.",
                            "left_values": ["joshua fields", "bob brown", "ron ryan"],
                            "right_values": [
                                "ron ryan",
                                "colby mules",
                                "bob brown (ice hockey)",
                                "josh fields (pitcher)",
                            ],
                            "mapping": {
                                "joshua fields": "josh fields (pitcher)",
                                "bob brown": "bob brown (ice hockey)",
                                "ron ryan": "ron ryan",
                            },
                        }
                    ]
                    ```
            use_skrub_joiner (bool): Whether to use the skrub joiner. Defaults to True.

        Returns:
            Type[JoinIngredient]: A partial class of JoinIngredient with predefined arguments.
        """
        if few_shot_examples:
            few_shot_examples = [AnnotatedJoinExample(**d) for d in few_shot_examples]
        few_shot_retriever = initialize_retriever(examples=few_shot_examples, k=k)
        return partialclass(
            cls,
            few_shot_retriever=few_shot_retriever,
            use_skrub_joiner=use_skrub_joiner,
        )

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
                *args, **kwargs, few_shot_retriever=self.few_shot_retriever
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
    context_formatter: Callable[[pd.DataFrame], str] = attrib(
        default=lambda df: df.to_markdown(index=False)
    )

    ingredient_type: str = IngredientType.QA.value
    allowed_output_types: Tuple[Type] = (Union[str, int, float],)

    @classmethod
    def from_args(
        cls,
        few_shot_examples: Optional[List[dict]] = None,
        context_formatter: Callable[[pd.DataFrame], str] = lambda df: df.to_markdown(
            index=False
        ),
        k: Optional[int] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            few_shot_examples (List[AnnotatedQAExample]): A list of AnnotatedQAExample dictionaries for few-shot learning.
                If not specified, will use the below as default:
                    ```python
                    DEFAULT_QA_FEW_SHOT: List[AnnotatedQAExample] = [
                        {
                            "question": "Who is the oldest?",
                            "context": pd.DataFrame(
                                data=[["Parker", 26], ["Andrew", 22], ["Paul", 18]], columns=["Name", "Age"]
                            ),
                            "options": ["Parker", "Andrew", "Paul"],
                            "answer": "Parker",
                        }
                    ]
                    ```
            context_formatter (Callable[[pd.DataFrame], str]): A callable that formats a pandas DataFrame into a string.
                Defaults to a lambda function that converts the DataFrame to markdown without index.

        Returns:
            Type[QAIngredient]: A partial class of QAIngredient with predefined arguments.
        """
        if few_shot_examples:
            few_shot_examples = [AnnotatedQAExample(**d) for d in few_shot_examples]
        few_shot_retriever = initialize_retriever(examples=few_shot_examples, k=k)
        return partialclass(
            cls,
            few_shot_retriever=few_shot_retriever,
            context_formatter=context_formatter,
        )

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
        response: Union[str, int, float] = self._run(
            *args,
            **kwargs,
            few_shot_retriever=self.few_shot_retriever,
            context_formatter=self.context_formatter,
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
