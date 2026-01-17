import copy
import logging
import time
import uuid
import pandas as pd
import re
from typing import Callable, Type, Generator
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
import sqlglot
from functools import partial
from sqlglot import exp
import string
from pathlib import Path
import polars as pl

from blendsql.common.logger import logger, Color
from blendsql.common.utils import (
    get_temp_session_table,
    get_temp_subquery_table,
)
from blendsql.common.exceptions import InvalidBlendSQL
from blendsql.db.database import Database
from blendsql.db.utils import (
    double_quote_escape,
    select_all_from_table_query,
    LazyTable,
)
from blendsql.parse import (
    get_dialect,
    QueryContextManager,
    SubqueryContextManager,
    _parse_one,
    transform,
    check,
    get_reversed_subqueries,
    get_first_child,
)
from blendsql.parse.cascade_filter import get_qa_cascade_filter, get_map_cascade_filter
from blendsql.parse.constants import MODIFIERS
from blendsql.ingredients.ingredient import Ingredient, LMFunctionException
from blendsql.smoothie import Smoothie, SmoothieMeta
from blendsql.common.typing import (
    IngredientType,
    Subquery,
    ColumnRef,
    StringConcatenation,
)
from blendsql.models.model import Model

format_blendsql_function = lambda name: "{{" + name + "()}}"


@dataclass
class Kitchen(list):
    """Superset of list. A collection of ingredients."""

    db: Database = field()
    session_uuid: str = field()

    name_to_ingredient: dict[str, Ingredient] = field(init=False)

    def __post_init__(self):
        self.name_to_ingredient = {}

    def get_from_name(self, name: str):
        try:
            return self.name_to_ingredient[name.upper()]
        except KeyError:
            raise InvalidBlendSQL(
                f"Ingredient '{name}' called, but not found in passed `ingredient` arg!"
            ) from None

    def extend(self, ingredients: Iterable[Type[Ingredient]]) -> None:
        """Initializes ingredients class with base attributes, for use in later operations."""
        try:
            if not all(issubclass(x, Ingredient) for x in ingredients):
                raise LMFunctionException(
                    "All arguments passed to `Kitchen` must be ingredients!"
                )
        except TypeError:
            raise LMFunctionException(
                "All arguments passed to `Kitchen` must be ingredients!"
            ) from None
        for ingredient in ingredients:
            name = ingredient.__name__.upper()
            # Initialize the ingredient, going from `Type[Ingredient]` to `Ingredient`
            initialized_ingredient: Ingredient = ingredient(
                name=name,
                # Add db and session_uuid as default kwargs
                # This way, ingredients are able to interact with data
                db=self.db,
                session_uuid=self.session_uuid,
            )
            self.name_to_ingredient[name] = initialized_ingredient
            self.append(initialized_ingredient)


def autowrap_query(
    query: str, kitchen: Kitchen, ingredient_alias_to_parsed_dict: dict[str, dict]
) -> str:
    """
    Check to see if we have some BlendSQL ingredient syntax that needs to be formatted differently
        before passing to sqlglot.parse_one.
        - A single `QAIngredient` should be prefaced by a `SELECT`.
        - A `JoinIngredient` needs to include a reference to the left tablename.

    Args:
        query: The BlendSQL query to wrap in SQLite logic
        kitchen: Our collection of ingredients
        ingredient_alias_to_parsed_dict: Mapping from abbreviates alias to pyparsing output

    Returns:
        _query: sqlglot Expression for the new BlendSQL query, wrapped in SQLite logic
        original_query: A copy of the original (string) _query
    """
    for alias, d in ingredient_alias_to_parsed_dict.items():
        current_function: Ingredient = kitchen.get_from_name(d["function"])
        if current_function.ingredient_type == IngredientType.QA:
            # If the query only contains the function alias
            # E.g: '{{A()}}'
            if query == format_blendsql_function(alias):
                query = f"""SELECT {query}"""
        else:
            continue
    return query


def preprocess_blendsql(
    node: exp.Exp,
    dialect: sqlglot.Dialect,
    kitchen: Kitchen,
    ingredients: Collection[Ingredient],
    default_model: Model,
) -> tuple[str, dict, set, Kitchen, set[Ingredient]]:
    """Parses BlendSQL string with our pyparsing grammar and returns objects
    required for interpretation and execution.

    Args:
        query: The BlendSQL query to preprocess
        default_model: Model object, which we attach to each parsed_dict

    Returns:
        Tuple, containing:

            - ingredient_alias_to_parsed_dict

            - tables_in_ingredients

    Examples:
        ```python
        preprocess_blendsql(
            query="SELECT * FROM documents JOIN {{LLMJoin(left_on='w::player', right_on='documents::title')}} WHERE rank = 2",
            default_model=default_model
        )
        ```
        ```text
        (
            {
                'A': {
                    'function': 'LLMJoin',
                    'args': [],
                    'raw': "{{ LLMJoin ( left_on= 'w::player' , right_on= 'documents::title' ) }}",
                    'kwargs_dict': {
                        'left_on': 'w::player',
                        'right_on': 'documents::title'
                    }
                }
            },
            {'documents', 'w'}
        )
        ```
    """

    def process_arg_value(n: exp.Expression):
        if isinstance(n, exp.Tuple):
            return [i.to_py() for i in n.find_all(exp.Literal)]
        elif isinstance(n, exp.Paren):
            # This happens when we try to define a tuple with a single item
            # e.g. `options=('something')`
            return [n.this.to_py()]
        elif isinstance(n, (exp.Literal, exp.Boolean)):
            return n.to_py()
        elif isinstance(n, exp.Column):
            return ColumnRef(n.sql(dialect=dialect).strip('"'))
        elif isinstance(n, exp.Subquery):
            return Subquery(n.sql(dialect=dialect).removesuffix(")").removeprefix("("))
        elif isinstance(n, exp.DPipe):
            # Some concatenation, e.g. `People.Name || ' ' || People.Known_For`
            # These get parsed kind of weird in sqlglot
            def flatten_dpipe(node: exp.DPipe) -> list[exp.Expression]:
                args = []
                current = node

                while isinstance(current, exp.DPipe):
                    args.append(current.expression)  # Collect right side
                    current = current.this  # Move left

                args.append(current)  # Don't forget the leftmost element
                return args[::-1]  # Reverse to restore left-to-right order

            parts = flatten_dpipe(n)
            return StringConcatenation(
                [
                    ColumnRef(part.sql(dialect=dialect).strip('"'))
                    for part in parts
                    if isinstance(part, exp.Column)
                ],
                raw_expr=n.sql(),
            )
        elif isinstance(n, exp.Concat):
            return StringConcatenation(
                [
                    ColumnRef(part.sql(dialect=dialect).strip('"'))
                    for part in n.expressions
                    if isinstance(part, exp.Column)
                ],
                raw_expr=n.sql(),
            )
        raise ValueError(f"Not sure what to do with {type(n)} here")

    ingredient_alias_to_parsed_dict: dict[str, dict] = {}
    function_hash_to_alias: dict[str, str] = {}
    alias_counter = 0
    for function_node in node.find_all(exp.BlendSQLFunction):
        function_name = function_node.name
        function_args = [process_arg_value(arg) for arg in function_node.fn_args]
        function_kwargs = {
            kw.name.this: process_arg_value(kw.value) for kw in function_node.fn_kwargs
        }
        function_hash = hash(f"{function_name} {function_args} {function_kwargs}")
        if function_hash in function_hash_to_alias:
            ingredient_aliasname = function_hash_to_alias[function_hash]
        else:
            ingredient_aliasname = string.ascii_uppercase[alias_counter]
            function_hash_to_alias[function_hash] = ingredient_aliasname
            alias_counter += 1

            # Bind arguments to function
            from inspect import signature

            current_ingredient = kitchen.get_from_name(function_name)
            sig = signature(current_ingredient)
            bound = sig.bind(*function_args, **function_kwargs)
            # Below we track the 'raw' representation, in case we need to pass into
            #   a recursive BlendSQL call later
            ingredient_alias_to_parsed_dict[ingredient_aliasname] = {
                "function": function_name,
                "raw": function_node.sql(dialect=dialect),
                "kwargs_dict": {
                    "model": default_model,
                    **bound.arguments,
                    **bound.kwargs,
                },
            }

        substituted_ingredient_alias = format_blendsql_function(ingredient_aliasname)
        aliased_function_node = exp.Column(
            this=exp.Identifier(this=substituted_ingredient_alias)
        )
        if function_node == node:
            # For some reason, .replace doesn't work here
            node = aliased_function_node
            continue
        function_node.replace(aliased_function_node)

    return (
        node.sql(dialect=dialect),
        ingredient_alias_to_parsed_dict,
        kitchen,
        ingredients,
    )


def materialize_cte(
    subquery: exp.Expression,
    query_context: QueryContextManager,
    aliasname: str,
    db: Database,
    default_model: Model,
    ingredient_alias_to_parsed_dict: dict[str, dict],
    **kwargs,
) -> pd.DataFrame:
    str_subquery = subquery.sql(dialect=query_context.dialect)
    materialized_smoothie: pd.DataFrame = disambiguate_and_submit_blend(
        ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
        query=str_subquery,
        db=db,
        default_model=default_model,
        aliasname=aliasname,
        **kwargs,
    )
    db.to_temp_table(
        df=materialized_smoothie.pl,
        tablename=aliasname,
    )
    # Now, we need to remove subquery and instead insert direct reference to aliasname
    # Example:
    #   `SELECT Symbol FROM (SELECT DISTINCT Symbol FROM portfolio) AS w`
    #   Should become: `SELECT Symbol FROM w`
    query_context.node = query_context.node.transform(
        transform.replace_subquery_with_direct_alias_call,
        subquery=subquery.parent,
        aliasname=aliasname,
    )
    # Remove any hanging CTE statements referencing `aliasname`
    for n in query_context.node.find_all((exp.With, exp.CTE)):
        for _n in n.expressions:
            if isinstance(_n, exp.Table):
                if _n.this == exp.Identifier(this=aliasname):
                    if len(n.expressions) == 1:
                        # Remove the whole With/CTE clause
                        n.replace(None)
                    else:
                        # Just replace the current node
                        _n.replace(None)
    return materialized_smoothie


def get_sorted_blendsql_nodes(
    node: exp.Expression,
    ingredient_alias_to_parsed_dict: dict,
    kitchen: Kitchen,
) -> Generator[tuple[exp.Expression, bool], None, None]:
    """
    Yields parsed matches from grammar, according to a specified order of operations.

    Args:
        q: str, the current BlendSQL query to parse. Should contain ingredient aliases.
        ingredient_alias_to_parsed_dict: Mapping from ingredient alias to their parsed representations.
            Example:
                {"{{A()}}": {"function": "LLMMap", "args": ...}
        kitchen: Contains inventory of ingredients (aka BlendSQL ingredients)

    Returns:
        Generator yielding expression node
    """
    ooo = [
        IngredientType.STRING,
        IngredientType.QA,
        IngredientType.MAP,
        IngredientType.JOIN,
    ]

    parse_results = list(node.find_all(exp.BlendSQLFunction))

    # Pre-scan to count total distinct mAP ingredients
    total_maps_left = set()
    for function_node in parse_results:
        parse_results_dict = ingredient_alias_to_parsed_dict[function_node.name]
        _function = kitchen.get_from_name(parse_results_dict["function"])
        if _function.ingredient_type == IngredientType.MAP:
            total_maps_left.add(function_node.name)

    while len(parse_results) > 0:
        curr_ingredient_target = ooo.pop(0)
        remaining_parse_results = []
        for function_node in parse_results:
            # Fetch parsed ingredient dict from our cache
            parse_results_dict = ingredient_alias_to_parsed_dict[function_node.name]
            _function: Ingredient = kitchen.get_from_name(
                parse_results_dict["function"]
            )
            if _function.ingredient_type == curr_ingredient_target:
                is_final_map = False
                if curr_ingredient_target == IngredientType.MAP:
                    total_maps_left.discard(function_node.name)
                    is_final_map = len(total_maps_left) == 0
                yield function_node, is_final_map
                continue
            elif _function.ingredient_type not in IngredientType:
                raise ValueError(
                    f"Not sure what to do with ingredient_type '{_function.ingredient_type}' yet"
                )
            remaining_parse_results.append(function_node)
        parse_results = remaining_parse_results


def disambiguate_and_submit_blend(
    ingredient_alias_to_parsed_dict: dict[str, dict],
    query: str,
    aliasname: str,
    **kwargs,
):
    """
    Used to disambiguate anonymized BlendSQL function and execute in a recursive context.
    """
    for alias, d in ingredient_alias_to_parsed_dict.items():
        # https://stackoverflow.com/a/12127534
        query = re.sub(
            re.escape(format_blendsql_function(alias)),
            lambda _: d["raw"],  # noqa
            query,
        )
    logger.debug(
        Color.update(f"Executing ")
        + Color.sql(query, ignore_prefix=True)
        + Color.update(f" and setting to `{aliasname}`...", ignore_prefix=True)
    )
    return _blend(query=query, **kwargs)


def _blend(
    query: str,
    db: Database,
    ingredients: Collection[Type[Ingredient]],
    default_model: Model | None = None,
    verbose: bool = False,
    infer_gen_constraints: bool = True,
    enable_cascade_filter: bool = True,
    enable_early_exit: bool = True,
    enable_constrained_decoding: bool = True,
    table_to_title: dict[str, str] | None = None,
    _prev_passed_values: int = 0,
) -> Smoothie:
    """Invoked from blend(), this contains the recursive logic to execute
    a BlendSQL query and return a `Smoothie` object.
    """
    # The QueryContextManager class is used to track all manipulations done to
    # the original query, prior to the final execution on the underlying DBMS.
    original_query = copy.deepcopy(query)
    dialect: sqlglot.Dialect = get_dialect(db.__class__.__name__)

    query_context = QueryContextManager(dialect)
    # sqlglot will add `"".column` if we pass an empty dict as a schema instead of `None`
    query_context.parse(
        query, schema=db.sqlglot_schema if len(db.sqlglot_schema) > 0 else None
    )

    session_uuid = uuid.uuid4().hex[:4]

    # Create our Kitchen
    kitchen = Kitchen(db=db, session_uuid=session_uuid)
    kitchen.extend(ingredients)
    # Replace ingredient calls with short aliases (e.g. '{{A()}}'),
    # and use _peg_grammar to extract ingredient types
    (
        query,
        ingredient_alias_to_parsed_dict,
        kitchen,
        ingredients,
    ) = preprocess_blendsql(
        node=query_context.node,
        dialect=dialect,
        kitchen=kitchen,
        ingredients=ingredients,
        default_model=default_model,
    )
    query = autowrap_query(
        query=query,
        kitchen=kitchen,
        ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
    )
    # Parse to our QueryContextManager object
    query_context.parse(query)

    # Preliminary check - we can't have anything that modifies database state
    if query_context.node.find(MODIFIERS):
        raise InvalidBlendSQL("BlendSQL query cannot have `DELETE` clause!")

    # If we don't have any ingredient calls, execute as normal SQL
    if len(ingredients) == 0 or len(ingredient_alias_to_parsed_dict) == 0:
        # Check to see if there is a table we haven't materialized yet
        # Need to `try`, `except` for cases like DuckDB's `...FROM read_text(x)`
        try:
            for tablename in [
                i.name for i in query_context.node.find_all(exp.Table) if i.name != ""
            ]:
                if tablename not in db.tables():
                    materialized_smoothie = db.lazy_tables.pop(tablename).collect()
                    _prev_passed_values += materialized_smoothie.meta.num_values_passed
        except Exception as e:
            logger.error(f"Error while materializing tables: {e}")
        logger.debug(Color.warning(f"No BlendSQL ingredients found in query:"))
        logger.debug(Color.quiet_sql(query))
        logger.debug(Color.warning(f"Executing as vanilla SQL..."))
        return Smoothie(
            _df=db.execute_to_df(query_context.to_string()),
            meta=SmoothieMeta(
                num_values_passed=0,
                num_generation_calls=(
                    default_model.num_generation_calls
                    if default_model is not None
                    else 0
                ),
                prompt_tokens=(
                    default_model.prompt_tokens if default_model is not None else 0
                ),
                completion_tokens=(
                    default_model.completion_tokens if default_model is not None else 0
                ),
                prompts=default_model.prompts if default_model is not None else [],
                raw_prompts=default_model.raw_prompts
                if default_model is not None
                else [],
                ingredients=[],
                query=original_query,
                db_url=str(db.db_url),
                db_type=db.__class__.__name__,
                contains_ingredient=False,
            ),
        )

    _get_temp_session_table: Callable = partial(get_temp_session_table, session_uuid)
    alias_function_name_to_result: dict[str, str] = {}
    session_modified_tables = set()
    scm = None
    # TODO: Currently, as we traverse upwards from deepest subquery,
    #   if any lower subqueries have an ingredient, we deem the current
    #   as ineligible for optimization. Maybe this can be improved in the future.
    prev_subquery_has_ingredient = False
    for subquery_idx, subquery in enumerate(
        get_reversed_subqueries(query_context.node)
    ):
        # At this point, we should have already handled cte statements and created associated tables
        if subquery.find(exp.With) is not None:
            subquery = subquery.transform(transform.remove_nodetype, exp.With)
        # Only cache executed_ingredients within the same subquery
        # The same ingredient may have different results within a different subquery context
        executed_subquery_ingredients: set[str] = set()
        prev_subquery_map_columns: set[str] = set()
        _get_temp_subquery_table: Callable = partial(
            get_temp_subquery_table, session_uuid, subquery_idx
        )
        if subquery is None:
            continue
        if not isinstance(subquery, exp.Select):
            # We need to create a select query from this parentheses expression
            # So we find the parent select, and grab that table
            parent_select_tablenames = [
                i.name for i in subquery.find_ancestor(exp.Select).find_all(exp.Table)
            ]
            if len(parent_select_tablenames) == 1:
                subquery_str = (
                    f"SELECT * FROM {parent_select_tablenames[0]} WHERE "
                    + get_first_child(subquery).sql(dialect=dialect)
                )

            else:
                logger.debug(
                    Color.warning(
                        "Encountered subquery without `SELECT`, and more than 1 table!\nCannot optimize yet, skipping this step."
                    )
                )
                continue
        else:
            subquery_str = subquery.sql(dialect=dialect)

        subquery_processed_tablenames = set()
        in_cte, cte_table_alias_name = check.in_cte(subquery, return_name=True)
        scm = SubqueryContextManager(
            dialect=dialect,
            node=_parse_one(
                subquery_str, dialect=dialect
            ),  # Need to do this so we don't track parents into construct_abstracted_selects
            prev_subquery_has_ingredient=prev_subquery_has_ingredient,
            alias_to_subquery={cte_table_alias_name: subquery} if in_cte else {},
            ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
        )
        for (
            tablename,
            _postprocess_columns,
            abstracted_query_str,
        ) in scm.abstracted_table_selects(db):
            if in_cte:  # Don't execute CTEs until we need them
                continue
            # # If this table isn't being used in any ingredient calls, there's no
            # #   need to create a temporary session table
            # if (tablename not in tables_in_ingredients) and (
            #     scm.tablename_to_alias.get(tablename, None) not in tables_in_ingredients
            # ):
            #     continue
            aliased_subquery = scm.alias_to_subquery.pop(tablename, None)
            if aliased_subquery is not None:
                # First, we need to explicitly create the aliased subquery as a table
                # For example, `SELECT Symbol FROM (SELECT DISTINCT Symbol FROM portfolio) AS w WHERE w...`
                # We can't assign `abstracted_query` for non-existent `w`
                #   until we set `w` to `SELECT DISTINCT Symbol FROM portfolio`
                db.lazy_tables.add(
                    LazyTable(
                        tablename=tablename,
                        collect_fn=partial(
                            materialize_cte,
                            query_context=query_context,
                            subquery=aliased_subquery,
                            aliasname=tablename,
                            default_model=default_model,
                            db=db,
                            ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
                            # Below are in case we need to call blend() again
                            ingredients=ingredients,
                            infer_gen_constraints=infer_gen_constraints,
                            table_to_title=table_to_title,
                            verbose=verbose,
                            _prev_passed_values=_prev_passed_values,
                        ),
                        has_blendsql_function=aliased_subquery.find(
                            exp.BlendSQLFunction
                        )
                        is not None,
                    )
                )
            if abstracted_query_str is not None:
                if tablename in db.lazy_tables:
                    # We need to materialize here, in the case of something like:
                    #   `WITH a AS (restrictive_condition) SELECT * FROM a WHERE {{LLMMap(..., a.column)}}`
                    # We want the LM function to only get the output from the restrictive condition.
                    materialized_smoothie = db.lazy_tables.pop(tablename).collect()
                    _prev_passed_values += materialized_smoothie.meta.num_values_passed

                tablename_to_write = _get_temp_subquery_table(tablename)
                logger.debug(
                    Color.update("Executing ")
                    + Color.sql(abstracted_query_str, ignore_prefix=True)
                    + Color.update(
                        f" and setting to `{tablename_to_write}`...", ignore_prefix=True
                    )
                )
                abstracted_df = db.execute_to_df(abstracted_query_str)

                db.to_temp_table(
                    df=abstracted_df,
                    tablename=tablename_to_write,
                )
                subquery_processed_tablenames.add(tablename)

        # Be sure to handle those remaining aliases, which didn't have abstracted queries
        for aliasname, aliased_subquery in scm.alias_to_subquery.items():
            db.lazy_tables.add(
                LazyTable(
                    tablename=aliasname,
                    collect_fn=partial(
                        materialize_cte,
                        query_context=query_context,
                        subquery=aliased_subquery,
                        aliasname=aliasname,
                        default_model=default_model,
                        db=db,
                        ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
                        # Below are in case we need to call blend() again
                        ingredients=ingredients,
                        infer_gen_constraints=infer_gen_constraints,
                        table_to_title=table_to_title,
                        verbose=verbose,
                        _prev_passed_values=_prev_passed_values,
                    ),
                    has_blendsql_function=aliased_subquery.find(exp.BlendSQLFunction)
                    is not None,
                )
            )
        if prev_subquery_has_ingredient:
            scm.set_node(scm.node.transform(transform.maybe_set_subqueries_to_true))

        # Now, 1) Find all ingredients to execute (e.g. '{{f(a, b, c)}}')
        # 2) Track when we've created a new table from a MapIngredient call
        #   only at the end of parsing a subquery, we can merge to the original session_uuid table
        tablename_to_map_out: dict[str, tuple[pd.DataFrame, str]] = {}
        cascade_filter: pl.LazyFrame = None
        previous_cascade_filter_failed = False
        for function_node, is_final_map in get_sorted_blendsql_nodes(
            node=scm.node,
            ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
            kitchen=kitchen,
        ):
            if in_cte:  # Don't execute CTEs until we need them
                continue
            curr_function_parsed_results = ingredient_alias_to_parsed_dict[
                function_node.name
            ]
            curr_ingredient = kitchen.get_from_name(
                curr_function_parsed_results["function"]
            )
            prev_subquery_has_ingredient = True
            if function_node.name in executed_subquery_ingredients:
                # Don't execute same ingredient twice
                continue

            executed_subquery_ingredients.add(function_node.name)
            kwargs_dict = curr_function_parsed_results["kwargs_dict"]

            if (
                enable_early_exit
                and curr_ingredient.ingredient_type == IngredientType.MAP
            ):
                # We can ONLY apply this exit condition if we're executing the final Map function of the subquery
                if is_final_map:
                    # We can't apply exit conditions if a previous cascade filter application failed
                    if not previous_cascade_filter_failed:
                        # Fetch an exit condition, if we can extract one from the expression context
                        # i.e. `SELECT * FROM t WHERE a() = TRUE LIMIT 5`
                        # The exit condition would be at least 5 `a()` evaluate to `TRUE`
                        kwargs_dict["exit_condition"] = scm.get_exit_condition(
                            function_node
                        )
            # Immediately set false, unless proven otherwise
            previous_cascade_filter_failed = True

            logger.debug(
                Color.update("\nExecuting ")
                + Color.sql(
                    f"{curr_function_parsed_results['raw']}", ignore_prefix=True
                )
                + Color.update("...", ignore_prefix=True)
            )
            Color.in_block = True

            if infer_gen_constraints:
                # Latter is the winner.
                # So if we already define something in kwargs_dict,
                #   It's not overridden here.
                kwargs_dict = (
                    scm.infer_gen_constraints(
                        function_node=function_node,
                        schema=db.sqlglot_schema,
                        alias_to_tablename=scm.alias_to_tablename,
                        has_user_regex=bool(kwargs_dict.get("regex", None) is not None),
                    )
                    | kwargs_dict
                )

            if table_to_title is not None:
                kwargs_dict["table_to_title"] = table_to_title

            # Optionally, recursively call blend() again to get subtable from args
            # This applies to `context` and `options`, since they could be `Subquery` types.
            for _i, unpack_kwarg in enumerate(["context", "options"]):
                unpack_value = kwargs_dict.get(unpack_kwarg, None)
                if unpack_value is None:
                    continue

                if unpack_kwarg == "context":
                    if not isinstance(unpack_value, (tuple, list)):
                        kwargs_dict[unpack_kwarg] = (kwargs_dict[unpack_kwarg],)
                    unpack_values = kwargs_dict[unpack_kwarg]
                elif unpack_kwarg == "options":
                    unpack_values = [unpack_value]

                for value_idx, value in enumerate(unpack_values):
                    if isinstance(value, Subquery):
                        _smoothie = _blend(
                            query=value,
                            db=db,
                            default_model=default_model,
                            ingredients=ingredients,
                            infer_gen_constraints=infer_gen_constraints,
                            table_to_title=table_to_title,
                            verbose=verbose,
                            _prev_passed_values=_prev_passed_values,
                        )
                        _prev_passed_values += _smoothie.meta.num_values_passed
                        subtable = _smoothie.pl
                        if unpack_kwarg == "options":
                            if len(subtable.columns) == 1 or len(subtable) == 1:
                                # Here, we need to format as a flat set
                                kwargs_dict[unpack_kwarg] = list(
                                    subtable.to_numpy().flatten()
                                )
                            else:
                                raise InvalidBlendSQL(
                                    f"Invalid subquery passed to `options`!\nNeeds to return exactly one column or row, got {len(subtable.columns)} columns and {len(subtable)} rows instead"
                                )
                        elif unpack_kwarg == "context":
                            if curr_ingredient.ingredient_type == IngredientType.QA:
                                # `QAIngredient` can potentially receive multiple context subtables
                                tup = kwargs_dict[unpack_kwarg]
                                new_tup = (
                                    tup[:value_idx] + (subtable,) + tup[value_idx + 1 :]
                                )
                                kwargs_dict[unpack_kwarg] = new_tup
                            else:
                                kwargs_dict[unpack_kwarg] = subtable
                        else:
                            raise LMFunctionException(
                                f"Invalid kwarg {unpack_kwarg}\nAlso, we should have never hit this error..."
                            )

            if getattr(curr_ingredient, "model", None) is not None:
                kwargs_dict["model"] = curr_ingredient.model

            # Execute our ingredient function
            function_out = curr_ingredient(
                **kwargs_dict
                | {
                    "get_temp_subquery_table": _get_temp_subquery_table,
                    "get_temp_session_table": _get_temp_session_table,
                    "aliases_to_tablenames": scm.alias_to_tablename,
                    "prev_subquery_map_columns": prev_subquery_map_columns,
                    "cascade_filter": cascade_filter,
                    "enable_constrained_decoding": enable_constrained_decoding,
                },
            )
            # Check how to handle output, depending on ingredient type
            if curr_ingredient.ingredient_type == IngredientType.MAP:
                # Parse so we replace this function in blendsql with 1st arg
                #   (new_col, which is the question we asked)
                #  But also update our underlying table, so we can execute correctly at the end
                (new_col, tablename, colname, new_table) = function_out
                prev_subquery_map_columns.add(new_col)
                if tablename in tablename_to_map_out:
                    tablename_to_map_out[tablename].append((new_table, new_col))
                else:
                    tablename_to_map_out[tablename] = [(new_table, new_col)]
                session_modified_tables.add(tablename)
                alias_function_name_to_result[
                    function_node.name
                ] = f'"{double_quote_escape(tablename)}"."{double_quote_escape(new_col)}"'

                if enable_cascade_filter:
                    if (
                        scm.is_eligible_for_cascade_filter()
                        and len(scm.stateful_columns_referenced_by_lm_ingredients) == 1
                    ):
                        previous_cascade_filter_failed = False
                        cascade_filter = LazyTable(
                            collect_fn=partial(
                                get_map_cascade_filter,
                                function_node=function_node,
                                tablename=tablename,
                                new_table=new_table,
                                new_col=new_col,
                                scm=scm,
                            ),
                            has_blendsql_function=True,
                        )

            elif curr_ingredient.ingredient_type in (
                IngredientType.STRING,
                IngredientType.QA,
            ):
                # Here, we can simply insert the function's output
                alias_function_name_to_result[function_node.name] = function_out
                if enable_cascade_filter:
                    if (
                        scm.is_eligible_for_cascade_filter()
                        and len(scm.stateful_columns_referenced_by_lm_ingredients) == 1
                    ):
                        previous_cascade_filter_failed = False
                        cascade_filter = LazyTable(
                            collect_fn=partial(
                                get_qa_cascade_filter,
                                function_node=function_node,
                                function_result=function_out,
                                scm=scm,
                                db=db,
                            ),
                            has_blendsql_function=True,
                        )
            elif curr_ingredient.ingredient_type == IngredientType.JOIN:
                # 1) Get the `JOIN` clause containing function
                # 2) Replace with just the function alias
                # 3) Assign `function_out` to `alias_function_str`
                (
                    left_tablename,
                    right_tablename,
                    join_clause,
                    temp_join_tablename,
                ) = function_out
                # Special case for when we have more than 1 ingredient in `JOIN` node left at this point
                join_node = query_context.node.find(exp.Join)
                join_node.replace(exp.BlendSQLFunction(this=function_node.name))
                alias_function_name_to_result[function_node.name] = join_clause
            else:
                raise ValueError(
                    f"Not sure what to do with ingredient_type '{curr_ingredient.ingredient_type}' yet\n(Also, we should have never hit this error....)"
                )
            Color.in_block = False

        # Combine all the retrieved map outputs
        # The below assumes the `mapped_dfs` are in the same row-order!
        # Which is the case for LLMMap, since it does a left-join to make sure
        # the return order is consistent with the input order.
        for tablename, outputs in tablename_to_map_out.items():
            if not outputs:
                continue

            temp_name = _get_temp_session_table(tablename)
            source = temp_name if db.has_temp_table(temp_name) else tablename

            # Fetch the base table to modify with our new columns.
            # This ensures parity with the existing database once we
            #   swap in our reference to the new temporary table.
            base = db.execute_to_df(
                select_all_from_table_query(source), close_conn=False
            )
            base_cols = set(base.collect_schema().names())
            mapped_dfs, new_cols = map(list, zip(*outputs))

            # The data in `mapped_dfs` will have the new column (in `new_cols`), along with
            #   any native columns passed to the Map function.
            # These would be things like the `value` column, any `context`, etc.
            frames = [df.select(col) for df, col in zip(mapped_dfs, new_cols)]
            new_data = pl.concat(frames, how="horizontal").collect()  # Collect once

            to_add = [c for c in new_cols if c not in base_cols]
            to_coalesce = [c for c in new_cols if c in base_cols]

            # Build result with coalesce for overlapping columns
            if to_coalesce:
                coalesce_exprs = [
                    pl.coalesce(new_data[c], pl.col(c)).alias(c) for c in to_coalesce
                ]
                base = base.with_columns(coalesce_exprs)

            if to_add:
                base = base.with_columns(new_data.select(to_add))

            db.to_temp_table(df=base.collect(), tablename=temp_name)
            session_modified_tables.add(tablename)

    # Now insert the function outputs to the original query
    # We need to re-sync if we did some operation on the underlying query,
    #   like with a JoinIngredient
    query = query_context.to_string()
    if alias_function_name_to_result:
        # Process regex replacements in a batch
        replacements = {
            format_blendsql_function(alias): f" {str(res)} "
            for alias, res in alias_function_name_to_result.items()
        }
        pattern = re.compile("|".join(map(re.escape, replacements.keys())))
        query = pattern.sub(lambda m: replacements[m.group(0)], query)

    query_context.parse(
        query
    )  # Sync query to string, after replacing aliases with function outputs

    temp_table_cache = {t: _get_temp_session_table(t) for t in session_modified_tables}
    for t in session_modified_tables:
        query_context.node = query_context.node.transform(
            transform.replace_tablename, t, temp_table_cache[t]
        )
    if scm is not None:
        for a, t in scm.alias_to_tablename.items():
            if t in session_modified_tables:
                query_context.node = query_context.node.transform(
                    transform.replace_tablename, a, temp_table_cache[t]
                )

    # Finally, iter through tables in query and see if we need to collect LazyTable
    # TODO: this is materializing tables, even if we don't need them
    #   but, without it, test_cte_qa_multi_exec fails
    for table in query_context.node.find_all((exp.Table, exp.TableAlias)):
        if table.name in db.lazy_tables:
            lazy_table = db.lazy_tables.pop(table.name)
            if lazy_table.has_blendsql_function:
                materialized_smoothie = lazy_table.collect()
                _prev_passed_values += materialized_smoothie.meta.num_values_passed

    query = query_context.to_string()

    logger.debug(
        Color.success(f"Final Query:\n") + Color.sql(query, ignore_prefix=True)
    )

    df = db.execute_to_df(query).collect()

    return Smoothie(
        _df=df,
        meta=SmoothieMeta(
            num_values_passed=sum(
                [
                    i.num_values_passed
                    for i in kitchen
                    if hasattr(i, "num_values_passed")
                ]
            )
            + _prev_passed_values,
            num_generation_calls=(
                default_model.num_generation_calls if default_model is not None else 0
            ),
            prompt_tokens=(
                default_model.prompt_tokens if default_model is not None else 0
            ),
            completion_tokens=(
                default_model.completion_tokens if default_model is not None else 0
            ),
            prompts=default_model.prompts if default_model is not None else [],
            raw_prompts=default_model.raw_prompts if default_model is not None else [],
            ingredients=ingredients,
            query=original_query,
            db_url=str(db.db_url),
            db_type=db.__class__.__name__,
        ),
    )


@dataclass
class BlendSQL:
    """Core `BlendSQL` class that provides high level interface for executing BlendSQL queries.

    Args:
        db (Union[pd.DataFrame, dict, str, Database]): Database to connect to. Can be:

            - pandas DataFrame or dict of DataFrames

            - Path to SQLite database file

            - PostgreSQL connection string

            - `Database` object
        model (Optional[Model]): Model instance to use for LLM operations. Can also be
            provided during query execution.
        ingredients (Optional[Collection[Type[Ingredient]]]): Collection of ingredients to
            make available for queries. Can also be
                provided during query execution.
        verbose (bool): Whether to output debug logging information. Defaults to False.
        infer_gen_constraints (bool): Whether to automatically infer constraints for
            LLM generation based on query context. Defaults to True.
        table_to_title (Optional[Dict[str, str]]): Optional mapping from table names to
            descriptive titles, useful for datasets where table titles contain metadata.
    """

    db: pd.DataFrame | dict | str | Path | Database = field(default=None)
    model: Model | None = field(default=None)
    ingredients: Collection[Type[Ingredient]] | None = field(default_factory=list)

    verbose: bool = field(default=False)

    infer_gen_constraints: bool = field(default=True)
    enable_constrained_decoding: bool = field(default=True)
    enable_cascade_filter: bool = field(default=True)
    enable_early_exit: bool = field(default=True)

    table_to_title: dict[str, str] | None = field(default=None)

    def __post_init__(self):
        if not isinstance(self.db, Database):
            self.db = self._infer_db_type(self.db)
        if self.db is None:
            raise ValueError("df_or_db_path must be provided")
        self.ingredients = self._merge_default_ingredients(self.ingredients)
        self._toggle_verbosity(self.verbose)

    @staticmethod
    def _toggle_verbosity(verbose_in_use: bool):
        if verbose_in_use:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.ERROR)

    @staticmethod
    def _merge_default_ingredients(
        ingredients: Collection[Type[Ingredient]] | None,
    ) -> set[Type[Ingredient]]:
        from blendsql.ingredients import LLMQA, LLMMap, LLMJoin

        DEFAULT_INGREDIENTS = {LLMQA, LLMMap, LLMJoin}
        try:
            _ingredient_names = [i.__name__.upper() for i in ingredients]
        except AttributeError as e:
            raise LMFunctionException(
                "All arguments passed to `ingredients` should be `Ingredient` classes!"
            ) from e
        ingredient_names = set(_ingredient_names)
        if len(ingredient_names) != len(_ingredient_names):
            raise LMFunctionException(
                f"Duplicate ingredient names passed! These are case insensitive, be careful.\n{_ingredient_names=}"
            )
        ingredients = set(ingredients)
        for default_ingredient in DEFAULT_INGREDIENTS:
            if default_ingredient.__name__.upper() not in ingredient_names:
                ingredients.add(default_ingredient)
        return ingredients

    @staticmethod
    def _infer_db_type(df_or_db_path: pd.DataFrame | dict | str | Path) -> Database:
        if df_or_db_path is None:
            from .db.pandas import Pandas

            return Pandas({})  # Load an empty DuckDB connection

        elif isinstance(df_or_db_path, (pd.DataFrame, dict)):
            from .db.pandas import Pandas

            if isinstance(df_or_db_path, dict):
                if not isinstance(next(iter(df_or_db_path.values())), pd.DataFrame):
                    logger.debug(
                        Color.update("Converting dict values to pd.DataFrames...")
                    )
                    df_or_db_path = {
                        k: pd.DataFrame(v) for k, v in df_or_db_path.items()
                    }
            return Pandas(df_or_db_path)

        elif isinstance(df_or_db_path, (str, Path)):
            if Path(df_or_db_path).exists():
                if Path(df_or_db_path).suffix == ".duckdb":
                    from .db.duckdb import DuckDB

                    return DuckDB.from_file(df_or_db_path)
                else:
                    from .db.sqlite import SQLite

                    return SQLite(df_or_db_path)
            elif "://" in df_or_db_path:
                from .db.postgresql import PostgreSQL

                return PostgreSQL(df_or_db_path)
        else:
            raise ValueError(
                f"Could not resolve '{df_or_db_path}' to a valid database type!"
            )

    def visualize(self, query: str, output_path: str | None = None, format="pdf"):
        """Visualize query as a DAG with graphviz."""
        from .visualize import SQLGlotASTVisualizer

        visualizer = SQLGlotASTVisualizer()

        dialect: sqlglot.Dialect = get_dialect(self.db.__class__.__name__)

        # Generate visualization
        dot = visualizer.visualize(
            _parse_one(query, dialect=dialect, schema=self.db.sqlglot_schema)
        )

        if output_path is not None:
            # Save as PDF
            dot.render(output_path, format=format, cleanup=True)
        return dot

    def execute(
        self,
        query: str,
        ingredients: Collection[Type[Ingredient]] | None = None,
        model: str | None = None,
        infer_gen_constraints: bool | None = None,
        enable_cascade_filter: bool | None = None,
        enable_early_exit: bool | None = None,
        enable_constrained_decoding: bool | None = None,
        verbose: bool | None = None,
    ) -> Smoothie:
        '''The `execute()` function is used to execute a BlendSQL query against a database and
        return the final result, in addition to the intermediate reasoning steps taken.
        Execution is done on a database given an ingredient context.

        Args:
            query: The BlendSQL query to execute
            ingredients: Collection of ingredient objects, to use in interpreting BlendSQL query.
                {LLMQA, LLMMap, LLMJoin} are supplied by default.
            verbose: Boolean defining whether to run with logger in debug mode
            default_model: Which BlendSQL model to use in performing ingredient tasks in the current query
            infer_gen_constraints: Optionally infer the output format of an `IngredientMap` call, given the predicate context
                For example, in `{{LLMMap('convert to date', 'w::listing date')}} <= '1960-12-31'`
                We can infer the output format should look like '1960-12-31' and both:
                    1) Put this string in the `example_outputs` kwarg
                    2) If we have a LocalModel, pass the date regex pattern to guidance
            enable_cascade_filter: Enable cascade filtering optimization.
            enable_early_exit: Enable early exit optimization.
            enable_constrained_decoding: Enable constrained decoding for local models.

        Returns:
            smoothie: `Smoothie` dataclass containing pd.DataFrame output and execution metadata

        Examples:
            ```python
            import psutil
            import pandas as pd

            from blendsql import BlendSQL
            from blendsql.models import LlamaCpp

            # Prepare our BlendSQL connection
            bsql = BlendSQL(
                {
                    "People": pd.DataFrame(
                        {
                            "Name": [
                                "George Washington",
                                "John Quincy Adams",
                                "Thomas Jefferson",
                                "James Madison",
                                "James Monroe",
                                "Alexander Hamilton",
                                "Sabrina Carpenter",
                                "Charli XCX",
                                "Elon Musk",
                                "Michelle Obama",
                                "Elvis Presley",
                            ],
                            "Known_For": [
                                "Established federal government, First U.S. President",
                                "XYZ Affair, Alien and Sedition Acts",
                                "Louisiana Purchase, Declaration of Independence",
                                "War of 1812, Constitution",
                                "Monroe Doctrine, Missouri Compromise",
                                "Created national bank, Federalist Papers",
                                "Nonsense, Emails I Cant Send, Mean Girls musical",
                                "Crash, How Im Feeling Now, Boom Clap",
                                "Tesla, SpaceX, Twitter/X acquisition",
                                "Lets Move campaign, Becoming memoir",
                                "14 Grammys, King of Rock n Roll",
                            ],
                        }
                    ),
                    "Eras": pd.DataFrame({"Years": ["1800-1900", "1900-2000", "2000-Now"]}),
                },
                model = LlamaCpp(
                    model_name_or_path="unsloth/gemma-3-4b-it-GGUF",
                    filename="gemma-3-4b-it-Q4_K_M.gguf",
                    config={
                        "n_gpu_layers": -1,
                        "n_ctx": 1028,
                        "seed": 100,
                        "n_threads": psutil.cpu_count(logical=False),
                    }
                )
            )

            smoothie = bsql.execute(
                """
                SELECT * FROM People P
                WHERE P.Name IN {{
                    LLMQA('First 3 presidents of the U.S?', quantifier='{3}')
                }}
                """
            )

            print(smoothie.df)
            # ┌───────────────────┬───────────────────────────────────────────────────────┐
            # │ Name              │ Known_For                                             │
            # ├───────────────────┼───────────────────────────────────────────────────────┤
            # │ George Washington │ Established federal government, First U.S. Preside... │
            # │ John Quincy Adams │ XYZ Affair, Alien and Sedition Acts                   │
            # │ Thomas Jefferson  │ Louisiana Purchase, Declaration of Independence       │
            # └───────────────────┴───────────────────────────────────────────────────────┘
            smoothie.print_summary()
            # ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
            # │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
            # ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
            # │    1.25158 │                    1 │             296 │                  16 │
            # └────────────┴──────────────────────┴─────────────────┴─────────────────────┘
            ```
        '''
        self._toggle_verbosity(verbose if verbose is not None else self.verbose)
        logger.debug(Color.horizontal_line())
        start = time.time()
        model_in_use = model or self.model
        try:
            smoothie = _blend(
                query=query,
                db=self.db,
                default_model=model_in_use,
                ingredients=self._merge_default_ingredients(
                    ingredients or self.ingredients
                ),
                infer_gen_constraints=infer_gen_constraints
                if infer_gen_constraints is not None
                else self.infer_gen_constraints,
                enable_constrained_decoding=enable_constrained_decoding
                if enable_constrained_decoding is not None
                else self.enable_constrained_decoding,
                enable_cascade_filter=enable_cascade_filter
                if enable_cascade_filter is not None
                else self.enable_cascade_filter,
                enable_early_exit=enable_early_exit
                if enable_early_exit is not None
                else self.enable_early_exit,
                table_to_title=self.table_to_title,
            )
        except Exception as error:
            raise error
        finally:
            # In the case of a recursive `_blend()` call,
            #   this logic allows temp tables to persist until
            #   the final base case is fulfilled.
            self.db._reset_connection()
            # Reset model stats, so future executions don't add here
            if model_in_use is not None:
                model_in_use.reset_stats()
        smoothie.meta.process_time_seconds = time.time() - start
        logger.debug(Color.horizontal_line())
        return smoothie
