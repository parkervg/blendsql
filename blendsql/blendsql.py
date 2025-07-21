import copy
import logging
import time
import uuid
import pandas as pd
import re
import typing as t
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
import sqlglot
from attr import attrs, attrib
from functools import partial
from sqlglot import exp
from colorama import Fore
import string

from blendsql.common.logger import logger
from blendsql.common.utils import (
    get_temp_session_table,
    get_temp_subquery_table,
)
from blendsql.common.exceptions import InvalidBlendSQL
from blendsql.db.database import Database
from blendsql.db.duckdb import DuckDB
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
from blendsql.parse.constants import MODIFIERS
from blendsql.ingredients.ingredient import Ingredient, IngredientException
from blendsql.smoothie import Smoothie, SmoothieMeta
from blendsql.common.constants import IngredientType, Subquery, ColumnRef
from blendsql.models.model import Model

format_blendsql_function = lambda name: "{{" + name + "()}}"


@attrs
class Kitchen(list):
    """Superset of list. A collection of ingredients."""

    db: Database = attrib()
    session_uuid: str = attrib()

    name_to_ingredient: t.Dict[str, Ingredient] = attrib(init=False)

    def __attrs_post_init__(self):
        self.name_to_ingredient = {}

    def names(self):
        return [i.name for i in self]

    def get_from_name(self, name: str, flag_duplicates: bool = True):
        try:
            return self.name_to_ingredient[name.upper()]
        except KeyError:
            raise InvalidBlendSQL(
                f"Ingredient '{name}' called, but not found in passed `ingredient` arg!"
            ) from None

    def extend(
        self, ingredients: Iterable[t.Type[Ingredient]], flag_duplicates: bool = True
    ) -> None:
        """Initializes ingredients class with base attributes, for use in later operations."""
        try:
            if not all(issubclass(x, Ingredient) for x in ingredients):
                raise IngredientException(
                    "All arguments passed to `Kitchen` must be ingredients!"
                )
        except TypeError:
            raise IngredientException(
                "All arguments passed to `Kitchen` must be ingredients!"
            ) from None
        for ingredient in ingredients:
            name = ingredient.__name__.upper()
            if name in self.name_to_ingredient:
                if flag_duplicates:
                    raise IngredientException(
                        f"Duplicate ingredient names passed! These are case insensitive, be careful.\n{name}"
                    )
                return
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
    query: str, kitchen: Kitchen, ingredient_alias_to_parsed_dict: t.Dict[str, dict]
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
) -> t.Tuple[str, dict, set, Kitchen, t.Set[Ingredient]]:
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
            return [i.name for i in n.find_all(exp.Literal)]
        elif isinstance(n, exp.Paren):
            # This happens when we try to define a tuple with a single item
            # e.g. `options=('something')`
            return [n.this.name]
        elif isinstance(n, (exp.Literal, exp.Boolean)):
            return n.to_py()
        elif isinstance(n, exp.Column):
            return ColumnRef(n.sql(dialect=dialect).strip('"'))
        elif isinstance(n, exp.Subquery):
            return Subquery(n.sql(dialect=dialect).removesuffix(")").removeprefix("("))
        raise ValueError(f"Not sure what to do with {type(n.expression)} here")

    ingredient_alias_to_parsed_dict: t.Dict[str, dict] = {}
    function_hash_to_alias: t.Dict[str, str] = {}
    for idx, function_node in enumerate(node.find_all(exp.BlendSQLFunction)):
        parsed_results_dict = {}
        kwargs_dict = {}
        function_name = function_node.name
        parsed_results_dict["function"] = function_name
        function_args, function_kwargs = [], {}
        function_args.extend(
            [process_arg_value(arg_node) for arg_node in function_node.fn_args]
        )
        for kwarg_node in function_node.fn_kwargs:
            function_kwargs = {
                **function_kwargs,
                **{kwarg_node.name.this: process_arg_value(kwarg_node.value)},
            }
        function_hash = hash(f"{function_name} {function_args} {function_kwargs}")
        if function_hash in function_hash_to_alias:
            # If we've already processed this function, no need to do it again
            ingredient_aliasname = function_hash_to_alias[function_hash]
        else:
            ingredient_aliasname = string.ascii_uppercase[idx]
            function_hash_to_alias[function_hash] = ingredient_aliasname

        substituted_ingredient_alias = format_blendsql_function(ingredient_aliasname)

        kwargs_dict["model"] = default_model

        # Bind arguments to function
        from inspect import signature

        current_ingredient = kitchen.get_from_name(function_name)
        sig = signature(current_ingredient)
        bound = sig.bind(*function_args, **function_kwargs)
        kwargs_dict = {**kwargs_dict, **bound.arguments, **bound.kwargs}

        # Below we track the 'raw' representation, in case we need to pass into
        #   a recursive BlendSQL call later
        ingredient_alias_to_parsed_dict[ingredient_aliasname] = parsed_results_dict | {
            "raw": function_node.sql(dialect=dialect),
            "kwargs_dict": kwargs_dict,
        }
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
    ingredient_alias_to_parsed_dict: t.Dict[str, dict],
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
    materialized_cte_df = materialized_smoothie.df
    db.to_temp_table(
        df=materialized_cte_df,
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
    ).transform(transform.remove_nodetype, exp.With)
    return materialized_smoothie


def get_sorted_blendsql_nodes(
    node: exp.Expression,
    ingredient_alias_to_parsed_dict: dict,
    kitchen: Kitchen,
) -> t.Generator[exp.Expression, None, None]:
    """
    Yields parsed matches from grammar, according to a specified order of operations.

    Args:
        q: str, the current BlendSQL query to parse. Should contain ingredient aliases.
        ingredient_alias_to_parsed_dict: Mapping from ingredient alias to their parsed representations.
            Example:
                {"{{A()}}": {"function": "LLMMap", "args": ...}
        kitchen: Contains inventory of ingredients (aka BlendSQL ingredients)

    Returns:
        Generator containing tuple of:

            - start index of grammar match

            - end index of grammar match

            - parsed_results_dict

            - `Function` object that is matched
    """
    ooo = [
        IngredientType.STRING,
        IngredientType.MAP,
        IngredientType.QA,
        IngredientType.JOIN,
    ]
    parse_results = list(node.find_all(exp.BlendSQLFunction))
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
                yield function_node
                continue
            elif _function.ingredient_type not in IngredientType:
                raise ValueError(
                    f"Not sure what to do with ingredient_type '{_function.ingredient_type}' yet"
                )
            remaining_parse_results.append(function_node)
        parse_results = remaining_parse_results


def disambiguate_and_submit_blend(
    ingredient_alias_to_parsed_dict: t.Dict[str, dict],
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
        Fore.CYAN + f"Executing `{query}` and setting to `{aliasname}`" + Fore.RESET
    )
    return _blend(query=query, **kwargs)


def _blend(
    query: str,
    db: Database,
    default_model: t.Optional[Model] = None,
    ingredients: t.Optional[Collection[t.Type[Ingredient]]] = None,
    verbose: bool = False,
    infer_gen_constraints: bool = True,
    table_to_title: t.Optional[t.Dict[str, str]] = None,
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
    query_context.parse(query, schema=db.sqlglot_schema)

    session_uuid = str(uuid.uuid4())[:4]
    if ingredients is None:
        ingredients = []

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
            for tablename in [i.name for i in query_context.node.find_all(exp.Table)]:
                if tablename not in db.tables():
                    materialized_smoothie = db.lazy_tables.pop(tablename).collect()
                    _prev_passed_values += materialized_smoothie.meta.num_values_passed
        except Exception as e:
            logger.error(f"Error while materializing tables: {e}")
        logger.debug(
            Fore.YELLOW + f"No BlendSQL ingredients found in query:" + Fore.RESET
        )
        logger.debug(Fore.LIGHTYELLOW_EX + query + Fore.RESET)
        logger.debug(Fore.YELLOW + f"Executing as vanilla SQL..." + Fore.RESET)
        return Smoothie(
            df=db.execute_to_df(query_context.to_string()),
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
                contains_ingredient=False,
            ),
        )

    _get_temp_session_table: t.Callable = partial(get_temp_session_table, session_uuid)
    alias_function_name_to_result: t.Dict[str, str] = {}
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
        executed_subquery_ingredients: t.Set[str] = set()
        prev_subquery_map_columns: t.Set[str] = set()
        _get_temp_subquery_table: t.Callable = partial(
            get_temp_subquery_table, session_uuid, subquery_idx
        )
        if not isinstance(subquery, exp.Select):
            # We need to create a select query from this subquery
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
                    Fore.YELLOW
                    + "Encountered subquery without `SELECT`, and more than 1 table!\nCannot optimize yet, skipping this step."
                )
                continue
        else:
            subquery_str = subquery.sql(dialect=dialect)

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
            postprocess_columns,
            abstracted_query_str,
        ) in scm.abstracted_table_selects():
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
                    materialized_smoothie = db.lazy_tables.pop(tablename).collect()
                    _prev_passed_values += materialized_smoothie.meta.num_values_passed

                tablename_to_write = _get_temp_subquery_table(tablename)

                logger.debug(
                    Fore.CYAN
                    + "Executing "
                    + Fore.LIGHTCYAN_EX
                    + f"`{abstracted_query_str}` "
                    + Fore.CYAN
                    + f"and setting to `{tablename_to_write}`..."
                    + Fore.RESET
                )
                abstracted_df = db.execute_to_df(abstracted_query_str)
                if aliased_subquery is None:
                    if postprocess_columns:
                        if isinstance(db, DuckDB):
                            # TODO: fix this
                            # `self.db.execute_to_df("SELECT * FROM League AS l JOIN Country AS c ON l.country_id = c.id WHERE TRUE")`
                            # Gives:
                            #   id  country_id                    name   id_1   name_1
                            #   0      1           1  Belgium Jupiler League      1  Belgium
                            #   1   1729        1729  England Premier League   1729  England
                            #   2   4769        4769          France Ligue 1   4769   France
                            #   3   7809        7809   Germany 1. Bundesliga   7809  Germany
                            #   4  10257       10257           Italy Serie A  10257    Italy
                            # But, below we remove the columns with underscores. we need those.
                            set_of_column_names = set(db.sqlglot_schema[tablename])
                            # In case of a join, duckdb formats columns with 'column_1'
                            # But some columns (e.g. 'parent_category') just have underscores in them already
                            abstracted_df = abstracted_df.rename(
                                columns=lambda x: re.sub(r"_\d$", "", x)
                                if x not in set_of_column_names  # noqa: B023
                                else x
                            )
                        # In case of a join, we could have duplicate column names in our pandas dataframe
                        # This will throw an error when we try to write to the database
                        abstracted_df = abstracted_df.loc[
                            :, ~abstracted_df.columns.duplicated()
                        ]
                db.to_temp_table(
                    df=abstracted_df,
                    tablename=tablename_to_write,
                )
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

        # lazy_limit: Union[int, None] = scm.get_lazy_limit()

        # Now, 1) Find all ingredients to execute (e.g. '{{f(a, b, c)}}')
        # 2) Track when we've created a new table from a MapIngredient call
        #   only at the end of parsing a subquery, we can merge to the original session_uuid table
        tablename_to_map_out: t.Dict[str, t.List[pd.DataFrame]] = {}
        for function_node in get_sorted_blendsql_nodes(
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
            logger.debug(
                Fore.CYAN
                + "Executing "
                + Fore.LIGHTCYAN_EX
                + f" `{curr_function_parsed_results['raw']}`..."
                + Fore.RESET
            )
            if infer_gen_constraints:
                # Latter is the winner.
                # So if we already define something in kwargs_dict,
                #   It's not overriden here
                kwargs_dict = (
                    scm.infer_gen_constraints(function_node=function_node) | kwargs_dict
                )

            if table_to_title is not None:
                kwargs_dict["table_to_title"] = table_to_title

            # Optionally, recursively call blend() again to get subtable from args
            # This applies to `context` and `options`
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
                        subtable = _smoothie.df
                        if unpack_kwarg == "options":
                            if len(subtable.columns) == 1 or len(subtable) == 1:
                                # Here, we need to format as a flat set
                                kwargs_dict[unpack_kwarg] = list(subtable.values.flat)
                            else:
                                raise InvalidBlendSQL(
                                    f"Invalid subquery passed to `options`!\nNeeds to return exactly one column or row, got {len(subtable.columns)} columns and {len(subtable)} rows instead"
                                )
                        elif unpack_kwarg == "context":
                            tup = kwargs_dict[unpack_kwarg]
                            new_tup = (
                                tup[:value_idx] + (subtable,) + tup[value_idx + 1 :]
                            )
                            kwargs_dict[unpack_kwarg] = new_tup
                        else:
                            raise IngredientException(
                                f"Invalid kwarg {unpack_kwarg}\nAlso, we should have never hit this error..."
                            )

            if getattr(curr_ingredient, "model", None) is not None:
                kwargs_dict["model"] = curr_ingredient.model
            # Execute our ingredient function
            function_out = curr_ingredient(
                # *parsed_results_dict["args"],
                **kwargs_dict
                | {
                    "get_temp_subquery_table": _get_temp_subquery_table,
                    "get_temp_session_table": _get_temp_session_table,
                    "aliases_to_tablenames": scm.alias_to_tablename,
                    "prev_subquery_map_columns": prev_subquery_map_columns,
                },
            )
            # Check how to handle output, depending on ingredient type
            if curr_ingredient.ingredient_type == IngredientType.MAP:
                # Parse so we replace this function in blendsql with 1st arg
                #   (new_col, which is the question we asked)
                #  But also update our underlying table, so we can execute correctly at the end
                (new_col, tablename, colname, new_table) = function_out
                prev_subquery_map_columns.add(new_col)
                new_table[new_table[new_col].notnull()]
                if tablename in tablename_to_map_out:
                    tablename_to_map_out[tablename].append(new_table)
                else:
                    tablename_to_map_out[tablename] = [new_table]
                session_modified_tables.add(tablename)
                alias_function_name_to_result[
                    function_node.name
                ] = f'"{double_quote_escape(tablename)}"."{double_quote_escape(new_col)}"'
            elif curr_ingredient.ingredient_type in (
                IngredientType.STRING,
                IngredientType.QA,
            ):
                # Here, we can simply insert the function's output
                alias_function_name_to_result[function_node.name] = function_out
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
                assert join_node is not None
                join_node.replace(exp.BlendSQLFunction(this=function_node.name))
                alias_function_name_to_result[function_node.name] = join_clause
            else:
                raise ValueError(
                    f"Not sure what to do with ingredient_type '{curr_ingredient.ingredient_type}' yet\n(Also, we should have never hit this error....)"
                )
        # Combine all the retrieved ingredient outputs
        for tablename, ingredient_outputs in tablename_to_map_out.items():
            if len(ingredient_outputs) > 0:
                logger.debug(
                    Fore.CYAN
                    + f"Combining {len(ingredient_outputs)} outputs for table `{tablename}`"
                    + Fore.RESET
                )
                # Once we finish parsing this subquery, write to our session_uuid table
                # Below, we differ from Binder, which seems to replace the old table
                # On their left join merge command: https://github.com/HKUNLP/Binder/blob/9eede69186ef3f621d2a50572e1696bc418c0e77/nsql/database.py#L196
                # We create a new temp table to avoid a potentially self-destructive operation
                base_tablename = tablename
                _base_table: pd.DataFrame = db.execute_to_df(
                    select_all_from_table_query(base_tablename)
                )
                base_table = _base_table
                if db.has_temp_table(_get_temp_session_table(tablename)):
                    base_tablename = _get_temp_session_table(tablename)
                    base_table: pd.DataFrame = db.execute_to_df(
                        select_all_from_table_query(base_tablename)
                    )
                previously_added_columns = base_table.columns.difference(
                    _base_table.columns
                )
                assert len(set([len(x) for x in ingredient_outputs])) == 1
                llm_out_df = pd.concat(ingredient_outputs, axis=1)
                llm_out_df = llm_out_df.loc[:, ~llm_out_df.columns.duplicated()]
                # Handle duplicate columns, e.g. in test_nested_duplicate_ingredient_calls()
                for column in previously_added_columns:
                    if all(
                        column in x for x in [llm_out_df.columns, base_table.columns]
                    ):
                        # Fill nan in llm_out_df with those values in base_table
                        try:
                            pd.testing.assert_index_equal(
                                base_table.index, llm_out_df.index
                            )
                        except AssertionError:
                            logger.debug(
                                Fore.RED + "pd.testing.assert_index_equal error"
                            )
                        llm_out_df[column] = llm_out_df[column].fillna(
                            base_table[column]
                        )
                        base_table = base_table.drop(columns=column)
                llm_out_df = llm_out_df[
                    llm_out_df.columns.difference(base_table.columns)
                ]
                try:
                    pd.testing.assert_index_equal(base_table.index, llm_out_df.index)
                except AssertionError:
                    logger.debug(Fore.RED + "pd.testing.assert_index_equal error")
                merged = base_table.merge(
                    llm_out_df, how="left", right_index=True, left_index=True
                )
                db.to_temp_table(
                    df=merged, tablename=_get_temp_session_table(tablename)
                )
                session_modified_tables.add(tablename)
    # Now insert the function outputs to the original query
    # We need to re-sync if we did some operation on the underlying query,
    #   like with a JoinIngredient
    query = query_context.to_string()
    for alias, res in alias_function_name_to_result.items():
        query = re.sub(
            re.escape(format_blendsql_function(alias)),
            f" {str(res)} ",
            query,
        )
    query_context.parse(query)
    for t in session_modified_tables:
        query_context.node = query_context.node.transform(
            transform.replace_tablename, t, _get_temp_session_table(t)
        )
    if scm is not None:
        for a, t in scm.alias_to_tablename.items():
            if t in session_modified_tables:
                query_context.node = query_context.node.transform(
                    transform.replace_tablename, a, _get_temp_session_table(t)
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

    logger.debug(Fore.LIGHTGREEN_EX + f"Final Query:\n{query}" + Fore.RESET)

    df = db.execute_to_df(query)

    return Smoothie(
        df=df,
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

    db: t.Union[pd.DataFrame, dict, str, Database] = field(default=None)
    model: t.Optional[Model] = field(default=None)
    ingredients: t.Optional[Collection[t.Type[Ingredient]]] = field(
        default_factory=list
    )

    verbose: bool = field(default=False)
    infer_gen_constraints: bool = field(default=True)
    table_to_title: t.Optional[t.Dict[str, str]] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.db, Database):
            self.db = self._infer_db_type(self.db)
        if self.db is None:
            raise ValueError("df_or_db_path must be provided")
        self.ingredients = self._merge_default_ingredients(self.ingredients)
        self._toggle_verbosity(self.verbose)

    @staticmethod
    def _toggle_verbosity(verbose_in_use: bool):
        def set_level(l: int):
            logger.setLevel(l)
            for handler in logger.handlers:
                handler.setLevel(l)

        if verbose_in_use:
            set_level(logging.DEBUG)
        else:
            set_level(logging.ERROR)

    @staticmethod
    def _merge_default_ingredients(
        ingredients: t.Optional[Collection[t.Type[Ingredient]]],
    ):
        from blendsql.ingredients import LLMQA, LLMMap, LLMJoin

        DEFAULT_INGREDIENTS = {LLMQA, LLMMap, LLMJoin}
        ingredients = set(ingredients)
        try:
            ingredient_names = {i.__name__ for i in ingredients}
        except AttributeError as e:
            raise IngredientException(
                "All arguments passed to `ingredients` should be `Ingredient` classes!"
            ) from e
        for default_ingredient in DEFAULT_INGREDIENTS:
            if default_ingredient.__name__ not in ingredient_names:
                ingredients.add(default_ingredient)
        return ingredients

    @staticmethod
    def _infer_db_type(df_or_db_path) -> Database:
        from pathlib import Path

        if df_or_db_path is None:
            from .db.pandas import Pandas

            return Pandas({})  # Load an empty DuckDB connection

        elif isinstance(df_or_db_path, (pd.DataFrame, dict)):
            from .db.pandas import Pandas

            return Pandas(df_or_db_path)
        elif isinstance(df_or_db_path, (str, Path)):
            if Path(df_or_db_path).exists():
                from .db.sqlite import SQLite

                return SQLite(df_or_db_path)
            else:
                from .db.postgresql import PostgreSQL

                return PostgreSQL(df_or_db_path)
        else:
            raise ValueError(
                f"Could not resolve '{df_or_db_path}' to a valid database type!"
            )

    def visualize(self, query: str, output_path: t.Optional[str] = None, format="pdf"):
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
        ingredients: t.Optional[Collection[t.Type[Ingredient]]] = None,
        model: t.Optional[str] = None,
        infer_gen_constraints: t.Optional[bool] = None,
        verbose: t.Optional[bool] = None,
    ) -> Smoothie:
        '''The `execute()` function is used to execute a BlendSQL query against a database and
        return the final result, in addition to the intermediate reasoning steps taken.
        Execution is done on a database given an ingredient context.

        Args:
            query: The BlendSQL query to execute
            ingredients: Collection of ingredient objects, to use in interpreting BlendSQL query
            verbose: Boolean defining whether to run with logger in debug mode
            default_model: Which BlendSQL model to use in performing ingredient tasks in the current query
            infer_gen_constraints: Optionally infer the output format of an `IngredientMap` call, given the predicate context
                For example, in `{{LLMMap('convert to date', 'w::listing date')}} <= '1960-12-31'`
                We can infer the output format should look like '1960-12-31' and both:
                    1) Put this string in the `example_outputs` kwarg
                    2) If we have a LocalModel, pass the r'\d{4}-\d{2}-\d{2}' pattern to guidance
            table_to_title: Optional mapping from table name to title of table.
                Useful for datasets like WikiTableQuestions, where relevant info is stored in table title.

        Returns:
            smoothie: `Smoothie` dataclass containing pd.DataFrame output and execution metadata

        Examples:
            ```python
            import pandas as pd

            from blendsql import BlendSQL, config
            from blendsql.ingredients import LLMMap, LLMQA, LLMJoin
            from blendsql.models import LiteLLM, TransformersLLM

            # Optionally set how many async calls to allow concurrently
            # This depends on your OpenAI/Anthropic/etc. rate limits
            config.set_async_limit(10)

            # Load model
            model = LiteLLM("openai/gpt-4o-mini") # requires .env file with `OPENAI_API_KEY`
            # model = LiteLLM("anthropic/claude-3-haiku-20240307") # requires .env file with `ANTHROPIC_API_KEY`
            # model = TransformersLLM(
            #    "meta-llama/Llama-3.2-1B-Instruct",
            #    config={"chat_template": Llama3ChatTemplate, "device_map": "auto"},
            # ) # run with any local Transformers model

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
                ingredients={LLMMap, LLMQA, LLMJoin},
                model=model,
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
            print(smoothie.summary())
            # ┌────────────┬──────────────────────┬─────────────────┬─────────────────────┐
            # │   Time (s) │   # Generation Calls │   Prompt Tokens │   Completion Tokens │
            # ├────────────┼──────────────────────┼─────────────────┼─────────────────────┤
            # │    1.25158 │                    1 │             296 │                  16 │
            # └────────────┴──────────────────────┴─────────────────┴─────────────────────┘
            ```
        '''
        self._toggle_verbosity(verbose if verbose is not None else self.verbose)

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
                table_to_title=self.table_to_title,
            )
        except Exception as error:
            raise error
        finally:
            # In the case of a recursive `_blend()` call,
            #   this logic allows temp tables to persist until
            #   the final base case is fulfilled.
            self.db._reset_connection()
        smoothie.meta.process_time_seconds = time.time() - start
        # Reset model stats, so future executions don't add here
        if model_in_use is not None:
            model_in_use.reset_stats()
        return smoothie
