import copy
import logging
import time
import uuid
import pandas as pd
import re
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Generator,
    Optional,
    Callable,
    Type,
)
from collections.abc import Collection, Iterable
from sqlite3 import OperationalError
from attr import attrs, attrib
from functools import partial
from sqlglot import exp
from colorama import Fore
import string

from ._logger import logger, msg_box
from .utils import (
    sub_tablename,
    get_temp_session_table,
    get_temp_subquery_table,
    recover_blendsql,
    get_tablename_colname,
)
from ._exceptions import InvalidBlendSQL
from .db import Database
from .db.utils import double_quote_escape, select_all_from_table_query, LazyTable
from ._sqlglot import (
    MODIFIERS,
    QueryContextManager,
    SubqueryContextManager,
    get_first_child,
    get_reversed_subqueries,
    replace_join_with_ingredient_single_ingredient,
    replace_join_with_ingredient_multiple_ingredient,
    prune_true_where,
    prune_with,
    replace_subquery_with_direct_alias_call,
    maybe_set_subqueries_to_true,
    remove_ctes,
    is_in_cte,
    get_scope_nodes,
)
from ._dialect import _parse_one, FTS5SQLite
from .grammars._peg_grammar import grammar
from .ingredients.ingredient import Ingredient, IngredientException
from ._smoothie import Smoothie, SmoothieMeta
from ._constants import IngredientType, IngredientKwarg
from .models._model import Model


@attrs
class Kitchen(list):
    """Superset of list. A collection of ingredients."""

    db: Database = attrib()
    session_uuid: str = attrib()

    name_to_ingredient: Dict[str, Ingredient] = attrib(init=False)

    def __attrs_post_init__(self):
        self.name_to_ingredient = {}

    def names(self):
        return [i.name for i in self]

    def get_from_name(self, name: str):
        try:
            return self.name_to_ingredient[name.upper()]
        except KeyError:
            raise InvalidBlendSQL(
                f"Ingredient '{name}' called, but not found in passed `ingredient` arg!"
            ) from None

    def extend(self, ingredients: Iterable[Type[Ingredient]]) -> None:
        """ "Initializes ingredients class with base attributes, for use in later operations."""
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
            assert (
                name not in self.name_to_ingredient
            ), f"Duplicate ingredient names passed! These are case insensitive, be careful.\n{name}"
            # Initialize the ingredient, going from `Type[Ingredient]` to `Ingredient`
            initialied_ingredient: Ingredient = ingredient(
                name=name,
                # Add db and session_uuid as default kwargs
                # This way, ingredients are able to interact with data
                db=self.db,
                session_uuid=self.session_uuid,
            )
            self.name_to_ingredient[name] = initialied_ingredient
            self.append(initialied_ingredient)


def autowrap_query(
    query: str, kitchen: Kitchen, ingredient_alias_to_parsed_dict: Dict[str, dict]
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
        _function: Ingredient = kitchen.get_from_name(d["function"])
        if _function.ingredient_type == IngredientType.QA:
            # If the query only contains the function alias
            # E.g: '{{A()}}'
            if query == alias:
                query = query.replace(
                    alias,
                    f"""SELECT {alias}""",
                )
        elif _function.ingredient_type == IngredientType.JOIN:
            left_table, _ = get_tablename_colname(d["kwargs_dict"]["left_on"])
            query = query.replace(
                alias,
                f'"{left_table}" ON {alias}',
            )
        else:
            continue
    return query


def preprocess_blendsql(query: str, default_model: Model) -> Tuple[str, dict, set]:
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
                '{{A()}}': {
                    'function': 'LLMJoin',
                    'args': [],
                    'ingredient_aliasname': 'A',
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
    ingredient_alias_to_parsed_dict: Dict[str, dict] = {}
    ingredient_str_to_alias: Dict[str, str] = {}
    tables_in_ingredients: Set[str] = set()
    query = re.sub(r"(\s+)", " ", query)
    reversed_scan_res = [scan_res for scan_res in grammar.scanString(query)][::-1]
    for idx, (parse_results, start, end) in enumerate(reversed_scan_res):
        original_ingredient_string = query[start:end]
        # If we're in between parentheses, add a `SELECT`
        # This way it gets picked up as a subquery to parse later
        # TODO: is this safe to do?
        if query[start - 2] == "(" and query[end + 1] == ")":
            inserted_select = " SELECT "
            query = query[:start] + inserted_select + query[start:end] + query[end:]
            start += len(inserted_select)
            end += len(inserted_select)
        if (
            original_ingredient_string in ingredient_str_to_alias
        ):  # If we've already processed this function, no need to do it again
            substituted_ingredient_alias = ingredient_str_to_alias[
                original_ingredient_string
            ]
        else:
            parsed_results_dict = parse_results.as_dict()
            ingredient_aliasname = string.ascii_uppercase[idx]
            parsed_results_dict["ingredient_aliasname"] = ingredient_aliasname
            substituted_ingredient_alias = "{{" + f"{ingredient_aliasname}()" + "}}"
            ingredient_str_to_alias[
                original_ingredient_string
            ] = substituted_ingredient_alias
            # Remove parentheses at beginning and end of arg/kwarg
            # TODO: this should be handled by pyparsing
            for arg_type in {"args", "kwargs"}:
                for idx in range(len(parsed_results_dict[arg_type])):
                    curr_arg = parsed_results_dict[arg_type][idx]
                    curr_arg = curr_arg[-1] if arg_type == "kwargs" else curr_arg
                    if not isinstance(curr_arg, str):
                        continue
                    formatted_curr_arg = re.sub(
                        r"(^\()(.*)(\)$)", r"\2", curr_arg
                    ).strip()
                    if arg_type == "args":
                        parsed_results_dict[arg_type][idx] = formatted_curr_arg
                    else:
                        parsed_results_dict[arg_type][idx][
                            -1
                        ] = formatted_curr_arg  # kwargs gets returned as ['limit', '=', 10] sort of list
            # So we need to parse by indices in dict expression
            # maybe if I was better at pp.Suppress we wouldn't need this
            kwargs_dict = {x[0]: x[-1] for x in parsed_results_dict["kwargs"]}
            kwargs_dict[IngredientKwarg.MODEL] = default_model
            context_arg = kwargs_dict.get(
                IngredientKwarg.CONTEXT,
                (
                    parsed_results_dict["args"][1]
                    if len(parsed_results_dict["args"]) > 1
                    else (
                        parsed_results_dict["args"][1]
                        if len(parsed_results_dict["args"]) > 1
                        else None
                    )
                ),
            )
            for arg in {
                context_arg,
                kwargs_dict.get("left_on", None),
                kwargs_dict.get("right_on", None),
            }:
                if arg is None:
                    continue
                if not arg.upper().startswith(("SELECT", "WITH")):
                    tablename, _ = get_tablename_colname(arg)
                    tables_in_ingredients.add(tablename)
            # We don't need raw kwargs anymore
            # in the future, we just refer to kwargs_dict
            parsed_results_dict.pop("kwargs")
            # Below we track the 'raw' representation, in case we need to pass into
            #   a recursive BlendSQL call later
            ingredient_alias_to_parsed_dict[
                substituted_ingredient_alias
            ] = parsed_results_dict | {
                "raw": query[start:end],
                "kwargs_dict": kwargs_dict,
            }
        query = query[:start] + substituted_ingredient_alias + query[end:]
    return (query.strip(), ingredient_alias_to_parsed_dict, tables_in_ingredients)


def materialize_cte(
    subquery: exp.Expression,
    query_context: QueryContextManager,
    aliasname: str,
    db: Database,
    default_model: Model,
    ingredient_alias_to_parsed_dict: Dict[str, dict],
    **kwargs,
) -> pd.DataFrame:
    str_subquery = recover_blendsql(subquery.sql(dialect=FTS5SQLite))
    materialized_cte_df: pd.DataFrame = disambiguate_and_submit_blend(
        ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
        query=str_subquery,
        db=db,
        default_model=default_model,
        aliasname=aliasname,
        **kwargs,
    ).df
    db.to_temp_table(
        df=materialized_cte_df,
        tablename=aliasname,
    )
    # Now, we need to remove subquery and instead insert direct reference to aliasname
    # Example:
    #   `SELECT Symbol FROM (SELECT DISTINCT Symbol FROM portfolio) AS w`
    #   Should become: `SELECT Symbol FROM w`
    query_context.node = query_context.node.transform(
        replace_subquery_with_direct_alias_call,
        subquery=subquery.parent,
        aliasname=aliasname,
    ).transform(prune_with)
    return materialized_cte_df


def get_sorted_grammar_matches(
    q: str,
    ingredient_alias_to_parsed_dict: dict,
    kitchen: Kitchen,
) -> Generator[Tuple[int, int, str, dict, Ingredient], None, None]:
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
    parse_results = [i for i in grammar.scanString(q)]
    while len(parse_results) > 0:
        curr_ingredient_target = ooo.pop(0)
        remaining_parse_results = []
        for _, start, end in parse_results:
            alias_function_str = q[start:end]
            # Fetch parsed ingredient dict from our cache
            parse_results_dict = ingredient_alias_to_parsed_dict[alias_function_str]
            _function: Ingredient = kitchen.get_from_name(
                parse_results_dict["function"]
            )
            if _function.ingredient_type == curr_ingredient_target:
                yield (start, end, alias_function_str, parse_results_dict, _function)
                continue
            elif _function.ingredient_type not in IngredientType:
                raise ValueError(
                    f"Not sure what to do with ingredient_type '{_function.ingredient_type}' yet"
                )
            remaining_parse_results.append((_, start, end))
        parse_results = remaining_parse_results


def disambiguate_and_submit_blend(
    ingredient_alias_to_parsed_dict: Dict[str, dict],
    query: str,
    aliasname: str,
    **kwargs,
):
    """
    Used to disambiguate anonymized BlendSQL function and execute in a recursive context.
    """
    for alias, d in ingredient_alias_to_parsed_dict.items():
        query = re.sub(re.escape(alias), d["raw"], query)
    logger.debug(
        Fore.CYAN + f"Executing `{query}` and setting to `{aliasname}`..." + Fore.RESET
    )
    return _blend(query=query, **kwargs)


def _blend(
    query: str,
    db: Database,
    default_model: Optional[Model] = None,
    ingredients: Optional[Collection[Type[Ingredient]]] = None,
    verbose: bool = False,
    infer_gen_constraints: bool = True,
    table_to_title: Optional[Dict[str, str]] = None,
    schema_qualify: bool = True,
    _prev_passed_values: int = 0,
) -> Smoothie:
    """Invoked from blend(), this contains the recursive logic to execute
    a BlendSQL query and return a `Smoothie` object.
    """
    # The QueryContextManager class is used to track all manipulations done to
    # the original query, prior to the final execution on the underlying DBMS.
    original_query = copy.deepcopy(query)
    query_context = QueryContextManager()
    naive_execution = False
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
        tables_in_ingredients,
    ) = preprocess_blendsql(query=query, default_model=default_model)
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
        logger.debug(
            Fore.YELLOW + f"No BlendSQL ingredients found in query:" + Fore.RESET
        )
        logger.debug(Fore.LIGHTYELLOW_EX + query + Fore.RESET)
        logger.debug(Fore.YELLOW + f"Executing as vanilla SQL..." + Fore.RESET)
        return Smoothie(
            df=db.execute_to_df(query_context.to_string()),
            meta=SmoothieMeta(
                num_values_passed=0,
                prompt_tokens=default_model.prompt_tokens
                if default_model is not None
                else 0,
                completion_tokens=(
                    default_model.completion_tokens if default_model is not None else 0
                ),
                prompts=default_model.prompts if default_model is not None else [],
                ingredients=[],
                query=original_query,
                db_url=str(db.db_url),
                contains_ingredient=False,
            ),
        )

    schema = None
    if schema_qualify:
        # Only construct sqlglot schema if we need to
        schema = db.sqlglot_schema
    query_context.parse(query, schema=schema)

    _get_temp_session_table: Callable = partial(get_temp_session_table, session_uuid)
    # Mapping from {"QA('does this company...', 'constituents::Name')": 'does this company'...})
    function_call_to_res: Dict[str, str] = {}
    session_modified_tables = set()
    # TODO: Currently, as we traverse upwards from deepest subquery,
    #   if any lower subqueries have an ingredient, we deem the current
    #   as ineligible for optimization. Maybe this can be improved in the future.
    prev_subquery_has_ingredient = False
    for subquery_idx, subquery in enumerate(
        get_reversed_subqueries(query_context.node)
    ):
        # At this point, we should have already handled cte statements and created associated tables
        if subquery.find(exp.With) is not None:
            subquery = subquery.transform(remove_ctes)
        # Only cache executed_ingredients within the same subquery
        # The same ingredient may have different results within a different subquery context
        executed_subquery_ingredients: Set[str] = set()
        prev_subquery_map_columns: Set[str] = set()
        _get_temp_subquery_table: Callable = partial(
            get_temp_subquery_table, session_uuid, subquery_idx
        )
        if not isinstance(subquery, exp.Select):
            # We need to create a select query from this subquery
            # So we find the parent select, and grab that table
            parent_select_tablenames = [
                i.name for i in subquery.find_ancestor(exp.Select).find_all(exp.Table)
            ]
            if len(parent_select_tablenames) == 1:
                subquery_str = recover_blendsql(
                    f"SELECT * FROM {parent_select_tablenames[0]} WHERE "
                    + get_first_child(subquery).sql(dialect=FTS5SQLite)
                )
            else:
                logger.debug(
                    Fore.YELLOW
                    + "Encountered subquery without `SELECT`, and more than 1 table!\nCannot optimize yet, skipping this step."
                )
                continue
        else:
            subquery_str = recover_blendsql(subquery.sql(dialect=FTS5SQLite))

        in_cte, table_alias_name = is_in_cte(subquery, return_name=True)
        scm = SubqueryContextManager(
            node=_parse_one(
                subquery_str
            ),  # Need to do this so we don't track parents into construct_abstracted_selects
            prev_subquery_has_ingredient=prev_subquery_has_ingredient,
            alias_to_subquery={table_alias_name: subquery} if in_cte else {},
            tables_in_ingredients=tables_in_ingredients,
        )
        for tablename, abstracted_query in scm.abstracted_table_selects():
            # If this table isn't being used in any ingredient calls, there's no
            #   need to create a temporary session table
            if (tablename not in tables_in_ingredients) and (
                scm.tablename_to_alias.get(tablename, None) not in tables_in_ingredients
            ):
                continue
            aliased_subquery = scm.alias_to_subquery.pop(tablename, None)
            if aliased_subquery is not None:
                # First, we need to explicitly create the aliased subquery as a table
                # For example, `SELECT Symbol FROM (SELECT DISTINCT Symbol FROM portfolio) AS w WHERE w...`
                # We can't assign `abstracted_query` for non-existent `w`
                #   until we set `w` to `SELECT DISTINCT Symbol FROM portfolio`
                db.lazy_tables.add(
                    LazyTable(
                        tablename,
                        partial(
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
                    )
                )
            if abstracted_query is not None:
                if tablename in db.lazy_tables:
                    db.lazy_tables.pop(tablename).collect()
                logger.debug(
                    Fore.CYAN
                    + f"Executing `{abstracted_query}` and setting to `{_get_temp_subquery_table(tablename)}`..."
                    + Fore.RESET
                )
                try:
                    db.to_temp_table(
                        df=db.execute_to_df(abstracted_query),
                        tablename=_get_temp_subquery_table(tablename),
                    )
                except OperationalError as e:
                    # Fallback to naive execution
                    logger.debug(Fore.RED + str(e) + Fore.RESET)
                    logger.debug(
                        Fore.RED + "Falling back to naive execution..." + Fore.RESET
                    )
                    naive_execution = True
        # Be sure to handle those remaining aliases, which didn't have abstracted queries
        for aliasname, aliased_subquery in scm.alias_to_subquery.items():
            db.lazy_tables.add(
                LazyTable(
                    aliasname,
                    partial(
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
                )
            )
        if prev_subquery_has_ingredient:
            scm.set_node(scm.node.transform(maybe_set_subqueries_to_true))

        # lazy_limit: Union[int, None] = scm.get_lazy_limit()
        # After above processing of AST, sync back to string repr
        subquery_str = scm.sql()
        # Now, 1) Find all ingredients to execute (e.g. '{{f(a, b, c)}}')
        # 2) Track when we've created a new table from a MapIngredient call
        #   only at the end of parsing a subquery, we can merge to the original session_uuid table
        tablename_to_map_out: Dict[str, List[pd.DataFrame]] = {}
        for (
            start,
            end,
            alias_function_str,
            parsed_results_dict,
            ingredient,
        ) in get_sorted_grammar_matches(
            q=subquery_str,
            ingredient_alias_to_parsed_dict=ingredient_alias_to_parsed_dict,
            kitchen=kitchen,
        ):
            prev_subquery_has_ingredient = True
            if alias_function_str in executed_subquery_ingredients:
                # Don't execute same ingredient twice
                continue
            executed_subquery_ingredients.add(alias_function_str)
            kwargs_dict = parsed_results_dict["kwargs_dict"]

            if infer_gen_constraints:
                # Latter is the winner.
                # So if we already define something in kwargs_dict,
                #   It's not overriden here
                kwargs_dict = (
                    scm.infer_gen_constraints(
                        start=start,
                        end=end,
                    )
                    | kwargs_dict
                )

            if table_to_title is not None:
                kwargs_dict["table_to_title"] = table_to_title
            # Heuristic check to see if we should snag the singleton arg as context
            if (
                len(parsed_results_dict["args"]) == 1
                and "::" in parsed_results_dict["args"][0]
            ):
                kwargs_dict[IngredientKwarg.CONTEXT] = parsed_results_dict["args"].pop()
            # Optionally, recursively call blend() again to get subtable from args
            # This applies to `context` and `options`
            for i, unpack_kwarg in enumerate(
                [IngredientKwarg.CONTEXT, IngredientKwarg.OPTIONS]
            ):
                unpack_value = kwargs_dict.get(
                    unpack_kwarg,
                    (
                        parsed_results_dict["args"][i + 1]
                        if len(parsed_results_dict["args"]) > i + 1
                        else (
                            parsed_results_dict["args"][i]
                            if len(parsed_results_dict["args"]) > i
                            else ""
                        )
                    ),
                )
                if isinstance(unpack_value, str) and unpack_value.upper().startswith(
                    ("SELECT", "WITH")
                ):
                    _smoothie = _blend(
                        query=unpack_value,
                        db=db,
                        default_model=default_model,
                        ingredients=ingredients,
                        infer_gen_constraints=infer_gen_constraints,
                        table_to_title=table_to_title,
                        verbose=verbose,
                        _prev_passed_values=_prev_passed_values,
                    )
                    _prev_passed_values = _smoothie.meta.num_values_passed
                    subtable = _smoothie.df
                    if unpack_kwarg == IngredientKwarg.OPTIONS:
                        if len(subtable.columns) != 1:
                            raise InvalidBlendSQL(
                                f"Invalid subquery passed to `options`!\nNeeds to return exactly one column, got {len(subtable.columns)} instead"
                            )
                        # Here, we need to format as a flat set
                        kwargs_dict[unpack_kwarg] = list(subtable.values.flat)
                    else:
                        kwargs_dict[unpack_kwarg] = subtable
                        # Below, we can remove the optional `context` arg we passed in args
                        parsed_results_dict["args"] = parsed_results_dict["args"][:1]
            if getattr(ingredient, "model", None) is not None:
                kwargs_dict["model"] = ingredient.model
            # Execute our ingredient function
            function_out = ingredient(
                *parsed_results_dict["args"],
                **kwargs_dict
                | {
                    "get_temp_subquery_table": _get_temp_subquery_table,
                    "get_temp_session_table": _get_temp_session_table,
                    "aliases_to_tablenames": scm.alias_to_tablename,
                    "prev_subquery_map_columns": prev_subquery_map_columns,
                },
            )
            # Check how to handle output, depending on ingredient type
            if ingredient.ingredient_type == IngredientType.MAP:
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
                function_call_to_res[
                    alias_function_str
                ] = f'"{double_quote_escape(tablename)}"."{double_quote_escape(new_col)}"'
            elif ingredient.ingredient_type in (
                IngredientType.STRING,
                IngredientType.QA,
            ):
                # Here, we can simply insert the function's output
                function_call_to_res[alias_function_str] = function_out
            elif ingredient.ingredient_type == IngredientType.JOIN:
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
                num_ingredients_in_join = len(list(join_node.find_all(exp.Struct))) // 2
                if num_ingredients_in_join > 1:
                    # Case where we have
                    # `SELECT * FROM w0 JOIN w0 ON {{B()}} > 1 AND {{A()}} WHERE TRUE`
                    # Since we haven't executed and saved `{{B()}}` to temp table yet,
                    #   we need to keep. So we get:
                    temp_uuid = str(uuid.uuid4())
                    query_context.node = query_context.node.transform(
                        replace_join_with_ingredient_multiple_ingredient,
                        ingredient_name=parsed_results_dict["ingredient_aliasname"],
                        ingredient_alias=alias_function_str,
                        temp_uuid=temp_uuid,
                    ).transform(prune_true_where)
                    query_context.node = query_context.parse(
                        query_context.to_string().replace(f'SELECT "{temp_uuid}", ', "")
                    )
                else:
                    # Case where we have
                    # `SELECT * FROM w0 JOIN w0 ON w0.x > 1 AND {{A()}} WHERE TRUE`
                    # Since we've already applied SQL operation `w0.x > 1` and set to temp table
                    # we can remove this. So below we transform to:
                    # `SELECT * FROM w0 {{A()}}  WHERE TRUE`
                    # This way, `{{A()}}` can get replaced with our new join
                    # TODO: since we're not removing predicates in other areas, probably not best to do it here.
                    #   Should probably modify in the future.
                    query_context.node = query_context.node.transform(
                        replace_join_with_ingredient_single_ingredient,
                        ingredient_name=parsed_results_dict["ingredient_aliasname"],
                        ingredient_alias=alias_function_str,
                    )
                function_call_to_res[alias_function_str] = join_clause
            else:
                raise ValueError(
                    f"Not sure what to do with ingredient_type '{ingredient.ingredient_type}' yet\n(Also, we should have never hit this error....)"
                )
            if naive_execution:
                break
        # Combine all the retrieved ingredient outputs
        for tablename, llm_outs in tablename_to_map_out.items():
            if len(llm_outs) > 0:
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
                assert len(set([len(x) for x in llm_outs])) == 1
                llm_out_df = pd.concat(llm_outs, axis=1)
                llm_out_df = llm_out_df.loc[:, ~llm_out_df.columns.duplicated()]
                # Handle duplicate columns, e.g. in test_nested_duplicate_ingredient_calls()
                for column in previously_added_columns:
                    if all(
                        column in x for x in [llm_out_df.columns, base_table.columns]
                    ):
                        # Fill nan in llm_out_df with those values in base_table
                        pd.testing.assert_index_equal(
                            base_table.index, llm_out_df.index
                        )
                        llm_out_df[column] = llm_out_df[column].fillna(
                            base_table[column]
                        )
                        base_table = base_table.drop(columns=column)
                llm_out_df = llm_out_df[
                    llm_out_df.columns.difference(base_table.columns)
                ]
                pd.testing.assert_index_equal(base_table.index, llm_out_df.index)
                merged = base_table.merge(
                    llm_out_df, how="left", right_index=True, left_index=True
                )
                db.to_temp_table(
                    df=merged, tablename=_get_temp_session_table(tablename)
                )
                session_modified_tables.add(tablename)

    # Now insert the function outputs to the original query
    query = query_context.to_string()
    for function_str, res in function_call_to_res.items():
        query = query.replace(function_str, str(res))
    for t in session_modified_tables:
        query = sub_tablename(
            t, f'"{double_quote_escape(_get_temp_session_table(t))}"', query
        )
    if scm is not None:
        for a, t in scm.alias_to_tablename.items():
            if t in session_modified_tables:
                query = sub_tablename(
                    a, f'"{double_quote_escape(_get_temp_session_table(t))}"', query
                )
    # Finally, iter through tables in query and see if we need to collect LazyTable
    for table in get_scope_nodes(
        nodetype=exp.Table, node=query_context.node, restrict_scope=False
    ):
        if table.name in db.lazy_tables:
            db.lazy_tables.pop(table.name).collect()

    logger.debug(Fore.LIGHTGREEN_EX + msg_box(f"Final Query:\n{query}") + Fore.RESET)

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
            prompt_tokens=default_model.prompt_tokens
            if default_model is not None
            else 0,
            completion_tokens=default_model.completion_tokens
            if default_model is not None
            else 0,
            prompts=default_model.prompts if default_model is not None else [],
            ingredients=ingredients,
            query=original_query,
            db_url=str(db.db_url),
        ),
    )


def blend(
    query: str,
    db: Database,
    default_model: Optional[Model] = None,
    ingredients: Optional[Collection[Type[Ingredient]]] = None,
    verbose: bool = False,
    infer_gen_constraints: bool = True,
    table_to_title: Optional[Dict[str, str]] = None,
    schema_qualify: bool = True,
) -> Smoothie:
    '''The `blend()` function is used to execute a BlendSQL query against a database and
    return the final result, in addition to the intermediate reasoning steps taken.
    Execution is done on a database given an ingredient context.

    Args:
        query: The BlendSQL query to execute
        db: Database connector object
        ingredients: Collection of ingredient objects, to use in interpreting BlendSQL query
        verbose: Boolean defining whether to run with logger in debug mode
        default_model: Which BlendSQL model to use in performing ingredient tasks in the current query
        infer_gen_constraints: Optionally infer the output format of an `IngredientMap` call, given the predicate context
            For example, in `{{LLMMap('convert to date', 'w::listing date')}} <= '1960-12-31'`
            We can infer the output format should look like '1960-12-31' and both:
                1) Put this string in the `example_outputs` kwarg
                2) If we have a LocalModel, pass the '\d{4}-\d{2}-\d{2}' pattern to outlines.generate.regex
        table_to_title: Optional mapping from table name to title of table.
            Useful for datasets like WikiTableQuestions, where relevant info is stored in table title.
        schema_qualify: Optional bool, determines if we run qualify_columns() from sqlglot
            This enables us to write BlendSQL scripts over multi-table databases without manually qualifying columns ourselves
            However, we need to call `db.sqlglot_schema` if schema_qualify=True, which may add some latency.
            With single-table queries, we can set this to False.

    Returns:
        smoothie: `Smoothie` dataclass containing pd.DataFrame output and execution metadata

    Examples:
        ```python
        import pandas as pd

        from blendsql import blend, LLMMap, LLMQA, LLMJoin
        from blendsql.db import Pandas
        from blendsql.models import TransformersLLM

        # Load model
        model = TransformersLLM('Qwen/Qwen1.5-0.5B')

        # Prepare our local database
        db = Pandas(
            {
                "w": pd.DataFrame(
                    (
                        ['11 jun', 'western districts', 'bathurst', 'bathurst ground', '11-0'],
                        ['12 jun', 'wallaroo & university nsq', 'sydney', 'cricket ground',
                         '23-10'],
                        ['5 jun', 'northern districts', 'newcastle', 'sports ground', '29-0']
                    ),
                    columns=['date', 'rival', 'city', 'venue', 'score']
                ),
                "documents": pd.DataFrame(
                    (
                        ['bathurst, new south wales', 'bathurst /ˈbæθərst/ is a city in the central tablelands of new south wales , australia . it is about 200 kilometres ( 120 mi ) west-northwest of sydney and is the seat of the bathurst regional council .'],
                        ['sydney', 'sydney ( /ˈsɪdni/ ( listen ) sid-nee ) is the state capital of new south wales and the most populous city in australia and oceania . located on australia s east coast , the metropolis surrounds port jackson.'],
                        ['newcastle, new south wales', 'the newcastle ( /ˈnuːkɑːsəl/ new-kah-səl ) metropolitan area is the second most populated area in the australian state of new south wales and includes the newcastle and lake macquarie local government areas .']
                    ),
                    columns=['title', 'content']
                )
            }
        )

        # Write BlendSQL query
        blendsql = """
        SELECT * FROM w
        WHERE city = {{
            LLMQA(
                'Which city is located 120 miles west of Sydney?',
                (SELECT * FROM documents WHERE content LIKE '%sydney%'),
                options='w::city'
            )
        }}
        """
        smoothie = blend(
            query=blendsql,
            db=db,
            ingredients={LLMMap, LLMQA, LLMJoin},
            default_model=model,
            # Optional args below
            infer_gen_constraints=True,
            verbose=True
        )
        print(smoothie.df)
        # ┌────────┬───────────────────┬──────────┬─────────────────┬─────────┐
        # │ date   │ rival             │ city     │ venue           │ score   │
        # ├────────┼───────────────────┼──────────┼─────────────────┼─────────┤
        # │ 11 jun │ western districts │ bathurst │ bathurst ground │ 11-0    │
        # └────────┴───────────────────┴──────────┴─────────────────┴─────────┘
        print(smoothie.meta.prompts)
        # [
        #   {
        #       'answer': 'sydney',
        #       'question': 'Which city is located 120 miles west of Sydney?',
        #       'context': [
        #           {'title': 'bathurst, new south wales', 'content': 'bathurst /ˈbæθərst/ is a city in the central tablelands of new south wales , australia . it is about...'},
        #           {'title': 'sydney', 'content': 'sydney ( /ˈsɪdni/ ( listen ) sid-nee ) is the state capital of new south wales and the most populous city in...'}
        #       ]
        #    }
        # ]
        ```
    '''
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    start = time.time()
    try:
        smoothie = _blend(
            query=query,
            db=db,
            default_model=default_model,
            ingredients=ingredients,
            infer_gen_constraints=infer_gen_constraints,
            table_to_title=table_to_title,
            schema_qualify=schema_qualify,
        )
    except Exception as error:
        # if not isinstance(error, (InvalidBlendSQL, IngredientException)):
        #     from .grammars.minEarley.parser import EarleyParser
        #     from .grammars.utils import load_cfg_parser
        #
        #     # Parse with CFG and try to get helpful recommendations
        #     parser: EarleyParser = load_cfg_parser(ingredients)
        #     try:
        #         parser.parse(query)
        #     except Exception as parser_error:
        #         raise parser_error
        raise error
    finally:
        # In the case of a recursive `_blend()` call,
        #   this logic allows temp tables to persist until
        #   the final base case is fulfilled.
        db._reset_connection()
    smoothie.meta.process_time_seconds = time.time() - start
    return smoothie
