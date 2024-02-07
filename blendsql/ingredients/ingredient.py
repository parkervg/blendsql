from attr import attrs, attrib
from abc import abstractmethod, ABC
import pandas as pd
from typing import Any, Iterable, Union, Dict, Tuple, Type, Callable
import uuid
from typeguard import check_type

from .. import utils
from .._constants import MAIN_INGREDIENT_KWARG, IngredientType
from ..db.sqlite_db_connector import SQLiteDBConnector


class IngredientException(ValueError):
    pass


def map_base_unpack_default_kwargs(**kwargs):
    """Unpack default kwargs for all ingredients where
    we get some list of column values as default kwargs.
    Includes MapIngredient.
    """
    return (
        kwargs.get("values"),
        kwargs.get("original_table"),
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


def string_base_unpack_default_kwargs(**kwargs):
    """Here we don't need values or table,
    so we just get basic tablename, colname.
    """
    return (
        kwargs.get("tablename"),
        kwargs.get("colname"),
    )


def align_to_real_columns(db: SQLiteDBConnector, colname: str, tablename: str) -> str:
    table_columns = db.execute_query(f'SELECT * FROM "{tablename}" LIMIT 1').columns
    if colname not in table_columns:
        # Try to align with column, according to some normalization rules
        cleaned_to_original = {
            col.replace("\\n", " ").replace("\xa0", " "): col for col in table_columns
        }
        colname = cleaned_to_original[colname]
    return colname


@attrs
class Ingredient(ABC):
    name: str = attrib()
    ingredient_type: str = attrib(init=False)
    allowed_output_types: Tuple[Type] = attrib(init=False)
    # Below gets passed via `Kitchen.extend()`
    db: SQLiteDBConnector = attrib(init=False)
    session_uuid: str = attrib(init=False)

    def __repr__(self):
        return f"{self.ingredient_type} {self.name}"

    def __str__(self):
        return f"{self.ingredient_type} {self.name}"

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        ...

    def _run(self, *args, **kwargs):
        return check_type(self.run(*args, **kwargs), self.allowed_output_types)

    def maybe_get_temp_table(
        self, temp_table_func: Callable, tablename: str
    ) -> Tuple[str, bool]:
        temp_tablename = temp_table_func(tablename)
        _tablename = tablename
        if self.db.has_table(temp_tablename):
            # We've already applied some operation to this table
            # We want to use this as our base
            _tablename = temp_tablename
        return (_tablename, True) if _tablename != tablename else (_tablename, False)


@attrs
class MapIngredient(Ingredient):
    """For a given table/column pair, maps an external function
    to each of the given values, creating a new column."""

    ingredient_type: str = IngredientType.MAP.value
    num_values_passed: int = 0
    allowed_output_types: Tuple[Type] = (Iterable[Any],)

    def unpack_default_kwargs(self, **kwargs):
        return map_base_unpack_default_kwargs(**kwargs)

    def __call__(
        self, question: str = None, context: str = None, *args, **kwargs
    ) -> tuple:
        """Returns tuple with format (arg, tablename, colname, new_table)"""
        tablename, colname = utils.get_tablename_colname(context)
        tablename = kwargs.get("aliases_to_tablenames").get(tablename, tablename)
        kwargs["tablename"] = tablename
        kwargs["colname"] = colname
        # Check for previously created temporary tables
        value_source_tablename, _ = self.maybe_get_temp_table(
            temp_table_func=kwargs.get("get_temp_subquery_table"), tablename=tablename
        )
        temp_session_tablename, temp_session_table_exists = self.maybe_get_temp_table(
            temp_table_func=kwargs.get("get_temp_session_table"), tablename=tablename
        )

        # Need to be sure the new column doesn't already exist here
        new_arg_column = (
            f"___{question}"
            if question in set(self.db.iter_columns(tablename))
            else question
        )

        original_table = self.db.execute_query(
            f"SELECT * FROM '{tablename}'", silence_errors=False
        )

        # Get a list of values to map
        # First, check if we've already dumped some `MapIngredient` output to the main session table
        if temp_session_table_exists:
            temp_session_table = self.db.execute_query(
                f"SELECT * FROM '{temp_session_tablename}'"
            )
            colname = align_to_real_columns(
                db=self.db, colname=colname, tablename=temp_session_tablename
            )
            # We don't need to run this function on everything,
            #   if a previous subquery already got to certain values
            if new_arg_column in temp_session_table.columns:
                values = self.db.execute_query(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}" WHERE "{new_arg_column}" IS NULL',
                    silence_errors=False,
                )[colname].tolist()
            # Base case: this is the first time we've used this particular ingredient
            # BUT, temp_session_tablename still exists
            else:
                values = self.db.execute_query(
                    f'SELECT DISTINCT "{colname}" FROM "{temp_session_tablename}"',
                    silence_errors=False,
                )[colname].tolist()
        else:
            colname = align_to_real_columns(
                db=self.db, colname=colname, tablename=value_source_tablename
            )
            values = self.db.execute_query(
                f'SELECT DISTINCT "{colname}" FROM "{value_source_tablename}"',
                silence_errors=False,
            )[colname].tolist()

        # No need to run ingredient if we have no values to map onto
        if len(values) == 0:
            original_table[new_arg_column] = None
            return (new_arg_column, tablename, colname, original_table)

        kwargs["values"] = values
        kwargs["original_table"] = original_table
        kwargs[MAIN_INGREDIENT_KWARG] = question
        mapped_values: Iterable[Any] = self._run(*args, **kwargs)
        self.num_values_passed += len(mapped_values)
        df_as_dict = {colname: [], new_arg_column: []}
        for value, mapped_value in zip(values, mapped_values):
            df_as_dict[colname].append(value)
            df_as_dict[new_arg_column].append(mapped_value)
        subtable = pd.DataFrame(df_as_dict)
        if all(isinstance(x, (int, type(None))) for x in mapped_values):
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

    ingredient_type: str = IngredientType.JOIN.value
    num_values_passed: int = 0
    allowed_output_types: Tuple[Type] = (dict,)

    def unpack_default_kwargs(self, **kwargs):
        return map_base_unpack_default_kwargs(**kwargs)

    def __call__(
        self,
        question: str = None,
        left_on: str = None,
        right_on: str = None,
        *args,
        **kwargs,
    ) -> tuple:
        values = []
        original_lr_identifiers = []
        modified_lr_identifiers = []
        for on_arg in [left_on, right_on]:
            tablename, colname = utils.get_tablename_colname(on_arg)
            tablename = kwargs.get("aliases_to_tablenames").get(tablename, tablename)
            original_lr_identifiers.append((tablename, colname))
            tablename, _ = self.maybe_get_temp_table(
                temp_table_func=kwargs.get("get_temp_subquery_table"),
                tablename=tablename,
            )
            values.append(
                set(
                    self.db.execute_query(
                        f'SELECT DISTINCT "{colname}" FROM "{tablename}"'
                    )[colname].tolist()
                )
            )
            modified_lr_identifiers.append((tablename, colname))

        if kwargs.get("join_criteria", None) is None:
            # First, check which values we actually need to call LLM on
            # We don't want to join when there's already an intuitive alignment
            mapping = {}
            left_values, right_values = values
            for l in left_values:
                if l in right_values:
                    # Define this mapping, and remove from LLM inference call
                    mapping[l] = l

            processed_values = set(list(mapping.keys()))
            left_values = left_values.difference(processed_values)
            right_values = right_values.difference(processed_values)

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
            # Some alignment still left
            self.num_values_passed += len(kwargs["left_values"])

            kwargs[MAIN_INGREDIENT_KWARG] = question
            _mapping: Dict[str, str] = self._run(*args, **kwargs)
            mapping = mapping | _mapping

        # Using mapped left/right values, create intermediary mapping table
        # This needs a new unique id. But start with our session uuid, so it gets cleaned up
        temp_join_tablename = kwargs.get("get_temp_session_table")(
            str(uuid.uuid4())[:4]
        )
        joined_values_df = pd.DataFrame(
            data={"left": mapping.keys(), "right": mapping.values()}
        )
        joined_values_df.to_sql(
            name=temp_join_tablename, con=self.db.con, if_exists="fail", index=False
        )
        return (
            left_tablename,
            right_tablename,
            f"""JOIN "{temp_join_tablename}" ON "{right_tablename}"."{right_colname}" = "{temp_join_tablename}".right
                           JOIN "{left_tablename}" ON "{left_tablename}"."{left_colname}" = "{temp_join_tablename}".left
                           """,
            temp_join_tablename,
        )


@attrs
class QAIngredient(Ingredient):
    ingredient_type: str = IngredientType.QA.value
    allowed_output_types: Tuple[Type] = (Union[str, int, float],)

    def unpack_default_kwargs(self, **kwargs):
        return kwargs.get("subtable")

    def __call__(
        self,
        question: str = None,
        context: Union[str, pd.DataFrame] = None,
        *args,
        **kwargs,
    ) -> Union[str, int, float]:
        if context is not None:
            subtable = context
            if isinstance(context, str):
                tablename, colname = utils.get_tablename_colname(context)
                subtable = self.db.execute_query(
                    f'SELECT "{colname}" FROM "{tablename}"'
                )
            elif not isinstance(context, pd.DataFrame):
                raise ValueError(
                    f"Unknown type for `identifier` arg in QAIngredient: {type(context)}"
                )
            if subtable.empty:
                raise IngredientException("Empty subtable passed to QAIngredient!")
            kwargs["subtable"] = subtable
        kwargs[MAIN_INGREDIENT_KWARG] = question
        response: Union[str, int, float] = self._run(*args, **kwargs)
        return response


@attrs
class StringIngredient(Ingredient):
    """Outputs a string to be placed directly into the SQL query."""

    ingredient_type: str = IngredientType.STRING.value
    allowed_output_types: Tuple[Type] = (str,)

    def unpack_default_kwargs(self, **kwargs):
        return string_base_unpack_default_kwargs(**kwargs)

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
