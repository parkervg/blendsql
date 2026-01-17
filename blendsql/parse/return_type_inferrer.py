"""
Infers regex patterns and type constraints to guide downstream
Model generations based on the syntax of BlendSQL queries.
"""
from abc import ABC, abstractmethod
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Any, Optional
import sqlglot.expressions as exp

from blendsql.types import DataTypes, DB_TYPE_TO_STR, STR_TO_DATATYPE, prepare_datatype
from blendsql.types.utils import try_infer_datatype_from_collection, DataType
from blendsql.common.logger import logger, Color
from blendsql.common.typing import QuantifierType, ColumnRef


@dataclass
class InferenceContext:
    """Holds the parsed context for type inference."""

    function_node: exp.Expression
    parent_node: exp.Expression
    schema: dict
    alias_to_tablename: dict
    has_user_regex: bool
    _predicate_literals: list = field(default=None, init=False, repr=False)

    @property
    def column_in_predicate(self) -> Optional[exp.Column]:
        """Find the column reference in the parent predicate."""
        if self.parent_node is None:
            return None
        return self.parent_node.find(exp.Column)

    @property
    def predicate_literals(self) -> list:
        """Extract literals from predicate expressions."""
        if self._predicate_literals is not None:
            return self._predicate_literals

        self._predicate_literals = self._extract_literals()
        return self._predicate_literals

    def _extract_literals(self) -> list:
        """Extract all literal values from the parent expression."""
        if self.parent_node is None:
            return []

        expressions = self._get_expressions()
        literals = []

        for expression in expressions:
            if expression is None:
                continue

            # Extract Literals
            for lit in expression.find_all(exp.Literal):
                if lit.is_string:
                    literals.append(lit.this)
                else:
                    literals.append(literal_eval(lit.this))

            # Extract Booleans
            for bool_node in expression.find_all(exp.Boolean):
                literals.append(bool_node.args["this"])

        return literals

    def _get_expressions(self) -> list:
        """Get the list of expressions from the parent node."""
        if self.parent_node is None:
            return []

        if self.parent_node.expressions:
            return self.parent_node.expressions
        if self.parent_node.expression is not None:
            return [self.parent_node.expression]
        return []

    def has_expressions(self) -> bool:
        """Check if parent node has any expressions."""
        if self.parent_node is None:
            return False
        return (
            self.parent_node.expression is not None
            or self.parent_node.expressions is not None
        )


class TypeInferenceStrategy(ABC):
    """Base class for type inference strategies."""

    @abstractmethod
    def infer(self, ctx: InferenceContext) -> Optional[DataType]:
        """
        Attempt to infer a return type from the context.

        Returns:
            The inferred DataType, or None to defer to the next strategy.
        """


class ColumnTypeStrategy(TypeInferenceStrategy):
    """Infer type from the column's native database type."""

    def infer(self, ctx: InferenceContext) -> Optional[DataType]:
        column = ctx.column_in_predicate
        if column is None:
            return None

        tablename = ctx.alias_to_tablename.get(column.table, column.table)
        native_type = ctx.schema.get(tablename, {}).get(column.this.name)

        if native_type is None:
            return None

        if native_type not in DB_TYPE_TO_STR:
            logger.debug(
                Color.warning(f"No type logic for native DB type {native_type}!")
            )
            return None

        resolved = DB_TYPE_TO_STR[native_type]
        if resolved == "str":
            return None  # Defer to other strategies for string types

        if not ctx.has_user_regex:
            logger.debug(
                Color.quiet_update(
                    f"The column in this predicate (`{column.this.name}`) has type "
                    f"`{native_type}`, so using regex for {resolved}..."
                )
            )

        return STR_TO_DATATYPE[resolved]


class LiteralTypeStrategy(TypeInferenceStrategy):
    """Infer type from predicate literals."""

    def infer(self, ctx: InferenceContext) -> Optional[DataType]:
        literals = ctx.predicate_literals
        if not literals:
            return None

        logger.debug(Color.quiet_update(f"Extracted predicate literals `{literals}`"))
        return try_infer_datatype_from_collection(literals)


class NumericContextStrategy(TypeInferenceStrategy):
    """Infer numeric type from ordering/aggregation context."""

    NUMERIC_CONTEXTS = (
        exp.Order,
        exp.Ordered,
        exp.AggFunc,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.Sum,
    )

    def infer(self, ctx: InferenceContext) -> Optional[DataType]:
        if ctx.predicate_literals:
            # Don't apply if we have literals - let LiteralTypeStrategy handle it
            return None

        if isinstance(ctx.parent_node, self.NUMERIC_CONTEXTS):
            return DataTypes.NUMERIC()
        return None


class QuantifierDetector:
    """Determines if the return type should be a list."""

    DEFAULT_QUANTIFIER = "+"
    LIST_CONTEXTS = (exp.In, exp.Tuple, exp.Values, exp.Unnest)

    def detect(
        self, parent_node: exp.Expression, function_node: exp.Expression
    ) -> QuantifierType:
        """
        Detect if a list quantifier should be applied.

        Args:
            parent_node: The parent expression node
            function_node: The BlendSQL function node

        Returns:
            The quantifier string (e.g., "+") or None
        """
        if not isinstance(parent_node, self.LIST_CONTEXTS):
            return None

        if isinstance(parent_node, exp.In):
            return self._handle_in_clause(parent_node)

        return self.DEFAULT_QUANTIFIER

    def _handle_in_clause(self, node: exp.In) -> QuantifierType:
        """
        Handle IN clause quantifier detection.

        Only apply quantifier when the ingredient is in the field position,
        not when it's the target of the IN.

        E.g., `column IN {{LLMQA()}}` should get a quantifier,
        but `{{LLMMap()}} IN ('a', 'b')` should not.
        """
        field = node.args.get("field")
        if field is None:
            return None

        # Check if the field itself is the node or is a BlendSQL function
        if node == field or isinstance(field, exp.BlendSQLFunction):
            return self.DEFAULT_QUANTIFIER

        return None


class OptionsDetector:
    """Detects if we can infer column options for the ingredient."""

    OPTION_CONTEXTS = (exp.EQ, exp.In)

    def detect(self, parent_node: exp.Expression) -> Optional[ColumnRef]:
        """
        Detect if column options can be inferred from the context.

        Args:
            parent_node: The parent expression node

        Returns:
            A ColumnRef for the options, or None
        """
        if not isinstance(parent_node, self.OPTION_CONTEXTS):
            return None

        column = parent_node.args.get("this")
        if not isinstance(column, exp.Column):
            return None

        if "table" not in column.args:
            self._warn_missing_table(parent_node)
            return None

        table_name = column.args["table"].name
        column_name = column.args["this"].name
        return ColumnRef(f"{table_name}.{column_name}")

    def _warn_missing_table(self, parent_node: exp.Expression) -> None:
        """Log a warning when table is not specified."""
        if not isinstance(parent_node, exp.BlendSQLFunction):
            logger.debug(
                "When inferring `options` in infer_gen_kwargs, encountered column "
                "node with no table specified!\n"
                "Should probably mark `schema_qualify` arg as True"
            )


class ReturnTypeInferrer:
    """
    Infers generation constraints from BlendSQL query structure.

    This class analyzes the syntax of a BlendSQL query to infer regex patterns
    and type constraints that guide downstream Model generations.

    Example:
        ```sql
        SELECT * FROM w WHERE {{LLMMap('Is this true?', w.column)}} = TRUE
        ```

        Given this structure, we can infer that `LLMMap` should return a boolean.

    Attributes:
        type_strategies: Ordered list of strategies for inferring return types
        quantifier_detector: Detector for list quantifiers
        options_detector: Detector for column options
    """

    DEFAULT_QUANTIFIER = "+"

    def __init__(self):
        self.type_strategies: list[TypeInferenceStrategy] = [
            ColumnTypeStrategy(),
            LiteralTypeStrategy(),
            NumericContextStrategy(),
        ]
        self.quantifier_detector = QuantifierDetector()
        self.options_detector = OptionsDetector()

    def __call__(
        self,
        function_node: exp.Expression,
        schema: dict,
        alias_to_tablename: dict,
        has_user_regex: bool,
    ) -> dict:
        """
        Infer generation constraints from a BlendSQL query's syntax.

        Args:
            function_node: The expression node containing the BlendSQL function
            schema: Database schema mapping table names to column types
            alias_to_tablename: Mapping of table aliases to actual table names
            has_user_regex: Whether the user has provided a custom regex

        Returns:
            A dict with keys:
                - return_type: 'boolean' | 'integer' | 'float' | 'string' | etc.
                - regex: Regular expression pattern lambda for constrained decoding
                - options: Optional ColumnRef for QAIngredient options ('{table}.{column}')
                - example_outputs: Optional list of example output values
                - wrap_tuple_in_parentheses: Optional bool for tuple handling
        """
        parent_node = self._get_parent_node(function_node)

        # Build inference context
        ctx = InferenceContext(
            function_node=function_node,
            parent_node=parent_node,
            schema=schema,
            alias_to_tablename=alias_to_tablename,
            has_user_regex=has_user_regex,
        )

        # Build result dictionary
        result: dict[str, Any] = {}

        # Step 1: Detect options from column context
        options = self.options_detector.detect(parent_node)
        if options is not None:
            result["options"] = options

        # Step 2: Detect quantifier (list type)
        quantifier = self.quantifier_detector.detect(parent_node, function_node)

        # Step 3: Handle special tuple parentheses flag
        if isinstance(parent_node, (exp.Tuple, exp.Values)):
            result["wrap_tuple_in_parentheses"] = False

        # Step 4: Infer return type using strategy chain
        return_type = self._infer_type(ctx)

        # Step 5: Apply results based on what we found
        if return_type is not None:
            return_type.quantifier = quantifier
            result["return_type"] = return_type
            self._log_inferred_type(return_type, result.get("options"), quantifier)
        elif ctx.predicate_literals:
            # Special case: literals exist but type couldn't be inferred
            result.update(
                self._handle_untyped_literals(ctx.predicate_literals, quantifier)
            )
        elif quantifier:
            # Fallback to string list when we only know it's a list
            result["return_type"] = DataTypes.STR(quantifier)

        return result

    def _infer_type(self, ctx: InferenceContext) -> Optional[DataType]:
        """
        Run through inference strategies in priority order.

        Args:
            ctx: The inference context

        Returns:
            The first successfully inferred DataType, or None
        """
        for strategy in self.type_strategies:
            result = strategy.infer(ctx)
            if result is not None:
                return result
        return None

    def _get_parent_node(self, function_node: exp.Expression) -> exp.Expression:
        """
        Get the appropriate parent node for inference.

        For SELECT clauses, we don't traverse up to avoid incorrect inference.
        """
        if isinstance(function_node.parent, exp.Select):
            return function_node
        return function_node.parent

    def _handle_untyped_literals(
        self, literals: list, quantifier: QuantifierType
    ) -> dict:
        """
        Handle the case where literals exist but type couldn't be inferred.

        Creates example outputs from the literals for guidance.
        """
        str_literals = [str(i) for i in literals]

        # Ensure we have at least 2 examples
        if len(str_literals) == 1:
            str_literals = str_literals * 2

        return {
            "return_type": DataTypes.ANY(quantifier),
            "example_outputs": str_literals,
        }

    def _log_inferred_type(
        self,
        return_type: DataType,
        options: Optional[ColumnRef],
        quantifier: QuantifierType,
    ) -> None:
        """Log the inferred type for debugging."""
        logger.debug(
            lambda: Color.quiet_update(
                f"Inferred return_type='"
                f"{prepare_datatype(return_type=return_type, options=options, quantifier=quantifier, log=False).name}' "
                "given expression context"
            )
        )
