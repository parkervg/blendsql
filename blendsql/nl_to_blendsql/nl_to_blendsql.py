from typing import Tuple, Optional, Union, Type
from collections.abc import Collection
from textwrap import dedent
from colorama import Fore
import re

from .._logger import logger
from ..ingredients import Ingredient, IngredientException
from ..models import Model
from ..db import Database, double_quote_escape
from .._program import Program
from ..prompts import FewShot
from .. import generate
from .args import NLtoBlendSQLArgs


PARSER_STOP_TOKENS = ["---", ";", "\n\n", "Q:"]
PARSER_SYSTEM_PROMPT = dedent(
    """
Generate BlendSQL given the question, table, and passages to answer the question correctly.
BlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.
These external functions should be wrapped in double curly brackets.
{ingredient_descriptions}
ONLY use these BlendSQL ingredients if necessary. Answer parts of the question in vanilla SQL, if possible.

Additionally, we have the table `documents` at our disposal, which contains Wikipedia articles providing more details about the values in our table.
The `documents` table for each database has the same schema:

CREATE TABLE documents (
  "title" TEXT,
  "content" TEXT
)
"""
)


class ParserProgram(Program):
    def __call__(
        self,
        model: Model,
        system_prompt: str,
        serialized_db: str,
        question: str,
        **kwargs,
    ) -> Tuple[str, str]:
        prompt = ""
        prompt += system_prompt
        prompt += f"{serialized_db}\n"
        prompt += f"Question: {question}\n"
        prompt += f"BlendSQL: "
        logger.debug(
            Fore.LIGHTYELLOW_EX
            + f"Using parsing prompt:"
            + Fore.YELLOW
            + prompt
            + Fore.RESET
        )
        response = generate.text(
            model,
            prompt=prompt,
            stop_at=PARSER_STOP_TOKENS,
        )
        return (response, prompt)


def create_system_prompt(
    ingredients: Optional[Collection[Type[Ingredient]]],
    db: Database,
    question: str,
    args: NLtoBlendSQLArgs,
    few_shot_examples: Optional[Union[str, FewShot]] = "",
) -> str:
    ingredient_descriptions = []
    for ingredient in ingredients:
        if not hasattr(ingredient, "DESCRIPTION"):
            raise IngredientException(
                "In order to use an ingredient in parse_blendsql, you need to provide an explanation of it in the `DESCRIPTION` attribute"
            )
        ingredient_descriptions.append(ingredient.DESCRIPTION)
    serialized_db = db.to_serialized(
        use_tables=args.use_tables,
        num_rows=args.num_serialized_rows,
        include_content=args.include_db_content_tables,
        use_bridge_encoder=args.use_bridge_encoder,
        question=question,
    )
    return (
        PARSER_SYSTEM_PROMPT.format(
            ingredient_descriptions=dedent("\n".join(ingredient_descriptions))
        )
        + "\n"
        + str(few_shot_examples)
        + "\n\n---\n\n"
        + f"{serialized_db}\n"
        + f"Question: {question}\n"
        + f"BlendSQL:\n"
    )


def post_process_blendsql(
    blendsql: str, db: Database, use_tables: Optional[Collection[str]] = None
) -> str:
    """Applies any relevant post-processing on the generated BlendSQL query.
    Currently, only adds double-quotes around column references.
    This helps to ensure the query will successfully execute on the given DBMS.
    For example:
        `SELECT * FROM w WHERE index = 0;` is invalid in SQLite, because 'index' is a keyword
        So, it becomes `SELECT * FROM w WHERE "index" = 0;` after this function
    """
    quotes_start_end = [i.start() for i in re.finditer(r"(\'|\")", blendsql)]
    quotes_start_end_spans = list(zip(*(iter(quotes_start_end),) * 2))
    for tablename in db.tables():
        if use_tables is not None:
            if tablename not in use_tables:
                continue
        for columnname in sorted(
            list(db.iter_columns(tablename)), key=lambda x: len(x), reverse=True
        ):
            # Reverse finditer so we don't mess up indices when replacing
            # Only sub if surrounded by: whitespace, comma, or parentheses
            # Or, prefaced by period (e.g. 'p.Current_Value')
            # AND it's not already in quotes
            for m in list(
                re.finditer(
                    r"(?<=(\s|,|\(|\.)){}(?=(\s|,|\)|;|$))".format(
                        re.escape(columnname)
                    ),
                    blendsql,
                )
            )[::-1]:
                # Check if m.start already occurs within quotes (' or ")
                # If it does, don't add quotes
                if any(
                    start + 1 < m.start() < end
                    for (start, end) in quotes_start_end_spans
                ):
                    continue
                blendsql = (
                    blendsql[: m.start()]
                    + '"'
                    + double_quote_escape(
                        blendsql[m.start() : m.start() + (m.end() - m.start())]
                    )
                    + '"'
                    + blendsql[m.end() :]
                )
    return blendsql
