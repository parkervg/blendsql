from typing import Tuple, Set, Optional, Union, Type
from collections.abc import Collection
from textwrap import dedent
from colorama import Fore
import re
import logging

from .._logger import logger
from ..ingredients import Ingredient, IngredientException
from ..models import Model, OllamaLLM
from ..db import Database, double_quote_escape
from .._program import Program
from ..grammars.minEarley.parser import EarleyParser
from ..grammars.utils import load_cfg_parser
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


class CorrectionProgram(Program):
    def __call__(
        self,
        model: Model,
        system_prompt: str,
        serialized_db: str,
        question: str,
        partial_completion: str,
        candidates: Set[str],
        **kwargs,
    ) -> Tuple[str, str]:
        if isinstance(model, OllamaLLM):
            raise ValueError("CorrectionProgram can't use OllamaLLM!")
        prompt = ""
        prompt += (
            system_prompt
            + "\n For this setting, you will ONLY generate the completion to the partially-generate query below.\n"
        )
        prompt += f"{serialized_db}\n\n"
        prompt += f"Question: {question}\n"
        prompt += f"BlendSQL:\n"
        prompt += partial_completion
        response = generate.choice(
            model, prompt=prompt, choices=[re.escape(str(i)) for i in candidates]
        )
        return (response, prompt)


def validate_program(prediction: str, parser: EarleyParser) -> bool:
    try:
        parser.parse(prediction)
        return True
    except Exception as runtime_e:
        logger.debug(Fore.LIGHTCYAN_EX + prediction + Fore.RESET)
        logger.debug(f"Error: {str(runtime_e)}")
        return False


def obtain_correction_pairs(
    prediction: str, parser: EarleyParser
) -> Tuple[str, Set[str], int]:
    """
    Returns a list of candidates in the form of (prefix, candidates, error_position_index).
    """
    try:
        parser.parse(prediction)
        raise ValueError(
            "When calling obtain_correction_pairs, the passed prediction should already be assumed to fail the grammar constraints"
        )
    except Exception as runtime_e:
        return parser.handle_error(runtime_e)


def create_system_prompt(
    ingredients: Optional[Collection[Type[Ingredient]]],
    few_shot_examples: Optional[Union[str, FewShot]] = "",
) -> str:
    ingredient_descriptions = []
    for ingredient in ingredients:
        if not hasattr(ingredient, "DESCRIPTION"):
            raise IngredientException(
                "In order to use an ingredient in parse_blendsql, you need to provide an explanation of it in the `DESCRIPTION` attribute"
            )
        ingredient_descriptions.append(ingredient.DESCRIPTION)
    return (
        PARSER_SYSTEM_PROMPT.format(
            ingredient_descriptions="\n".join(ingredient_descriptions)
        )
        + "\n"
        + str(few_shot_examples)
        + "\n\n---\n\n"
    )


def nl_to_blendsql(
    question: str,
    db: Database,
    model: Model,
    ingredients: Optional[Collection[Type[Ingredient]]],
    correction_model: Optional[Model] = None,
    few_shot_examples: Union[str, FewShot] = "",
    args: Optional[NLtoBlendSQLArgs] = None,
    verbose: bool = False,
) -> str:
    """Takes a natural language question, and attempts to parse BlendSQL representation for answering against a databse.

    Args:
        question: The natural language question to parse
        db: Database to use in translating
        model: BlendSQL model to use in translating the question
        ingredients: Which ingredients to treat as valid in the output parse.
            Only these ingredient descriptions are included in the system prompt.
        few_shot_examples: String prompt introducing few shot nl-to-blendsql examples.
        args: Optional NLtoBlendSQLArgs object, containing additional parameters.
        verbose: Boolean defining whether to run in logger mode

    Returns:
        ret_prediction: Final BlendSQL query prediction

    Examples:
        ```python
        from blendsql import LLMMap, LLMQA
        from blendsql.models import TransformersLLM, OllamaLLM
        from blendsql.nl_to_blendsql import nl_to_blendsql, NLtoBlendSQLArgs
        from blendsql.db import SQLite
        from blendsql.utils import fetch_from_hub
        from blendsql.prompts import FewShot

        db = SQLite(
            fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
        )
        parser_model = OllamaLLM("phi3", caching=False)
        correction_model = TransformersLLM("Qwen/Qwen1.5-0.5B")

        ingredients = {LLMMap, LLMQA}
        filtered_few_shot = FewShot.hybridqa.filter(ingredients)

        blendsql = nl_to_blendsql(
            "What was the result of the game played 120 miles west of Sydney?",
            db=db,
            model=parser_model,
            correction_model=correction_model,
            ingredients=ingredients,
            few_shot_examples=filtered_few_shot,
            verbose=True,
            args=NLtoBlendSQLArgs(
                max_grammar_corrections=5,
                use_tables=["w"],
                include_db_content_tables=["w"],
                num_serialized_rows=3,
                use_bridge_encoder=True,
            ),
        )
        ```
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    if args is None:
        args = NLtoBlendSQLArgs()
    if correction_model is None:
        correction_model = model
    parser: EarleyParser = load_cfg_parser(ingredients)
    system_prompt: str = create_system_prompt(
        ingredients=ingredients, few_shot_examples=few_shot_examples
    )
    serialized_db = db.to_serialized(
        use_tables=args.use_tables,
        num_rows=args.num_serialized_rows,
        include_content=args.include_db_content_tables,
        use_bridge_encoder=args.use_bridge_encoder,
        question=question,
    )
    if args.max_grammar_corrections == 0:
        return model.predict(
            program=ParserProgram,
            system_prompt=system_prompt,
            question=question,
            serialized_db=serialized_db,
            stream=verbose,
        )
    num_correction_left = args.max_grammar_corrections
    partial_program_prediction = ""
    ret_prediction, initial_prediction = None, None
    while num_correction_left > 0 and ret_prediction is None:
        residual_program_prediction = model.predict(
            program=ParserProgram,
            system_prompt=system_prompt,
            question=question,
            serialized_db=serialized_db,
            stream=verbose,
        )

        # if the prediction is empty, return the initial prediction
        if initial_prediction is None:
            initial_prediction = residual_program_prediction
        program_prediction = (
            partial_program_prediction + " " + residual_program_prediction
        )

        if validate_program(program_prediction, parser):
            ret_prediction = program_prediction
            continue

        # find the max score from a list of score
        prefix, candidates, pos_in_stream = obtain_correction_pairs(
            program_prediction, parser
        )
        # candidates = [i for i in candidates if i.strip() != ""]
        if len(candidates) == 0:
            logger.debug(
                Fore.LIGHTMAGENTA_EX + "No correction pairs found" + Fore.RESET
            )
            return prefix
        elif len(candidates) == 1:
            # If we only have 1 candidate, no need to call LLM
            selected_candidate = candidates.pop()
        else:
            # Generate the continuation candidate with the highest probability
            selected_candidate = correction_model.predict(
                program=CorrectionProgram,
                system_prompt=system_prompt,
                question=question,
                serialized_db=serialized_db,
                partial_completion=prefix,
                candidates=candidates,
            )

        # Try to use our selected candidate in a few ways
        # 1) Insert our selection into the index where the error occurred, and add left/right context
        #   Example: SELECT a b FROM table -> SELECT a, b FROM table
        inserted_candidate = (
            prefix + selected_candidate + program_prediction[pos_in_stream:]
        )
        if validate_program(inserted_candidate, parser):
            ret_prediction = inserted_candidate
            continue
        # 2) If rest of our query is also broken, we just keep up to the prefix + candidate
        partial_program_prediction = prefix + selected_candidate
        for p in {inserted_candidate, partial_program_prediction}:
            if validate_program(p, parser):
                ret_prediction = p

        num_correction_left -= 1

    if ret_prediction is None:
        logger.debug(
            Fore.RED
            + f"cannot find a valid prediction after {args.max_grammar_corrections} retries"
            + Fore.RESET
        )
        ret_prediction = initial_prediction
    ret_prediction = post_process_blendsql(
        ret_prediction, db, use_tables=args.use_tables
    )
    logger.debug(Fore.GREEN + ret_prediction + Fore.RESET)
    return ret_prediction


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
