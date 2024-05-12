from typing import Collection, List, Tuple, Set, Optional
from textwrap import dedent
from guidance import gen, select
from colorama import Fore
from pathlib import Path
import logging

from .ingredients import Ingredient, IngredientException
from .models import Model
from ._program import Program
from .grammars.minEarley.parser import EarleyParser

CFG_PARSER = EarleyParser.open(
    Path(__file__).parent / "./grammars/_cfg_grammar.lark",
    start="start",
    keep_all_tokens=True,
)
PARSER_STOP_TOKENS = ["---", ";", "\n\n"]
PARSER_SYSTEM_PROMPT = dedent(
    """
Generate BlendSQL given the question, table, and passages to answer the question correctly.
BlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.
These external functions should be wrapped in double curly brackets.
{ingredient_descriptions}
ONLY use these BlendSQL ingredients if necessary.
Answer parts of the question in vanilla SQL, if possible.
Don't forget to use the `options` argument when necessary!
"""
)


class ParserProgram(Program):
    def __call__(self, system_prompt: str, serialized_db: str, question: str, **kwargs):
        _model = self.model
        with self.systemcontext:
            _model += system_prompt + "\n"
        with self.usercontext:
            _model += f"{serialized_db}\n\n"
            _model += f"Question: {question}\n"
            _model += f"BlendSQL: "
        with self.assistantcontext:
            _model = _model + gen(
                name="result",
                max_tokens=128,
                stop=PARSER_STOP_TOKENS,
                **self.gen_kwargs,
            )
        return _model


class CorrectionProgram(Program):
    def __call__(
        self,
        system_prompt: str,
        serialized_db: str,
        question: str,
        partial_completion: str,
        candidates: List[str],
        **kwargs,
    ):
        _model = self.model
        with self.systemcontext:
            _model += (
                system_prompt
                + "\n For this setting, you will ONLY generate the completion to the partially-generate query below.\n"
            )
        with self.usercontext:
            _model += f"{serialized_db}\n\n"
            _model += f"Question: {question}\n"
            _model += f"BlendSQL: "
            _model += partial_completion
        with self.assistantcontext:
            _model += select(candidates, name="result")
        return _model


def validate_program(prediction: str, parser: EarleyParser) -> bool:
    try:
        parser.parse(prediction)
        return True
    except Exception as runtime_e:
        logging.debug(Fore.CYAN + prediction + Fore.RESET)
        logging.debug(f"Error: {str(runtime_e)}")
        return False


def obtain_correction_pairs(
    prediction: str, parser: EarleyParser
) -> Tuple[str, Set[str], int]:
    """
    Returns a list of candidates in the form of (prefix, suffix).
    """
    try:
        parser.parse(prediction)
        return []
    except Exception as runtime_e:
        return parser.handle_error(runtime_e)


def create_system_prompt(
    ingredients: Collection[Ingredient], few_shot_examples: Optional[str] = ""
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
        + few_shot_examples
    )


def nl_to_blendsql(
    question: str,
    serialized_db: str,
    model: Model,
    ingredients: Collection[Ingredient],
    few_shot_examples: str = "",
    max_grammar_corrections: int = 0,
    verbose: bool = False,
) -> str:
    """Takes a natural language question, and attempts to parse BlendSQL representation for answering against a databse.

    Args:
        question: The natural language question to parse
        serialized_db: Database in a serialized string format.
            This can be achieved by using db.to_serialized()
        model: BlendSQL model to use in translating the question
        ingredients: Which ingredients to treat as valid in the output parse.
            Only these ingredient descriptions are included in the system prompt.
        few_shot_examples: String prompt introducing few shot nl-to-blendsql examples.
        max_grammar_corrections: Optional int defining maximum CFG-guided correction steps to be taken.
            This is based on the method in https://arxiv.org/pdf/2305.19234.
        verbose: Boolean defining whether to run in logging.debug mode

    Returns:
        ret_prediction: Final BlendSQL query prediction
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.ERROR)
    system_prompt: str = create_system_prompt(
        ingredients=ingredients, few_shot_examples=few_shot_examples
    )
    logging.debug(Fore.YELLOW + f"Using system prompt: '{system_prompt}'" + Fore.RESET)
    if max_grammar_corrections == 0:
        return model.predict(
            program=ParserProgram,
            system_prompt=system_prompt,
            question=question,
            serialized_db=serialized_db,
        )["result"]
    num_correction_left = max_grammar_corrections
    partial_program_prediction = ""
    ret_prediction, initial_prediction = None, None
    while num_correction_left > 0 and ret_prediction is None:
        residual_program_prediction = model.predict(
            program=ParserProgram,
            system_prompt=system_prompt,
            question=question,
            serialized_db=serialized_db,
        )["result"]

        # if the prediction is empty, return the initial prediction
        if initial_prediction is None:
            initial_prediction = residual_program_prediction
        program_prediction = (
            partial_program_prediction + " " + residual_program_prediction
        )

        if validate_program(program_prediction, CFG_PARSER):
            ret_prediction = program_prediction
            continue

        # find the max score from a list of score
        prefix, candidates, pos_in_stream = obtain_correction_pairs(
            program_prediction, CFG_PARSER
        )
        candidates = [i for i in candidates if i.strip() != ""]
        if len(candidates) == 0:
            logging.debug(
                Fore.LIGHTMAGENTA_EX + "No correction pairs found" + Fore.RESET
            )
            return prefix
        # Generate the continuation candidate with the highest probability
        selected_candidate = model.predict(
            program=CorrectionProgram,
            system_prompt=system_prompt,
            question=question,
            serialized_db=serialized_db,
            partial_completion=prefix,
            candidates=candidates,
        )["result"]

        # Try to use our selected candidate in a few ways
        # 1) Insert our selection into the index where the error occured, and add left/right context
        #   Example: SELECT a b FROM table -> SELECT a, b FROM table
        inserted_candidate = (
            prefix + selected_candidate + program_prediction[pos_in_stream:]
        )
        if validate_program(inserted_candidate, CFG_PARSER):
            ret_prediction = inserted_candidate
            continue
        # 2) If rest of our query is also broken, we just keep up to the prefix + candidate
        partial_program_prediction = prefix + selected_candidate
        for p in {inserted_candidate, partial_program_prediction}:
            if validate_program(p, CFG_PARSER):
                ret_prediction = p

        num_correction_left -= 1

    if ret_prediction is None:
        logging.debug(
            Fore.RED
            + f"cannot find a valid prediction after {max_grammar_corrections} retries"
            + Fore.RESET
        )
        ret_prediction = initial_prediction
    logging.debug(Fore.GREEN + ret_prediction + Fore.RESET)
    return ret_prediction


if __name__ == "__main__":
    from blendsql import nl_to_blendsql, LLMMap
    from blendsql.models import TransformersLLM
    from blendsql.db import SQLite
    from blendsql.utils import fetch_from_hub

    model = TransformersLLM("Qwen/Qwen1.5-0.5B")
    db = SQLite(
        fetch_from_hub("1884_New_Zealand_rugby_union_tour_of_New_South_Wales_1.db")
    )

    query = nl_to_blendsql(
        question="Which venues in Sydney saw more than 30 points scored?",
        model=model,
        ingredients={LLMMap},
        serialized_db=db.to_serialized(num_rows=3, use_tables=["w", "documents"]),
        verbose=True,
        max_grammar_corrections=5,
    )
