from blendsql.grammars.minEarley.parser import EarleyParser
from blendsql._program import Program
from blendsql._constants import PARSER_STOP_TOKENS
from colorama import Fore
from guidance import gen, select
from typing import List, Tuple, Set
from blendsql.models import Model
from blendsql.db import SQLite
from textwrap import dedent
from dotenv import load_dotenv

MAX_NUM_CORRECTION = 6
PARSER_SYSTEM_PROMPT = dedent(
    """
Generate BlendSQL given the question, table, and passages to answer the question correctly.
BlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.
These external functions should be wrapped in double curly brackets.

If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column using the scalar function:
    `{{LLMMap('question', 'table::column')}}`

If mapping to a new column still cannot answer the question with valid SQL, turn to an end-to-end solution using the aggregate function:
    `{{LLMQA('question', (blendsql))}}`
    Optionally, this function can take an `options` argument to restrict its output to an existing SQL column.
    For example: `... WHERE column = {{LLMQA('question', (blendsql), options='table::column)}}`

If we need to do a `join` operation where there is imperfect alignment between table values, use the new function:
    `{{LLMJoin(left_on='table::column', right_on='table::column')}}`

ONLY use these BlendSQL ingredients if necessary.
Answer parts of the question in vanilla SQL, if possible.
Don't forget to use the `options` argument when necessary! 

Examples:

CREATE TABLE "w" (
  "index" INTEGER,
  "no" INTEGER,
  "rider" TEXT,
  "team" TEXT,
  "motorcycle" TEXT
)
/*
3 example rows:
SELECT * FROM w LIMIT 3
 index  no          rider                 team      motorcycle
     0   1   carl fogarty   ducati performance      ducati 996
     1   4 akira yanagawa kawasaki racing team kawasaki zx-7rr
     2   5  colin edwards        castrol honda      honda rc45
*/

CREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')

Q: After what season did the number 7 competitor retire ?
BlendSQL:
{{
    LLMQA(
        'When did the competitor retire?',
        (
            SELECT documents.title AS 'Competitor', documents.content FROM documents
            JOIN {{
                LLMJoin(
                    left_on='w::rider',
                    right_on='documents::title'
                )
            }}
            WHERE w.no = 7
        )
    )
}}

---

CREATE TABLE "w" (
  "index" INTEGER,
  "year" TEXT,
  "winner" TEXT,
  "position" TEXT,
  "school" TEXT
)
/*
3 example rows:
SELECT * FROM w LIMIT 3
 index    year         winner   position     school
     0 1961-62       ron ryan right wing      colby
     1 1962-63 bob brinkworth     center rensselaer
     2 1963-64 bob brinkworth     center rensselaer
*/

CREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')

Here are some values that may be useful: w.year ( 1971-72 )
Q: What year was the 1971-72 ECAC Hockey Player of the Year born ?
BlendSQL:
{{
    LLMQA(
        'What year was the player born?',
        (
            SELECT documents.title AS 'Player', documents.content FROM documents
            JOIN {{
                LLMJoin(
                    left_on = 'w::winner',
                    right_on = 'documents::title'
                )
            }}
            WHERE w.year = '1971-72'
        )
    )
}}

---
"""
)
load_dotenv(".env")


class ParserProgram(Program):
    def __call__(self, serialized_db: str, question: str, **kwargs):
        _model = self.model
        with self.systemcontext:
            _model += PARSER_SYSTEM_PROMPT + "\n"
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
        serialized_db: str,
        question: str,
        partial_completion: str,
        candidates: List[str],
        **kwargs,
    ):
        _model = self.model
        with self.systemcontext:
            _model += (
                PARSER_SYSTEM_PROMPT
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
        print(Fore.YELLOW + prediction + Fore.RESET)
        print(f"Error: {str(runtime_e)}")
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


def predict_program_with_earley_correction(
    question: str,
    model: Model,
    db: SQLite,
    parser_program: Program,
    correction_program: Program,
    parser: EarleyParser,
):
    num_correction_left = MAX_NUM_CORRECTION

    partial_program_prediction = ""
    ret_prediction, initial_prediction = None, None
    while num_correction_left > 0 and ret_prediction is None:
        residual_program_prediction = model.predict(
            program=parser_program,
            question=question,
            serialized_db=db.to_serialized(num_rows=3, use_tables=["w", "documents"]),
        )["result"]

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
        candidates = [i for i in candidates if i.strip() != ""]
        if len(candidates) == 0:
            print(Fore.LIGHTMAGENTA_EX + "No correction pairs found" + Fore.RESET)
            return prefix
        # Generate the continuation candidate with the highest probability
        selected_candidate = model.predict(
            program=correction_program,
            question=question,
            serialized_db=db.to_serialized(use_tables={"w", "documents"}),
            partial_completion=prefix,
            candidates=candidates,
        )["result"]

        # Try to use our selected candidate in a few ways
        # 1) Insert our selection into the index where the error occured, and add left/right context
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
        print(
            Fore.RED
            + f"cannot find a valid prediction after {MAX_NUM_CORRECTION} retries"
            + Fore.RESET
        )
        ret_prediction = initial_prediction

    return ret_prediction


if __name__ == "__main__":
    """
    https://github.com/zbrookle/sql_to_ibis/blob/main/sql_to_ibis/grammar/sql.lark
    https://lark-parser.readthedocs.io/en/latest/grammar.html
    https://arxiv.org/pdf/2305.19234.pdf
    https://pypi.org/project/lark-dynamic/

    The question-mark prefixing value (”?value”) tells the tree-builder to inline this branch if it has only one member. In this case, value will always have only one member, and will always be inlined.
    """
    # from select_parser import select_stmt
    parser = EarleyParser.open(
        "./blendsql/grammars/_cfg_grammar.lark", start="start", keep_all_tokens=True
    )

    q = """
    {{
     LLMQA(
         'Who won the race?',
         (
             SELECT documents.title AS 'Driver', documents.content FROM documents
             JOIN {{
                 LLMJoin(
                     left_on = 'driver::title',
                     right_on = 'driver::content'
                 )
             }}
             WHERE driver = 'Rubens Barrichello'
         )
     )
     Optionally, this function can take an `options` argument to restrict its output to an existing SQL column.
     For example: `... WHERE driver =
    """
    prefix, candidates, pos_in_stream = obtain_correction_pairs(q, parser)
    print(candidates)

    from blendsql.models import TransformersLLM

    # model = LlamaCppLLM(
    #     model_name_or_path="./lark-constrained-parsing/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    #     caching=False
    # )
    model = TransformersLLM("Qwen/Qwen1.5-0.5B", caching=False)
    db = SQLite("./research/db/hybridqa/2004_United_States_Grand_Prix_0.db")
    print(
        Fore.GREEN
        + predict_program_with_earley_correction(
            question="Of the top 3 drivers, who is the youngest?",
            model=model,
            db=db,
            parser_program=ParserProgram,
            correction_program=CorrectionProgram,
            parser=parser,
        )
        + Fore.RESET
    )
