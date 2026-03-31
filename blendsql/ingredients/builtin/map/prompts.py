from collections.abc import Collection
from enum import Enum
from textwrap import indent
import json

from blendsql.types import STR_TO_DATATYPE
from blendsql.common.typing import DataType, AdditionalMapArg
from blendsql.common.constants import INDENT

PYTHON_INSTRUCTION = (
    "Complete the docstring for the provided Python function. "
    "On each newline, it will follow the format of `f({value}) == {answer}`. "
    "Your task is to ONLY generate `{answer}` for the provided `f({value}) ==`."
    "An example is shown below."
)

BASIC_INSTRUCTION = "You are a helpful assistant. You will be presented with some context and a question. "
BASE_RETURN_TYPE_TO_INSTRUCTION: dict[str, str] = {
    "bool": BASIC_INSTRUCTION
    + "Output 'True' if the context satisfies the filter condition presented in the question, and 'False' otherwise.",
    "int": BASIC_INSTRUCTION + "Return your answer as a valid Python `int`.",
    "float": BASIC_INSTRUCTION + "Return your answer as a valid Python `float`.",
    "str": BASIC_INSTRUCTION
    + "Answer the question accurately. Do not add any additional commentary - for example, don't say 'The answer is golf', simply say 'golf'.",
    "list[str]": BASIC_INSTRUCTION
    + "Return your answer as a valid Python `list[str]`.",
    "date": BASIC_INSTRUCTION + "Return your answer as a date, formatted `YYYY-MM-DD`.",
    "list[int]": BASIC_INSTRUCTION
    + "Return your answer as a valid Python `list[int]`.",
    "literal": BASIC_INSTRUCTION
    + "Your answer should be a valid selection from the provided `OPTIONS`.",
}

BASE_RETURN_TYPE_TO_EXAMPLE: dict[str, dict] = {
    "bool": {
        "question": "Is this city in the California Bay Area?",
        "examples": [{"value": "San Jose", "column_name": "city", "answer": True}],
    },
    "int": {
        "question": "Approximately how tall is this person in cm?",
        "examples": [{"value": "Barack Obama", "column_name": "person", "answer": 185}],
    },
    "float": {
        "question": "What is the approximate distance in thousands of miles?",
        "examples": [
            {"value": "Earth to Moon", "column_name": "trip", "answer": 238.9}
        ],
    },
    "str": {
        "question": "What is the capital of this country?",
        "examples": [{"value": "France", "column_name": "country", "answer": "Paris"}],
    },
    "list[str]": {
        "question": "Who played in the game?",
        "examples": [
            {
                "value": "Lebron James played very well, but Steph Curry struggled.",
                "column_name": "game_summary",
                "answer": ["Lebron James", "Steph Curry"],
            }
        ],
    },
    "date": {
        "question": "When was {} born?",
        "examples": [
            {
                "value": "Michael Phelps",
                "column_name": "summary",
                "context": "Michael Phelps (born June 30, 1985) is a swimmer.",
                "answer": "1985-06-30",
            }
        ],
    },
    "list[int]": {
        "question": "How many points did Steph Curry and Lebron James score? Return the answer in the order [steph_points, lebron_points].",
        "examples": [
            {
                "value": "Steph had 34pts on 5/16 shooting, whereas Lebron had only 4.",
                "column_name": "game_summary",
                "answer": [34, 4],
            }
        ],
    },
    "literal": {
        "question": "Does the review have positive or negative sentiment?",
        "options": ["POSITIVE", "NEGATIVE"],
        "examples": [
            {
                "value": "I love this movie! It's so good.",
                "column_name": "movieReview",
                "answer": "POSITIVE",
            }
        ],
    },
}


def _return_type_converter(value):
    return STR_TO_DATATYPE[value.lower()] if isinstance(value, str) else value


class FeatureType(Enum):
    """Distinguishes between features that are passed for each value (LOCAL),
    vs. ones that can be shared and prefix-cached for an entire inference session (GLOBAL).
    """

    GLOBAL = "global"
    LOCAL = "local"


def get_return_type_annotation(
    return_type: DataType,
    options: Collection[str] | None = None,
    list_options: bool = True,
) -> str:
    if list_options and options is not None:
        return (
            "Literal["
            + ", ".join(
                [
                    f'"{option}"' if return_type.requires_quotes else str(option)
                    for option in options
                ]
            )
            + "]"
        )
    return return_type.name


def format_python_signature(
    question: str,
    return_type: DataType | str,
    context: str | None = None,
    context_type: FeatureType | None = None,
    table_name: str = None,
    column_name: str = None,
    options_type: FeatureType | None = None,
    additional_args: list[AdditionalMapArg] = None,
    options: Collection[str] | None = None,
    list_options: bool = True,
    add_leading_newlines: bool = True,
) -> str:
    return_type = _return_type_converter(return_type)
    return_type_annotation = get_return_type_annotation(
        return_type=return_type,
        options=options,
        list_options=list_options,
    )
    if additional_args is None:
        additional_args = []

    use_context = context_type is not None

    s = "\n\n" if add_leading_newlines else ""
    s += "```python\n"

    if table_name and column_name:
        args_str = f'Values from the "{table_name}" table in a SQL database.'
    else:
        args_str = "Value from a column in a SQL database."

    var_name = column_name or "s"
    s += f"""def f({var_name}: str"""

    for arg in additional_args:
        s += f""", {arg.columnname}: str"""

    if context_type == FeatureType.LOCAL:
        s += f""", context: List[str]"""

    if options_type == FeatureType.LOCAL:
        s += f""", options: List[str]"""
    s += ")"
    s += f" -> {return_type_annotation}:\n" + indent(f'"""{question}', prefix=INDENT())

    if context_type == FeatureType.GLOBAL:
        indented_context = context.replace("\n", "\n" + INDENT())
        s += (
            f"""\n{INDENT()}All function outputs are based on the following context:\n{INDENT()}"""
            + f"\n{INDENT()}{indented_context}"
        )
    arg_name = column_name or "s"
    s += f"""\n\n{INDENT()}Args:\n{INDENT(2)}{arg_name} (str): {args_str}"""
    for arg in additional_args:
        s += f"""\n{INDENT(2)}{arg.columnname} (str): Values from the "{arg.tablename}" table in a SQL database."""
    if context_type == FeatureType.LOCAL:
        s += f"""\n{INDENT(2)}context (List[str]): Context to use in answering the question."""
    if options_type == FeatureType.LOCAL:
        s += f"""\n{INDENT(2)}options (List[str]): Candidate strings for use in your response."""
    s += f"""\n\n{INDENT()}Returns:\n{INDENT(2)}{return_type_annotation}: Answer to the above question for each input."""
    s += f"""\n\n{INDENT()}Examples:\n{INDENT(2)}```python"""
    _question = '"' + question + '"'
    if "\n" in question:
        _question = "\n" + indent(question, prefix=INDENT(2))
        _question = '"""' + _question + INDENT(2) + '"""'
    s += (
        f"\n{INDENT(2)}# f() returns the output to the question {_question}"
        + ("" if not use_context else f" given the supplied context")
        + "\n"
    )
    return s


def format_python_continuation(
    value: str,
    additional_args: tuple[str] | None,
    context: str | list[str] | None,
    context_in_use_type: FeatureType | None,
    local_options: list[str] | None,
) -> str:
    """Builds the per-item assistant continuation for the 'code' prompt style.

    Returns a string like `  f("value") ==` that the model completes.
    """

    def get_quote(s: str):
        return '"""' if any(c in s for c in ["\n", '"']) else '"'

    value_quote = get_quote(value)
    has_more_than_one_arg = bool(
        context_in_use_type == FeatureType.LOCAL
        or additional_args is not None
        or local_options is not None
    )
    arg_prefix = " "
    newline_args = False
    if has_more_than_one_arg:
        newline_args = False
        if len(value) > 20:
            newline_args = True
        elif local_options is not None:
            if len(str(local_options)) > 20:
                newline_args = True
        elif additional_args is not None:
            for a in additional_args:
                if len(a) > 20:
                    newline_args = True
        if newline_args:
            arg_prefix = f"\n{INDENT(3)}"
        gen_str = f"""{INDENT(2)}f({arg_prefix if newline_args else ''}{value_quote}{value}{value_quote}"""
        if additional_args is not None:
            for arg in additional_args:
                gen_str += f',{arg_prefix}"{arg}"'
        if context_in_use_type == FeatureType.LOCAL:
            json_str = json.dumps(context, ensure_ascii=False, indent=16)[:-1]
            gen_str += f",{arg_prefix}" + json_str + f"{INDENT(3)}]"
        if local_options is not None:
            gen_str += f",{arg_prefix}{local_options}"
    else:
        indented_value = value.replace("\n", f"\n{INDENT(2)}")
        gen_str = f"""{INDENT(2)}f({value_quote}{indented_value}{value_quote}"""
    if has_more_than_one_arg:
        if newline_args:
            gen_str += f"\n{INDENT(2)}"
    gen_str += ") == "
    return gen_str


def format_basic_continuation(
    question: str,
    value: str | None,
    column_name: str | None,
    additional_args: list[str] | None,
    additional_args_columnnames: list[str] | None,
    context: str | list[str] | None,
    options: list[str] | None,
    table_name: str | None = None,
    additional_args_tablenames: list[str] | None = None,
    skip_value_in_inputs: bool = False,
) -> str:
    s = ""
    if question is not None:
        s += f"QUESTION:\n{question}\n\n"
    context_dict = {}
    has_multiple_tables = False
    if additional_args is not None:
        has_multiple_tables = (
            len(set([table_name]).union(set(additional_args_tablenames))) > 1
        )
    if value is not None and not skip_value_in_inputs:
        if has_multiple_tables:
            context_dict[f"{table_name}.{column_name}"] = value
        else:
            context_dict[column_name] = value
    if additional_args is not None:
        for a, a_tablename, a_columnname in zip(
            additional_args, additional_args_tablenames, additional_args_columnnames
        ):
            if has_multiple_tables:
                context_dict[f"{a_tablename}.{a_columnname}"] = a
            else:
                context_dict[a_columnname] = a
    if context is not None:
        context_dict["extra_context"] = json.loads(context)
    if context_dict:
        s += f"CONTEXT:\n{json.dumps(context_dict, ensure_ascii=False, indent=4 if len(context_dict) > 1 else None)}\n\n"
    else:
        s += f"CONTEXT: See attached content.\n"
    if options is not None:
        s += f"OPTIONS:\n{options}\n\n"
    s += "ANSWER:\n"
    return s
