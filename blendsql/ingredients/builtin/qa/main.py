import os
import copy
import re
from ast import literal_eval
from pathlib import Path
from typing import Union, Optional, Callable, List
from collections.abc import Collection
import pandas as pd
import json
from colorama import Fore
from attr import attrs, attrib
import guidance

from blendsql._logger import logger
from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import maybe_load_lm, LMString
from blendsql.models._utils import user, assistant
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape
from blendsql._exceptions import IngredientException
from blendsql.ingredients.utils import (
    initialize_retriever,
    cast_responses_to_datatypes,
    prepare_datatype,
    partialclass,
)
from blendsql._configure import MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
from blendsql._constants import ModifierType, DataType
from .examples import QAExample, AnnotatedQAExample

MAIN_INSTRUCTION = "Answer the question given the table context.\n"
LONG_ANSWER_INSTRUCTION = "Make the answer as concrete as possible, providing more context and reasoning using the entire table.\n"
SHORT_ANSWER_INSTRUCTION = "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'.\n"
DEFAULT_QA_FEW_SHOT: List[AnnotatedQAExample] = [
    AnnotatedQAExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]


def get_modifier_wrapper(
    modifier: ModifierType,
) -> Callable[[guidance.models.Model], guidance.models.Model]:
    modifier_wrapper = lambda x: x
    if modifier is not None:
        if modifier == "*":
            modifier_wrapper = guidance.zero_or_more
        elif modifier == "+":
            modifier_wrapper = guidance.one_or_more
        elif re.match("{\d+}", modifier):
            repeats = [
                int(i) for i in modifier.replace("}", "").replace("{", "").split(",")
            ]
            if len(repeats) == 1:
                repeats = repeats * 2
            min_length, max_length = repeats
            modifier_wrapper = lambda f: guidance.sequence(
                f, min_length=min_length, max_length=max_length
            )
    return modifier_wrapper


@guidance(stateless=True)
def gen_list(
    lm, force_quotes: bool, modifier=None, options: List[str] = None, regex: str = None
):
    if options:
        single_item = guidance.select(options, list_append=True, name="response")
    else:
        single_item = guidance.gen(
            max_tokens=100,
            # If not regex is passed, default to all characters except these specific to list-syntax
            regex=regex or "[^],']+",
            list_append=True,
            name="response",
        )
    quote = "'"
    if not force_quotes:
        quote = guidance.optional(quote)
    single_item = quote + single_item + quote
    single_item += guidance.optional(", ")
    return lm + "[" + get_modifier_wrapper(modifier)(single_item) + "]"


def get_option_aliases(options: Optional[List[str]], is_list_output: bool):
    options_alias_to_original = {}
    options_with_aliases = None
    if options is not None:
        # Since 'options' is a mutable list, create a copy to retain the originals
        options_with_aliases = copy.deepcopy(options)
        # Below we check to see if our options have a unique first word
        # sometimes, the model will generate 'Frank' instead of 'Frank Smith'
        # We still want to align that, in this case
        add_first_word = False
        if len(set([i.split(" ")[0] for i in options])) == len(options):
            add_first_word = True
        for option in options:
            option = str(option)
            for option_alias in [option.title(), option.lower(), option.upper()]:
                options_with_aliases.add(option_alias)
                options_alias_to_original[option_alias] = option
            if add_first_word:
                option_alias = option.split(" ")[0]
                options_alias_to_original[option_alias] = option
                options_with_aliases.add(option_alias)
    return options_with_aliases or options, options_alias_to_original


@attrs
class LLMQA(QAIngredient):
    DESCRIPTION = """
    If mapping to a new column still cannot answer the question with valid SQL, turn to an end-to-end solution using the aggregate function:
        `{{LLMQA('question', (blendsql))}}`
        Optionally, this function can take an `options` argument to restrict its output to an existing SQL column.
        For example: `... WHERE column = {{LLMQA('question', (blendsql), options='table::column)}}`
    """
    model: Model = attrib(default=None)
    context_formatter: Callable[[pd.DataFrame], str] = attrib(
        default=lambda df: df.to_markdown(index=False)
    )
    list_options_in_prompt: bool = attrib(default=True)
    few_shot_retriever: Callable[[str], List[AnnotatedQAExample]] = attrib(default=None)
    k: Optional[int] = attrib(default=None)

    @classmethod
    def from_args(
        cls,
        model: Optional[Model] = None,
        few_shot_examples: Optional[List[dict]] = None,
        context_formatter: Callable[[pd.DataFrame], str] = lambda df: df.to_markdown(
            index=False
        ),
        list_options_in_prompt: bool = True,
        k: Optional[int] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            few_shot_examples: A list of Example dictionaries for few-shot learning.
                If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/qa/default_examples.json) as default.
            context_formatter: A callable that formats a pandas DataFrame into a string.
                Defaults to a lambda function that converts the DataFrame to markdown without index.
             k: Determines number of few-shot examples to use for each ingredient call.
                Default is None, which will use all few-shot examples on all calls.
                If specified, will initialize a haystack-based DPR retriever to filter examples.

        Returns:
            Type[QAIngredient]: A partial class of QAIngredient with predefined arguments.

        Examples:
            ```python
            from blendsql import blend, LLMQA
            from blendsql.ingredients.builtin import DEFAULT_QA_FEW_SHOT

            ingredients = {
                LLMQA.from_args(
                    few_shot_examples=[
                        *DEFAULT_QA_FEW_SHOT,
                        {
                            "question": "Which weighs the most?",
                            "context": {
                                {
                                    "Animal": ["Dog", "Gorilla", "Hamster"],
                                    "Weight": ["20 pounds", "350 lbs", "100 grams"]
                                }
                            },
                            "answer": "Gorilla",
                            # Below are optional
                            "options": ["Dog", "Gorilla", "Hamster"]
                        }
                    ],
                    # Will fetch `k` most relevant few-shot examples using embedding-based retriever
                    k=2,
                    # Lambda to turn the pd.DataFrame to a serialized string
                    context_formatter=lambda df: df.to_markdown(
                        index=False
                    )
                )
            }
            smoothie = blend(
                query=blendsql,
                db=db,
                ingredients=ingredients,
                default_model=model,
            )
            ```
        """
        if few_shot_examples is None:
            few_shot_examples = DEFAULT_QA_FEW_SHOT
        else:
            few_shot_examples = [
                AnnotatedQAExample(**d) if isinstance(d, dict) else d
                for d in few_shot_examples
            ]
        few_shot_retriever = initialize_retriever(
            examples=few_shot_examples, k=k, context_formatter=context_formatter
        )
        return partialclass(
            cls,
            model=model,
            few_shot_retriever=few_shot_retriever,
            context_formatter=context_formatter,
            list_options_in_prompt=list_options_in_prompt,
        )

    def run(
        self,
        model: Model,
        question: str,
        context_formatter: Callable[[pd.DataFrame], str],
        few_shot_retriever: Callable[[str], List[AnnotatedQAExample]] = None,
        options: Optional[Collection[str]] = None,
        list_options_in_prompt: bool = None,
        modifier: ModifierType = None,
        output_type: Optional[Union[DataType, str]] = None,
        context: Optional[pd.DataFrame] = None,
        value_limit: Optional[int] = None,
        long_answer: bool = False,
        **kwargs,
    ) -> Union[str, int, float]:
        """
        Args:
            question: The question to map onto the values. Will also be the new column name
            context: Table subset to use as context in answering question
            model: The Model (blender) we will make calls to.
            context_formatter: Callable defining how we want to serialize table context.
            few_shot_retriever: Callable which takes a string, and returns n most similar few-shot examples
            options: Optional collection with which we try to constrain generation.
            list_options_in_prompt: Defines whether we include options in the prompt for the current inference example
            modifier: If we expect an array of scalars, this defines the regex we want to apply.
                Used directly for constrained decoding at inference time if we have a guidance model.
            output_type: In the absence of example_outputs, give the Model some signal as to what we expect as output.
            regex: Optional regex to constrain answer generation.
            value_limit: Optional limit on how many rows from context we use
            long_answer: If true, we more closely mimic long-form end-to-end question answering.
                If false, we just give the answer with no explanation or context

        Returns:
            Union[str, int, float, tuple] containing the response from the model.
                Response will only be a tuple if `modifier` is not None.
        """
        if model is None:
            raise IngredientException(
                "LLMQA requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: DEFAULT_QA_FEW_SHOT
        if context is not None:
            if value_limit is not None:
                context = context.iloc[:value_limit]
        output_type: DataType = prepare_datatype(
            output_type=output_type, options=options, modifier=modifier
        )
        current_example = QAExample(
            **{
                "question": question,
                "context": context,
                "options": options,
                "output_type": output_type,
            }
        )
        few_shot_examples: List[AnnotatedQAExample] = few_shot_retriever(
            current_example.to_string(context_formatter)
        )

        is_list_output = "list" in current_example.output_type.name.lower()
        regex = current_example.output_type.regex
        options = current_example.options
        modifier = current_example.output_type.modifier
        options_with_aliases, options_alias_to_original = get_option_aliases(
            options, is_list_output=is_list_output
        )
        if options is not None and list_options_in_prompt:
            if len(options) > os.getenv(
                MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
            ):
                logger.debug(
                    Fore.YELLOW
                    + f"Number of options ({len(options)}) is greater than the configured MAX_OPTIONS_IN_PROMPT.\nWill run inference without explicitly listing these options in the prompt text."
                )
                list_options_in_prompt = False
        if isinstance(model, ConstrainedModel):
            lm = LMString()

            instruction_str = MAIN_INSTRUCTION
            if long_answer:
                instruction_str += LONG_ANSWER_INSTRUCTION
            else:
                instruction_str += SHORT_ANSWER_INSTRUCTION

            curr_example_str = current_example.to_string(
                context_formatter, list_options=list_options_in_prompt
            )

            if is_list_output:
                gen_kwargs = {
                    "force_quotes": bool("str" in current_example.output_type.name),
                    "regex": regex,
                    "options": options_with_aliases,
                    "modifier": modifier,
                }
                gen_f = gen_list
            else:
                if options:
                    gen_kwargs = {"options": options, "name": "response"}
                    gen_f = guidance.select
                else:
                    gen_kwargs = {
                        "max_tokens": kwargs.get("max_tokens", 200),
                        "regex": regex,
                        "name": "response",
                        "stop": ["\n", "Question:"],
                    }
                    gen_f = guidance.gen

            # First check - do we need to load the model?
            in_cache = False
            if model.caching:
                response, key = model.check_cache(
                    instruction_str,
                    curr_example_str,
                    "\n".join(
                        [
                            f"{example.to_string(context_formatter)}\n {example.answer}"
                            for example in few_shot_examples
                        ]
                    ),
                    funcs=[gen_f],
                )
                if response is not None:
                    in_cache = True
            if not in_cache:
                # Load our underlying guidance model, if we need to
                lm: guidance.models.Model = maybe_load_lm(model, lm)
                model.num_generation_calls += 1
                with guidance.user():
                    lm += instruction_str
                for example in few_shot_examples:
                    with guidance.user():
                        lm += example.to_string(context_formatter)
                    with guidance.assistant():
                        lm += example.answer
                with guidance.user():
                    lm += curr_example_str

                model.prompt_tokens += len(model.tokenizer.encode(lm._current_prompt()))

                with guidance.assistant():
                    lm += gen_f(**gen_kwargs)

                if is_list_output and modifier == "*":
                    response = lm.get("response", [])
                else:
                    response = lm["response"]

                model.completion_tokens += len(model.tokenizer.encode(str(response)))

                if model.caching:
                    model.cache[key] = response
        else:
            messages = []
            intro_prompt = MAIN_INSTRUCTION
            if long_answer:
                intro_prompt += LONG_ANSWER_INSTRUCTION
            else:
                intro_prompt += SHORT_ANSWER_INSTRUCTION
            messages.append(user(intro_prompt))
            # Add few-shot examples
            for example in few_shot_examples:
                messages.append(user(example.to_string(context_formatter)))
                messages.append(assistant(example.answer))
            # Add current question + context for inference
            messages.append(
                user(
                    current_example.to_string(
                        context_formatter, list_options=list_options_in_prompt
                    )
                )
            )
            response = model.generate(
                messages_list=[messages],
                max_tokens=kwargs.get("max_tokens", None),
            )[0].strip()
            "".join([i["content"] for i in messages])
        if isinstance(response, str):
            # If we have specified a modifier, we try to parse it to a tuple
            if is_list_output:
                try:
                    response = response.strip("'")
                    response = literal_eval(response)
                    assert isinstance(response, (list, tuple))
                    response = tuple(response)
                except (ValueError, SyntaxError, AssertionError):
                    response = [i.strip() for i in response.split(",")]
                    response = tuple(
                        [
                            "'{}'".format(single_quote_escape(val.strip()))
                            if isinstance(val, str)
                            else val
                            for val in cast_responses_to_datatypes(response)
                        ]
                    )
            else:
                response = cast_responses_to_datatypes([response])[0]
        # Map from modified options to original, as they appear in DB
        if not isinstance(response, (list, tuple, set)):
            response = [response]
        response: List[str] = [
            options_alias_to_original.get(str(r), r) for r in response
        ]
        if len(response) == 1 and not is_list_output:
            response = response[0]
            if options and response not in options:
                print(
                    Fore.RED
                    + f"Model did not select from a valid option!\nExpected one of {options}, got '{response}'"
                    + Fore.RESET
                )
            if isinstance(response, str):
                response = f"'{single_quote_escape(response)}'"
        else:
            response = tuple(response)
        return response
