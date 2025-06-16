import os
import copy
import re
from ast import literal_eval
from pathlib import Path
import typing as t
import pandas as pd
import json
from colorama import Fore
from attr import attrs, attrib

from blendsql.common.logger import logger
from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import maybe_load_lm, LMString
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape
from blendsql.common.exceptions import IngredientException
from blendsql.ingredients.utils import (
    initialize_retriever,
    partialclass,
)
from blendsql.configure import MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
from blendsql.types import DataType, QuantifierType, prepare_datatype
from .examples import QAExample, AnnotatedQAExample
from blendsql.search.searcher import Searcher

MAIN_INSTRUCTION = "Answer the question given the table context, if provided.\n"
LONG_ANSWER_INSTRUCTION = "Make the answer as concrete as possible, providing more context and reasoning using the entire table.\n"
SHORT_ANSWER_INSTRUCTION = "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'.\n"
DEFAULT_QA_FEW_SHOT: t.List[AnnotatedQAExample] = [
    AnnotatedQAExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]


def get_option_aliases(options: t.Optional[t.List[str]]):
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
    context_formatter: t.Callable[[pd.DataFrame], str] = attrib(
        default=lambda df: json.dumps(df.to_dict(orient="records"), indent=4),
    )
    list_options_in_prompt: bool = attrib(default=True)
    few_shot_retriever: t.Callable[[str], t.List[AnnotatedQAExample]] = attrib(
        default=None
    )
    k: t.Optional[int] = attrib(default=None)

    @classmethod
    def from_args(
        cls,
        model: t.Optional[Model] = None,
        few_shot_examples: t.Optional[
            t.Union[t.List[dict], t.List[AnnotatedQAExample]]
        ] = None,
        context_formatter: t.Callable[[pd.DataFrame], str] = lambda df: df.to_markdown(
            index=False
        ),
        list_options_in_prompt: bool = True,
        num_few_shot_examples: t.Optional[int] = None,
        searcher: t.Optional[Searcher] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            few_shot_examples: A list of Example dictionaries for few-shot learning.
                If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/qa/default_examples.json) as default.
            context_formatter: A callable that formats a pandas DataFrame into a string.
                Defaults to a lambda function that converts the DataFrame to markdown without index.
             num_few_shot_examples: Determines number of few-shot examples to use for each ingredient call.
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
                    num_few_shot_examples=2,
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
            examples=few_shot_examples,
            num_few_shot_examples=num_few_shot_examples,
            context_formatter=context_formatter,
        )
        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                few_shot_retriever=few_shot_retriever,
                context_formatter=context_formatter,
                list_options_in_prompt=list_options_in_prompt,
                searcher=searcher,
            )
        )

    def run(
        self,
        model: Model,
        question: str,
        context_formatter: t.Callable[[pd.DataFrame], str],
        list_options_in_prompt: bool,
        few_shot_retriever: t.Optional[
            t.Callable[[str], t.List[AnnotatedQAExample]]
        ] = None,
        searcher: t.Optional[Searcher] = None,
        options: t.Optional[t.List[str]] = None,
        quantifier: QuantifierType = None,
        return_type: t.Optional[t.Union[DataType, str]] = None,
        regex: t.Optional[str] = None,
        context: t.Optional[pd.DataFrame] = None,
        long_answer: bool = False,
        use_option_aliases: bool = False,
        **kwargs,
    ) -> t.Union[str, int, float, tuple]:
        """
        Args:
            question: The question to map onto the values. Will also be the new column name
            context: Table subset to use as context in answering question
            model: The Model (blender) we will make calls to.
            context_formatter: Callable defining how we want to serialize table context.
            few_shot_retriever: Callable which takes a string, and returns n most similar few-shot examples
            options: Optional collection with which we try to constrain generation.
            list_options_in_prompt: Defines whether we include options in the prompt for the current inference example
            quantifier: If we expect an array of scalars, this defines the regex we want to apply.
                Used directly for constrained decoding at inference time if we have a guidance model.
            return_type: In the absence of example_outputs, give the Model some signal as to what we expect as output.
            regex: Optional regex to constrain answer generation. Takes precedence over `return_type`
            long_answer: If true, we more closely mimic long-form end-to-end question answering.
                If false, we just give the answer with no explanation or context

        Returns:
            Union[str, int, float, tuple] containing the response from the model.
                Response will only be a tuple if `quantifier` is not None.
        """
        if model is None:
            raise IngredientException(
                "LLMQA requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if few_shot_retriever is None:
            # Default to no few-shot examples in LLMQA
            few_shot_retriever = lambda *_: []

        # If we explicitly passed `context`, this should take precedence over the vector store.
        if searcher is not None and context is None:
            docs = searcher(question)[0]
            context = pd.DataFrame(docs, columns=["content"])
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Retrieved contexts '{[doc[:50] + '...' for doc in docs]}'"
                + Fore.RESET
            )

        resolved_return_type: DataType = prepare_datatype(
            return_type=return_type, options=options, quantifier=quantifier
        )
        current_example = QAExample(
            question=question,
            context=context,
            options=options,
            return_type=resolved_return_type,
        )
        few_shot_examples: t.List[AnnotatedQAExample] = few_shot_retriever(
            current_example.to_string(context_formatter)
        )

        is_list_output = resolved_return_type.quantifier is not None
        regex = regex or resolved_return_type.regex
        quantifier = resolved_return_type.quantifier

        options_with_aliases, options_alias_to_original = None, dict()
        if use_option_aliases:
            options_with_aliases, options_alias_to_original = get_option_aliases(
                options
            )
        elif options is not None:
            options_with_aliases, options_alias_to_original = options, {
                o: o for o in options
            }

        if options is not None and list_options_in_prompt:
            max_options_in_prompt = os.getenv(
                MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
            )
            if len(options) > max_options_in_prompt:  # type: ignore
                logger.debug(
                    Fore.YELLOW
                    + f"Number of options ({len(options)}) is greater than the configured MAX_OPTIONS_IN_PROMPT={max_options_in_prompt}.\nWill run inference without explicitly listing these options in the prompt text."
                    + Fore.RESET
                )
                list_options_in_prompt = False

        if isinstance(model, ConstrainedModel):
            import guidance

            def get_quantifier_wrapper(
                quantifier: QuantifierType,
            ) -> t.Callable[[guidance.models.Model], guidance.models.Model]:
                quantifier_wrapper = lambda x: x
                if quantifier is not None:
                    if quantifier == "*":
                        quantifier_wrapper = guidance.zero_or_more
                    elif quantifier == "+":
                        quantifier_wrapper = guidance.one_or_more
                    elif re.match(r"{\d+}", quantifier):
                        repeats = [
                            int(i)
                            for i in quantifier.replace("}", "")
                            .replace("{", "")
                            .split(",")
                        ]
                        if len(repeats) == 1:
                            repeats = repeats * 2
                        min_length, max_length = repeats
                        quantifier_wrapper = lambda f: guidance.sequence(
                            f, min_length=min_length, max_length=max_length
                        )  # type: ignore
                return quantifier_wrapper  # type: ignore

            @guidance(stateless=True, dedent=False)
            def gen_list(
                lm,
                force_quotes: bool,
                quantifier=None,
                options: t.Optional[t.List[str]] = None,
                regex: t.Optional[str] = None,
            ):
                if options:
                    single_item = guidance.select(
                        options, list_append=True, name="response"
                    )
                else:
                    single_item = guidance.gen(
                        max_tokens=100,
                        # If not regex is passed, default to all characters except these specific to list-syntax
                        regex=regex or "[^],']+",
                        list_append=True,
                        name="response",
                    )  # type: ignore
                quote = "'"
                if not force_quotes:
                    quote = guidance.optional(quote)  # type: ignore
                single_item = quote + single_item + quote
                single_item += guidance.optional(", ")  # type: ignore
                return lm + "[" + get_quantifier_wrapper(quantifier)(single_item) + "]"

            lm = LMString()  # type: ignore

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
                    "force_quotes": bool("str" in resolved_return_type.name),
                    "regex": regex,
                    "options": options_with_aliases,
                    "quantifier": quantifier,
                }
                gen_f = gen_list
            else:
                if options:
                    # Too many options here raises:
                    # ValueError: Parser Error: Current row has 10850 items; max is 2000; consider making your grammar left-recursive if it's right-recursive
                    gen_kwargs = {"options": options, "name": "response"}
                    gen_f = guidance.select
                else:
                    gen_kwargs = {
                        "max_tokens": kwargs.get("max_tokens", 200),
                        "regex": regex,
                        "name": "response",
                        # guidance=0.2.1 doesn't allow both `stop` and `regex` to be passed
                        # "stop": ["\n"] if regex is None else None,
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
                    regex,
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
                if len(few_shot_examples) > 0:
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
                if is_list_output:
                    response: list = lm.get("response", [])[::-1]  # type: ignore
                else:
                    response: str = lm["response"]  # type: ignore
                model.completion_tokens += len(model.tokenizer.encode(str(response)))

                if model.caching:
                    model.cache[key] = response  # type: ignore
        else:
            # Use DSPy to get LLM output
            import dspy

            if options is not None and list_options_in_prompt:
                return_type_annotation = (
                    f"Literal[" + ", ".join([f'"{option}"' for option in options]) + "]"
                )
                if is_list_output:
                    return_type_annotation = f"List[{return_type_annotation}]"
            else:
                return_type_annotation = resolved_return_type.name

            instructions = MAIN_INSTRUCTION + (
                LONG_ANSWER_INSTRUCTION if long_answer else SHORT_ANSWER_INSTRUCTION
            )

            if quantifier is not None:
                if quantifier == "*":
                    instructions += (
                        "You may generate zero or more responses in your list.\n"
                    )
                elif quantifier == "+":
                    instructions += (
                        "You may generate one or more responses in your list.\n"
                    )
                else:
                    repeats = [
                        int(i)
                        for i in quantifier.replace("}", "").replace("{", "").split(",")
                    ]
                    if len(repeats) == 1:
                        repeats = repeats * 2
                    min_length, max_length = repeats
                    if min_length == max_length:
                        instructions += (
                            f"You may generate {min_length} responses in your list.\n"
                        )
                    else:
                        instructions += f"You may generate between {min_length} and {max_length} responses in your list.\n"

            qa_fn = dspy.Predict(
                dspy.Signature(
                    f"question: str, context: Optional[str] -> answer: {return_type_annotation}",
                    instructions=instructions,
                )
            )
            qa_fn.demos = [
                dspy.Example(
                    {
                        "question": example.question,
                        "context": context_formatter(example.context),
                        "answer": example.answer,
                    }
                )
                for example in few_shot_examples
            ]

            signature = qa_fn.dump_state()["signature"]
            fn_kwargs = {"question": question, "context": context}
            # First check - do we need to load the model?
            in_cache = False
            if model.caching:
                cached_response_data, cache_key = model.check_cache(
                    signature, fn_kwargs
                )
                if cached_response_data is not None:
                    response, token_stats = (
                        cached_response_data["response"],
                        cached_response_data["token_stats"],
                    )
                    prompt_tokens = token_stats["prompt_tokens"]
                    completion_tokens = token_stats["completion_tokens"]
                    in_cache = True
            if not in_cache:
                response = model.generate(
                    qa_fn, kwargs_list=[{"question": question, "context": context}]
                )[0].answer
                model.num_generation_calls += 1
                prompt_tokens, completion_tokens = model.get_token_usage(1)
                if model.caching:
                    model.cache[cache_key] = {
                        "response": response,
                        "token_stats": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    }
            model.completion_tokens += completion_tokens
            model.prompt_tokens += prompt_tokens

        if isinstance(response, str):  # type: ignore
            # If we have specified a quantifier, we try to parse it to a tuple
            if is_list_output:
                try:
                    response = response.strip("'")
                    response = literal_eval(response)
                    assert isinstance(response, (list, tuple))
                    response = tuple(response)
                except (ValueError, SyntaxError, AssertionError):
                    response = [i.strip() for i in response.strip("[]()").split(",")]
                    response = [
                        current_example.return_type.coerce_fn(r) for r in response
                    ]
                    response = tuple(
                        [
                            single_quote_escape(val) if isinstance(val, str) else val
                            for val in response
                        ]
                    )
            else:
                if isinstance(response, str):
                    response = current_example.return_type.coerce_fn(response)
                elif isinstance(response, list):
                    response = [
                        current_example.return_type.coerce_fn(r) for r in response
                    ]
        # Map from modified options to original, as they appear in DB
        if not isinstance(response, (list, tuple, set)):
            response = [response]
        response: t.List[str] = [
            options_alias_to_original.get(str(r), r) for r in response
        ]
        if len(response) == 1 and not is_list_output:
            response = response[0]  # type: ignore
            if options and response not in options:
                logger.debug(
                    Fore.RED
                    + f"Model did not select from a valid option!\nExpected one of {options}, got '{response}'"
                    + Fore.RESET
                )
            if isinstance(response, str):
                response = f"'{single_quote_escape(response)}'"  # type: ignore
        else:
            response = tuple(response)  # type: ignore
        return response  # type: ignore
