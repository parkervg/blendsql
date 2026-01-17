import os
import copy
from ast import literal_eval
from pathlib import Path
from typing import Callable
import pandas as pd
import polars as pl
import json
from dataclasses import dataclass, field
from textwrap import dedent

from blendsql.configure import add_to_global_history
from blendsql.common.logger import logger, Color
from blendsql.common.constants import DEFAULT_CONTEXT_FORMATTER
from blendsql.models.utils import user, assistant
from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import maybe_load_lm, LMString
from blendsql.ingredients.ingredient import QAIngredient
from blendsql.db.utils import single_quote_escape
from blendsql.common.exceptions import LMFunctionException
from blendsql.common.typing import DataType, QuantifierType
from blendsql.ingredients.utils import initialize_retriever, partialclass, gen_list
from blendsql.configure import (
    MAX_OPTIONS_IN_PROMPT_KEY,
    DEFAULT_MAX_OPTIONS_IN_PROMPT,
    MAX_TOKENS_KEY,
    DEFAULT_MAX_TOKENS,
)
from blendsql.types import prepare_datatype, apply_type_conversion
from blendsql.search.searcher import Searcher
from .examples import QAExample, AnnotatedQAExample

MAIN_INSTRUCTION = "Answer the question given the context, if provided.\n"
LONG_ANSWER_INSTRUCTION = "Make the answer as concrete as possible, providing more context and reasoning using the entire context.\n"
SHORT_ANSWER_INSTRUCTION = "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'.\n"
DEFAULT_QA_FEW_SHOT: list[AnnotatedQAExample] = [
    AnnotatedQAExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]


def get_option_aliases(options: list[str] | None):
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


@dataclass
class LLMQA(QAIngredient):
    model: Model = field(default=None)
    context_formatter: Callable[[pd.DataFrame], str] = field(
        default=DEFAULT_CONTEXT_FORMATTER,
    )
    list_options_in_prompt: bool = field(default=True)
    few_shot_retriever: Callable[[str], list[AnnotatedQAExample]] = field(default=None)
    k: int | None = field(default=None)

    @classmethod
    def from_args(
        cls,
        model: Model | None = None,
        few_shot_examples: list[dict] | list[AnnotatedQAExample] | None = None,
        context_formatter: Callable[[pd.DataFrame], str] = DEFAULT_CONTEXT_FORMATTER,
        list_options_in_prompt: bool = True,
        num_few_shot_examples: int | None = 0,
        context_searcher: Searcher | None = None,
        options_searcher: Searcher | None = None,
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
            few_shot_retriever = lambda *_: []
        else:
            few_shot_retriever = initialize_retriever(
                examples=few_shot_examples, num_few_shot_examples=num_few_shot_examples
            )

        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                few_shot_retriever=few_shot_retriever,
                context_formatter=context_formatter,
                list_options_in_prompt=list_options_in_prompt,
                context_searcher=context_searcher,
                options_searcher=options_searcher,
            )
        )

    def run(
        self,
        model: Model,
        question: str,
        context_formatter: Callable[[pd.DataFrame], str],
        list_options_in_prompt: bool,
        few_shot_retriever: Callable[[str], list[AnnotatedQAExample]] | None = None,
        context_searcher: Searcher | None = None,
        options: list[str] | None = None,
        options_searcher: Searcher | None = None,
        quantifier: QuantifierType = None,
        return_type: DataType | str | None = None,
        regex: str | None = None,
        context: list[pd.DataFrame] | None = None,
        long_answer: bool = False,
        use_option_aliases: bool = False,
        enable_constrained_decoding: bool = True,
        **kwargs,
    ) -> str | int | float | tuple:
        """
        Args:
            question: The question to map onto the values. Will also be the new column name
            context: Table subset(s) to use as context in answering question
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
            raise LMFunctionException(
                "LLMQA requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: []

        question = dedent(question.removeprefix("\n"))

        # If we explicitly passed `context`, this should take precedence over the vector store.
        if context_searcher is not None and context is None:
            docs = context_searcher(question)[0]
            context = [pl.DataFrame(docs, columns=["content"])]
            logger.debug(
                Color.quiet_update(
                    f"Retrieved contexts '{[doc[:50] + '...' for doc in docs]}'"
                )
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
        few_shot_examples: list[AnnotatedQAExample] = [
            AnnotatedQAExample(**example.__dict__)
            if not isinstance(example, dict)
            else AnnotatedQAExample(**example)
            for example in few_shot_retriever(
                current_example.to_string(context_formatter)
            )
        ]

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

        if self.options_searcher is None:
            if options is not None and list_options_in_prompt:
                max_options_in_prompt = int(
                    os.getenv(MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT)
                )
                if len(options) > max_options_in_prompt:  # type: ignore
                    logger.debug(
                        Color.warning(
                            f"Number of options ({len(options):,}) is greater than the configured MAX_OPTIONS_IN_PROMPT={max_options_in_prompt:,}.\nWill run inference without explicitly listing these options in the prompt text."
                        )
                    )
                    list_options_in_prompt = False
        else:
            logger.debug(
                Color.warning(
                    f"Calling provided `options_searcher` to retrieve {self.options_searcher.k} options, out of {len(self.options_searcher.documents):,} total options..."
                )
            )
            options = (
                self.options_searcher(
                    f"Context: {context_formatter(context)}\nQuestion: {question}"
                )[0]
                if context is not None
                else self.options_searcher(question)[0]
            )

        if isinstance(model, ConstrainedModel):
            import guidance

            lm = LMString()  # type: ignore

            instruction_str = MAIN_INSTRUCTION
            if long_answer:
                instruction_str += LONG_ANSWER_INSTRUCTION
            else:
                instruction_str += SHORT_ANSWER_INSTRUCTION

            curr_example_str = current_example.to_string(
                context_formatter, list_options=list_options_in_prompt
            )

            gen_f = None
            if enable_constrained_decoding:
                if is_list_output:
                    gen_f = lambda _: guidance.capture(
                        gen_list(
                            force_quotes=bool("str" in resolved_return_type.name),
                            regex=regex,
                            options=options_with_aliases,
                            quantifier=quantifier,
                        ),
                        name="response",
                    )
                elif options:
                    gen_f = lambda _: guidance.select(options=options, name="response")
            else:
                logger.debug(
                    Color.warning(
                        "Not applying constraints, since `enable_constrained_decoding==False`"
                    )
                )

            if gen_f is None:
                gen_f = lambda _: guidance.gen(
                    max_tokens=kwargs.get(
                        "max_tokens",
                        int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS)),
                    ),
                    regex=regex if enable_constrained_decoding else None,
                    name="response",
                )

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
                    options,
                    quantifier,
                    kwargs.get(
                        "max_tokens",
                        int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS)),
                    ),
                    funcs=[gen_f],
                )
                if response is not None:
                    in_cache = True
            if not in_cache:
                # Load our underlying guidance model, if we need to
                lm: guidance.models.Model = maybe_load_lm(model, lm)
                model.num_generation_calls += 1
                lm = model.maybe_add_system_prompt(lm)
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

                with guidance.assistant():
                    lm += gen_f(question)
                add_to_global_history(str(lm))

                response = apply_type_conversion(
                    lm["response"],
                    return_type=resolved_return_type,
                    db=self.db,
                )

                if model.caching:
                    model.cache[key] = response  # type: ignore

                model.completion_tokens += lm._get_usage().output_tokens
                model.prompt_tokens += lm._get_usage().input_tokens
                lm._reset_usage()
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
                max_tokens=kwargs.get(
                    "max_tokens", int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS))
                ),
            )[0].strip()
            model.num_generation_calls += 1
            add_to_global_history(messages)

            if isinstance(response, str):  # type: ignore
                # If we have specified a quantifier, we try to parse it to a tuple
                if is_list_output:
                    try:
                        response = response.strip("'")
                        response = literal_eval(response)
                        assert isinstance(response, (list, tuple))
                        response = tuple(response)
                    except (ValueError, SyntaxError, AssertionError):
                        response = [
                            i.strip() for i in response.strip("[]()").split(",")
                        ]
                        response = [
                            current_example.return_type.coerce_fn(r, self.db)
                            for r in response
                        ]
                        response = tuple(
                            [
                                single_quote_escape(val)
                                if isinstance(val, str)
                                else val
                                for val in response
                            ]
                        )
                else:
                    if isinstance(response, str):
                        response = current_example.return_type.coerce_fn(
                            response, self.db
                        )
                    elif isinstance(response, list):
                        response = [
                            current_example.return_type.coerce_fn(r, self.db)
                            for r in response
                        ]
        # Map from modified options to original, as they appear in DB
        if not isinstance(response, (list, tuple, set)):
            response = [response]
        response: list[str] = [
            options_alias_to_original.get(str(r), r) for r in response
        ]
        if len(response) == 1 and not is_list_output:
            response = response[0]  # type: ignore
            if options and response not in options:
                logger.debug(
                    Color.error(
                        f"Model did not select from a valid option!\nExpected one of {options}, got '{response}'"
                    )
                )
            if resolved_return_type.name.lower() in ["str", "any"]:
                response = f"'{single_quote_escape(response)}'"  # type: ignore
        else:
            response = tuple(response)  # type: ignore
        if os.getenv("BLENDSQL_ALWAYS_LOWERCASE_RESPONSE") == "1":
            # Basic transforms not handled by SQLite type affinity
            if response == "True":
                return True
            if response == "False":
                return False
            return response.lower()
        logger.debug(
            lambda: Color.warning(
                f"Finished LLMQA with value: {str(response)[:200]}{'...' if len(str(response)) > 200 else ''}"
            )
        )
        return response  # type: ignore
