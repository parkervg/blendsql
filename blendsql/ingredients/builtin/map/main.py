import os
import typing as t
from pathlib import Path
import json
import pandas as pd
from colorama import Fore
from attr import attrs, attrib

from blendsql.common.logger import logger
from blendsql.models import Model, ConstrainedModel
from blendsql.models.constrained.utils import LMString, maybe_load_lm
from blendsql.ingredients.ingredient import MapIngredient
from blendsql.common.exceptions import IngredientException
from blendsql.ingredients.utils import (
    initialize_retriever,
    partialclass,
)
from blendsql.configure import MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT
from blendsql.types import DataType, prepare_datatype
from .examples import (
    AnnotatedMapExample,
    ConstrainedMapExample,
    ConstrainedAnnotatedMapExample,
)
from blendsql.search.searcher import Searcher

DEFAULT_MAP_FEW_SHOT: t.List[AnnotatedMapExample] = [
    AnnotatedMapExample(**d)
    for d in json.loads(
        open(Path(__file__).resolve().parent / "./default_examples.json", "r").read()
    )
]
CONSTRAINED_MAIN_INSTRUCTION = "Complete the docstring for the provided Python function. The output should correctly answer the question provided for each input value. "
CONSTRAINED_MAIN_INSTRUCTION = (
    CONSTRAINED_MAIN_INSTRUCTION
    + "On each newline, you will follow the format of f({value}) == {answer}.\n"
)
DEFAULT_CONSTRAINED_MAP_BATCH_SIZE = 100

# OPTIONS_INSTRUCTION = "Your responses MUST select from one of the following values:\n"


@attrs
class LLMMap(MapIngredient):
    DESCRIPTION = """
    If question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column using the scalar function:
        `{{LLMMap('question', 'table::column')}}`
    """
    model: Model = attrib(default=None)
    few_shot_retriever: t.Callable[[str], t.List[AnnotatedMapExample]] = attrib(
        default=None
    )
    list_options_in_prompt: bool = attrib(default=True)
    few_shot_retriever: t.Callable[[str], t.List[AnnotatedMapExample]] = attrib(
        default=None
    )
    context_formatter: t.Callable[[pd.DataFrame], str] = attrib(
        default=lambda df: json.dumps(df.to_dict(orient="records"), indent=4),
    )
    batch_size: int = attrib(default=None)

    @classmethod
    def from_args(
        cls,
        model: t.Optional[Model] = None,
        few_shot_examples: t.Optional[
            t.Union[t.List[dict], t.List[AnnotatedMapExample]]
        ] = None,
        list_options_in_prompt: bool = True,
        batch_size: t.Optional[int] = None,
        num_few_shot_examples: t.Optional[int] = None,
        searcher: t.Optional[Searcher] = None,
    ):
        """Creates a partial class with predefined arguments.

        Args:
            model: The model to be used. Defaults to None.
            few_shot_examples: A list of dictionary MapExample few-shot examples.
               If not specified, will use [default_examples.json](https://github.com/parkervg/blendsql/blob/main/blendsql/ingredients/builtin/map/default_examples.json) as default.
            list_options_in_prompt: Whether to list options in the prompt. Defaults to True.
            batch_size: The batch size for processing. Defaults to 5.
            num_few_shot_examples: Determines number of few-shot examples to use for each ingredient call.
               Default is None, which will use all few-shot examples on all calls.
               If specified, will initialize a haystack-based embedding retriever to filter examples.

        Returns:
            Type[MapIngredient]: A partial class of MapIngredient with predefined arguments.

        Examples:
            ```python
            from blendsql import BlendSQL
            from blendsql.ingredients.builtin import LLMQA, DEFAULT_QA_FEW_SHOT

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

            bsql = BlendSQL(db, ingredients=ingredients)
            ```
        """
        if few_shot_examples is None:
            few_shot_examples = DEFAULT_MAP_FEW_SHOT
        else:
            # Sort of guessing here - the user could change the `model` type later,
            #   or pass the model at the `BlendSQL(...)` level instead of the ingredient level.
            if model is not None:
                few_shot_examples = [
                    ConstrainedAnnotatedMapExample(**d)
                    if isinstance(d, dict)
                    else ConstrainedAnnotatedMapExample(**d.__dict__)
                    for d in few_shot_examples
                ]

        few_shot_retriever = initialize_retriever(
            examples=few_shot_examples, num_few_shot_examples=num_few_shot_examples
        )
        return cls._maybe_set_name_to_var_name(
            partialclass(
                cls,
                model=model,
                few_shot_retriever=few_shot_retriever,
                list_options_in_prompt=list_options_in_prompt,
                batch_size=batch_size,
                searcher=searcher,
            )
        )

    def run(
        self,
        model: Model,
        question: str,
        values: t.List[str],
        context_formatter: t.Callable[[pd.DataFrame], str],
        list_options_in_prompt: bool,
        unpacked_questions: t.List[str] = None,
        searcher: t.Optional[Searcher] = None,
        options: t.Optional[t.List[str]] = None,
        few_shot_retriever: t.Callable[
            [str], t.List[ConstrainedAnnotatedMapExample]
        ] = None,
        value_limit: t.Optional[int] = None,
        example_outputs: t.Optional[str] = None,
        return_type: t.Optional[t.Union[DataType, str]] = None,
        regex: t.Optional[str] = None,
        context: t.Optional[pd.DataFrame] = None,
        batch_size: int = None,
        **kwargs,
    ) -> t.List[t.Union[float, int, str, bool]]:
        """For each value in a given column, calls a Model and retrieves the output.

        Args:
            question: The question(s) to map onto the values. Will also be the new column name
            model: The Model (blender) we will make calls to.
            values: The list of values to apply question to.
            value_limit: Optional limit on the number of values to pass to the Model
            example_outputs: This gives the Model an example of the output we expect.
            return_type: In the absence of example_outputs, give the Model some signal as to what we expect as output.
            regex: Optional regex to constrain answer generation.

        Returns:
            Iterable[Any] containing the output of the Model for each value.
        """
        if model is None:
            raise IngredientException(
                "LLMMap requires a `Model` object, but nothing was passed!\nMost likely you forgot to set the `default_model` argument in `blend()`"
            )
        if few_shot_retriever is None:
            few_shot_retriever = lambda *_: DEFAULT_MAP_FEW_SHOT
        use_context = context is not None or searcher is not None
        # If we explicitly passed `context`, this should take precedence over the vector store.
        if searcher is not None and context is None:
            if unpacked_questions:
                docs = searcher(unpacked_questions)
            else:
                docs = searcher(question) * len(values)
            context = ["\n\n".join(d) for d in docs]
            logger.debug(
                Fore.LIGHTBLACK_EX
                + f"Retrieved contexts '{[d[:50] + '...' for d in context]}'"
                + Fore.RESET
            )
        elif context is None:
            context = [None] * len(values)

        # Unpack default kwargs
        table_name, column_name = self.unpack_default_kwargs(**kwargs)
        if value_limit is not None:
            values = values[:value_limit]
        values = [value if not pd.isna(value) else "-" for value in values]
        resolved_return_type: DataType = prepare_datatype(
            return_type=return_type, options=options, quantifier=None
        )

        if isinstance(model, ConstrainedModel):
            current_example = ConstrainedMapExample(
                question=question,
                column_name=column_name,
                table_name=table_name,
                return_type=resolved_return_type,
                example_outputs=example_outputs,
                options=options,
                use_context=use_context,
            )

        regex = regex or resolved_return_type.regex

        if isinstance(model, ConstrainedModel):
            batch_size = batch_size or DEFAULT_CONSTRAINED_MAP_BATCH_SIZE
            few_shot_examples: t.List[ConstrainedAnnotatedMapExample] = [
                ConstrainedAnnotatedMapExample(**example.__dict__)
                for example in few_shot_retriever()
            ]
        else:
            few_shot_examples: t.List[AnnotatedMapExample] = [
                AnnotatedMapExample(**example.__dict__)
                for example in few_shot_retriever()
            ]

        if options is not None and list_options_in_prompt:
            if len(options) > int(
                os.getenv(MAX_OPTIONS_IN_PROMPT_KEY, DEFAULT_MAX_OPTIONS_IN_PROMPT)
            ):
                logger.debug(
                    Fore.YELLOW
                    + f"Number of options ({len(options)}) is greater than the configured MAX_OPTIONS_IN_PROMPT.\nWill run inference without explicitly listing these options in the prompt text."
                )
                list_options_in_prompt = False

        if isinstance(model, ConstrainedModel):
            import guidance

            if all(x is not None for x in [options, regex]):
                raise IngredientException(
                    "MapIngredient exception!\nCan't have both `options` and `regex` argument passed."
                )

            lm = LMString()  # type: ignore

            if options is not None:
                gen_f = lambda _: guidance.select(options=options)  # type: ignore
            elif resolved_return_type.name == "substring":
                # Special case for substring datatypes
                gen_f = lambda s: guidance.substring(target_string=s)
            else:
                gen_f = lambda _: guidance.gen(
                    max_tokens=kwargs.get("max_tokens", 200),
                    # guidance=0.2.1 doesn't allow both `stop` and `regex` to be passed
                    stop=[")", "\n\t"]
                    + (['"'] if resolved_return_type.name == "str" else []),
                    regex=regex,
                )  # type: ignore

            def make_prediction(
                value: str,
                context: t.Optional[str],
                str_output: bool,
                gen_f: t.Callable,
            ) -> str:
                def get_quote(s: str):
                    return '"""' if any(c in s for c in ["\n", '"']) else '"'

                value_quote = get_quote(value)
                gen_str = f"""\t\tf({value_quote}{value}{value_quote}"""
                if context is not None:
                    context_quote = get_quote(context)
                    gen_str += f""", {context_quote}{context}{context_quote}"""
                gen_str += f""") == {'"' if str_output else ''}{guidance.capture(gen_f(value), name=value)}{'"' if str_output else ''}"""
                return gen_str

            example_str = ""
            if len(few_shot_examples) > 0:
                for example in few_shot_examples:
                    example_str += example.to_string()

            loaded_lm = False
            batch_inference_strings = []
            value_to_cache_key = {}
            # Due to guidance's prefix caching, this is a one-time cost
            model.prompt_tokens += len(
                model.tokenizer.encode(CONSTRAINED_MAIN_INSTRUCTION + example_str)
            )
            for c, v in zip(context, values):
                current_example.context = c
                current_example_str = current_example.to_string(
                    list_options=list_options_in_prompt,
                    add_leading_newlines=True,
                )

                # First check - do we need to load the model?
                in_cache = False
                if model.caching:
                    cached_response, cache_key = model.check_cache(
                        CONSTRAINED_MAIN_INSTRUCTION,
                        example_str,
                        current_example_str,
                        question,
                        options,
                        c,
                        v,
                        funcs=[make_prediction, gen_f],
                    )
                    if cached_response is not None:
                        lm.set(v, cached_response)
                        in_cache = True
                    else:
                        value_to_cache_key[v] = cache_key
                if not in_cache and not loaded_lm:
                    lm: guidance.models.Model = maybe_load_lm(model, lm)
                    loaded_lm = True
                    with guidance.user():
                        lm += CONSTRAINED_MAIN_INSTRUCTION
                        lm += example_str
                        lm += current_example_str

                model.prompt_tokens += len(model.tokenizer.encode(current_example_str))

                if not in_cache:
                    batch_inference_strings.append(
                        make_prediction(
                            value=v,
                            context=c,
                            str_output=(resolved_return_type.name == "str"),
                            gen_f=gen_f,
                        )
                    )

            with guidance.assistant():
                for i in range(0, len(batch_inference_strings), batch_size):
                    model.num_generation_calls += 1
                    batch_lm = lm + "\n".join(
                        batch_inference_strings[i : i + batch_size]
                    )
                    lm._variables.update(batch_lm._variables)
            if model.caching:
                for value, cache_key in value_to_cache_key.items():
                    model.cache[cache_key] = lm.get(value)  # type: ignore

            lm_mapping: t.List[str] = [lm[value] for value in values]  # type: ignore
            model.completion_tokens += sum(
                [len(model.tokenizer.encode(v)) for v in lm_mapping]
            )
            # For each value, call the DataType's `coerce_fn()`
            mapped_values = [resolved_return_type.coerce_fn(s) for s in lm_mapping]
        else:
            import dspy

            if options is not None and list_options_in_prompt:
                type_annotation = (
                    f"Literal[" + ", ".join([f'"{option}"' for option in options]) + "]"
                )
            else:
                type_annotation = resolved_return_type.name

            map_fn = dspy.Predict(
                dspy.Signature(
                    f"question: str, value: str, source_column: str, context: Optional[str] -> answer: {type_annotation}",
                    instructions="Answer the provided question for the value from the database given the context.",
                )
            )
            demos = []
            for example in few_shot_examples:
                for value, answer in example.mapping.items():
                    demos.append(
                        dspy.Example(
                            {
                                "question": example.question,
                                "value": value,
                                "source_column": f'"{example.table_name}"."{example.column_name}"',
                                "answer": answer,
                            }
                        )
                    )
                map_fn.demos = demos

            signature = map_fn.dump_state()["signature"]

            value_to_kwargs_list: t.Dict[str, t.List[dict]] = {}
            value_to_cache_key: t.Dict[str, str] = {}
            value_to_mapped_value: t.Dict[str, str] = {}
            value_to_token_stats = {}
            # Determine which values we actually need to pass
            # Note that dspy has an in-memory cache by default as well
            # But, we 1) Want this to be library-agnostic
            #   and 2) Still want to track hypothetical token throughput
            #   for research purposes.
            for c, v in zip(context, values):
                in_cache = False
                kwargs_list = {
                    "question": question,
                    "value": v,
                    "source_column": f'"{column_name}"."{table_name}"',
                    "context": c,
                }
                if model.caching:
                    cached_response_data, cache_key = model.check_cache(
                        signature, kwargs_list
                    )
                    if cached_response_data is not None:
                        cached_response, cached_token_stats = (
                            cached_response_data["response"],
                            cached_response_data["token_stats"],
                        )
                        value_to_token_stats[v] = {
                            "prompt_tokens": cached_token_stats["prompt_tokens"],
                            "completion_tokens": cached_token_stats[
                                "completion_tokens"
                            ],
                        }
                        value_to_mapped_value[v] = cached_response
                        in_cache = True
                    else:
                        value_to_cache_key[v] = cache_key
                if not in_cache:
                    value_to_kwargs_list[v] = kwargs_list

            responses = [
                i.answer
                for i in model.generate(
                    map_fn,
                    kwargs_list=list(value_to_kwargs_list.values()),
                )
            ]
            model.num_generation_calls += len(responses)
            passed_values_list = list(value_to_kwargs_list.keys())
            for value, usage in zip(
                passed_values_list, model.get_usage(-len(responses))
            ):
                if usage != {}:
                    value_to_token_stats[value] = {
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                    }
                else:
                    logger.debug(
                        Fore.RED
                        + "DSPy program has empty usage. Is caching on by accident?"
                        + Fore.RESET
                    )
                    value_to_token_stats[value] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }

            new_mapped_values = dict(zip(passed_values_list, responses))

            if model.caching:
                for value, cache_key in value_to_cache_key.items():
                    model.cache[cache_key] = {
                        "response": new_mapped_values[value],
                        "token_stats": value_to_token_stats[value],
                    }

            value_to_mapped_value = {**value_to_mapped_value, **new_mapped_values}
            mapped_values = [value_to_mapped_value[v] for v in values]

            model.prompt_tokens += sum(
                [i["prompt_tokens"] for i in value_to_token_stats.values()]
            )
            model.completion_tokens += sum(
                [i["completion_tokens"] for i in value_to_token_stats.values()]
            )

        logger.debug(
            Fore.YELLOW
            + f"Finished LLMMap with values:\n{json.dumps(dict(zip(values[:10], mapped_values[:10])), indent=4)}"
            + Fore.RESET
        )
        return mapped_values
