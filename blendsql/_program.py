from __future__ import annotations
from typing import Tuple, Callable
import inspect
import ast
import textwrap
import logging
from colorama import Fore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Model
from ._logger import logger


class Program:
    """A Program is the base class used for dynamically formatting prompts
    and returning unconstrained/constrained generation results.
    At it's core, a Program should be a callable that takes a BlendSQL model as a named arg,
    along with any number of other positional or keyword arguments.
    It should then return a Tuple[str, str], containing the (response, prompt) pair from the
    internal program logic.

    Examples:
        ```python
        import pandas as pd
        import outlines
        from typing import Tuple

        from blendsql.models import Model
        from blendsql._program import Program

        class SummaryProgram(Program):
            def __call__(self, model: Model, serialized_db: pd.DataFrame) -> Tuple[str, str]:
                prompt = f"Summarize the table below. {serialized_db}"
                # Below we follow the outlines pattern for unconstrained text generation
                # https://github.com/outlines-dev/outlines
                # Finally, return (response, prompt) tuple
                # Returning the prompt here allows the underlying BlendSQL classes to track token usage
                generator = outlines.generate.text(model.logits_generator)
                return (generator(prompt), prompt)
        ```
        We could also write the same `Program` as a function:
        ```python
        def summary_program(model: Model, serialized_db: pd.DataFrame) -> Tuple[str, str]:
            ...
        ```
    """

    def __new__(
        self,
        model: Model,
        **kwargs,
    ):
        return self.__call__(self, model, **kwargs)

    def __call__(self, model: Model, *args, **kwargs) -> Tuple[str, str]:
        """Logic for formatting prompt and calling the underlying model.
        Should return tuple of (response, prompt).
        """
        ...


def return_ollama_response(
    logits_generator: Callable, prompt, **kwargs
) -> Tuple[str, str]:
    """Helper function to work with Ollama models,
    since they're not recognized in the Outlines ecosystem.
    """
    from ollama import Options

    options = Options(**kwargs)
    if options.get("temperature") is None:
        options["temperature"] = 0.0
    stream = logger.level <= logging.DEBUG
    response = logits_generator(
        messages=[{"role": "user", "content": prompt}],
        options=options,
        stream=stream,
    )
    if stream:
        chunked_res = []
        for chunk in response:
            chunked_res.append(chunk["message"]["content"])
            print(
                Fore.CYAN + chunk["message"]["content"] + Fore.RESET,
                end="",
                flush=True,
            )
        print("\n")
        return ("".join(chunked_res), prompt)
    return (response["message"]["content"], prompt)


def program_to_str(program: Program):
    """Create a string representation of a program.
    It is slightly tricky, since in addition to getting the code content, we need to
        1) identify all global variables referenced within a function, and then
        2) evaluate the variable value
    This is required, since if we have some global constant `PROMPT` called,
    we don't want to fetch from a previously created cache if the value of `PROMPT` changes.

    To avoid extreme messiness, we don't traverse into globals pointing at functions.

    Example:
        >>> PROMPT = "Here is my question: {question}"
        >>> class CorrectionProgram(Program):
        >>>     def __call__(self, question: str, **kwargs):
        >>>         return PROMPT.format(question)

    Some helpful refs:
        - https://github.com/universe-proton/universe-topology/issues/15
    """
    source_func = program.__call__
    call_content = textwrap.dedent(inspect.getsource(source_func))
    root = ast.parse(call_content)
    root_names = {node.id for node in ast.walk(root) if isinstance(node, ast.Name)}
    co_varnames = set(source_func.__code__.co_varnames)
    names_to_resolve = sorted(root_names.difference(co_varnames))
    resolved_names = ""
    if len(names_to_resolve) > 0:
        globals_as_dict = dict(inspect.getmembers(source_func))["__globals__"]
        for name in names_to_resolve:
            if name in globals_as_dict:
                if name.startswith("__"):
                    continue
                val = globals_as_dict[name]
                # Ignore functions - we really only want scalars here
                if any(x for x in [callable(val), hasattr(val, "__module__")]):
                    continue
                resolved_names += f"{val}\n"
    return f"{call_content}{resolved_names}"
