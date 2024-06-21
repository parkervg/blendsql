from functools import singledispatch
import logging
from colorama import Fore
from typing import Optional, List, Union
import outlines

from .._logger import logger
from ..models import Model, OllamaLLM


@singledispatch
def text(
    model: Model,
    prompt: str,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[List[str], str]] = None,
    **kwargs
) -> str:
    generator = outlines.generate.text(model.model_obj)
    return generator(prompt, max_tokens=max_tokens, stop_at=stop_at)


@text.register(OllamaLLM)
def text_ollama(model: OllamaLLM, prompt, **kwargs) -> str:
    """Helper function to work with Ollama models,
    since they're not recognized in the Outlines ecosystem.
    """
    from ollama import Options

    # Turn outlines kwargs into Ollama
    if "stop_at" in kwargs:
        stop_at = kwargs.pop("stop_at")
        if isinstance(stop_at, str):
            stop_at = [stop_at]
        kwargs["stop"] = stop_at
    options = Options(**kwargs)
    if options.get("temperature") is None:
        options["temperature"] = 0.0
    stream = logger.level <= logging.DEBUG
    response = model.model_obj(
        messages=[{"role": "user", "content": prompt}],
        options=options,
        stream=stream,
    )  # type: ignore
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
        return "".join(chunked_res)
    return response["message"]["content"]
