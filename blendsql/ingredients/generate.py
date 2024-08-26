from functools import singledispatch
import logging
from colorama import Fore
from typing import Optional
from collections.abc import Collection

from .._logger import logger
from ..models import Model, OllamaLLM, OpenaiLLM


@singledispatch
def generate(model: Model, *args, **kwargs) -> str:
    pass


@generate.register(OpenaiLLM)
def generate_openai(
    model: OpenaiLLM, prompt, max_tokens: Optional[int], **kwargs
) -> str:
    client = model.model_obj.engine.client
    return (
        client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model.model_obj.engine.model_name,
            max_tokens=max_tokens,
            **model.load_model_kwargs,
        )
        .choices[0]
        .message.content
    )


@generate.register(OllamaLLM)
def generate_ollama(
    model: OllamaLLM, prompt, options: Optional[Collection[str]] = None, **kwargs
) -> str:
    """Helper function to work with Ollama models,
    since they're not recognized in the Outlines ecosystem.
    """
    if options:
        raise NotImplementedError(
            "Cannot use choice generation with an Ollama model"
            + "due to the limitations of the Ollama API."
        )
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
