from functools import singledispatch
import logging
from colorama import Fore
from typing import Optional, List

from .._logger import logger
from ..models import Model, OllamaLLM, OpenaiLLM, AnthropicLLM

system = lambda x: {"role": "system", "content": x}
assistant = lambda x: {"role": "assistant", "content": x}
user = lambda x: {"role": "user", "content": x}


@singledispatch
def generate(model: Model, messages: List[dict], *args, **kwargs) -> str:
    pass


@generate.register(OpenaiLLM)
def generate_openai(
    model: OpenaiLLM,
    messages: List[dict],
    max_tokens: Optional[int] = None,
    stop_at: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """This function only exists because of a bug in guidance
    https://github.com/guidance-ai/guidance/issues/881
    """
    client = model.model_obj.engine.client
    return (
        client.chat.completions.create(
            model=model.model_obj.engine.model_name,
            messages=messages,
            max_tokens=max_tokens,
            stop=stop_at,
            **model.load_model_kwargs,
        )
        .choices[0]
        .message.content
    )


@generate.register(AnthropicLLM)
def generate_anthropic(
    model: AnthropicLLM,
    messages: List[dict],
    max_tokens: Optional[int] = None,
    stop_at: Optional[List[str]] = None,
    **kwargs,
):
    client = model.model_obj.engine.anthropic

    return (
        client.messages.create(
            model=model.model_obj.engine.model_name,
            messages=messages,
            max_tokens=max_tokens or 4000,
            # stop_sequences=stop_at
            **model.load_model_kwargs,
        )
        .content[0]
        .text
    )


@generate.register(OllamaLLM)
def generate_ollama(model: OllamaLLM, messages: List[dict], **kwargs) -> str:
    """Helper function to work with Ollama models,
    since they're not recognized natively in the guidance ecosystem.
    """
    # if options:
    #     raise NotImplementedError(
    #         "Cannot use choice generation with an Ollama model"
    #         + "due to the limitations of the Ollama API."
    #     )
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
        messages=messages,
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
