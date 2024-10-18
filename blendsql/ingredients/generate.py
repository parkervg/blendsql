import os
from functools import singledispatch
import asyncio
from asyncio import Semaphore
import logging
from colorama import Fore
from typing import Optional, List

from blendsql._configure import ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT
from .._logger import logger
from ..models import Model, OllamaLLM, OpenaiLLM, AnthropicLLM

system = lambda x: {"role": "system", "content": x}
assistant = lambda x: {"role": "assistant", "content": x}
user = lambda x: {"role": "user", "content": x}


@singledispatch
def generate(model: Model, *args, **kwargs) -> str:
    pass


async def run_openai_async_completions(
    model: OpenaiLLM,
    messages_list: List[List[dict]],
    max_tokens: Optional[int] = None,
    stop_at: Optional[List[str]] = None,
    **kwargs,
):
    sem = Semaphore(int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT)))
    client: "AsyncOpenAI" = model.model_obj
    async with sem:
        responses = [
            client.chat.completions.create(
                model=model.model_name_or_path,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop_at,
                **model.load_model_kwargs,
            )
            for messages in messages_list
        ]
    return [m.choices[0].message.content for m in await asyncio.gather(*responses)]


@generate.register(OpenaiLLM)
def generate_openai(model: OpenaiLLM, *args, **kwargs) -> List[str]:
    """This function only exists because of a bug in guidance
    https://github.com/guidance-ai/guidance/issues/881

    https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a
    """
    return asyncio.get_event_loop().run_until_complete(
        run_openai_async_completions(model, *args, **kwargs)
    )


async def run_anthropic_async_completions(
    model: AnthropicLLM,
    messages_list: List[List[dict]],
    max_tokens: Optional[int] = None,
    stop_at: Optional[List[str]] = None,
    **kwargs,
):
    sem = Semaphore(int(os.getenv(ASYNC_LIMIT_KEY, DEFAULT_ASYNC_LIMIT)))
    client: "AsyncAnthropic" = model.model_obj
    async with sem:
        responses = [
            client.messages.create(
                model=model.model_name_or_path,
                messages=messages,
                max_tokens=max_tokens or 4000,
                # stop_sequences=stop_at
                **model.load_model_kwargs,
            )
            for messages in messages_list
        ]
    return [m.content[0].text for m in await asyncio.gather(*responses)]


@generate.register(AnthropicLLM)
def generate_anthropic(
    model: AnthropicLLM,
    *args,
    **kwargs,
) -> List[str]:
    return asyncio.get_event_loop().run_until_complete(
        run_anthropic_async_completions(model, *args, **kwargs)
    )


@generate.register(OllamaLLM)
def generate_ollama(model: OllamaLLM, messages_list: List[List[dict]], **kwargs) -> str:
    """Helper function to work with Ollama models,
    since they're not recognized natively in the guidance ecosystem.
    """
    # if options:
    #     raise NotImplementedError(
    #         "Cannot use choice generation with an Ollama model"
    #         + "due to the limitations of the Ollama API."
    #     )
    from ollama import Options

    # Turn guidance kwargs into Ollama
    if "stop_at" in kwargs:
        stop_at = kwargs.pop("stop_at")
        if isinstance(stop_at, str):
            stop_at = [stop_at]
        kwargs["stop"] = stop_at
    options = Options(**kwargs)
    if options.get("temperature") is None:
        options["temperature"] = 0.0
    stream = logger.level <= logging.DEBUG
    responses = []
    for messages in messages_list:
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
            responses.append("".join(chunked_res))
            continue
        responses.append(response["message"]["content"])
    return responses
