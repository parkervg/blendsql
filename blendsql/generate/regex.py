from functools import singledispatch
from typing import Optional, List, Union
import outlines

from ..models import Model, OllamaLLM, TransformersVisionModel


@singledispatch
def regex(
    model: Model,
    prompt: str,
    regex: str,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[List[str], str]] = None,
) -> str:
    generator = outlines.generate.regex(model.model_obj, regex_str=regex)
    return generator(prompt, max_tokens=max_tokens, stop_at=stop_at)


@regex.register(TransformersVisionModel)
def regex_transformers_vision(
    model: TransformersVisionModel,
    prompt: str,
    regex: str,
    media=None,
    max_tokens: Optional[int] = None,
    stop_at: Optional[Union[List[str], str]] = None,
    **kwargs
):
    generator = outlines.generate.regex(model.model_obj, regex_str=regex)
    return generator(prompt, media=media, max_tokens=max_tokens, stop_at=stop_at)


@regex.register(OllamaLLM)
def regex_ollama(*_, **__) -> str:
    """Helper function to work with Ollama models,
    since they're not recognized in the Outlines ecosystem.
    """
    raise NotImplementedError(
        "Cannot use regex-structured generation with an Ollama model"
        + "due to the limitations of the Ollama API."
    )
