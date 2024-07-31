from functools import singledispatch
from typing import List
import outlines

from ..models import Model, OllamaLLM, TransformersVisionModel


@singledispatch
def choice(model: Model, prompt: str, choices: List[str], **kwargs) -> str:
    generator = outlines.generate.choice(model.model_obj, choices=choices)
    return generator(prompt)


@choice.register(TransformersVisionModel)
def regex_transformers_vision(
    model: TransformersVisionModel,
    prompt: str,
    choices: List[str],
    media=None,
    **kwargs
):
    generator = outlines.generate.choice(model.model_obj, choices=choices)
    return generator(prompt, media=media, choices=choices)


@choice.register(OllamaLLM)
def choice_ollama(*_, **__) -> str:
    """Helper function to work with Ollama models,
    since they're not recognized in the Outlines ecosystem.
    """
    raise NotImplementedError(
        "Cannot use choice generation with an Ollama model"
        + "due to the limitations of the Ollama API."
    )
