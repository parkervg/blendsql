from typing import List, Any

from blendsql.models import OpenaiLLM, TransformersLLM
from blendsql.models._model import Model


def initialize_llm(llm_type: str, model_name_or_path: str) -> Model:
    if llm_type == "hf":
        return TransformersLLM(model_name_or_path=model_name_or_path)
    elif llm_type == "openai":
        return OpenaiLLM(model_name_or_path=model_name_or_path)
    else:
        raise ValueError(f"Unknown llm_type '{llm_type}'")


def construct_gen_clause(
    gen_type: str = "gen",
    pattern: str = None,
    max_tokens: int = None,
    stop: str = None,
    options: List[Any] = None,
    temperature: float = 0.0,
) -> str:
    """Construct a {{gen}} clause for use with guidance"""
    gen_clause = "{{" + f'{gen_type} "result"'
    if options is not None and gen_type == "select":
        gen_clause += f" options={str(options)}"
    if pattern is not None:
        gen_clause += f' pattern="{pattern}"'
    if temperature is not None and gen_type == "gen":
        gen_clause += f" temperature={temperature}"
    if max_tokens is not None and gen_type == "gen":
        gen_clause += f" max_tokens={max_tokens}"
    if stop is not None and gen_type == "gen":
        gen_clause += f' stop="{stop}"'
    gen_clause += "}}"
    return gen_clause
