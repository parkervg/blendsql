from typing import List, Any
from .openai_endpoint import OpenaiEndpoint


def initialize_endpoint(endpoint_name: str) -> "Endpoint":
    return OpenaiEndpoint(endpoint_name)


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
