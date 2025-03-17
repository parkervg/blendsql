from typing import Any
from litellm import LlmProviders

system = lambda x: {"role": "system", "content": x}
assistant = lambda x: {"role": "assistant", "content": x}
user = lambda x: {"role": "user", "content": x}


def get_tokenizer(model_name_or_path: str) -> Any:
    provider_route, name = model_name_or_path.split("/")
    if provider_route == LlmProviders.OPENAI:
        import tiktoken

        return tiktoken.encoding_for_model(name)
    elif provider_route == LlmProviders.GEMINI:
        return None
    elif provider_route == LlmProviders.ANTHROPIC:
        return None
    return None
