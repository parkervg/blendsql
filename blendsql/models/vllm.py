import asyncio
import os

from blendsql.configure import MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS
from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationResult, GenerationItem
from blendsql.configure import add_to_global_history

DEFAULT_BODY = {"temperature": 0.0}


class VLLM(ModelBase):
    """Class for vLLM endpoints.

    Args:
        model_name_or_path: Name of the model
        base_url: Base URL for http requests

    Examples:
        ```python
        from blendsql.models import VLLM

        model = VLLM("RedHatAI/gemma-3-12b-it-quantized.w4a16", base_url="http://localhost:8000/v1/")
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        base_url: str,
        api_key: str = "N/A",
        tokenizer: "BaseTokenizer" = None,
        extra_body: dict | None = None,
        caching: bool = False,
        **kwargs,
    ):
        from openai import AsyncOpenAI

        self.extra_body = extra_body or dict()

        super().__init__(
            model_name_or_path=model_name_or_path,
            caching=caching,
            _allows_parallel_requests=True,
            **kwargs,
        )
        if tokenizer is None:
            from huggingface_hub import hf_hub_download
            import json

            with open(
                hf_hub_download(
                    repo_id=model_name_or_path, filename="tokenizer_config.json"
                ),
                "r",
            ) as f:
                config = json.load(f)
            self.chat_template = config["chat_template"]
            with open(
                hf_hub_download(
                    repo_id=model_name_or_path, filename="special_tokens_map.json"
                ),
                "r",
            ) as f:
                special_tokens_map = json.load(f)
            self.special_tokens_map = {
                k: v["content"] if isinstance(v, dict) else v
                for k, v in special_tokens_map.items()
            }
        self.tokenizer = tokenizer
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(
        self, item: GenerationItem, cancel_event: asyncio.Event | None = None
    ):
        buffer = ""
        extra_body = (
            DEFAULT_BODY
            | {"max_tokens": int(os.getenv(MAX_TOKENS_KEY, DEFAULT_MAX_TOKENS))}
            | self.extra_body
        )
        if item.grammar:
            extra_body |= {
                "guided_decoding_backend": "guidance",
                "guided_grammar": item.grammar,
                "structured_outputs": {"grammar": item.grammar},
            }
        messages = [{"role": "user", "content": item.prompt}]
        if item.assistant_continuation is not None:
            messages.append(
                {"role": "assistant", "content": item.assistant_continuation}
            )

        if self.tokenizer is None:
            from .tokenization import render_jinja_template

            prompt_to_send = render_jinja_template(
                messages=messages,
                chat_template=self.chat_template,
                continue_final_message=item.assistant_continuation is not None,
                add_generation_prompt=item.assistant_continuation is None,
                **self.special_tokens_map,
            )
        else:
            prompt_to_send = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=item.assistant_continuation is not None,
                add_generation_prompt=item.assistant_continuation is None,
            )

        stream = await self.client.completions.create(
            model=self.model_name_or_path,
            prompt=prompt_to_send,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )
        self.num_generation_calls += 1
        add_to_global_history(prompt_to_send)

        try:
            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    return GenerationResult(item.identifier, buffer, completed=False)

                if chunk.choices and chunk.choices[0].text:
                    buffer += chunk.choices[0].text

                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self.prompt_tokens += chunk.usage.prompt_tokens
                    self.completion_tokens += chunk.usage.completion_tokens
                    if chunk.usage.prompt_tokens_details is not None:
                        self.cached_tokens += (
                            chunk.usage.prompt_tokens_details.cached_tokens
                        )

        finally:
            await stream.close()

        return GenerationResult(item.identifier, buffer, completed=True)
