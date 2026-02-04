import asyncio

from blendsql.models.model_base import ModelBase
from blendsql.common.typing import GenerationResult, GenerationItem
from blendsql.configure import add_to_global_history

DEFAULT_BODY = {"temperature": 0.0, "max_tokens": 512}


class VLLM(ModelBase):
    def __init__(
        self,
        model_name_or_path: str,
        base_url: str,
        api_key: str = "N/A",
        extra_body: dict | None = None,
        caching: bool = False,
        **kwargs,
    ):
        from openai import AsyncOpenAI
        from transformers import AutoTokenizer

        self.extra_body = extra_body or dict()

        super().__init__(
            model_name_or_path=model_name_or_path,
            caching=caching,
            _allows_parallel_requests=True,
            **kwargs,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def generate(
        self, item: GenerationItem, cancel_event: asyncio.Event | None = None
    ):
        buffer = ""
        extra_body = DEFAULT_BODY | self.extra_body
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

        finally:
            await stream.close()

        return GenerationResult(item.identifier, buffer, completed=True)
