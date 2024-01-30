import logging
import os
import time
from typing import Dict, Union

import guidance
from attr import attrs, attrib
import openai

from blendsql._constants import OPENAI_CHAT_LLM
from blendsql.ingredients.builtin.llm.endpoint import Endpoint

logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("guidance").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def library_call_with_engine(**kwargs):
    kwargs["engine"] = kwargs.get("model")
    # Needed since openai.Completion.acreate() calls aiohttp,
    # which passes the proxy from this attribute, not the typical HTTP_PROXY environment variable
    openai.proxy = os.environ.get("HTTP_PROXY")
    return guidance.llm._library_call(**kwargs)


@attrs(auto_detect=True)
class OpenaiEndpoint(Endpoint):
    """Class for OpenAI Endpoints."""

    endpoint_name: str = attrib()
    gen_kwargs: dict = attrib(init=False)

    def __attrs_post_init__(self):
        self.gen_kwargs = {}

    def predict(self, program: str, **kwargs) -> Union[str, Dict]:
        """
        Runs a guidance program, and returns the output variables.
        """
        program = guidance(program)
        # Initialize guidance
        guidance.llm = guidance.llms.OpenAI(
            # tiktoken.encoding_for_model has different name for gpt-35-turbo
            model=self.endpoint_name
            if self.endpoint_name != "gpt-35-turbo"
            else "gpt-3.5-turbo",
            api_type=os.getenv("API_TYPE", "azure"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION", "2023-03-15-preview"),
            api_base=os.getenv("OPENAI_API_BASE"),
            max_retries=50,
            chat_mode=self.endpoint_name in OPENAI_CHAT_LLM,
        )
        # Override the caller, so we can add the 'engine' arg
        guidance.llm.caller = library_call_with_engine
        time.time()
        program = program(**kwargs)
        time.time()
        guidance.llm = None
        return {k: v for k, v in program.variables().items() if k != "llm"}
