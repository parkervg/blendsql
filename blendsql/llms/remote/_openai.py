import logging
import os
import time
from typing import Dict, Union

import guidance
import tiktoken
import openai
from azure.identity import ClientSecretCredential

from blendsql._constants import OPENAI_CHAT_LLM
from .._llm import LLM

logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("guidance").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def library_call_with_engine(**kwargs):
    kwargs["engine"] = kwargs.get("model")
    # Needed since openai.Completion.acreate() calls aiohttp,
    # which passes the proxy from this attribute, not the typical HTTP_PROXY environment variable
    openai.proxy = os.environ.get("HTTP_PROXY")
    return guidance.llm._library_call(**kwargs)


class OpenaiLLM(LLM):
    """Class for OpenAI LLM API."""

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=True,
            refresh_interval_min=30,
            encoding=tiktoken.encoding_for_model(model_name_or_path),
            **kwargs
        )

    def _setup(self, **kwargs) -> None:
        try:
            credential = ClientSecretCredential(
                tenant_id=os.environ["TENANT_ID"],
                client_id=os.environ["CLIENT_ID"],
                client_secret=os.environ["CLIENT_SECRET"],
                disable_instance_discovery=True,
            )
            access_token = credential.get_token(
                os.environ["TOKEN_SCOPE"],
                tenant_id=os.environ["TENANT_ID"],
            )
            os.environ["OPENAI_API_KEY"] = access_token.token
        except KeyError:
            raise ValueError(
                "Error authenticating with OpenAI\n Without explicit `OPENAI_API_KEY`, you need to provide ['TENANT_ID', 'CLIENT_ID', 'CLIENT_SECRET']"
            ) from None

    def _predict(
        self, program: guidance._program.Program, **kwargs
    ) -> Union[str, Dict]:
        """
        Runs a guidance program, and returns the output variables.
        """
        # Initialize guidance
        guidance.llm = guidance.llms.OpenAI(
            # tiktoken.encoding_for_model has different name for gpt-35-turbo
            model=self.model_name_or_path,
            api_type=os.getenv("API_TYPE"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            api_base=os.getenv("OPENAI_API_BASE"),
            max_retries=50,
            chat_mode=self.model_name_or_path in OPENAI_CHAT_LLM,
        )
        # Override the caller, so we can add the 'engine' arg
        guidance.llm.caller = library_call_with_engine
        time.time()
        program = program(**kwargs)
        time.time()
        guidance.llm = None
        return {k: v for k, v in program.variables().items() if k != "llm"}
