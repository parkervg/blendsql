import logging
import os
from guidance.models import OpenAI
import tiktoken

from .._llm import LLM

logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("guidance").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

class OpenaiLLM(LLM):
    """Class for OpenAI LLM API."""

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(
            modelclass=OpenAI,
            model_name_or_path=model_name_or_path,
            tokenizer=tiktoken.encoding_for_model(model_name_or_path),
            requires_config=True,
            refresh_interval_min=30,
            **kwargs
        )

    def _setup(self, **kwargs) -> None:
        if all(x is not None for x in {os.getenv("TENANT_ID"), os.getenv("CLIENT_ID"), os.getenv("CLIENT_SECRET")}):
            try:
                from azure.identity import ClientSecretCredential
            except ImportError:
                raise ValueError("In order to use Azure OpenAI, run `pip install azure-identity`!")
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
        elif os.getenv("OPENAI_API_KEY") is not None:
            pass
        else:
            raise ValueError(
                "Error authenticating with OpenAI\n Without explicit `OPENAI_API_KEY`, you need to provide ['TENANT_ID', 'CLIENT_ID', 'CLIENT_SECRET']"
            ) from None