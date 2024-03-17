import logging
import os
from guidance.models import OpenAI, AzureOpenAI, Model
import tiktoken

from .._model import Model

logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("guidance").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def openai_setup() -> None:
    """Setup helper for AzureOpenAI and OpenAI models."""
    if all(
        x is not None
        for x in {
            os.getenv("TENANT_ID"),
            os.getenv("CLIENT_ID"),
            os.getenv("CLIENT_SECRET"),
        }
    ):
        try:
            from azure.identity import ClientSecretCredential
        except ImportError:
            raise ValueError(
                "Found ['TENANT_ID', 'CLIENT_ID', 'CLIENT_SECRET'] in .env file, using Azure OpenAI\nIn order to use Azure OpenAI, run `pip install azure-identity`!"
            ) from None
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


class AzureOpenaiLLM(Model):
    """Class for Azure OpenAI Model API.

    Args:
        model_name_or_path: Name of the Azure deployment to use
        env: Path to directory of .env file, or to the file itself to load as a dotfile.
            Should either contain the variable `OPENAI_API_KEY`,
                or all of `TENANT_ID`, `CLIENT_ID`, `CLIENT_SECRET`
    """

    def __init__(self, model_name_or_path: str, env: str = None, **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tiktoken.encoding_for_model(model_name_or_path),
            requires_config=True,
            refresh_interval_min=30,
            env=env,
            **kwargs
        )

    def _load_model(self) -> Model:
        return AzureOpenAI(
            self.model_name_or_path,
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            azure_deployment=os.getenv("API_VERSION"),
            echo=False,
        )

    def _setup(self, **kwargs) -> None:
        openai_setup()
        self.model = self._load_model()


class OpenaiLLM(Model):
    """Class for OpenAI Model API.

    Args:
        model_name_or_path: Name of the OpenAI model to use
        env: Path to directory of .env file, or to the file itself to load as a dotfile.
            Should contain the variable `OPENAI_API_KEY`
    """

    def __init__(self, model_name_or_path: str, env: str = None, **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tiktoken.encoding_for_model(model_name_or_path),
            requires_config=True,
            refresh_interval_min=30,
            env=env,
            **kwargs
        )

    def _load_model(self) -> Model:
        return OpenAI(
            self.model_name_or_path, api_key=os.getenv("OPENAI_API_KEY"), echo=False
        )

    def _setup(self, **kwargs) -> None:
        openai_setup()
        self.model = self._load_model()
