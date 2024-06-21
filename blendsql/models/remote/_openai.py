import os
import importlib.util
from outlines.models.openai import openai, azure_openai, OpenAIConfig

from .._model import RemoteModel, ModelObj
from typing import Optional

DEFAULT_CONFIG = OpenAIConfig(temperature=0.0)

_has_openai = importlib.util.find_spec("openai") is not None


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


class AzureOpenaiLLM(RemoteModel):
    """Class for Azure OpenAI Model API.

    Args:
        model_name_or_path: Name of the Azure deployment to use
        env: Path to directory of .env file, or to the file itself to load as a dotfile.
            Should either contain the variable `OPENAI_API_KEY`,
            or all of `TENANT_ID`, `CLIENT_ID`, `CLIENT_SECRET`
        config: Optional outlines.models.openai.OpenAIConfig to use in loading model
        caching: Bool determining whether we access the model's cache

    Examples:
        Given the following `.env` file in the directory above current:
        ```text
        TENANT_ID=my_tenant_id
        CLIENT_ID=my_client_id
        CLIENT_SECRET=my_client_secret
        ```
        ```python
        from blendsql.models import AzureOpenaiLLM
        from outlines.models.openai import OpenAIConfig

        model = AzureOpenaiLLM(
            "gpt-3.5-turbo",
            env="..",
            config=OpenAIConig(
                temperature=0.7
            )
        )
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        env: str = ".",
        config: Optional[OpenAIConfig] = None,
        caching: bool = True,
        **kwargs
    ):
        if not _has_openai:
            raise ImportError(
                "Please install openai>=1.0.0 with `pip install openai>=1.0.0`!"
            ) from None

        import tiktoken

        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tiktoken.encoding_for_model(model_name_or_path),
            requires_config=True,
            refresh_interval_min=30,
            load_model_kwargs=kwargs | {"config": config or DEFAULT_CONFIG},
            env=env,
            caching=caching,
            **kwargs
        )

    def _load_model(self, config: Optional[OpenAIConfig] = None) -> ModelObj:
        return azure_openai(
            self.model_name_or_path,
            config=config,
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )  # type: ignore

    def _setup(self, **kwargs) -> None:
        openai_setup()


class OpenaiLLM(RemoteModel):
    """Class for OpenAI Model API.

    Args:
        model_name_or_path: Name of the OpenAI model to use
        env: Path to directory of .env file, or to the file itself to load as a dotfile.
            Should contain the variable `OPENAI_API_KEY`
        config: Optional outlines.models.openai.OpenAIConfig to use in loading model
        caching: Bool determining whether we access the model's cache

    Examples:
        Given the following `.env` file in the directory above current:
        ```text
        OPENAI_API_KEY=my_api_key
        ```
        ```python
        from blendsql.models import OpenaiLLM
        from outlines.models.openai import OpenAIConfig

        model = AzureOpenaiLLM(
            "gpt-3.5-turbo",
            env="..",
            config=OpenAIConig(
                temperature=0.7
            )
        )
        ```
    """

    def __init__(
        self,
        model_name_or_path: str,
        env: str = ".",
        config: Optional[OpenAIConfig] = None,
        caching: bool = True,
        **kwargs
    ):
        if not _has_openai:
            raise ImportError(
                'Please install openai>=1.0.0 and tiktoken with `pip install "openai>=1.0.0" tiktoken`!'
            ) from None

        import tiktoken

        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=tiktoken.encoding_for_model(model_name_or_path),
            requires_config=True,
            refresh_interval_min=30,
            load_model_kwargs={"config": config or DEFAULT_CONFIG},
            env=env,
            caching=caching,
            **kwargs
        )

    def _load_model(self, config: Optional[OpenAIConfig] = None) -> ModelObj:
        return openai(
            self.model_name_or_path, config=config, api_key=os.getenv("OPENAI_API_KEY")
        )  # type: ignore

    def _setup(self, **kwargs) -> None:
        openai_setup()
