import importlib.util
import os
from typing import Optional

from .._model import LocalModel, ModelObj

_has_transformers = importlib.util.find_spec("transformers") is not None


class AzurePhiModel(LocalModel):
    """Class for Azure Phi model with guidance serverside integration.

    https://github.com/guidance-ai/guidance?tab=readme-ov-file#azure-ai

    Args:
        env: Path to directory of .env file, or to the file itself to load as a dotfile.
            Should either contain `AZURE_PHI_KEY` and `AZURE_PHI_URL`
        caching: Bool determining whether we access the model's cache

    Examples:
        Given the following `.env` file in the directory above current:
        ```text
        AZURE_PHI_KEY=...
        AZURE_PHI_URL=...
        ```
        ```python
        from blendsql.models import AzurePhiModel

        model = AzurePhiModel(
            env="..",
        )
        ```
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if not _has_transformers:
            raise ImportError(
                "Please install transformers with `pip install transformers`!"
            ) from None
        import transformers

        transformers.logging.set_verbosity_error()
        if config is None:
            config = {}

        super().__init__(
            model_name_or_path="microsoft/Phi-3.5-mini-instruct",
            requires_config=True,
            tokenizer=transformers.AutoTokenizer.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct"
            ),
            load_model_kwargs=config,
            caching=caching,
            **kwargs,
        )

    def _load_model(self) -> ModelObj:
        # https://huggingface.co/blog/how-to-generate
        from guidance.models import AzureGuidance

        assert all(os.getenv(k) is not None for k in ["AZURE_PHI_KEY", "AZURE_PHI_URL"])
        lm = AzureGuidance(
            f"{os.getenv('AZURE_PHI_URL')}/guidance#auth={os.getenv('AZURE_PHI_KEY')}",
            echo=False,
            # **self.load_model_kwargs,
        )
        return lm
