import logging

from guidance.models import Transformers

from .._llm import LLM

logging.getLogger("guidance").setLevel(logging.CRITICAL)


class TransformersLLM(LLM):
    """Class for Transformers Local LLM."""

    def __init__(self, model_name_or_path: str, **kwargs):
        self._setup()
        try:
            import transformers
        except ImportError:
            raise Exception(
                "Please install transformers with `pip install transformers`!"
            ) from None
        super().__init__(
            modelclass=Transformers,
            model_name_or_path=model_name_or_path,
            requires_config=False,
            tokenizer=transformers.AutoTokenizer.from_pretrained(model_name_or_path),
            **kwargs
        )
