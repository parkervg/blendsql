import logging
import time
from typing import Dict, Union

import guidance

from .._llm import LLM

logging.getLogger("guidance").setLevel(logging.CRITICAL)


class TransformersLocalLLM(LLM):
    """Class for Transformers Local LLM."""

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(
            model_name_or_path=model_name_or_path,
            requires_config=False,
            encoding=None,
            **kwargs
        )

    def _predict(
        self, program: guidance._program.Program, **kwargs
    ) -> Union[str, Dict]:
        """
        Runs a guidance program, and returns the output variables.
        """
        # Initialize guidance
        guidance.llm = guidance.llms.Transformers(
            model=self.model_name_or_path if "model" not in kwargs else kwargs["model"],
            # if tokenizer object has not been given in kwargs then set it to None as guidance loads it using
            # model_name_or_path
            tokenizer=None if "tokenizer" not in kwargs else kwargs["tokenizer"],
        )
        time.time()
        program = program(**kwargs)
        time.time()
        guidance.llm = None
        return {k: v for k, v in program.variables().items() if k != "llm"}
