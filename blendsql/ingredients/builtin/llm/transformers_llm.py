import logging
import time
from typing import Dict, Union

import guidance
from attr import attrs, attrib

from blendsql.ingredients.builtin.llm.llm import LLM

logging.getLogger("guidance").setLevel(logging.CRITICAL)


@attrs(auto_detect=True)
class TransformersLocalLLM(LLM):
    """Class for Transformers Local LLM."""

    model_name_or_path: str = attrib()
    gen_kwargs: dict = attrib(init=False)

    def __attrs_post_init__(self):
        self.gen_kwargs = {}

    def _predict(self, program: str, **kwargs) -> Union[str, Dict]:
        """
        Runs a guidance program, and returns the output variables.
        """
        program = guidance(program)
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
