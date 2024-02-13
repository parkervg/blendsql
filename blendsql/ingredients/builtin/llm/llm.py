from abc import ABC, abstractmethod
from typing import Dict, Union
import guidance
from attr import attrib, attrs


@attrs
class LLM(ABC):
    """Abstract class for all LLM endpoints"""

    model_name_or_path: str = attrib()
    gen_kwargs: dict = attrib()

    num_llm_calls: int = 0

    def predict(self, program: guidance._program.Program, **kwargs) -> Union[str, Dict]:
        self.num_llm_calls += 1
        return self._predict(program, **kwargs)

    @abstractmethod
    def _predict(self, program: guidance._program.Program, **kwargs) -> Union[str, Dict]:
        ...
