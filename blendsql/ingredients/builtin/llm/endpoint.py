from abc import ABC, abstractmethod
from typing import Dict, Union
import guidance
from attr import attrib, attrs


@attrs
class Endpoint(ABC):
    """Abstract class for all LLM endpoints"""

    endpoint_name: str = attrib()
    gen_kwargs: dict = attrib()

    @abstractmethod
    def predict(self, program: guidance._program.Program, **kwargs) -> Union[str, Dict]:
        ...
