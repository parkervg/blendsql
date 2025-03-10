import os
import importlib.util
from itertools import cycle
from typing import Optional
from .._model import RemoteModel, ModelObj

DEFAULT_CONFIG = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 1,
    "max_output_tokens": 2048,
}

_has_genai = importlib.util.find_spec("google.generativeai") is not None

class GeminiLLM(RemoteModel):
    """Class for Google's Gemini API.
    
    Args:
        model_name_or_path: Name of the Gemini model (e.g., "gemini-pro")
        env: Path to directory of .env file containing GEMINI_API_KEYS
        config: Optional configuration for generation
        caching: Whether to cache responses
    """
    
    def __init__(
        self,
        model_name_or_path: str = "gemini-2.0-flash-exp",
        env: str = ".",
        config: Optional[dict] = None,
        caching: bool = True,
        **kwargs,
    ):
        if not _has_genai:
            raise ImportError(
                "Please install the Google Generative AI package with: pip install google-generativeai"
            )
        
        if config is None:
            config = {}
            
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer=None,
            requires_config=True,
            refresh_interval_min=30,
            load_model_kwargs=config | DEFAULT_CONFIG,
            env=env,
            caching=caching,
            **kwargs,
        )

    def _load_model(self) -> ModelObj:
        import google.generativeai as genai
        
        
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Please provide a GEMINI_API_KEY in your .env file"
            )
        genai.configure(api_key=self.api_key)
        
        
        model = genai.GenerativeModel(self.model_name_or_path)
        return model
