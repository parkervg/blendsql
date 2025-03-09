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
        env: Path to directory of .env file containing GOOGLE_API_KEYS
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

        
        self.api_keys = []
        try:
            with open(os.path.join(env, '.env'), 'r') as f:
                for line in f:
                    if line.startswith('GOOGLE_API_KEY='):
                        self.api_keys.append(line.split('=')[1].strip())
            if not self.api_keys:
                raise ValueError("No GOOGLE_API_KEY found in .env file")
        except FileNotFoundError:
            raise ValueError(f"Could not find .env file in {env}")
        except Exception as e:
            raise ValueError(f"Error reading .env file: {str(e)}")

        self.api_key_cycle = cycle(self.api_keys)

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
        
        
        api_key = next(self.api_key_cycle)
        genai.configure(api_key=api_key)
        
        
        model = genai.GenerativeModel(self.model_name_or_path)
        return model

    def _setup(self, **kwargs) -> None:
        if not self.api_keys:
            raise ValueError(
                "Error authenticating with Google Gemini API\n"
                "You need to provide at least one GOOGLE_API_KEY in your .env file"
            )
