import os

from guidance.chat import ChatMLTemplate
from dotenv import load_dotenv

from blendsql.db import Database
from blendsql.models import (
    TransformersLLM,
    LiteLLM,
    Model,
)
from litellm.exceptions import APIConnectionError

from blendsql import LLMQA, LLMMap, LLMJoin
from blendsql.ingredients.builtin import DEFAULT_MAP_FEW_SHOT

load_dotenv()

# Disable MPS for test cases
# This causes an 'MPS backend out of memory' error on github actions
os.environ["HAYSTACK_MPS_ENABLED"] = "false"


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Database):
        return val.__class__.__name__
    elif isinstance(val, Model):
        return val.model_name_or_path
    # return None to let pytest handle the formatting
    return None


def pytest_generate_tests(metafunc):
    if "model" in metafunc.fixturenames:
        model_list = [
            TransformersLLM(
                "HuggingFaceTB/SmolLM-135M-Instruct",
                caching=False,
                config={"chat_template": ChatMLTemplate, "device_map": "cpu"},
            )
        ]

        # Ollama check
        try:
            model = LiteLLM("ollama/qwen:0.5b", caching=False)
            model.generate(messages_list=[[{"role": "user", "content": "hello"}]])
            model_list.append(model)
        except APIConnectionError:
            print("Skipping OllamaLLM, as Ollama server is not running...")

        # OpenAI check
        if os.getenv("OPENAI_API_KEY") is not None:
            model_list.append(LiteLLM("openai/gpt-4o", caching=False))

        # Anthropic check
        if os.getenv("ANTHROPIC_API_KEY") is not None:
            model_list.append(
                LiteLLM("anthropic/claude-3-5-sonnet-20241022", caching=False)
            )

        # Gemini check
        if os.getenv("GEMINI_API_KEY") is not None:
            model_list.append(LiteLLM("gemini/gemini-2.0-flash-exp", caching=False))

        # Azure Phi check
        # if all(os.getenv(k) is not None for k in ["AZURE_PHI_KEY", "AZURE_PHI_URL"]):
        #     model_list.append(AzurePhiModel(caching=False))

        metafunc.parametrize("model", model_list)

    if "ingredients" in metafunc.fixturenames:
        ingredient_sets = [
            {LLMQA, LLMMap, LLMJoin},
            {
                LLMQA.from_args(
                    k=1,
                ),
                LLMMap.from_args(
                    few_shot_examples=[
                        *DEFAULT_MAP_FEW_SHOT,
                        {
                            "question": "What school type is this?",
                            "mapping": {
                                "A. L. Conner Elementary": "Traditional",
                                "Abraxas Continuation High": "Continuation School",
                            },
                        },
                    ],
                    k=2,
                    batch_size=3,
                ),
                LLMJoin.from_args(
                    k=2,
                    model=TransformersLLM(
                        "HuggingFaceTB/SmolLM-135M-Instruct",
                        caching=False,
                        config={"chat_template": ChatMLTemplate, "device_map": "cpu"},
                    ),
                ),
            },
        ]
        metafunc.parametrize("ingredients", ingredient_sets)
