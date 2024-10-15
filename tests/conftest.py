import os

import httpx
from guidance.chat import ChatMLTemplate
from dotenv import load_dotenv

from blendsql.db import Database
from blendsql.models import TransformersLLM, OllamaLLM, OpenaiLLM, AnthropicLLM, Model
from blendsql import LLMQA, LLMMap, LLMJoin
from blendsql.ingredients.builtin import DEFAULT_MAP_FEW_SHOT
from blendsql.ingredients.builtin import DEFAULT_QA_FEW_SHOT

load_dotenv()

# Disable MPS for test cases
# This causes an 'MPS backend out of memory' error on github actions
os.environ["HAYSTACK_MPS_ENABLED"] = "false"


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, (Database, Model)):
        return val.__class__.__name__
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
            model = OllamaLLM("qwen:0.5b", caching=False)
            model.model_obj(messages=[{"role": "user", "content": "hello"}])
            model_list.append(model)
        except httpx.ConnectError:
            print("Skipping OllamaLLM, as Ollama server is not running...")

        # OpenAI check
        if os.getenv("OPENAI_API_KEY") is not None:
            model_list.append(OpenaiLLM("gpt-3.5-turbo", caching=False))

        # Anthropic check
        if os.getenv("ANTHROPIC_API_KEY") is not None:
            model_list.append(AnthropicLLM("claude-3-haiku-20240307", caching=False))

        metafunc.parametrize("model", model_list)

    if "ingredients" in metafunc.fixturenames:
        ingredient_sets = [
            {LLMQA, LLMMap, LLMJoin},
            {
                LLMQA.from_args(few_shot_examples=DEFAULT_QA_FEW_SHOT, k=1),
                LLMMap.from_args(
                    model=TransformersLLM(
                        "HuggingFaceTB/SmolLM-135M-Instruct",
                        caching=False,
                        config={"chat_template": ChatMLTemplate, "device_map": "cpu"},
                    ),
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
                    batch_size=3,
                ),
                LLMJoin.from_args(
                    k=2,
                ),
            },
        ]
        metafunc.parametrize("ingredients", ingredient_sets)
