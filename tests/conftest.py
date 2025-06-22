import os
import pytest
from guidance.chat import ChatMLTemplate
from dotenv import load_dotenv
import torch

from blendsql.db import Database
from blendsql.models import (
    TransformersLLM,
    LiteLLM,
    TransformersVisionModel,
    Model,
)
from litellm.exceptions import APIConnectionError

from blendsql.ingredients import LLMQA, LLMMap, LLMJoin
from blendsql.ingredients.builtin import DEFAULT_MAP_FEW_SHOT

load_dotenv()


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Database):
        return val.__class__.__name__
    elif isinstance(val, Model):
        return val.model_name_or_path
    # return None to let pytest handle the formatting
    return None


# Define the model configurations
CONSTRAINED_MODEL_CONFIGS = [
    {
        "name": "llama",
        "class": TransformersLLM,
        "path": "meta-llama/Llama-3.2-1B-Instruct",
        "config": {"device_map": "cuda"},
        "requires_cuda": True,
    },
    {
        "name": "smollm",
        "class": TransformersLLM,
        "path": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "config": {"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
        "requires_cuda": False,
    },
    # {
    #     "name": "llamacpp",
    #     "class": LlamaCpp,
    #     "path": "QuantFactory/SmolLM-135M-GGUF",
    #     "filename": "SmolLM-135M.Q2_K.gguf",
    #     "config": {"n_gpu_layers": -1},
    #     "requires_cuda": True,
    # },
]

UNCONSTRAINED_MODEL_CONFIGS = [
    {
        "name": "ollama",
        "class": LiteLLM,
        "path": "ollama/qwen:0.5b",
        "requires_api": False,
    },
    {
        "name": "openai",
        "class": LiteLLM,
        "path": "openai/gpt-4o-mini",
        "requires_env": "OPENAI_API_KEY",
    },
    {
        "name": "anthropic",
        "class": LiteLLM,
        "path": "anthropic/claude-3-haiku-20240307",
        "requires_env": "ANTHROPIC_API_KEY",
    },
    {
        "name": "gemini",
        "class": LiteLLM,
        "path": "gemini/gemini-2.0-flash-exp",
        "requires_env": "GEMINI_API_KEY",
    },
]


def get_available_constrained_models():
    available_models = []
    for config in CONSTRAINED_MODEL_CONFIGS:
        if config["requires_cuda"] and not torch.cuda.is_available():
            continue
        args = (
            (config["filename"], config["path"])
            if "filename" in config
            else (config["path"],)
        )
        model = config["class"](*args, config=config.get("config", {}), caching=False)
        available_models.append(pytest.param(model, id=config["name"]))
    return available_models


def get_available_unconstrained_models():
    available_models = []
    for config in UNCONSTRAINED_MODEL_CONFIGS:
        if config.get("requires_env") and os.getenv(config["requires_env"]) is None:
            continue

        model = config["class"](config["path"], caching=False)

        # Test Ollama connectivity
        if config["name"] == "ollama":
            try:
                model.generate(messages_list=[[{"role": "user", "content": "hello"}]])
            except APIConnectionError:
                print(f"Skipping {config['name']}, as server is not running...")
                continue
        available_models.append(pytest.param(model, id=config["name"]))
    return available_models


def get_available_models():
    return get_available_constrained_models() + get_available_unconstrained_models()


@pytest.fixture(params=get_available_constrained_models(), scope="session")
def constrained_model(request):
    return request.param


@pytest.fixture(params=get_available_unconstrained_models(), scope="session")
def unconstrained_model(request):
    return request.param


@pytest.fixture(params=get_available_models(), scope="session")
def model(request):
    """Return all models"""
    return request.param


@pytest.fixture(scope="session")
def vision_model() -> TransformersVisionModel:
    return TransformersVisionModel(
        "Salesforce/blip-image-captioning-base",
        caching=False,
        config={"device_map": "cuda" if torch.cuda.is_available() else "cpu"},
    )


@pytest.fixture(autouse=True)
def cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


def pytest_generate_tests(metafunc):
    if "ingredients" in metafunc.fixturenames:
        ingredient_sets = [
            {LLMQA, LLMMap, LLMJoin},
            {
                LLMQA.from_args(
                    num_few_shot_examples=1,
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
                    num_few_shot_examples=2,
                    batch_size=3,
                ),
                LLMJoin.from_args(
                    num_few_shot_examples=2,
                    model=TransformersLLM(
                        "HuggingFaceTB/SmolLM-135M-Instruct",
                        config={
                            "chat_template": ChatMLTemplate,
                            "device_map": "cpu",
                        },
                        caching=False,
                    ),
                ),
            },
        ]
        metafunc.parametrize("ingredients", ingredient_sets)
