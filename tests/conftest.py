import os
import pytest
from guidance.chat import ChatMLTemplate
from dotenv import load_dotenv
import torch
from dataclasses import dataclass, field
import pandas as pd

from blendsql.db import Database
from blendsql.models import (
    TransformersLLM,
    TransformersVisionModel,
    Model,
    LlamaCpp,
)
from litellm.exceptions import APIConnectionError

from blendsql.ingredients import LLMQA, LLMMap, LLMJoin
from blendsql.ingredients.builtin import DEFAULT_MAP_FEW_SHOT
from blendsql.configure import set_default_max_tokens

set_default_max_tokens(20)
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
        "config": {
            "device_map": "cuda",
            # "torch_dtype": "bfloat16"
        },
        "requires_cuda": True,
    },
    {
        "name": "smollm",
        "class": TransformersLLM,
        "path": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "config": {
            "device_map": "cuda" if torch.cuda.is_available() else "cpu",
            # "torch_dtype": "bfloat16"
        },
        "requires_cuda": False,
    },
    {
        "name": "llamacpp",
        "class": LlamaCpp,
        "path": "bartowski/SmolLM2-360M-Instruct-GGUF",
        "filename": "SmolLM2-360M-Instruct-Q6_K.gguf",
        "config": {"n_gpu_layers": -1, "n_ctx": 8000},
        "requires_cuda": True,
    },
]

UNCONSTRAINED_MODEL_CONFIGS = [
    # {
    #     "name": "ollama",
    #     "class": LiteLLM,
    #     "path": "ollama/qwen:0.5b",
    #     "requires_api": False,
    # },
    # {
    #     "name": "openai",
    #     "class": LiteLLM,
    #     "path": "openai/gpt-4o-mini",
    #     "requires_env": "OPENAI_API_KEY",
    # },
    # {
    #     "name": "anthropic",
    #     "class": LiteLLM,
    #     "path": "anthropic/claude-3-haiku-20240307",
    #     "requires_env": "ANTHROPIC_API_KEY",
    # },
    # {
    #     "name": "gemini",
    #     "class": LiteLLM,
    #     "path": "gemini/gemini-2.0-flash-exp",
    #     "requires_env": "GEMINI_API_KEY",
    # },
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


@dataclass
class TimingCollector:
    """Collects timing data across all tests."""

    records: list[dict] = field(default_factory=list)

    def add(self, test_name: str, db_name: str, blendsql_time: float, sql_time: float):
        self.records.append(
            {
                "test_name": test_name,
                "db_name": db_name,
                "blendsql_time_ms": blendsql_time * 1000,
                "sql_time_ms": sql_time * 1000,
                "ratio": blendsql_time / sql_time if sql_time > 0 else None,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def print_summary(self):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        df = self.to_dataframe()

        if df.empty:
            console.print("[yellow]No timing data collected.[/yellow]")
            return

        # Create a table for each database
        for db_name in df["db_name"].unique():
            db_df = df[df["db_name"] == db_name]

            table = Table(
                title=f"Timing Results — {db_name}",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Test Name", style="white")
            table.add_column("BlendSQL (ms)", justify="right", style="magenta")
            table.add_column("SQL (ms)", justify="right", style="green")
            table.add_column("Ratio", justify="right", style="yellow")

            for _, row in db_df.iterrows():
                ratio_str = (
                    f"{row['ratio']:.2f}x" if row["ratio"] is not None else "N/A"
                )
                table.add_row(
                    row["test_name"],
                    f"{row['blendsql_time_ms']:.2f}",
                    f"{row['sql_time_ms']:.2f}",
                    ratio_str,
                )

            console.print()
            console.print(table)

            # Summary for this database
            summary = Text()
            summary.append(f"Total BlendSQL time: ", style="bold")
            summary.append(
                f"{db_df['blendsql_time_ms'].sum():.2f} ms\n", style="magenta"
            )
            summary.append(f"Total SQL time:      ", style="bold")
            summary.append(f"{db_df['sql_time_ms'].sum():.2f} ms\n", style="green")
            summary.append(f"Average ratio:       ", style="bold")
            summary.append(f"{db_df['ratio'].mean():.2f}x", style="yellow")

            console.print(
                Panel(summary, title=f"{db_name} Summary", border_style="blue")
            )

        # Overall summary across all databases
        if df["db_name"].nunique() > 1:
            console.print()
            overall = Text()
            overall.append("Overall Totals\n", style="bold underline")
            overall.append(f"Total BlendSQL time: ", style="bold")
            overall.append(f"{df['blendsql_time_ms'].sum():.2f} ms\n", style="magenta")
            overall.append(f"Total SQL time:      ", style="bold")
            overall.append(f"{df['sql_time_ms'].sum():.2f} ms\n", style="green")
            overall.append(f"Average ratio:       ", style="bold")
            overall.append(f"{df['ratio'].mean():.2f}x", style="yellow")

            console.print(
                Panel(overall, title="All Databases", border_style="bold white")
            )


# Global collector instance
_timing_collector = TimingCollector()


@pytest.fixture
def timing_collector(request) -> TimingCollector:
    """Fixture that provides the timing collector with current test name."""
    _timing_collector._current_test = request.node.name
    return _timing_collector


def pytest_sessionfinish(session, exitstatus):
    """Hook that runs after all tests complete."""
    _timing_collector.print_summary()
