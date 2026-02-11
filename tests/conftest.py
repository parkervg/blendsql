import pytest
from dataclasses import dataclass, field
import pandas as pd

from blendsql.db import Database
from blendsql.models import VLLM
from blendsql.models.model_base import ModelBase
from blendsql.configure import set_deterministic
from blendsql.ingredients import LLMQA, LLMMap, LLMJoin
from blendsql.ingredients.builtin import DEFAULT_MAP_FEW_SHOT

set_deterministic(True)


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, Database):
        return val.__class__.__name__
    elif isinstance(val, ModelBase):
        return val.model_name_or_path
    # return None to let pytest handle the formatting
    return None


def get_available_models():
    return [
        VLLM(
            model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
            base_url="http://localhost:8000/v1",
        )
    ]


@pytest.fixture(params=get_available_models(), scope="session")
def model(request):
    """Return all models"""
    return request.param


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
