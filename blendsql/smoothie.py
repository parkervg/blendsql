from dataclasses import dataclass, field
from typing import Iterable, Type
import polars as pl
from functools import cached_property

from blendsql.ingredients import Ingredient


@dataclass
class SmoothieMeta:
    # Number of values passed to a Map/Join/QA ingredient
    num_values_passed: int = field()
    num_generation_calls: int = field()  # Number of generation calls made to the model
    prompt_tokens: int = field()
    completion_tokens: int = field()
    prompts: list[dict] = field()  # Log of prompts submitted to model
    raw_prompts: list[str] = field()
    ingredients: Iterable[Type[Ingredient]] = field()
    query: str = field()
    db_url: str = field()
    db_type: str = field()
    contains_ingredient: bool = field(default=True)
    process_time_seconds: float = field(default="N.A.")


@dataclass
class Smoothie:
    _df: pl.DataFrame = field()
    meta: SmoothieMeta = field()

    def __post_init__(self):
        if isinstance(self._df, pl.LazyFrame):
            self._df = self._df.collect()

    @cached_property
    def df(self):
        return self._df.to_pandas()

    @cached_property
    def pl(self):
        return self._df

    def print_summary(self):
        from rich.console import Console, Group
        from rich.align import Align
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.columns import Columns
        from rich.box import ROUNDED
        from blendsql.parse.dialect import get_dialect
        import sqlglot

        console = Console(force_terminal=True)

        # Create SQL syntax highlighted query
        formatted_query = sqlglot.transpile(
            self.meta.query, read=get_dialect(self.meta.db_type), pretty=True
        )[0]
        query_syntax = Syntax(
            formatted_query, "sql", theme="default", dedent=True, word_wrap=True
        )

        # Create summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Time (s)")
        table.add_column("# Generation Calls")
        table.add_column("# DB Values Passed")
        table.add_column("Prompt Tokens")
        table.add_column("Completion Tokens")

        time_value = (
            str(self.meta.process_time_seconds)
            if hasattr(self.meta, "process_time_seconds")
            else "N.A."
        )

        table.add_row(
            time_value,
            str(self.meta.num_generation_calls),
            str(self.meta.num_values_passed),
            str(self.meta.prompt_tokens),
            str(self.meta.completion_tokens),
        )

        # Create side-by-side panels for query and result
        query_panel = Panel(query_syntax, title="Query", border_style="blue")

        def df_to_table(df: pl.DataFrame, title: str = "") -> Table:
            table = Table(title=title, show_header=True, show_lines=True)

            # Add columns
            for col in df.columns:
                table.add_column(str(col))

            # Add rows
            for row in df.iter_rows():
                table.add_row(*[str(v) for v in row])

            return table

        total_rows = len(self.pl)
        num_row_limit = 5
        if total_rows > num_row_limit:
            df_to_display = self.pl.head(num_row_limit)
            result_title = f"Result ({num_row_limit} out of {total_rows} Rows)"
        else:
            df_to_display = self.pl
            result_title = f"Result"
        result_panel = Panel(
            df_to_table(df_to_display),
            title=result_title,
            border_style="blue",
        )

        content = Group(Columns([query_panel, result_panel], equal=True), table)
        boxed = Panel(
            content,
            box=ROUNDED,  # box style: ROUNDED, DOUBLE, HEAVY, SIMPLE, etc.
            padding=(1, 2),  # (vertical, horizontal) padding inside
        )
        console.print(Align.center(boxed))

    def __str__(self):
        self.print_summary()
