from dataclasses import dataclass, field
from typing import Iterable, Type
import pandas as pd

from blendsql.ingredients import Ingredient
from blendsql.common.utils import tabulate
from blendsql.db.utils import truncate_df_content


class PrettyDataFrame(pd.DataFrame):
    def __str__(self):
        truncated = truncate_df_content(self, 50)
        try:
            return tabulate(truncated)
        except:
            return str(truncated)

    def __repr__(self):
        truncated = truncate_df_content(self, 50)
        try:
            return tabulate(truncated)
        except:
            return str(truncated)


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
    df: pd.DataFrame = field()
    meta: SmoothieMeta = field()

    def __post_init__(self):
        self.df = PrettyDataFrame(self.df)

    def print_summary(self):
        from rich.console import Console, Group
        from rich.align import Align
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.markdown import Markdown
        from rich.table import Table
        from rich.columns import Columns
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
        total_rows = len(self.df)
        num_row_limit = 5
        if total_rows > num_row_limit:
            df_to_display = self.df.head(num_row_limit)
            result_title = f"Result ({num_row_limit} out of {total_rows} Rows)"
        else:
            df_to_display = self.df
            result_title = f"Result"
        result_panel = Panel(
            Markdown(df_to_display.to_markdown(index=False)),
            title=result_title,
            border_style="blue",
        )

        content = Group(Columns([query_panel, result_panel], equal=True), table)

        console.print(Align.center(content))

    def __str__(self):
        return self.summary()
