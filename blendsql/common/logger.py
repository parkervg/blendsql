import logging
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter
from rich.console import Console
from rich.markup import escape

console = Console(force_terminal=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=False,
            show_level=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            highlighter=NullHighlighter(),
        )
        handler.setLevel(level)
        logger.addHandler(handler)

    return logger


logger = get_logger("blendsql", level=logging.DEBUG)


# Logging utilities
class Color:
    @staticmethod
    def update(s: str):
        return f"[cyan]{s}[/cyan]"

    @staticmethod
    def quiet_update(s: str):
        return f"[grey53]`{escape(s)}`[/grey53]"

    @staticmethod
    def light_update(s: str):
        return f"[light_cyan3]{escape(s)}[/light_cyan3]"

    @staticmethod
    def warning(s: str):
        return f"[yellow]{escape(s)}[/yellow]"

    @staticmethod
    def light_warning(s: str):
        return f"[light_yellow3]{escape(s)}[/light_yellow3]"

    @staticmethod
    def error(s: str):
        return f"[red]{escape(s)}[/red]"

    @staticmethod
    def success(s: str):
        return f"[green]{escape(s)}[/green]"

    @staticmethod
    def model_or_data_update(s: str):
        return f"[magenta]{escape(s)}[/magenta]"

    @staticmethod
    def sql(s: str):
        return f"[cornflower_blue]`{escape(s)}`[/cornflower_blue]"

    @staticmethod
    def quiet_sql(s: str):
        return f"[grey53]`{escape(s)}`[/grey53]"
