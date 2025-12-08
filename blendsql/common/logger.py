import logging
from typing import Callable
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


class LazyLogger(logging.LoggerAdapter):
    """Logger that only evaluates callables when logging is enabled"""

    def __init__(self, logger: logging.Logger):
        super().__init__(logger, {})

    def log(self, level: int, msg: str | Callable, *args, **kwargs):
        if self.isEnabledFor(level):
            if callable(msg):
                msg = msg()
            super().log(level, msg, *args, **kwargs)

    @property
    def level(self):
        """Expose level property from underlying logger"""
        return self.logger.level

    def debug(self, msg: str | Callable, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str | Callable, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str | Callable, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str | Callable, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)


logger = LazyLogger(get_logger("blendsql", level=logging.DEBUG))


# Logging utilities
class Color:
    @staticmethod
    def update(s: str):
        return f"[cyan]{s}[/cyan]"

    @staticmethod
    def quiet_update(s: str):
        return f"[grey53]{escape(s)}[/grey53]"

    @staticmethod
    def light_update(s: str):
        return f"[light_sky_blue1]{escape(s)}[/light_sky_blue1]"

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
        return f"[sky_blue1]`{escape(s)}`[/sky_blue1]"

    @staticmethod
    def horizontal_line(char: str = "─", width: int = None):
        if width is None:
            console = Console()
            width = console.width

        line = char * width
        return f"[black]{line}[/black]"
