import logging
from typing import Callable
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter
from rich.console import Console
from rich.markup import escape

console = Console(force_terminal=True, width=120)


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
    in_block = False
    prefix = "    "

    @classmethod
    def _apply_prefix(cls, s: str, ignore_prefix=False):
        if cls.in_block:
            return cls.prefix + s
        return s

    @staticmethod
    def update(s: str, ignore_prefix=False):
        formatted = f"[cyan]{s}[/cyan]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def optimization(s: str, ignore_prefix=False):
        formatted = f"[deep_pink3]{s}[/deep_pink3]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def quiet_update(s: str, ignore_prefix=False):
        formatted = f"[grey53]{escape(s)}[/grey53]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def light_update(s: str, ignore_prefix=False):
        formatted = f"[light_sky_blue1]{escape(s)}[/light_sky_blue1]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def warning(s: str, ignore_prefix=False):
        formatted = f"[yellow]{escape(s)}[/yellow]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def light_warning(s: str, ignore_prefix=False):
        formatted = f"[light_yellow3]{escape(s)}[/light_yellow3]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def error(s: str, ignore_prefix=False):
        formatted = f"[red]{escape(s)}[/red]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def success(s: str, ignore_prefix=False):
        formatted = f"[green]{escape(s)}[/green]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def model_or_data_update(s: str, ignore_prefix=False):
        formatted = f"[magenta]{escape(s)}[/magenta]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def sql(s: str, ignore_prefix=False):
        formatted = f"[cornflower_blue]`{escape(s)}`[/cornflower_blue]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def quiet_sql(s: str, ignore_prefix=False):
        formatted = f"[sky_blue1]`{escape(s)}`[/sky_blue1]"
        return formatted if ignore_prefix else Color._apply_prefix(formatted)

    @staticmethod
    def horizontal_line(char: str = "─", width: int = None):
        if width is None:
            console = Console()
            width = console.width

        line = char * width
        return f"[black]{line}[/black]"
