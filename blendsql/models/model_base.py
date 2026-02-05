from typing import Any, Sequence, Callable, Tuple

from pathlib import Path
from diskcache import Cache
import platformdirs
import hashlib
import inspect
from textwrap import dedent
from dataclasses import dataclass, field

from blendsql.common.logger import logger, Color

CONTEXT_TRUNCATION_LIMIT = 100


@dataclass
class ModelBase:
    """Parent class for all BlendSQL Models."""

    model_name_or_path: str = field()
    caching: bool = field(default=False)
    cache: Cache | None = field(default=None)
    _allows_parallel_requests: bool = field(default=False)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    num_generation_calls: int = 0
    num_cache_hits: int = 0

    def __post_init__(self):
        self.cache = Cache(
            Path(platformdirs.user_cache_dir("blendsql"))
            / f"{self.model_name_or_path}.diskcache"
        )

    def _create_key(
        self, *args, funcs: Sequence[Callable] | None = None, **kwargs
    ) -> str:
        """Generates a hash to use in diskcache Cache.
        This way, we don't need to send our prompts to the same Model
        if our context of Model + args + kwargs is the same.

        Returns:
            md5 hash used as key in diskcache
        """
        hasher = hashlib.md5()
        params_str = ""
        if len(kwargs) > 0:
            params_str += str(sorted([(k, str(v)) for k, v in kwargs.items()]))
        if len(args) > 0:
            params_str += str([arg for arg in args])
        if funcs:
            params_str += "\n".join([dedent(inspect.getsource(func)) for func in funcs])
        combined_str = "{}||{}".format(
            f"{self.model_name_or_path}||{type(self)}",
            params_str,
        ).encode()
        hasher.update(combined_str)
        return hasher.hexdigest()

    def check_cache(
        self, *args, funcs: Sequence[Callable] | None = None, **kwargs
    ) -> Tuple[Any, str]:
        response: dict[str, str] = None  # type: ignore
        key: str = self._create_key(funcs=funcs, *args, **kwargs)
        if key in self.cache:
            self.num_cache_hits += 1
            logger.debug(
                Color.model_or_data_update(
                    f"Using model cache ({self.num_cache_hits})..."
                )
            )
            response = self.cache.get(key)  # type: ignore
        return (response, key)

    def reset_stats(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.num_generation_calls = 0
        self.num_cache_hits = 0
