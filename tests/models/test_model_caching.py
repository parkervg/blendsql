import pytest

from blendsql import BlendSQL, config
from blendsql.common.utils import fetch_from_hub

config.set_async_limit(1)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(fetch_from_hub("codebase_community.sqlite"))


def test_llmmap_cache(bsql, model):
    model.caching = True
    model.cache.clear()
    first = bsql.execute(
        """
        SELECT p.Body, {{LLMMap('What is the sentiment of this post?', p.Body, options=('positive', 'negative'))}}
        FROM posts p LIMIT 10
        """,
        model=model,
    )
    second = bsql.execute(
        """
        SELECT p.Body, {{LLMMap('What is the sentiment of this post?', p.Body, options=('positive', 'negative'))}}
        FROM posts p LIMIT 10
        """,
        model=model,
    )

    assert first.meta.process_time_seconds > second.meta.process_time_seconds
