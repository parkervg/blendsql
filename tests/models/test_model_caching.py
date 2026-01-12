import pytest

from blendsql import BlendSQL, config
from blendsql.common.utils import fetch_from_hub
from blendsql.models import ConstrainedModel

config.set_async_limit(1)
config.set_deterministic(True)


@pytest.fixture(scope="module")
def bsql() -> BlendSQL:
    return BlendSQL(fetch_from_hub("codebase_community.sqlite"))


def test_llmmap_cache(bsql, model):
    try:
        if not isinstance(model, ConstrainedModel):
            # TODO: fix Unconstrained caching
            pytest.skip()
        model.caching = True
        model.cache.clear()
        query = """
        SELECT p.Body, {{LLMMap('What is the sentiment of this post?', p.Body, options=('positive', 'negative'))}}
        FROM posts p LIMIT 10
        """
        first = bsql.execute(
            query,
            model=model,
        )
        second = bsql.execute(
            query,
            model=model,
        )

        assert first.meta.process_time_seconds > second.meta.process_time_seconds
        assert second.meta.num_generation_calls == 0
    except Exception:
        raise
    finally:
        model.reset_stats()
        model.caching = False
