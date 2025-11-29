"""
https://github.com/meta-llama/llama-stack-apps/blob/a1b3d89ad7f47f4d8e5cb6510bb6ba0e3bbabb72/examples/client_tools/web_search.py#L18
"""
import os
from dataclasses import dataclass, field
import httpx
import asyncio

from blendsql.search.searcher import Searcher


@dataclass(kw_only=True)
class TavilySearch(Searcher):
    api_key: str = field(default=None)

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("TAVILY_API_KEY")

    @staticmethod
    def _cleanup_response(search_response):
        return [
            f"{res['title']} | {res['content']}" for res in search_response["results"]
        ]

    async def asearch(self, query: str, k: int) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": k},
            )
            response.raise_for_status()

        return self._cleanup_response(response.json())

    async def _search(self, queries: list[str], k: int):
        responses = [self.asearch(q, k) for q in queries]
        return [i for i in await asyncio.gather(*responses)]

    def __call__(self, query: list[str] | str, k: int | None = None) -> list[list[str]]:
        asyncio.set_event_loop(asyncio.new_event_loop())
        is_single_query = isinstance(query, str)
        queries = [query] if is_single_query else query
        return asyncio.run(self._search(queries, self.k or k))
