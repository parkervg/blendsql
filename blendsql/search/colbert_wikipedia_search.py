from dataclasses import dataclass, field
import httpx
import asyncio

from blendsql.search.searcher import Searcher


@dataclass(kw_only=True)
class ColbertWikipediaSearch(Searcher):
    url: str = field(default="http://20.102.90.50:2017/wiki17_abstracts")

    @staticmethod
    def _cleanup_response(search_response):
        return [res["text"] for res in search_response["topk"]]

    async def asearch(self, query: str, k: int) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.url,
                params={"query": query, "k": k},
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
