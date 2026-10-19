"""Web/news research through Tavily."""

from typing import List

from ..types import SourceResult


class TavilyWebSearch:
    def __init__(self, api_key: str, max_results: int = 4, search_depth: str = "advanced"):
        from tavily import TavilyClient

        self._client = TavilyClient(api_key=api_key)
        self.max_results = max_results
        self.search_depth = search_depth

    def __call__(self, task: str) -> List[SourceResult]:
        """Current web/news/company information through Tavily."""
        response = self._client.search(
            query=task,
            search_depth=self.search_depth,
            max_results=self.max_results,
        )
        return [
            {
                "title": result.get("title", "Untitled"),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            }
            for result in response.get("results", [])
        ]
