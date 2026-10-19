"""Research tools and the dispatcher that routes a plan task to the right one.

To add a source type: write a tool returning List[SourceResult], add it to
ResearchTools, and add the type to planner.SOURCE_TYPES and the Planner prompt.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from ..types import SourceResult

WebSearch = Callable[[str], List[SourceResult]]
TickerTool = Callable[[Optional[str]], List[SourceResult]]


@dataclass
class ResearchTools:
    web: WebSearch
    financials: TickerTool
    filings: TickerTool

    def run(self, source_type: str, task: str, ticker: Optional[str]) -> List[SourceResult]:
        if source_type == "web":
            return self.web(task)
        if source_type == "financials":
            return self.financials(ticker)
        if source_type == "filings":
            return self.filings(ticker)
        return [{"error": f"Unknown source type: {source_type}"}]


def default_tools(tavily_api_key: str, sec_user_agent: str) -> ResearchTools:
    """Live tools: Tavily web search, yfinance snapshots, SEC EDGAR filings."""
    from .financials import get_financials
    from .sec import get_sec_filings
    from .web import TavilyWebSearch

    return ResearchTools(
        web=TavilyWebSearch(tavily_api_key),
        financials=get_financials,
        filings=lambda ticker: get_sec_filings(ticker, sec_user_agent),
    )
