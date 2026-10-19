"""Public-company market snapshot through yfinance."""

import json
from typing import Any, Callable, List, Optional

from ..types import SourceResult

SNAPSHOT_FIELDS = ("last_price", "market_cap", "year_high", "year_low")


def _yfinance_ticker(symbol: str) -> Any:
    import yfinance as yf

    return yf.Ticker(symbol)


def get_financials(
    ticker: Optional[str],
    ticker_factory: Callable[[str], Any] = _yfinance_ticker,
) -> List[SourceResult]:
    """Small public-company snapshot using yfinance."""
    if not ticker:
        return [{"error": "Planner did not provide a ticker."}]

    fast = ticker_factory(ticker).fast_info

    # Some values may be missing, so only keep the ones that resolve.
    fields = {}
    for name in SNAPSHOT_FIELDS:
        try:
            value = getattr(fast, name)
        except Exception:
            continue
        if value is not None:
            fields[name] = value

    if not fields:
        return [{"error": f"No financial snapshot returned for {ticker}."}]

    return [
        {
            "title": f"{ticker} market snapshot",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "content": json.dumps(fields, default=str),
        }
    ]
