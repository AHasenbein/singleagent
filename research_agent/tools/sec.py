"""Recent SEC filing metadata through the public EDGAR JSON endpoints."""

from typing import Any, Callable, List, Optional

from ..types import SourceResult

TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
KEPT_FORMS = {"10-K", "10-Q", "8-K"}
MAX_FILINGS = 3


def _requests_get(url: str, headers: dict, timeout: int) -> Any:
    import requests

    return requests.get(url, headers=headers, timeout=timeout)


def get_sec_filings(
    ticker: Optional[str],
    user_agent: str,
    http_get: Callable[..., Any] = _requests_get,
) -> List[SourceResult]:
    """Recent SEC filing metadata for a public-company ticker."""
    if not ticker:
        return [{"error": "Planner did not provide a ticker."}]

    # SEC asks automated clients to identify themselves.
    headers = {"User-Agent": user_agent}

    # Find the company's SEC CIK number.
    ticker_response = http_get(TICKER_INDEX_URL, headers=headers, timeout=20)
    ticker_response.raise_for_status()
    companies = ticker_response.json()

    match = next(
        (
            item
            for item in companies.values()
            if str(item.get("ticker", "")).upper() == ticker.upper()
        ),
        None,
    )

    if not match:
        return [{"error": f"Could not find SEC CIK for ticker {ticker}."}]

    cik = str(match["cik_str"]).zfill(10)

    # Get the company's recent SEC filings.
    submissions_response = http_get(
        SUBMISSIONS_URL.format(cik=cik), headers=headers, timeout=20
    )
    submissions_response.raise_for_status()

    recent = (
        submissions_response.json()
        .get("filings", {})
        .get("recent", {})
    )

    results = []
    for form, date, accession, document in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if form not in KEPT_FORMS:
            continue

        url = FILING_URL.format(
            cik=int(cik),
            accession=str(accession).replace("-", ""),
            document=document,
        )
        results.append(
            {
                "title": f"{ticker} {form} ({date})",
                "url": url,
                "content": (
                    f"{form} filed on {date}. "
                    f"Accession number {accession}."
                ),
            }
        )
        if len(results) >= MAX_FILINGS:
            break

    return results or [
        {"error": f"No recent 10-K, 10-Q, or 8-K found for {ticker}."}
    ]
