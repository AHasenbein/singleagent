"""Shared record shapes passed between pipeline stages.

These are TypedDicts (plain dicts at runtime) so they serialize with json.dumps,
insert into SQLite, and render with st.json without conversion. Optional[...] is
used instead of `X | None` because the project runs on Python 3.9.
"""

from typing import List, Optional, TypedDict


class PlanTask(TypedDict):
    task: str
    source_type: str  # "web" | "financials" | "filings"
    ticker: Optional[str]


class SourceResult(TypedDict, total=False):
    title: str
    url: str
    content: str
    published_date: Optional[str]
    error: str  # present instead of the other fields when a tool fails


class ResearchItem(TypedDict):
    task: str
    source_type: str
    ticker: Optional[str]
    results: List[SourceResult]


class EvidenceRecord(TypedDict, total=False):
    claim: str
    source: str
    url: str
    publication_date: Optional[str]
    source_type: str
    evidence: str
    confidence: Optional[str]  # "high" | "medium" | "low"
    related_company: Optional[str]
    report_section: Optional[str]


EVIDENCE_COLUMNS = [
    "claim",
    "source",
    "url",
    "publication_date",
    "source_type",
    "evidence",
    "confidence",
    "related_company",
    "report_section",
]
