"""Pure display helpers for the Streamlit UI (no Streamlit import, unit-testable)."""

import re
from typing import Dict, List, Optional

from research_agent.types import EvidenceRecord

_CITATION_GROUP = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]")
_SOURCES_HEADING = "\n## Sources / References"


def citation_urls(evidence_records: List[EvidenceRecord]) -> Dict[str, str]:
    """Positional E-labels (E1 = first record) to source URLs."""
    return {
        f"E{index}": str(record.get("url") or "")
        for index, record in enumerate(evidence_records, start=1)
        if str(record.get("url") or "").startswith(("http://", "https://"))
    }


def link_citations(text: str, evidence_records: List[EvidenceRecord]) -> str:
    """Turn [E3] and [E1, E4] into markdown links to each cited source.

    Labels without a known URL are left as plain text.
    """
    urls = citation_urls(evidence_records)

    def replace(match: "re.Match") -> str:
        labels = re.findall(r"E\d+", match.group(1))
        parts = []
        for label in labels:
            label = f"E{int(label[1:])}"
            url = urls.get(label)
            parts.append(f"[[{label}]](<{url}>)" if url else f"[{label}]")
        return " ".join(parts)

    return _CITATION_GROUP.sub(replace, text or "")


def escape_dollars(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; '$46B to $54B' must stay literal."""
    return re.sub(r"(?<!\\)\$", r"\\$", text or "")


def for_display(text: str, evidence_records: List[EvidenceRecord]) -> str:
    return link_citations(escape_dollars(text), evidence_records)


def without_sources(report_markdown: str) -> str:
    """The report body; the reference list is shown in its own tab."""
    text = report_markdown or ""
    index = text.find(_SOURCES_HEADING)
    return text[:index].rstrip() if index != -1 else text


def evidence_rows(evidence_records: List[EvidenceRecord]) -> List[dict]:
    return [
        {
            "id": f"E{index}",
            "claim": record.get("claim"),
            "source": record.get("source"),
            "confidence": record.get("confidence"),
            "type": record.get("source_type"),
            "date": record.get("publication_date"),
            "company": record.get("related_company"),
            "evidence": record.get("evidence"),
            "url": record.get("url"),
        }
        for index, record in enumerate(evidence_records, start=1)
    ]


def source_rows(evidence_records: List[EvidenceRecord]) -> List[dict]:
    """One row per unique URL, with the E-labels that point at it."""
    by_url: Dict[str, dict] = {}
    for row in evidence_rows(evidence_records):
        url = row["url"] or ""
        entry = by_url.setdefault(url, {
            "source": row["source"], "type": row["type"], "date": row["date"],
            "records": 0, "labels": [], "url": url,
        })
        entry["records"] += 1
        entry["labels"].append(row["id"])
    rows = sorted(by_url.values(), key=lambda entry: -entry["records"])
    for row in rows:
        row["labels"] = ", ".join(row["labels"])
    return rows


def short(text: Optional[str], limit: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
