"""Report stage: write the nine-section report from evidence and render it,
with Sources / References built in Python as section ten."""

import json
from typing import Dict, List

from .llm import LLM
from .types import EvidenceRecord

# The model writes these nine sections; Sources / References is appended in
# Python so URLs come from the evidence, never from the model.
REPORT_SECTION_KEYS = [
    ("executive_summary", "Executive Summary"),
    ("key_findings", "Key Findings"),
    ("competitor_comparison", "Competitor Comparison"),
    ("market_industry_trends", "Market / Industry Trends"),
    ("emerging_technologies", "Emerging Technologies"),
    ("regulatory_developments", "Regulatory Developments"),
    ("risks_opportunities", "Risks / Opportunities"),
    ("recommended_next_steps", "Recommended Next Steps"),
    ("limitations_information_gaps", "Limitations / Information Gaps"),
]

REPORT_SYSTEM_PROMPT = """
You are the report-generation stage inside a SINGLE AI research-agent workflow.

Use ONLY the supplied structured Evidence Records as factual support.
Return ONLY valid JSON with exactly these keys:
{
  "executive_summary": "...",
  "key_findings": "...",
  "competitor_comparison": "...",
  "market_industry_trends": "...",
  "emerging_technologies": "...",
  "regulatory_developments": "...",
  "risks_opportunities": "...",
  "recommended_next_steps": "...",
  "limitations_information_gaps": "..."
}

Rules:
- Cite factual statements using evidence labels such as [E1].
- Do not invent facts, dates, numbers, source names, URLs, or citations.
- Clearly distinguish facts from analysis and recommendations.
- If evidence is missing for a section, say so.
- Do NOT build citation verification today. That is Week 6.
""".strip()

REPORT_USER_TEMPLATE = """
Research question:
{question}

Evidence Records (cite them by evidence_id, e.g. [E1]):
{packet}
""".strip()


def build_evidence_packet(evidence_records: List[EvidenceRecord]) -> List[dict]:
    """Number this run's evidence as E1, E2, ... for citation."""
    return [
        {
            "evidence_id": f"E{index}",
            "claim": record.get("claim"),
            "source": record.get("source"),
            "url": record.get("url"),
            "publication_date": record.get("publication_date"),
            "source_type": record.get("source_type"),
            "evidence": record.get("evidence"),
            "confidence": record.get("confidence"),
            "related_company": record.get("related_company"),
            "report_section": record.get("report_section"),
        }
        for index, record in enumerate(evidence_records, start=1)
    ]


def generate_report(llm: LLM, question: str, evidence_records: List[EvidenceRecord]) -> Dict[str, str]:
    """Generate the report from evidence — not from the full conversation history."""
    prompt = REPORT_USER_TEMPLATE.format(
        question=question,
        packet=json.dumps(build_evidence_packet(evidence_records), indent=2, default=str),
    )

    output = llm.complete_json(REPORT_SYSTEM_PROMPT, prompt)
    if not isinstance(output, dict):
        raise ValueError("Report generator must return a JSON object.")

    report = {}
    for key, title in REPORT_SECTION_KEYS:
        value = output.get(key)
        # The model sometimes returns a section as a list of points.
        if isinstance(value, list):
            value = "\n".join(f"- {point}" for point in value)
        value = str(value or "").strip()
        report[key] = value or f"No {title.lower()} could be supported by the collected evidence."

    return report


def build_references(evidence_records: List[EvidenceRecord]) -> str:
    """Build the source list directly from evidence so URLs are not invented by the model."""
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        lines.append(f"- [E{index}] {source} — {date} — {url}")
    return "\n".join(lines) if lines else "No source records were available."


def render_report_markdown(report: Dict[str, str], evidence_records: List[EvidenceRecord]) -> str:
    """Render all ten required report sections."""
    blocks = [f"## {title}\n\n{report.get(key, '')}" for key, title in REPORT_SECTION_KEYS]
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)
