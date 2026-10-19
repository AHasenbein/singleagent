"""Evidence stage: turn research results into validated, traceable records."""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .llm import LLM
from .types import EvidenceRecord, ResearchItem, SourceResult

CONFIDENCE_LEVELS = {"high", "medium", "low"}
REQUIRED_FIELDS = ["claim", "source", "url", "source_type", "evidence"]
MAX_PARALLEL_EXTRACTIONS = 8

EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a business research workflow.

Convert useful research results into structured Evidence Records.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "...",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Rules:
- claim = a factual statement supported by the source.
- evidence = the passage that supports the claim.
- source and url must come from the supplied research.
- Do not guess a publication date.
- confidence must be high, medium, or low.
- Do not create evidence for unsupported conclusions.
""".strip()


def extract_from_source(llm: LLM, item: ResearchItem, result: SourceResult) -> List[EvidenceRecord]:
    """Ask the model for evidence records from a single source result."""
    source_payload = {
        "task": item.get("task"),
        "source_type": item.get("source_type"),
        "title": result.get("title"),
        "url": result.get("url"),
        "content": result.get("content"),
        "published_date": result.get("published_date"),
    }

    try:
        output = llm.complete_json(
            EVIDENCE_SYSTEM_PROMPT,
            json.dumps(source_payload, indent=2, default=str),
        )
    except Exception:
        # One failed model call should not sink the whole ledger.
        return []

    extracted = output.get("records") if isinstance(output, dict) else None
    if not isinstance(extracted, list):
        return []

    records = []
    for record in extracted:
        if not isinstance(record, dict):
            continue
        # Traceability: never let the model invent provenance.
        record["url"] = source_payload["url"]
        record["source"] = record.get("source") or source_payload["title"]
        record["source_type"] = source_payload["source_type"]
        record["publication_date"] = source_payload["published_date"]
        if not record.get("related_company"):
            record["related_company"] = item.get("ticker")
        records.append(record)
    return records


def extract_evidence(llm: LLM, research_results: List[ResearchItem]) -> List[EvidenceRecord]:
    """Turn research results into structured evidence records."""
    jobs = [
        (item, result)
        for item in research_results
        for result in item.get("results", [])
        if isinstance(result, dict) and "error" not in result
    ]

    # Model calls are slow and independent, so run them in parallel.
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as pool:
        batches = pool.map(lambda job: extract_from_source(llm, *job), jobs)

    return [record for batch in batches for record in batch]


def validate_evidence(records: List[EvidenceRecord]) -> List[EvidenceRecord]:
    """Keep only complete records, dropping (claim, url) duplicates."""
    validated = []
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue
        if not all(str(record.get(field) or "").strip() for field in REQUIRED_FIELDS):
            continue

        key = (record["claim"].strip().lower(), record["url"].strip())
        if key in seen:
            continue
        seen.add(key)

        if record.get("confidence") not in CONFIDENCE_LEVELS:
            record["confidence"] = None
        validated.append(record)

    return validated
