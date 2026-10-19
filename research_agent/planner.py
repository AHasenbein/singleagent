"""Planner stage: break a research question into source-typed tasks."""

from typing import List

from .llm import LLM
from .types import PlanTask

SOURCE_TYPES = {"web", "financials", "filings"}

PLANNER_SYSTEM_PROMPT = """
You are the Planner for a single-agent competitive-intelligence research system.

Break the user's business research question into 4-6 specific, non-overlapping
research tasks. For each task, choose the most appropriate source type:

- "web" for current websites, news, products, partnerships, market activity,
  customer information, and general competitive intelligence.
- "financials" for public-company market/financial metrics.
- "filings" for official SEC filings and regulatory disclosures.

If a task uses "financials" or "filings", include the public-company ticker
symbol when you can identify it confidently. Otherwise use null.

Return only valid JSON in exactly this shape:
{
  "tasks": [
    {
      "task": "specific research task",
      "source_type": "web",
      "ticker": null
    }
  ]
}

Do not include markdown or text outside the JSON object.
""".strip()


def parse_plan(output: dict) -> List[PlanTask]:
    """Validate the Planner's JSON and normalize each task."""
    tasks = output.get("tasks")

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner must return a non-empty 'tasks' list.")

    cleaned = []
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("Each Planner task must be a JSON object.")

        task = str(item.get("task", "")).strip()
        source_type = str(item.get("source_type", "")).strip().lower()
        ticker = item.get("ticker")

        if not task:
            raise ValueError("Every Planner task needs a non-empty 'task'.")
        if source_type not in SOURCE_TYPES:
            raise ValueError(
                "source_type must be web, financials, or filings."
            )

        cleaned.append(
            {
                "task": task,
                "source_type": source_type,
                "ticker": str(ticker).upper().strip() if ticker else None,
            }
        )

    return cleaned


def plan_question(llm: LLM, question: str) -> List[PlanTask]:
    return parse_plan(llm.complete_json(PLANNER_SYSTEM_PROMPT, question))
