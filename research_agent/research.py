"""Research stage: run every Planner task through its tool, and the Week 3
plain-language response written from those results."""

import json
import time
from typing import Callable, List

from .llm import LLM
from .tools import ResearchTools
from .types import PlanTask, ResearchItem, SourceResult

RESPONSE_PROMPT_TEMPLATE = """
You are the response step in a single-agent business research workflow.

Answer the original question using only the research results supplied below.
Combine useful information from multiple Planner tasks. If a tool failed or a
planned area lacks useful information, say so briefly rather than inventing it.

This is still a Week 3 research response, not the final semester report.

Question:
{question}

Planner research results:
{research_text}
""".strip()


RETRY_DELAY_SECONDS = 1.0
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}


def is_transient(error: Exception) -> bool:
    """Rate limits, 5xx responses, timeouts and connection drops are worth one retry."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return status in TRANSIENT_HTTP_STATUSES
    # requests' ConnectionError/Timeout are OSError subclasses; so is TimeoutError.
    return isinstance(error, OSError)


def run_task(
    tools: ResearchTools, item: PlanTask, sleep: Callable[[float], None] = time.sleep
) -> List[SourceResult]:
    """Run one task; a tool failure becomes an error result instead of an exception."""
    def attempt() -> List[SourceResult]:
        return tools.run(item["source_type"], item["task"], item.get("ticker"))

    try:
        try:
            return attempt()
        except Exception as error:
            if not is_transient(error):
                raise
            sleep(RETRY_DELAY_SECONDS)
            return attempt()
    except Exception as error:
        return [{"error": f"{type(error).__name__}: {error}"}]


def run_research(
    plan: List[PlanTask], tools: ResearchTools, sleep: Callable[[float], None] = time.sleep
) -> List[ResearchItem]:
    """Execute the Planner's FULL plan; one failing tool does not abort the others."""
    return [
        {
            "task": item["task"],
            "source_type": item["source_type"],
            "ticker": item.get("ticker"),
            "results": run_task(tools, item, sleep),
        }
        for item in plan
    ]


def write_response(llm: LLM, question: str, research_results: List[ResearchItem]) -> str:
    prompt = RESPONSE_PROMPT_TEMPLATE.format(
        question=question,
        research_text=json.dumps(research_results, indent=2, default=str),
    )
    return llm.complete_text(prompt, temperature=0.3)
