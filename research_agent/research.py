"""Research stage: run every Planner task through its tool, and the Week 3
plain-language response written from those results."""

import json
from typing import List

from .llm import LLM
from .tools import ResearchTools
from .types import PlanTask, ResearchItem

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


def run_research(plan: List[PlanTask], tools: ResearchTools) -> List[ResearchItem]:
    """Execute the Planner's FULL plan with the appropriate research tool."""
    research_results = []
    for item in plan:
        ticker = item.get("ticker")
        research_results.append(
            {
                "task": item["task"],
                "source_type": item["source_type"],
                "ticker": ticker,
                "results": tools.run(item["source_type"], item["task"], ticker),
            }
        )
    return research_results


def write_response(llm: LLM, question: str, research_results: List[ResearchItem]) -> str:
    prompt = RESPONSE_PROMPT_TEMPLATE.format(
        question=question,
        research_text=json.dumps(research_results, indent=2, default=str),
    )
    return llm.complete_text(prompt, temperature=0.3)
