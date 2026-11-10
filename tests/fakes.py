"""Offline fakes: an LLM routed by system prompt, research tools, and Services."""

import threading
from typing import Callable, Dict, List, Optional

from research_agent.pipeline import Services
from research_agent.report import REPORT_SECTION_KEYS
from research_agent.storage import ResearchStore
from research_agent.tools import ResearchTools

SOURCE_SENTENCE = "NVIDIA reported record data center revenue of 30 billion dollars in the quarter."
SOURCE_CONTENT = f"Quarterly update. {SOURCE_SENTENCE} Gaming revenue was flat year over year."

# System-prompt prefixes identify which stage is calling.
STAGES = {
    "planner": "You are the Planner",
    "extract": "You are the evidence extraction",
    "report": "You are the report-generation",
    "verify": "You are a citation-verification",
    "revise": "You are the revision step",
    "decide": "You are the FOLLOW-UP",
    "answer": "Answer the user's follow-up",
}

PASS = {"status": "PASS", "issues": []}


def fail(explanation: str = "too strong", issue_type: str = "overstated_evidence") -> dict:
    return {"status": "FAIL", "issues": [
        {"type": issue_type, "claim": "x", "evidence_id": "E1", "explanation": explanation}
    ]}


def default_extract(user_text: str) -> dict:
    return {"records": [{
        "claim": "NVIDIA had record data center revenue.",
        "source": "Example News",
        "url": "ignored — provenance comes from the tool",
        "source_type": "web",
        "evidence": SOURCE_SENTENCE,
        "confidence": "high",
    }]}


class FakeLLM:
    """Each stage's reply comes from a handler(user_text) -> dict (or raises).

    Calls are recorded thread-safely, since verification and extraction run in pools.
    """

    def __init__(self, **handlers: Callable[[str], dict]):
        self.handlers: Dict[str, Callable[[str], dict]] = {
            "planner": lambda _: {"tasks": [{"task": "AI chip revenue", "source_type": "web", "ticker": None}]},
            "extract": default_extract,
            "report": lambda _: {key: "NVIDIA had record data center revenue [E1]." for key, _ in REPORT_SECTION_KEYS},
            "verify": lambda _: PASS,
            "revise": lambda _: {"text": "Revised: NVIDIA had record data center revenue [E1]."},
            "decide": lambda _: {"evidence_sufficient": True, "reason": "enough", "targeted_query": ""},
            "answer": lambda _: {"answer": "Revenue hit a record [E1].", "cited_ids": ["E1"]},
        }
        self.handlers.update(handlers)
        self.calls: List[tuple] = []
        self._lock = threading.Lock()

    def complete_json(self, system_prompt: str, user_text: str) -> dict:
        stage = next((name for name, prefix in STAGES.items() if system_prompt.startswith(prefix)), None)
        if stage is None:
            raise AssertionError(f"Unexpected prompt: {system_prompt[:60]}")
        with self._lock:
            self.calls.append((stage, user_text))
        return self.handlers[stage](user_text)

    def complete_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        return "plain response"

    def count(self, stage: str) -> int:
        with self._lock:
            return sum(1 for name, _ in self.calls if name == stage)

    def prompts(self, stage: str) -> List[str]:
        with self._lock:
            return [text for name, text in self.calls if name == stage]


def fake_tools(web: Optional[Callable] = None, financials: Optional[Callable] = None,
               filings: Optional[Callable] = None) -> ResearchTools:
    counter = {"n": 0}

    def default_web(query: str):
        counter["n"] += 1
        return [{"title": "Example News", "url": f"https://example.com/{counter['n']}", "content": SOURCE_CONTENT}]

    return ResearchTools(
        web=web or default_web,
        financials=financials or (lambda ticker: []),
        filings=filings or (lambda ticker: []),
    )


def make_services(llm: FakeLLM, db_path: str, tools: Optional[ResearchTools] = None) -> Services:
    return Services(llm=llm, tools=tools or fake_tools(), store=ResearchStore(db_path))


def record(n: int, claim: str = None, evidence: str = None) -> dict:
    return {
        "claim": claim or f"Claim number {n}",
        "source": f"Source {n}",
        "url": f"https://example.com/r{n}",
        "publication_date": None,
        "source_type": "web",
        "evidence": evidence or f"Evidence passage number {n}",
        "confidence": "high",
        "related_company": None,
        "report_section": None,
    }
