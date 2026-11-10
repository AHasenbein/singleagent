"""LangGraph workflows that wire the stages together.

Each node is a small adapter: read state, call one stage function, return the
keys it produced. Stages get their dependencies from Services, so the same
graphs run against live APIs, fakes in tests, or a future backend service.
"""

from dataclasses import dataclass
from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .evidence import build_evidence
from .llm import LLM
from .planner import plan_question
from .report import generate_report, render_report_markdown
from .research import run_research, write_response
from .storage import ResearchStore
from .verification import verify_and_revise
from .tools import ResearchTools
from .types import CoverageEntry, EvidenceRecord, PlanTask, ResearchItem


# LangGraph resolves these annotations at runtime, so they must stay
# Python 3.9-compatible (List/Dict/Optional, not `X | None`).
class ResearchState(TypedDict):
    question: str
    research_plan: List[PlanTask]
    research_results: List[ResearchItem]
    response: str


class EvidenceState(ResearchState):
    evidence_records: List[EvidenceRecord]
    coverage: List[CoverageEntry]  # per plan task: attempted, failed, records kept


class ReportState(EvidenceState):
    report: Dict[str, str]
    report_markdown: str
    report_id: int


class VerifiedReportState(ReportState):
    verification: Dict
    final_report_markdown: str


@dataclass
class Services:
    llm: LLM
    tools: ResearchTools
    store: ResearchStore


def build_services(settings: Settings) -> Services:
    """Live services from settings. Raises MissingSettingsError without API keys."""
    from .llm import DeepSeekLLM
    from .tools import default_tools

    settings.require_api_keys()
    return Services(
        llm=DeepSeekLLM(settings.deepseek_api_key, model=settings.model),
        tools=default_tools(settings.tavily_api_key, settings.sec_user_agent),
        store=ResearchStore(settings.db_path),
    )


# ---------------- Nodes ----------------
def _nodes(services: Services) -> dict:
    def plan(state: ResearchState) -> dict:
        return {"research_plan": plan_question(services.llm, state["question"])}

    def research(state: ResearchState) -> dict:
        return {"research_results": run_research(state["research_plan"], services.tools)}

    def respond(state: ResearchState) -> dict:
        return {"response": write_response(services.llm, state["question"], state["research_results"])}

    def evidence(state: EvidenceState) -> dict:
        """Extract, ground-check, validate, and persist evidence from the research results."""
        records, coverage = build_evidence(services.llm, state["research_results"])
        services.store.save_evidence(records)
        return {"evidence_records": records, "coverage": coverage}

    def report(state: ReportState) -> dict:
        """Turn this run's evidence into a report, saved as 'pending' until verified."""
        records = state.get("evidence_records") or []
        coverage = state.get("coverage") or []
        if not records:
            raise ValueError(
                "No evidence records were extracted for this question, so no report "
                "can be written. Try a more specific question or check the research tools."
            )

        sections = generate_report(services.llm, state["question"], records, coverage)
        markdown = render_report_markdown(sections, records, coverage)
        report_id = services.store.save_report(
            state["question"], markdown, len(records), status="pending", evidence_records=records
        )
        return {"report": sections, "report_markdown": markdown, "report_id": report_id}

    def verify(state: VerifiedReportState) -> dict:
        """Quality gate with bounded self-revision; only a PASS report becomes final."""
        records = state["evidence_records"]
        sections, verification = verify_and_revise(services.llm, state["report"], records)
        markdown = state["report_markdown"]
        if verification["revisions"]:
            markdown = render_report_markdown(sections, records, state.get("coverage") or [])
        passed = verification["overall_status"] == "PASS"
        services.store.finalize_report(
            state["report_id"], markdown, "PASS" if passed else "FAIL", verification, records
        )
        return {
            "report": sections,
            "report_markdown": markdown,
            "verification": verification,
            "final_report_markdown": markdown if passed else "",
        }

    return {
        "plan": plan,
        "research": research,
        "respond": respond,
        "evidence": evidence,
        "report": report,
        "verify": verify,
    }


def _linear_graph(state_type: type, services: Services, steps: List[str]):
    nodes = _nodes(services)
    graph = StateGraph(state_type)
    for name in steps:
        graph.add_node(name, nodes[name])

    graph.add_edge(START, steps[0])
    for current, following in zip(steps, steps[1:]):
        graph.add_edge(current, following)
    graph.add_edge(steps[-1], END)

    return graph.compile()


# ---------------- Workflows ----------------
def build_response_graph(services: Services):
    """Week 3: Question -> Planner -> Research -> Response."""
    return _linear_graph(ResearchState, services, ["plan", "research", "respond"])


def build_evidence_graph(services: Services):
    """Week 4: Question -> Planner -> Research -> Evidence Ledger."""
    return _linear_graph(EvidenceState, services, ["plan", "research", "evidence"])


def build_report_graph(services: Services):
    """Week 5: Question -> Planner -> Research -> Evidence -> Report."""
    return _linear_graph(ReportState, services, ["plan", "research", "evidence", "report"])


def build_verified_report_graph(services: Services):
    """Week 6+: Question -> Planner -> Research -> Evidence -> Report -> Verification."""
    return _linear_graph(
        VerifiedReportState, services, ["plan", "research", "evidence", "report", "verify"]
    )


# ---------------- Initial states ----------------
def initial_research_state(question: str) -> ResearchState:
    return {"question": question, "research_plan": [], "research_results": [], "response": ""}


def initial_evidence_state(question: str) -> EvidenceState:
    return {**initial_research_state(question), "evidence_records": [], "coverage": []}


def initial_report_state(question: str) -> ReportState:
    return {**initial_evidence_state(question), "report": {}, "report_markdown": "", "report_id": 0}


def initial_verified_report_state(question: str) -> VerifiedReportState:
    return {**initial_report_state(question), "verification": {}, "final_report_markdown": ""}
