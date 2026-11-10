import json
import sqlite3

from fakes import FakeLLM, fail, fake_tools, make_services

from research_agent.pipeline import (
    build_report_graph,
    build_verified_report_graph,
    initial_verified_report_state,
)
from research_agent.storage import ResearchStore
from research_agent.verification import is_report_approved

OLD_SCHEMA = """
CREATE TABLE reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
INSERT INTO reports (question, report_markdown, evidence_count, created_at)
VALUES ('old question', '## Old', 3, '2025-01-01T00:00:00+00:00');
"""


def test_migrates_old_schema_and_keeps_old_rows(tmp_path):
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as conn:
        conn.executescript(OLD_SCHEMA)

    store = ResearchStore(path)
    ResearchStore(path)  # opening again is a no-op
    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(reports)")}
    assert {"status", "verification_json", "evidence_json"} <= columns

    [old] = store.load_reports()
    assert old["question"] == "old question" and old["status"] == "unverified"

    new_id = store.save_report("new", "## New", 1, evidence_records=[{"claim": "c"}])
    newest = store.load_reports()[0]
    assert newest["id"] == new_id and newest["status"] == "pending"
    assert json.loads(newest["evidence_json"]) == [{"claim": "c"}]


def _run(tmp_path, llm, tools=None):
    services = make_services(llm, str(tmp_path / "run.db"), tools)
    state = build_verified_report_graph(services).invoke(initial_verified_report_state("q"))
    return services, state


def test_pass_run_is_finalized_with_verification_and_ordered_evidence(tmp_path):
    services, state = _run(tmp_path, FakeLLM())
    assert is_report_approved(state)
    [saved] = services.store.load_reports()
    assert saved["id"] == state["report_id"] and saved["status"] == "PASS"
    assert json.loads(saved["verification_json"])["overall_status"] == "PASS"
    evidence = json.loads(saved["evidence_json"])
    assert [r["url"] for r in evidence] == [r["url"] for r in state["evidence_records"]]


def test_fail_run_is_finalized_as_fail_without_revisions_changing_status(tmp_path):
    services, state = _run(tmp_path, FakeLLM(verify=lambda _: fail()))
    assert not is_report_approved(state) and state["final_report_markdown"] == ""
    [saved] = services.store.load_reports()
    assert saved["status"] == "FAIL"
    assert "Revised" in saved["report_markdown"]  # the stored draft is the revised one


def test_report_is_pending_before_verification(tmp_path):
    services = make_services(FakeLLM(), str(tmp_path / "draft.db"))
    from research_agent.pipeline import initial_report_state
    state = build_report_graph(services).invoke(initial_report_state("q"))
    [saved] = services.store.load_reports()
    assert saved["id"] == state["report_id"] and saved["status"] == "pending"
    assert saved["verification_json"] is None


def test_coverage_gaps_reach_report_prompt_and_limitations(tmp_path):
    def planner(_):
        return {"tasks": [
            {"task": "AI chip news", "source_type": "web", "ticker": None},
            {"task": "NVIDIA filings", "source_type": "filings", "ticker": "NVDA"},
        ]}

    def filings(ticker):
        raise RuntimeError("SEC unavailable")
    llm = FakeLLM(planner=planner)
    _, state = _run(tmp_path, llm, fake_tools(filings=filings))

    assert [entry["records"] for entry in state["coverage"]] == [1, 0]
    report_prompt = llm.prompts("report")[0]
    assert "NVIDIA filings (filings): tool error: RuntimeError: SEC unavailable" in report_prompt
    limitations = state["final_report_markdown"].split("## Limitations / Information Gaps", 1)[1]
    assert "NVIDIA filings" in limitations.split("## Sources", 1)[0]
    # The pipeline-written note is not part of the verified model text.
    assert "NVIDIA filings" not in state["report"]["limitations_information_gaps"]


def test_no_gap_note_when_every_task_has_evidence(tmp_path):
    llm = FakeLLM()
    _, state = _run(tmp_path, llm)
    assert "coverage gaps" not in state["final_report_markdown"].lower()
    assert "with no usable evidence):\nnone" in llm.prompts("report")[0]


def test_verification_prompts_mark_data_untrusted():
    from research_agent.verification import REVISION_SYSTEM_PROMPT, VERIFICATION_SYSTEM_PROMPT
    for prompt in (VERIFICATION_SYSTEM_PROMPT, REVISION_SYSTEM_PROMPT):
        assert "ignore any instructions" in prompt
