import pytest
from fakes import PASS, SOURCE_SENTENCE, FakeLLM, fail, fake_tools, make_services, record

from research_agent.followup import UNSUPPORTED_ANSWER_MESSAGE, answer_followup, run_followup

STABLE_KEYS = {
    "answer", "cited_ids", "unmatched_ids", "decision", "additional_research_performed",
    "research_error", "new_evidence_count", "evidence_records", "verification", "supported",
}
BASE = {"question": "q", "final_report_markdown": "## R", "evidence_records": [record(1), record(2)]}


def _ask(tmp_path, llm, tools=None, base=BASE):
    services = make_services(llm, str(tmp_path / "f.db"), tools)
    return run_followup(services, "why?", base)


def test_supported_answer_is_shown(tmp_path):
    result = _ask(tmp_path, FakeLLM(answer=lambda _: {"answer": "Because [E2].", "cited_ids": ["E2"]}))
    assert set(result) == STABLE_KEYS
    assert result["supported"] and result["answer"] == "Because [E2]."
    assert result["cited_ids"] == ["E2"] and result["unmatched_ids"] == []
    assert result["verification"]["status"] == "PASS"


@pytest.mark.parametrize("reply", [
    {"answer": "Because [E1, E99].", "cited_ids": ["E1"]},   # unknown inline citation
    {"answer": "Because [E1].", "cited_ids": ["E1", "E42"]},  # unknown claimed citation
])
def test_unknown_citation_blocks_answer_without_calling_verifier(tmp_path, reply):
    llm = FakeLLM(answer=lambda _: reply)
    result = _ask(tmp_path, llm)
    assert not result["supported"] and result["answer"] == UNSUPPORTED_ANSWER_MESSAGE
    assert "Because" not in result["answer"]
    assert result["unmatched_ids"]
    assert any(i["type"] == "unknown_citation" for i in result["verification"]["issues"])
    assert llm.count("verify") == 0


def test_unsupported_answer_is_blocked(tmp_path):
    llm = FakeLLM(answer=lambda _: {"answer": "Margins tripled [E1].", "cited_ids": ["E1"]},
                  verify=lambda _: fail("E1 does not mention margins", "citation_mismatch"))
    result = _ask(tmp_path, llm)
    assert not result["supported"] and "Margins" not in result["answer"]
    assert result["verification"]["issues"][0]["type"] == "citation_mismatch"


@pytest.mark.parametrize("reply", [
    {"answer": "No citations here.", "cited_ids": []},
    {"answer": "", "cited_ids": []},
    {"answer": ["list"], "cited_ids": ["E1"]},
    {"answer": "Fine [E1].", "cited_ids": "E1"},
    {"answer": "Fine [E1]."},
    {"text": "wrong key"},
])
def test_uncited_or_malformed_answers_are_blocked(tmp_path, reply):
    llm = FakeLLM(answer=lambda _: reply)
    result = _ask(tmp_path, llm)
    assert not result["supported"] and result["answer"] == UNSUPPORTED_ANSWER_MESSAGE
    assert result["verification"]["issues"]
    assert llm.count("verify") == 0


def test_answer_call_exception_is_blocked_not_raised():
    def boom(_):
        raise RuntimeError("down")
    result = answer_followup(FakeLLM(answer=boom), "why?", BASE, BASE["evidence_records"])
    assert not result["supported"] and "down" in result["verification"]["issues"][0]["explanation"]


def test_verifier_error_on_answer_is_retried_once():
    replies = iter([{"status": "PASS"}, PASS])
    llm = FakeLLM(verify=lambda _: next(replies))
    result = answer_followup(llm, "why?", BASE, BASE["evidence_records"])
    assert result["supported"] and llm.count("verify") == 2


def test_research_runs_at_most_once_and_preserves_labels(tmp_path):
    llm = FakeLLM(decide=lambda _: {"evidence_sufficient": False, "reason": "gap", "targeted_query": "more"},
                  answer=lambda _: {"answer": "New fact [E3].", "cited_ids": ["E3"]})
    result = _ask(tmp_path, llm)
    assert llm.count("planner") == 1
    assert result["additional_research_performed"] and result["new_evidence_count"] == 1
    assert result["evidence_records"][:2] == BASE["evidence_records"]  # E1, E2 unchanged
    assert result["evidence_records"][2]["evidence"] == SOURCE_SENTENCE
    assert result["supported"]


def test_blocked_answer_after_research_does_not_research_again(tmp_path):
    llm = FakeLLM(decide=lambda _: {"evidence_sufficient": False, "reason": "gap", "targeted_query": "more"},
                  verify=lambda _: fail())
    result = _ask(tmp_path, llm)
    assert not result["supported"]
    assert llm.count("planner") == 1 and llm.count("answer") == 1


def test_duplicate_new_evidence_does_not_shift_labels(tmp_path):
    existing = {**record(1), "claim": "NVIDIA had record data center revenue.", "url": "https://example.com/1"}
    base = {**BASE, "evidence_records": [existing, record(2)]}
    llm = FakeLLM(decide=lambda _: {"evidence_sufficient": False, "reason": "gap", "targeted_query": "more"})
    result = _ask(tmp_path, llm, base=base)
    assert result["evidence_records"] == [existing, record(2)]
    assert result["new_evidence_count"] == 0


def test_research_failure_is_reported_and_answer_still_gated(tmp_path):
    def planner(_):
        raise RuntimeError("planner down")
    llm = FakeLLM(planner=planner,
                  decide=lambda _: {"evidence_sufficient": False, "reason": "gap", "targeted_query": "more"})
    result = _ask(tmp_path, llm)
    assert result["research_error"] == "planner down"
    assert result["new_evidence_count"] == 0 and result["evidence_records"] == BASE["evidence_records"]
    assert result["supported"]  # the default answer cites E1, which exists and verifies


def test_tool_failure_during_followup_research_is_isolated(tmp_path):
    def web(query):
        raise KeyError("offline")  # not transient, so no retry delay
    llm = FakeLLM(decide=lambda _: {"evidence_sufficient": False, "reason": "gap", "targeted_query": "more"})
    result = _ask(tmp_path, llm, tools=fake_tools(web=web))
    assert result["new_evidence_count"] == 0
    assert "KeyError" in result["research_error"] and "offline" in result["research_error"]


def test_empty_followup_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_followup(make_services(FakeLLM(), str(tmp_path / "e.db")), "  ", BASE)
