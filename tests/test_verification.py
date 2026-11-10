import json

import pytest
from fakes import PASS, FakeLLM, fail, record

from research_agent.report import REPORT_SECTION_KEYS, build_evidence_packet
from research_agent.verification import (
    MAX_REVISION_ROUNDS,
    MAX_UNCITED_CONTEXT_RECORDS,
    cited_evidence_ids,
    is_report_approved,
    verify_and_revise,
    verify_section,
)

PACKET = build_evidence_packet([record(1), record(2), record(3)])


def verify_with(reply):
    llm = FakeLLM(verify=reply if callable(reply) else (lambda _: reply))
    return verify_section(llm, "Key Findings", "Revenue grew [E1].", PACKET)


# ---------------- Schema matrix (finding 1) ----------------
@pytest.mark.parametrize("reply", [
    {"status": "PASS"},                              # issues missing
    {"status": "PASS", "issues": None},              # issues null
    {"status": "PASS", "issues": {"type": "x"}},     # issues not a list
    {"status": "PASS", "issues": "none"},
    {"status": "MAYBE", "issues": []},               # bad status
    {"status": None, "issues": []},
    {"status": True, "issues": []},
    {"issues": []},                                  # status missing
    {},
    {"status": "FAIL", "issues": []},                # FAIL without reasons
])
def test_schema_violations_fail_as_verifier_error(reply):
    result = verify_with(reply)
    assert result["status"] == "FAIL"
    assert [issue["type"] for issue in result["issues"]] == ["verifier_error"]


@pytest.mark.parametrize("raw_issue", ["just a string", 42, None, ["nested"]])
def test_malformed_issue_is_kept_not_dropped(raw_issue):
    result = verify_with({"status": "PASS", "issues": [raw_issue]})
    assert result["status"] == "FAIL"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["type"] == "unsupported_fact"


def test_model_cannot_claim_verifier_error_type():
    result = verify_with({"status": "FAIL", "issues": [{"type": "verifier_error", "explanation": "x"}]})
    assert result["issues"][0]["type"] == "unsupported_fact"


def test_verifier_exception_fails_closed():
    def boom(_):
        raise RuntimeError("API down")
    result = verify_with(boom)
    assert result["status"] == "FAIL"
    assert result["issues"][0]["type"] == "verifier_error"
    assert "API down" in result["issues"][0]["explanation"]


def test_unknown_citation_fails_even_if_model_passes():
    llm = FakeLLM()
    result = verify_section(llm, "Key Findings", "Claim [E1, E9].", PACKET)
    assert result["status"] == "FAIL"
    assert [(i["type"], i["evidence_id"]) for i in result["issues"]] == [("unknown_citation", "E9")]


def test_clean_pass():
    result = verify_with(PASS)
    assert result == {"status": "PASS", "issues": [], "cited_ids": ["E1"]}


def test_lowercase_status_is_accepted():
    assert verify_with({"status": "pass", "issues": []})["status"] == "PASS"


def test_cited_ids_grouped_forms():
    assert cited_evidence_ids("a [E1] b [E1, E3] c [E2; E4] [see E5]") == ["E1", "E3", "E2", "E4", "E5"]


# ---------------- Bounded evidence context (finding 9) ----------------
def _sent_ids(prompt: str):
    packet_json = prompt.split("Evidence Records", 1)[1].split(":\n", 1)[1]
    return [item["evidence_id"] for item in json.loads(packet_json)]


def test_section_gets_cited_records_plus_bounded_others():
    records = [record(n, claim=f"Revenue figure {n}", evidence=f"revenue was {n} dollars") for n in range(1, 101)]
    packet = build_evidence_packet(records)
    llm = FakeLLM()
    result = verify_section(llm, "Key Findings", "Revenue grew [E57] and [E99].", packet)
    assert result["status"] == "PASS"
    sent = _sent_ids(llm.prompts("verify")[0])
    assert sent[:2] == ["E57", "E99"]  # cited records are always included, labels unchanged
    assert len(sent) <= 2 + MAX_UNCITED_CONTEXT_RECORDS
    sent_items = {item["evidence_id"]: item for item in json.loads(
        llm.prompts("verify")[0].split("Evidence Records", 1)[1].split(":\n", 1)[1])}
    assert sent_items["E57"]["claim"] == "Revenue figure 57"


def test_known_ids_use_full_packet_not_the_slice():
    records = [record(n) for n in range(1, 60)]
    packet = build_evidence_packet(records)
    # E59 exists but shares no keywords; it must still count as a known citation.
    result = verify_section(FakeLLM(), "X", "zzz [E59]", packet)
    assert result["status"] == "PASS"


# ---------------- Bounded self-revision ----------------
def _report(text="Fact [E1]."):
    return {key: text for key, _ in REPORT_SECTION_KEYS}


def test_only_failed_sections_are_revised():
    def verify(prompt):
        return fail() if "Report section: Key Findings" in prompt and "Revised" not in prompt else PASS
    llm = FakeLLM(verify=verify)
    report, verification = verify_and_revise(llm, _report(), [record(1)])
    assert verification["overall_status"] == "PASS"
    assert llm.count("revise") == 1
    changed = [key for key, _ in REPORT_SECTION_KEYS if report[key] != "Fact [E1]."]
    assert changed == ["key_findings"]
    assert verification["revisions"][0]["revised_sections"] == ["Key Findings"]


def test_persistent_failure_stops_after_max_rounds_and_is_not_approved():
    llm = FakeLLM(verify=lambda _: fail())
    report, verification = verify_and_revise(llm, _report(), [record(1)])
    assert verification["overall_status"] == "FAIL"
    assert len(verification["revisions"]) == MAX_REVISION_ROUNDS == 2
    assert llm.count("revise") == 9 * MAX_REVISION_ROUNDS
    result = {"verification": verification, "final_report_markdown": "x", "evidence_records": [record(1)]}
    assert not is_report_approved(result)


def test_verifier_error_is_retried_not_revised():
    attempts = {"n": 0}

    def verify(prompt):
        if "Report section: Executive Summary" in prompt:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise TimeoutError("transient")
        return PASS
    llm = FakeLLM(verify=verify)
    report, verification = verify_and_revise(llm, _report(), [record(1)])
    assert verification["overall_status"] == "PASS"
    assert llm.count("revise") == 0
    assert report == _report()
    assert verification["revisions"][0]["reverified_sections"] == ["Executive Summary"]
    assert verification["revisions"][0]["revised_sections"] == []


def test_verifier_error_with_unknown_citation_is_revised():
    def verify(prompt):
        if "Revised" in prompt:
            return PASS
        return {"status": "PASS"}  # schema violation -> verifier_error
    llm = FakeLLM(verify=verify)
    report = _report()
    report["key_findings"] = "Fact [E7]."
    _, verification = verify_and_revise(llm, report, [record(1)])
    assert "Key Findings" in verification["revisions"][0]["revised_sections"]


# ---------------- Final gate stays fail-closed ----------------
GOOD = {
    "verification": {"overall_status": "PASS"},
    "final_report_markdown": "## Report",
    "evidence_records": [record(1)],
}


@pytest.mark.parametrize("result", [
    None,
    {},
    "PASS",
    {**GOOD, "verification": {"overall_status": "FAIL"}},
    {**GOOD, "verification": {}},
    {**GOOD, "verification": "PASS"},
    {**GOOD, "final_report_markdown": "  "},
    {**GOOD, "evidence_records": []},
    {**GOOD, "evidence_records": [{**record(1), "url": "ftp://x"}]},
    {**GOOD, "evidence_records": ["https://example.com"]},
])
def test_gate_rejects(result):
    assert not is_report_approved(result)


def test_gate_accepts_verified_report():
    assert is_report_approved(GOOD)


def test_revision_gets_cited_and_flagged_records_capped():
    from research_agent.verification import MAX_REVISION_CONTEXT_RECORDS
    records = [record(n, claim=f"Revenue figure {n}", evidence=f"revenue was {n}") for n in range(1, 101)]

    def verify(prompt):
        if "Revised" in prompt:
            return PASS
        return {"status": "FAIL", "issues": [
            {"type": "citation_mismatch", "claim": "revenue", "evidence_id": "E80", "explanation": "wrong"}]}
    llm = FakeLLM(verify=verify)
    report = _report("Revenue grew [E3].")
    verify_and_revise(llm, report, records)
    sent = _sent_ids(llm.prompts("revise")[0])
    assert sent[:2] == ["E3", "E80"]
    assert len(sent) <= 2 + MAX_REVISION_CONTEXT_RECORDS
