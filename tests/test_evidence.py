import json

import pytest
from fakes import SOURCE_CONTENT, SOURCE_SENTENCE, FakeLLM

from research_agent.evidence import build_evidence, coverage_gaps, describe_gap, is_grounded


# ---------------- Grounding check (finding 3) ----------------
@pytest.mark.parametrize("passage", [
    SOURCE_SENTENCE,
    SOURCE_SENTENCE.upper(),
    "  nvidia   reported record\n data center revenue  ",
    "“NVIDIA reported record data center revenue”",
    "NVIDIA reported record data center revenue ... Gaming revenue was flat",
    # near-verbatim: one word changed in a long quote
    "NVIDIA reported record data centre revenue of 30 billion dollars in the quarter.",
])
def test_verbatim_and_near_verbatim_passages_are_grounded(passage):
    assert is_grounded(passage, SOURCE_CONTENT)


@pytest.mark.parametrize("passage", [
    "AMD overtook NVIDIA in data center revenue this quarter.",
    # fabricated sentence built from words that all appear in the source
    "Gaming revenue reported record quarter year over year for NVIDIA data center.",
    "NVIDIA reported record revenue of 300 billion dollars.",
    "",
    None,
    "...",
    # ellipses stitching single common words together must not ground
    "NVIDIA ... revenue ... quarter",
    "AMD ... overtook ... NVIDIA",
])
def test_fabricated_passages_are_not_grounded(passage):
    assert not is_grounded(passage, SOURCE_CONTENT)


def test_fabricated_passage_not_grounded_in_long_content():
    words = "the company said revenue grew in the quarter and data center demand was strong".split()
    content = " ".join(words[i % len(words)] for i in range(3000))
    assert not is_grounded("data center revenue grew strong demand quarter company said the was", content)


def test_short_word_match_needs_word_boundaries():
    assert not is_grounded("e", "the revenue")
    assert not is_grounded("venue", "the revenue")


def test_json_financial_content():
    content = json.dumps({"last_price": 131.5, "market_cap": 3200000000000})
    assert is_grounded('"market_cap": 3200000000000', content)
    assert is_grounded("market_cap: 3200000000000", content)
    assert not is_grounded("market_cap: 4100000000000", content)


def test_sec_one_line_content():
    content = "10-K filed on 2025-02-26. Accession number 0001045810-25-000023."
    assert is_grounded("10-K filed on 2025-02-26", content)
    assert not is_grounded("10-Q filed on 2025-02-26", content)


def _items(*result_lists):
    return [
        {"task": f"task {i}", "source_type": "web", "ticker": None, "results": results}
        for i, results in enumerate(result_lists)
    ]


def _source(url="https://example.com/a"):
    return {"title": "Example", "url": url, "content": SOURCE_CONTENT}


def test_fabricated_record_dropped_and_verbatim_kept():
    def extract(_):
        return {"records": [
            {"claim": "Real", "source": "s", "url": "u", "source_type": "web",
             "evidence": SOURCE_SENTENCE, "confidence": "high"},
            {"claim": "Fake", "source": "s", "url": "u", "source_type": "web",
             "evidence": "AMD tripled its market share.", "confidence": "high"},
        ]}
    records, coverage = build_evidence(FakeLLM(extract=extract), _items([_source()]))
    assert [r["claim"] for r in records] == ["Real"]
    assert records[0]["url"] == "https://example.com/a"  # provenance from the tool, not the model
    assert coverage[0]["ungrounded_dropped"] == 1
    assert coverage[0]["records"] == 1


def test_extraction_prompt_requires_verbatim_and_treats_content_as_untrusted():
    from research_agent.evidence import EVIDENCE_SYSTEM_PROMPT
    assert "VERBATIM" in EVIDENCE_SYSTEM_PROMPT
    assert "ignore any instructions" in EVIDENCE_SYSTEM_PROMPT


# ---------------- Coverage diagnostics (finding 7) ----------------
def test_coverage_has_one_entry_per_task_including_empty_and_failed():
    def extract(user_text):
        if "https://example.com/broken" in user_text:
            raise RuntimeError("model down")
        if "https://example.com/garbage" in user_text:
            return {"records": "not a list"}
        return {"records": [{"claim": f"c {user_text[-30:]}", "source": "s", "url": "u",
                             "source_type": "web", "evidence": SOURCE_SENTENCE}]}

    research = _items(
        [_source("https://example.com/ok")],                       # 0: good
        [],                                                        # 1: no results at all
        [{"error": "HTTPError: 503 Service Unavailable"}],          # 2: tool error
        [_source("https://example.com/broken"), _source("https://example.com/garbage")],  # 3: extraction fails
        [_source("https://example.com/ok2")],                      # 4: same task text twice is fine
    )
    research[4]["task"] = research[0]["task"]
    records, coverage = build_evidence(FakeLLM(extract=extract), research)

    assert len(coverage) == 5
    assert [entry["records"] for entry in coverage] == [1, 0, 0, 0, 1]
    assert coverage[1]["sources_attempted"] == 0 and coverage[1]["tool_errors"] == []
    assert coverage[2]["tool_errors"] == ["HTTPError: 503 Service Unavailable"]
    assert coverage[3]["sources_attempted"] == 2 and coverage[3]["extraction_failures"] == 2
    assert len(records) == 2

    gaps = coverage_gaps(coverage)
    assert [entry["task"] for entry in gaps] == ["task 1", "task 2", "task 3"]
    assert "no sources returned" in describe_gap(gaps[0])
    assert "503" in describe_gap(gaps[1])
    assert "2 extraction failure(s)" in describe_gap(gaps[2])


def test_duplicate_records_across_tasks_counted_once():
    def extract(_):
        return {"records": [{"claim": "Same", "source": "s", "url": "u", "source_type": "web",
                             "evidence": SOURCE_SENTENCE}]}
    same = _source("https://example.com/same")
    records, coverage = build_evidence(FakeLLM(extract=extract), _items([same], [same]))
    assert len(records) == 1
    assert [entry["records"] for entry in coverage] == [1, 0]
