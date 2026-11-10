from fakes import FakeLLM, fake_tools, make_services

from research_agent.pipeline import build_evidence_graph, initial_evidence_state
from research_agent.research import run_research


class HTTPError(OSError):
    """Shaped like requests.HTTPError from raise_for_status()."""

    def __init__(self, status_code):
        super().__init__(f"{status_code} Server Error")
        self.response = type("Response", (), {"status_code": status_code})()


PLAN = [
    {"task": "news", "source_type": "web", "ticker": None},
    {"task": "filings", "source_type": "filings", "ticker": "NVDA"},
    {"task": "price", "source_type": "financials", "ticker": "NVDA"},
]


def _filings_failing(status_code, calls):
    def filings(ticker):
        calls.append(ticker)
        raise HTTPError(status_code)
    return filings


def test_tool_exception_is_isolated_to_its_task():
    calls, sleeps = [], []
    tools = fake_tools(filings=_filings_failing(503, calls),
                       financials=lambda t: [{"title": "snap", "url": "https://f", "content": "{}"}])
    results = run_research(PLAN, tools, sleep=sleeps.append)
    assert [item["task"] for item in results] == ["news", "filings", "price"]
    assert "error" not in results[0]["results"][0]
    assert results[1]["results"] == [{"error": "HTTPError: 503 Server Error"}]
    assert results[2]["results"][0]["title"] == "snap"
    assert len(calls) == 2 and len(sleeps) == 1  # one bounded retry for a transient 503


def test_non_transient_error_is_not_retried():
    calls, sleeps = [], []
    results = run_research(PLAN, fake_tools(filings=_filings_failing(404, calls)), sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []
    assert "404" in results[1]["results"][0]["error"]


def test_transient_error_recovers_on_retry():
    calls = []

    def flaky(ticker):
        calls.append(ticker)
        if len(calls) == 1:
            raise ConnectionError("reset")
        return [{"title": "ok", "url": "https://ok", "content": "x"}]
    results = run_research(PLAN, fake_tools(filings=flaky), sleep=lambda _: None)
    assert results[1]["results"][0]["title"] == "ok"


def test_programming_error_in_tool_does_not_abort_workflow(tmp_path):
    def broken_web(query):
        raise KeyError("results")
    services = make_services(FakeLLM(), str(tmp_path / "t.db"), fake_tools(web=broken_web))
    state = build_evidence_graph(services).invoke(initial_evidence_state("q"))
    assert state["evidence_records"] == []
    assert state["coverage"][0]["tool_errors"] == ["KeyError: 'results'"]
