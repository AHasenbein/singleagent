"""Streamlit AppTest of the research workspace with fake services (no network)."""

from pathlib import Path

import pytest
import streamlit as st
from fakes import FakeLLM, fail, make_services
from streamlit.testing.v1 import AppTest

import research_agent.pipeline as pipeline

APP = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Returns start(llm, run=True) -> AppTest, optionally after a research run."""
    def start(llm, run=True):
        st.cache_resource.clear()  # _services/_graph would otherwise leak between tests
        monkeypatch.setattr(pipeline, "build_services", lambda settings: make_services(llm, str(tmp_path / "ui.db")))
        at = AppTest.from_file(APP, default_timeout=30).run()
        assert not at.exception, at.exception
        if run:
            at.button(key="run").click().run()
            assert not at.exception, at.exception
        return at
    yield start
    st.cache_resource.clear()


def _ask(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at


def _markdown(at):
    return [m.value for m in at.markdown]


def _downloads(at):
    return at.get("download_button")


def test_pass_shows_report_downloads_and_linked_citations(app):
    at = app(FakeLLM())
    assert at.success and len(_downloads(at)) == 2
    md = _markdown(at)
    assert any("Executive Summary" in m for m in md)
    # [E1] is a link to the source; the reference list moved to the Sources tab
    assert any("[[E1]](<https://" in m for m in md)
    assert not any("## Sources / References" in m for m in md)


def test_fail_has_no_download_and_draft_is_hidden(app):
    at = app(FakeLLM(verify=lambda _: fail()))
    assert at.error and len(_downloads(at)) == 0
    assert any("Unapproved draft" in e.label for e in at.expander)


def test_followups_accumulate_and_survive_reruns(app):
    at = app(FakeLLM())
    _ask(at, "What drove revenue?")
    _ask(at, "And what about margins?")
    md = _markdown(at)
    assert any("What drove revenue?" in m for m in md)
    assert any("And what about margins?" in m for m in md)
    assert sum("Revenue hit a record" in m for m in md) == 2
    assert any("Executive Summary" in m for m in md), "report lost after rerun"


def test_unsupported_followup_is_hidden(app):
    llm = FakeLLM()
    at = app(llm)
    llm.handlers["answer"] = lambda _: {"answer": "Invented claim [E99].", "cited_ids": ["E99"]}
    _ask(at, "Something unsupported?")
    assert not any("Invented claim" in m for m in _markdown(at))
    assert any("Could not produce a supported answer" in e.value for e in at.error)


def test_changed_question_flags_stale_report(app):
    at = app(FakeLLM())
    assert not any("earlier question" in w.value for w in at.warning)
    at.text_area(key="question").input("A completely different question").run()
    assert any("earlier question" in w.value for w in at.warning)


def test_history_reopens_saved_report(app):
    at = app(FakeLLM())
    history = [b for b in at.button if b.key and b.key.startswith("history_")]
    assert len(history) == 1 and history[0].label.startswith("✅")

    at = app(FakeLLM(), run=False)  # fresh session, same database
    assert not at.success
    [button] = [b for b in at.button if b.key and b.key.startswith("history_")]
    button.click().run()
    assert not at.exception, at.exception
    assert at.success and len(_downloads(at)) == 2
    assert any("Saved report #" in c.value for c in at.caption)
    assert any("[[E1]](<https://" in m for m in _markdown(at))


def test_example_pill_fills_question(app):
    at = app(FakeLLM(), run=False)
    at.button(key="example_1").click().run()
    assert at.text_area(key="question").value.startswith("What are the main risks to Tesla")
