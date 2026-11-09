"""Streamlit front end for the Team 1 research agent.

This is the only module that imports Streamlit. It loads settings, builds the
services once per server process, and renders one section per workflow.
"""

from collections import Counter
from typing import List

import streamlit as st
from dotenv import load_dotenv

from research_agent.config import MissingSettingsError, Settings
from research_agent.pipeline import (
    Services,
    build_evidence_graph,
    build_report_graph,
    build_response_graph,
    build_services,
    initial_evidence_state,
    initial_report_state,
    initial_research_state,
)
from research_agent.report import build_evidence_packet
from research_agent.types import EvidenceRecord
from ui.styles import apply_styles

DEFAULT_QUESTION = "How is NVIDIA positioning itself against AMD in the AI chip market?"


@st.cache_resource
def _services() -> Services:
    load_dotenv()
    return build_services(Settings.from_env())


@st.cache_resource
def _graphs() -> dict:
    services = _services()
    return {
        "response": build_response_graph(services),
        "evidence": build_evidence_graph(services),
        "report": build_report_graph(services),
    }


def render_research_section(question: str) -> None:
    if not st.button("Research question", type="primary", disabled=not question):
        return

    try:
        with st.spinner("Planning and researching the full question..."):
            result = _graphs()["response"].invoke(initial_research_state(question))
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
        return

    st.subheader("Research response")
    st.write(result["response"])
    with st.expander("Planner output"):
        st.json(result["research_plan"])
    with st.expander("Research results"):
        st.json(result["research_results"])


def render_evidence_section(question: str) -> None:
    if st.button("Build evidence ledger", type="primary", disabled=not question):
        try:
            with st.spinner("Researching and building the evidence ledger..."):
                result = _graphs()["evidence"].invoke(initial_evidence_state(question))
            st.subheader("New evidence records")
            st.json(result["evidence_records"])
        except Exception as error:
            st.error(f"The evidence workflow could not finish: {error}")

    store = _services().store
    stored_records = store.load_evidence()
    st.subheader("Evidence ledger")
    if stored_records:
        render_dashboard(stored_records)
        st.dataframe(stored_records, width="stretch", hide_index=True)
    else:
        st.info("No evidence has been saved yet. Build a ledger to get started.")
    st.caption(f"{len(stored_records)} records · SQLite: {store.db_path}")


def render_dashboard(evidence_records: List[EvidenceRecord]) -> None:
    by_type = Counter(record.get("source_type") for record in evidence_records)
    by_confidence = Counter(record.get("confidence") for record in evidence_records)
    unique_sources = {record["url"] for record in evidence_records if record.get("url")}

    st.subheader("Research Dashboard")

    first_row = st.columns(3)
    first_row[0].metric("Evidence Records", len(evidence_records))
    first_row[1].metric("Unique Sources", len(unique_sources))
    first_row[2].metric("SEC Filings", by_type["filings"])

    second_row = st.columns(3)
    second_row[0].metric("Web Evidence", by_type["web"])
    second_row[1].metric("Financial Evidence", by_type["financials"])
    second_row[2].metric("High Confidence", by_confidence["high"])

    st.caption(
        f"Confidence breakdown: {by_confidence['high']} high, "
        f"{by_confidence['medium']} medium, {by_confidence['low']} low"
    )


def render_report_section(question: str) -> None:
    if not st.button("Generate report", type="primary", disabled=not question):
        return

    try:
        with st.spinner("Planning, researching, extracting evidence, and writing the report..."):
            result = _graphs()["report"].invoke(initial_report_state(question))
    except Exception as error:
        st.error(f"The report pipeline could not finish: {error}")
        return

    store = _services().store
    st.success(f"Report #{result['report_id']} saved.")
    render_dashboard(result["evidence_records"])
    st.markdown(result["report_markdown"])

    with st.expander(f"Evidence used ({len(result['evidence_records'])} records)"):
        st.json(build_evidence_packet(result["evidence_records"]))

    with st.expander("Saved reports"):
        saved_reports = store.load_reports()
        if not saved_reports:
            st.info("No saved reports yet.")
        for saved in saved_reports:
            st.markdown(
                f"**#{saved['id']}** — {saved['question']}  \n"
                f"{saved['evidence_count']} evidence records · {saved['created_at']}"
            )


def main() -> None:
    st.set_page_config(page_title="Research Workspace", page_icon="🔎", layout="wide")
    apply_styles()
    st.title("Research workspace")
    st.caption("Plan, gather evidence, and turn findings into a report.")

    try:
        _services()
    except MissingSettingsError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.header("Research question")
        question = st.text_area(
            "",
            value=DEFAULT_QUESTION,
            height=130,
            placeholder="What would you like to investigate?",
            label_visibility="collapsed",
        ).strip()
        st.caption("One question is shared across all workflows.")

    research_tab, evidence_tab, report_tab = st.tabs(
        ["Research", "Evidence ledger", "Report"]
    )
    with research_tab:
        render_research_section(question)
    with evidence_tab:
        render_evidence_section(question)
    with report_tab:
        render_report_section(question)


if __name__ == "__main__":
    main()
