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
    if not (st.button("Research", type="primary") and question):
        return

    try:
        with st.spinner("Planning and researching the full question..."):
            result = _graphs()["response"].invoke(initial_research_state(question))
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
        return

    st.subheader("Planner output")
    st.json(result["research_plan"])

    st.subheader("Research from all tasks")
    st.json(result["research_results"])

    st.subheader("Response")
    st.write(result["response"])


def render_evidence_section(question: str) -> None:
    st.divider()
    st.subheader("Week 4 — Evidence Ledger")
    st.caption("Week 3 research → Evidence Extraction → SQLite Evidence Ledger")

    if not (st.button("Build Evidence Ledger", type="primary") and question):
        return

    try:
        with st.spinner("Researching and building the evidence ledger..."):
            result = _graphs()["evidence"].invoke(initial_evidence_state(question))
    except Exception as error:
        st.error(f"The Week 4 evidence workflow could not finish: {error}")
        return

    st.subheader("New Evidence Records")
    st.json(result["evidence_records"])

    store = _services().store
    stored_records = store.load_evidence()
    st.subheader("Evidence Ledger (SQLite)")
    st.dataframe(stored_records, use_container_width=True)
    st.caption(f"{len(stored_records)} evidence records stored in {store.db_path}")


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
    st.divider()
    st.subheader("Week 5 — Complete Report Pipeline")
    st.caption("Question → Planner → Research → Evidence Ledger → Report")

    if not (st.button("Run Full Report Pipeline", type="primary") and question):
        return

    try:
        with st.spinner("Planning, researching, extracting evidence, and writing the report..."):
            result = _graphs()["report"].invoke(initial_report_state(question))
    except Exception as error:
        st.error(f"The Week 5 report pipeline could not finish: {error}")
        return

    store = _services().store
    st.success(f"Report #{result['report_id']} saved to {store.db_path}.")
    render_dashboard(result["evidence_records"])
    st.markdown(result["report_markdown"])

    with st.expander(f"Evidence used ({len(result['evidence_records'])} records)"):
        st.json(build_evidence_packet(result["evidence_records"]))

    with st.expander("Saved reports (SQLite)"):
        for saved in store.load_reports():
            st.markdown(
                f"**#{saved['id']}** — {saved['question']}  \n"
                f"{saved['evidence_count']} evidence records · {saved['created_at']}"
            )


def main() -> None:
    st.set_page_config(page_title="ELP Team 1 — Week 3", page_icon="🔎")
    st.title("Team 1 — Focused AI Research Agent")
    st.caption("Planner → Appropriate Tools → Research → Response")

    try:
        _services()
    except MissingSettingsError as error:
        st.error(str(error))
        st.stop()

    question = st.text_input(
        "Enter a business research question",
        value=DEFAULT_QUESTION,
    ).strip()

    render_research_section(question)
    render_evidence_section(question)
    render_report_section(question)


if __name__ == "__main__":
    main()
