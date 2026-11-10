"""Streamlit front end for the Team 1 research agent.

One workflow: ask a question, watch the verified pipeline run, read the report
with clickable citations, inspect sources / evidence / quality, and ask
evidence-first follow-ups. Past reports can be reopened from the sidebar.
This is the only module that imports Streamlit.
"""

import csv
import io
import sys
import time
from pathlib import Path
from typing import List

# `streamlit run ui/streamlit_app.py` puts only ui/ on sys.path; add the project
# root so `research_agent` imports however the app is launched.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from dotenv import load_dotenv

from research_agent.config import MissingSettingsError, Settings
from research_agent.evidence import coverage_gaps, describe_gap
from research_agent.followup import run_followup
from research_agent.pipeline import (
    Services,
    build_services,
    build_verified_report_graph,
    initial_verified_report_state,
)
from research_agent.verification import citation_check, is_report_approved
from ui.formatting import (
    evidence_rows,
    for_display,
    link_citations,
    short,
    source_rows,
    without_sources,
)
from ui.styles import apply_styles

EXAMPLE_QUESTIONS = [
    "How is NVIDIA positioning itself against AMD in the AI chip market?",
    "What are the main risks to Tesla's EV market share in 2026?",
    "How are Microsoft and Google competing in enterprise AI assistants?",
]

# Session-state keys
QUESTION = "question"          # the question text area
RUN = "current_run"  # {"question", "result", "seconds", "log", "origin"}
RUN_EVIDENCE = "run_evidence"  # report evidence + anything follow-ups added
FOLLOWUPS = "followups"        # follow-up answers, oldest first

HOW_IT_WORKS = """
1. **Plan** the question into 4–6 research tasks (web, financials, SEC filings).
2. **Research** each task and **extract evidence**. Every passage is checked
   against the source text, and ungrounded quotes are dropped.
3. **Write** a nine-section report that cites evidence as [E1], [E2], ...
4. **Verify** every section's citations. Failing sections are rewritten and
   re-checked up to twice.
5. Only a report that **passes** can be downloaded as quality-checked.

Follow-ups answer from the report's evidence first, run at most one extra
research pass, and are citation-checked before they are shown.
"""


# ---------------- Services ----------------
@st.cache_resource
def _services() -> Services:
    load_dotenv()
    return build_services(Settings.from_env())


@st.cache_resource
def _graph():
    return build_verified_report_graph(_services())


# ---------------- Running ----------------
def _describe_step(node: str, update: dict) -> str:
    if node == "plan":
        return f"Planned {len(update.get('research_plan') or [])} research tasks"
    if node == "research":
        results = update.get("research_results") or []
        sources = sum(len(item.get("results") or []) for item in results)
        errors = sum(1 for item in results for r in item.get("results") or [] if "error" in r)
        return f"Ran {len(results)} tasks · {sources - errors} sources retrieved, {errors} tool errors"
    if node == "evidence":
        records = update.get("evidence_records") or []
        gaps = coverage_gaps(update.get("coverage") or [])
        note = f" · {len(gaps)} task(s) with no evidence" if gaps else ""
        return f"Extracted {len(records)} grounded evidence records{note}"
    if node == "report":
        return "Drafted the nine-section report"
    if node == "verify":
        verification = update.get("verification") or {}
        rounds = len(verification.get("revisions") or [])
        return (
            f"Verification {verification.get('overall_status', 'FAIL')} · "
            f"{verification.get('initial_issue_count', 0)} issue(s) found, "
            f"{rounds} revision round(s), {verification.get('issue_count', 0)} left"
        )
    return node


_NEXT_STEP = {
    "plan": "Researching each task...",
    "research": "Extracting and grounding evidence...",
    "evidence": "Writing the report...",
    "report": "Checking every citation (and revising if needed)...",
}


def run_research(question: str) -> None:
    """Stream the verified pipeline with live progress, then store the run."""
    started = time.perf_counter()
    state = dict(initial_verified_report_state(question))
    log: List[str] = []
    with st.status("Planning the research...", expanded=True) as status:
        try:
            for chunk in _graph().stream(state, stream_mode="updates"):
                for node, update in chunk.items():
                    state.update(update or {})
                    line = _describe_step(node, update or {})
                    log.append(line)
                    st.write(f"✓ {line}")
                    status.update(label=_NEXT_STEP.get(node, "Finishing..."))
        except Exception as error:
            status.update(label="Research failed", state="error")
            st.error(f"Research failed; no approved report was produced: {error}")
            return
        status.update(label="Done", state="complete", expanded=False)

    _open_run({
        "question": question,
        "result": state,
        "seconds": round(time.perf_counter() - started, 1),
        "log": log,
        "origin": "live",
    })
    st.rerun()  # refresh the sidebar history with the new report


def _open_run(run: dict) -> None:
    st.session_state[RUN] = run
    st.session_state[RUN_EVIDENCE] = list(run["result"].get("evidence_records") or [])
    st.session_state[FOLLOWUPS] = []


def _open_saved(report_id: int) -> None:
    saved = _services().store.get_report(report_id)
    if not saved:
        st.toast("That report could not be found.")
        return
    passed = saved["status"] == "PASS"
    result = {
        "question": saved["question"],
        "report_id": saved["id"],
        "report": {},
        "report_markdown": saved["report_markdown"],
        "final_report_markdown": saved["report_markdown"] if passed else "",
        "verification": saved["verification"],
        "evidence_records": saved["evidence_records"],
        "coverage": [],
    }
    _open_run({
        "question": saved["question"],
        "result": result,
        "seconds": None,
        "log": [],
        "origin": f"Saved report #{saved['id']} · {saved['created_at']}",
    })
    st.session_state[QUESTION] = saved["question"]


# ---------------- Sidebar ----------------
def _status_mark(status: str) -> str:
    return {"PASS": "✅", "FAIL": "⛔", "pending": "⏳"}.get(status, "·")


def render_sidebar() -> None:
    store = _services().store
    with st.sidebar:
        st.header("History")
        reports = store.list_reports(limit=15)
        if not reports:
            st.caption("Reports you run will appear here.")
        current = (st.session_state.get(RUN) or {}).get("result", {}).get("report_id")
        for report in reports:
            st.button(
                f"{_status_mark(report['status'])} {short(report['question'], 38)}  \n"
                f"#{report['id']} · {report['created_at'][:10]}",
                key=f"history_{report['id']}",
                help=(f"#{report['id']} · {report['status']} · "
                      f"{report['evidence_count']} evidence · {report['created_at']}"),
                on_click=_open_saved,
                args=(report["id"],),
                type="primary" if report["id"] == current else "secondary",
                width="stretch",
            )
        st.caption(f"{store.count_evidence()} evidence records in the ledger")


# ---------------- Question ----------------
def _use_example(example: str) -> None:
    st.session_state[QUESTION] = example


def render_question() -> str:
    if QUESTION not in st.session_state:
        st.session_state[QUESTION] = EXAMPLE_QUESTIONS[0]
    question = st.text_area(
        "Research question",
        key=QUESTION,
        height=90,
        placeholder="What would you like to investigate? Name the companies, market, or event.",
        label_visibility="collapsed",
    ).strip()
    st.caption("Try an example:")
    examples = st.columns(len(EXAMPLE_QUESTIONS))
    for index, (column, example) in enumerate(zip(examples, EXAMPLE_QUESTIONS)):
        column.button(
            short(example, 48),
            key=f"example_{index}",
            help=example,
            on_click=_use_example,
            args=(example,),
            type="tertiary",
            width="stretch",
        )
    left, right = st.columns([1, 3], vertical_alignment="center")
    if left.button("Run research", type="primary", disabled=not question, key="run", width="stretch"):
        run_research(question)
    right.caption("A full run takes about 3–4 minutes: research, evidence, report, and citation checks.")
    return question


# ---------------- Results ----------------
def render_run(question: str) -> None:
    run = st.session_state.get(RUN)
    if not run:
        return
    result = run["result"]
    evidence = result.get("evidence_records") or []
    verification = result.get("verification") or {}
    approved = is_report_approved(result)

    st.divider()
    st.subheader(run["question"])
    if run["origin"] != "live":
        st.caption(run["origin"])
    if run["question"] != question:
        st.warning(
            "The question box has changed. The report below answers the earlier question. "
            "Click Run research to research the new one."
        )

    initial = verification.get("initial_issue_count", verification.get("issue_count", 0))
    remaining = verification.get("issue_count", 0)
    stats = st.columns(5)
    stats[0].metric("Status", verification.get("overall_status") or "Unverified")
    stats[1].metric("Sources", len({r.get("url") for r in evidence if r.get("url")}))
    stats[2].metric("Evidence", len(evidence))
    stats[3].metric("Issues fixed", f"{max(initial - remaining, 0)}/{initial}")
    stats[4].metric("Time", f"{run['seconds']:.0f}s" if run.get("seconds") else "—")

    if approved:
        st.success("Quality gate passed: every citation was checked against the evidence. "
                   "Prototype output, not independently fact-checked.", icon="✅")
    else:
        st.error("Quality gate did not pass, so this is NOT an approved report. "
                 "See the Quality check tab for the open issues.", icon="⛔")

    run_evidence = st.session_state.get(RUN_EVIDENCE) or []
    followups = st.session_state.get(FOLLOWUPS) or []
    report_tab, followup_tab, sources_tab, evidence_tab, quality_tab = st.tabs([
        "Report",
        f"Follow-ups ({len(followups)})",
        f"Sources ({len(source_rows(evidence))})",
        f"Evidence ({len(run_evidence)})",
        "Quality check",
    ])
    with report_tab:
        render_report(result, approved)
    with followup_tab:
        render_followups(result, approved)
    with sources_tab:
        render_sources(evidence)
    with evidence_tab:
        render_evidence(run_evidence)
    with quality_tab:
        render_quality(run, result, verification)


def render_report(result: dict, approved: bool) -> None:
    evidence = result.get("evidence_records") or []
    if not approved:
        with st.expander("Unapproved draft (for inspection only, not downloadable)"):
            st.markdown(for_display(without_sources(result.get("report_markdown", "")), evidence))
        return

    text = result["final_report_markdown"]
    downloads = st.columns([1, 1, 3])
    downloads[0].download_button(
        "Download report (.md)",
        data=link_citations(text, evidence),
        file_name=f"research_report_{result.get('report_id', 0)}.md",
        mime="text/markdown",
        key="download_report",
        width="stretch",
    )
    downloads[1].download_button(
        "Download evidence (.csv)",
        data=_csv(evidence_rows(evidence)),
        file_name=f"research_evidence_{result.get('report_id', 0)}.csv",
        mime="text/csv",
        key="download_evidence",
        width="stretch",
    )
    st.caption("Citations like [E3] link to their source.")
    st.markdown(for_display(without_sources(text), evidence))


def render_followups(result: dict, approved: bool) -> None:
    st.caption(
        "Answers come from this report's evidence first. If that isn't enough, the agent runs "
        "one extra targeted research pass. Every answer is citation-checked before it's shown."
    )
    if not approved:
        st.warning("The report did not pass quality review, so follow-ups are not certified.")

    for item in st.session_state.get(FOLLOWUPS) or []:
        render_followup_answer(item)

    asked = st.chat_input("Ask a follow-up about this report", key="followup_input")
    if not asked:
        return
    with st.chat_message("user"):
        st.markdown(asked)
    base = {**result, "evidence_records": st.session_state.get(RUN_EVIDENCE) or []}
    try:
        with st.spinner("Checking the evidence..."):
            answer = run_followup(_services(), asked, base)
    except Exception as error:
        st.error(f"Follow-up could not be answered: {error}")
        return
    st.session_state[RUN_EVIDENCE] = answer["evidence_records"]
    st.session_state[FOLLOWUPS] = (st.session_state.get(FOLLOWUPS) or []) + [
        {**answer, "followup_question": asked.strip(), "evidence_records": None}
    ]
    st.rerun()


def render_followup_answer(item: dict) -> None:
    evidence = st.session_state.get(RUN_EVIDENCE) or []
    with st.chat_message("user"):
        st.markdown(item["followup_question"])
    with st.chat_message("assistant"):
        if item.get("supported"):
            st.markdown(for_display(item["answer"], evidence))
        else:
            st.error(item["answer"])
            for issue in (item.get("verification") or {}).get("issues") or []:
                st.markdown(
                    f"- **{issue.get('type', 'issue')}** ({issue.get('evidence_id') or 'no evidence ID'}): "
                    f"{issue.get('explanation', '')}"
                )
        meta = ["extra research: " + ("yes" if item.get("additional_research_performed") else "no")]
        if item.get("new_evidence_count"):
            meta.append(f"{item['new_evidence_count']} new evidence records")
        if item.get("cited_ids"):
            meta.append("cited " + ", ".join(item["cited_ids"]))
        st.caption(" · ".join(meta))
        if item.get("research_error"):
            st.caption(f"Research note: {item['research_error']}")


def render_sources(evidence: list) -> None:
    rows = source_rows(evidence)
    if not rows:
        st.info("No sources were recorded for this report.")
        return
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "url": st.column_config.LinkColumn("Link", display_text="open"),
            "labels": st.column_config.TextColumn("Cited as"),
            "records": st.column_config.NumberColumn("Evidence"),
        },
        column_order=["source", "type", "date", "records", "labels", "url"],
    )


def render_evidence(evidence: list) -> None:
    rows = evidence_rows(evidence)
    if not rows:
        st.info("No evidence was recorded for this report.")
        return
    filters = st.columns([2, 1, 1])
    search = filters[0].text_input("Search", key="evidence_search", placeholder="Search claims and passages")
    picked_conf = filters[1].multiselect(
        "Confidence", sorted({r["confidence"] or "unrated" for r in rows}), key="evidence_conf")
    picked_type = filters[2].multiselect(
        "Type", sorted({r["type"] or "unknown" for r in rows}), key="evidence_type")

    needle = search.strip().lower()
    shown = [
        r for r in rows
        if (not needle or needle in f"{r['claim']} {r['evidence']} {r['source']}".lower())
        and (not picked_conf or (r["confidence"] or "unrated") in picked_conf)
        and (not picked_type or (r["type"] or "unknown") in picked_type)
    ]
    st.caption(f"Showing {len(shown)} of {len(rows)} records. E-labels match the report's citations.")
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "url": st.column_config.LinkColumn("Link", display_text="open"),
            "claim": st.column_config.TextColumn("Claim", width="large"),
            "evidence": st.column_config.TextColumn("Supporting passage", width="large"),
        },
        column_order=["id", "claim", "source", "confidence", "type", "date", "evidence", "url"],
    )


def render_quality(run: dict, result: dict, verification: dict) -> None:
    if not verification:
        st.info("This saved report has no verification record (it was created before verification existed).")
        return

    issues = verification.get("issues") or []
    if issues:
        st.markdown(f"**{len(issues)} open issue(s)**")
        st.markdown("\n".join(
            f"- **{issue.get('section', 'Section')} · {issue.get('type', 'issue')}** "
            f"({issue.get('evidence_id') or 'no evidence ID'}): {issue.get('explanation', '')}"
            for issue in issues
        ))
    else:
        st.markdown("**No open citation issues.**")

    revisions = verification.get("revisions") or []
    if revisions:
        lines = []
        for r in revisions:
            parts = []
            if r.get("revised_sections"):
                parts.append("rewrote " + ", ".join(r["revised_sections"]))
            if r.get("reverified_sections"):
                parts.append("re-checked " + ", ".join(r["reverified_sections"]))
            lines.append(f"- Round {r['round']}: {'; '.join(parts) or 'no changes'} "
                         f"({r['issues_before']} → {r['issues_after']} issues)")
        st.markdown("**Revision history**\n\n" + "\n".join(lines))

    coverage = result.get("coverage") or []
    if coverage:
        gaps = coverage_gaps(coverage)
        st.markdown("**Research coverage**" + (f": {len(gaps)} task(s) produced no evidence" if gaps else ""))
        if gaps:
            st.markdown("\n".join(f"- {describe_gap(entry)}" for entry in gaps))
        st.dataframe(
            [
                {
                    "task": entry["task"],
                    "type": entry["source_type"],
                    "sources": entry["sources_attempted"],
                    "evidence kept": entry["records"],
                    "ungrounded dropped": entry["ungrounded_dropped"],
                    "tool errors": len(entry["tool_errors"]),
                }
                for entry in coverage
            ],
            hide_index=True,
            width="stretch",
        )

    if run.get("log"):
        st.markdown("**Run log**\n\n" + "\n".join(f"- {line}" for line in run["log"]))

    with st.expander("Raw verification data"):
        if result.get("report"):
            st.json(citation_check(result))
        st.json(verification)


def _csv(rows: List[dict]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


# ---------------- Page ----------------
def main() -> None:
    st.set_page_config(page_title="Research Workspace", page_icon="🔎", layout="wide")
    apply_styles()

    title, help_col = st.columns([5, 1], vertical_alignment="bottom")
    title.title("Research workspace")
    with help_col.popover("How it works", width="stretch"):
        st.markdown(HOW_IT_WORKS)

    try:
        _services()
    except MissingSettingsError as error:
        st.error(str(error))
        st.stop()

    render_sidebar()
    question = render_question()
    render_run(question)


if __name__ == "__main__":
    main()
