"""ELP Week 3 — Team 1 Student Starter
Focused AI Research Agent: Planner + Tools

Week 2 already gave you:
    Question -> Planner -> Tavily -> DeepSeek Response

Week 3 goal:
    Question -> Planner -> Appropriate Research Tools -> DeepSeek Response

Your job is to complete the NEW Week 3 TODOs only.

Run:
    python -m pip install -r requirements_team1.txt
    streamlit run team1_week3_student.py

.env:
    TAVILY_API_KEY=your_key
    DEEPSEEK_API_KEY=your_key

Important:
- Never commit .env.
- SEC asks automated clients to identify themselves. SEC_USER_AGENT is optional here;
  if you add one, use something like: Your Name your-email@example.com
"""

import json
import os
from typing import TypedDict

import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tavily import TavilyClient


# ---------------- Configuration ----------------
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Nittany AI ELP educational-project contact@example.com",
)

if not TAVILY_API_KEY or not DEEPSEEK_API_KEY:
    st.error(
        "Missing API key. Add TAVILY_API_KEY and DEEPSEEK_API_KEY "
        "to a .env file beside this script."
    )
    st.stop()

tavily = TavilyClient(api_key=TAVILY_API_KEY)
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# ---------------- Team 1 Planner ----------------
# This extends Week 2 homework: each task now includes a suggested source type.
TEAM_1_SYSTEM_PROMPT = """
You are the Planner for a single-agent competitive-intelligence research system.

Break the user's business research question into 4-6 specific, non-overlapping
research tasks. For each task, choose the most appropriate source type:

- "web" for current websites, news, products, partnerships, market activity,
  customer information, and general competitive intelligence.
- "financials" for public-company market/financial metrics.
- "filings" for official SEC filings and regulatory disclosures.

If a task uses "financials" or "filings", include the public-company ticker
symbol when you can identify it confidently. Otherwise use null.

Return only valid JSON in exactly this shape:
{
  "tasks": [
    {
      "task": "specific research task",
      "source_type": "web",
      "ticker": null
    }
  ]
}

Do not include markdown or text outside the JSON object.
""".strip()


class ResearchState(TypedDict):
    question: str
    research_plan: list[dict]
    research_results: list[dict]
    response: str


def call_model_json(system_prompt: str, user_text: str) -> dict:
    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def plan_node(state: ResearchState) -> dict:
    output = call_model_json(TEAM_1_SYSTEM_PROMPT, state["question"])
    tasks = output.get("tasks")

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner must return a non-empty 'tasks' list.")

    allowed = {"web", "financials", "filings"}
    cleaned = []
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError("Each Planner task must be a JSON object.")

        task = str(item.get("task", "")).strip()
        source_type = str(item.get("source_type", "")).strip().lower()
        ticker = item.get("ticker")

        if not task:
            raise ValueError("Every Planner task needs a non-empty 'task'.")
        if source_type not in allowed:
            raise ValueError(
                "source_type must be web, financials, or filings."
            )

        cleaned.append(
            {
                "task": task,
                "source_type": source_type,
                "ticker": str(ticker).upper().strip() if ticker else None,
            }
        )

    return {"research_plan": cleaned}


# ---------------- Research tools ----------------
def search_web(task: str) -> list[dict]:
    """Current web/news/company information through Tavily."""
    response = tavily.search(
        query=task,
        search_depth="advanced",
        max_results=4,
    )
    return [
        {
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
        }
        for result in response.get("results", [])
    ]


def get_financials(ticker: str) -> list[dict]:
    """Small public-company snapshot using yfinance."""
    if not ticker:
        return [{"error": "Planner did not provide a ticker."}]

    # We already create the Yahoo Finance company object for you.
    company = yf.Ticker(ticker)
    fast = company.fast_info

    # TODO 1 — FINANCIAL DATA
    #
    # Build a dictionary called fields.
    #
    # Try to collect these four values from fast:
    #   - last_price
    #   - market_cap
    #   - year_high
    #   - year_low
    #
    # Hint:
    # getattr(fast, "last_price")
    #
    # Some values may be missing, so use try/except.
    # Only add a value to fields if it is not None.

    fields = {}
    for name in ("last_price", "market_cap", "year_high", "year_low"):
        try:
            value = getattr(fast, name)
        except Exception:
            continue
        if value is not None:
            fields[name] = value

    if not fields:
        return [{"error": f"No financial snapshot returned for {ticker}."}]

    return [
        {
            "title": f"{ticker} market snapshot",
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "content": json.dumps(fields, default=str),
        }
    ]


def get_sec_filings(ticker: str) -> list[dict]:
    """Recent SEC filing metadata for a public-company ticker."""
    if not ticker:
        return [{"error": "Planner did not provide a ticker."}]

    headers = {"User-Agent": SEC_USER_AGENT}

    # Find the company's SEC CIK number.
    ticker_response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers,
        timeout=20,
    )
    ticker_response.raise_for_status()
    companies = ticker_response.json()

    match = next(
        (
            item
            for item in companies.values()
            if str(item.get("ticker", "")).upper() == ticker.upper()
        ),
        None,
    )

    if not match:
        return [{"error": f"Could not find SEC CIK for ticker {ticker}."}]

    cik = str(match["cik_str"]).zfill(10)

    # Get the company's recent SEC filings.
    submissions_response = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=headers,
        timeout=20,
    )
    submissions_response.raise_for_status()

    recent = (
        submissions_response.json()
        .get("filings", {})
        .get("recent", {})
    )

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])

    results = []

    # TODO 2 — SEC FILINGS
    #
    # Loop through:
    # forms, dates, accessions, and documents together.
    #
    # Only keep these filing types:
    #   10-K
    #   10-Q
    #   8-K
    #
    # For each filing you keep:
    #
    # 1. Remove "-" from the accession number.
    #
    # 2. Build this SEC URL:
    #
    # https://www.sec.gov/Archives/edgar/data/
    # {CIK}/{accession_without_dashes}/{document}
    #
    # 3. Append a dictionary to results containing:
    #       title
    #       url
    #       content
    #
    # Stop after collecting 3 filings.
    #
    # Hint:
    # zip(forms, dates, accessions, documents)

    for form, date, accession, document in zip(
        forms, dates, accessions, documents
    ):
        if form not in {"10-K", "10-Q", "8-K"}:
            continue

        accession_without_dashes = str(accession).replace("-", "")
        url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession_without_dashes}/{document}"
        )
        results.append(
            {
                "title": f"{ticker} {form} ({date})",
                "url": url,
                "content": (
                    f"{form} filed on {date}. "
                    f"Accession number {accession}."
                ),
            }
        )
        if len(results) >= 3:
            break

    return results or [
        {"error": f"No recent 10-K, 10-Q, or 8-K found for {ticker}."}
    ]


# ---------------- Week 3 node ----------------
def research_node(state: ResearchState) -> dict:
    """Execute the Planner's FULL plan with the appropriate research tool."""
    research_results = []

    # TODO 3 — EXECUTE THE FULL RESEARCH PLAN
    #
    # Loop through EVERY item in:
    #
    # state["research_plan"]
    #
    # For each task:
    #
    # 1. Read:
    #       task
    #       source_type
    #       ticker
    #
    # 2. Choose the correct tool:
    #
    #       web
    #           -> search_web(...)
    #
    #       financials
    #           -> get_financials(...)
    #
    #       filings
    #           -> get_sec_filings(...)
    #
    # 3. Save each result inside research_results.
    #
    # Each saved item should contain:
    #
    # {
    #     "task": ...,
    #     "source_type": ...,
    #     "ticker": ...,
    #     "results": ...
    # }
    #
    # IMPORTANT:
    # Research ALL Planner tasks.
    # Do not use research_plan[0].

    for item in state["research_plan"]:
        task = item["task"]
        source_type = item["source_type"]
        ticker = item.get("ticker")

        if source_type == "web":
            results = search_web(task)
        elif source_type == "financials":
            results = get_financials(ticker)
        elif source_type == "filings":
            results = get_sec_filings(ticker)
        else:
            results = [{"error": f"Unknown source type: {source_type}"}]

        research_results.append(
            {
                "task": task,
                "source_type": source_type,
                "ticker": ticker,
                "results": results,
            }
        )

    return {"research_results": research_results}


def response_node(state: ResearchState) -> dict:
    research_text = json.dumps(state["research_results"], indent=2, default=str)

    prompt = f"""
You are the response step in a single-agent business research workflow.

Answer the original question using only the research results supplied below.
Combine useful information from multiple Planner tasks. If a tool failed or a
planned area lacks useful information, say so briefly rather than inventing it.

This is still a Week 3 research response, not the final semester report.

Question:
{state["question"]}

Planner research results:
{research_text}
""".strip()

    response = deepseek.chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"response": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("respond", response_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


research_graph = build_graph()


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ELP Team 1 — Week 3", page_icon="🔎")
st.title("Team 1 — Focused AI Research Agent")
st.caption("Planner → Appropriate Tools → Research → Response")

question = st.text_input(
    "Enter a business research question",
    value="How is NVIDIA positioning itself against AMD in the AI chip market?",
)

if st.button("Research", type="primary") and question.strip():
    initial_state: ResearchState = {
        "question": question.strip(),
        "research_plan": [],
        "research_results": [],
        "response": "",
    }

    try:
        with st.spinner("Planning and researching the full question..."):
            result = research_graph.invoke(initial_state)
    except Exception as error:
        st.error(f"The workflow could not finish: {error}")
    else:
        st.subheader("Planner output")
        st.json(result["research_plan"])

        st.subheader("Research from all tasks")
        st.json(result["research_results"])

        st.subheader("Response")
        st.write(result["response"])


# ======================================================================
# ELP Week 4 — Team 1 Student Extension
# Evidence & Traceability + SQLite
# ======================================================================

# ---------------- Week 4: Evidence & Traceability ----------------

import sqlite3
from concurrent.futures import ThreadPoolExecutor

EVIDENCE_DB = "team1_evidence.db"


def init_evidence_db():
    """Create the Week 4 evidence table if it does not exist."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                publication_date TEXT,
                source_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence TEXT,
                related_company TEXT,
                report_section TEXT,
                UNIQUE(claim, url)
            )
        """)
        conn.commit()


def save_evidence_records(records: list[dict]):
    """Save validated evidence records to SQLite."""
    columns = [
        "claim",
        "source",
        "url",
        "publication_date",
        "source_type",
        "evidence",
        "confidence",
        "related_company",
        "report_section",
    ]
    rows = [tuple(record.get(col) for col in columns) for record in records]

    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.executemany(
            f"INSERT OR IGNORE INTO evidence ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            rows,
        )
        conn.commit()


def load_evidence_records() -> list[dict]:
    """Read the evidence ledger back from SQLite."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()


class Week4ResearchState(ResearchState):
    evidence_records: list[dict]


EVIDENCE_SYSTEM_PROMPT = """
You are the evidence extraction step in a business research workflow.

Convert useful research results into structured Evidence Records.

Return ONLY valid JSON in exactly this shape:
{
  "records": [
    {
      "claim": "...",
      "source": "...",
      "url": "...",
      "publication_date": null,
      "source_type": "...",
      "evidence": "...",
      "confidence": "high",
      "related_company": null,
      "report_section": null
    }
  ]
}

Rules:
- claim = a factual statement supported by the source.
- evidence = the passage that supports the claim.
- source and url must come from the supplied research.
- Do not guess a publication date.
- confidence must be high, medium, or low.
- Do not create evidence for unsupported conclusions.
""".strip()


def _extract_one(item: dict, result: dict) -> list[dict]:
    """Ask the model for evidence records from a single source result."""
    source_payload = {
        "task": item.get("task"),
        "source_type": item.get("source_type"),
        "title": result.get("title"),
        "url": result.get("url"),
        "content": result.get("content"),
        "published_date": result.get("published_date"),
    }

    try:
        output = call_model_json(
            EVIDENCE_SYSTEM_PROMPT,
            json.dumps(source_payload, indent=2, default=str),
        )
    except Exception:
        # One failed model call should not sink the whole ledger.
        return []

    extracted = output.get("records") if isinstance(output, dict) else None
    if not isinstance(extracted, list):
        return []

    records = []
    for record in extracted:
        if not isinstance(record, dict):
            continue
        # Traceability: never let the model invent provenance.
        record["url"] = source_payload["url"]
        record["source"] = record.get("source") or source_payload["title"]
        record["source_type"] = source_payload["source_type"]
        record["publication_date"] = source_payload["published_date"]
        if not record.get("related_company"):
            record["related_company"] = item.get("ticker")
        records.append(record)
    return records


def extract_evidence_records(research_results: list[dict]) -> list[dict]:
    """Turn Week 3 research results into structured evidence records."""
    jobs = [
        (item, result)
        for item in research_results
        for result in item.get("results", [])
        if isinstance(result, dict) and "error" not in result
    ]

    # Model calls are slow and independent, so run them in parallel.
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = pool.map(lambda job: _extract_one(*job), jobs)

    return [record for batch in batches for record in batch]


def validate_evidence_records(records: list[dict]) -> list[dict]:
    """Keep only records with the minimum Week 4 evidence fields."""
    required = ["claim", "source", "url", "source_type", "evidence"]

    validated = []
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue
        if not all(str(record.get(field) or "").strip() for field in required):
            continue

        key = (record["claim"].strip().lower(), record["url"].strip())
        if key in seen:
            continue
        seen.add(key)

        if record.get("confidence") not in {"high", "medium", "low"}:
            record["confidence"] = None
        validated.append(record)

    return validated


def evidence_node(state: ResearchState) -> dict:
    """Week 4: extract, validate, and persist evidence from Week 3 research."""
    raw_records = extract_evidence_records(state["research_results"])
    validated = validate_evidence_records(raw_records)
    save_evidence_records(validated)
    return {"evidence_records": validated}


def build_week4_graph():
    """
    Extend the existing Week 3 Team 1 flow:

    Question -> Planner -> Research -> Evidence Extraction -> SQLite Ledger
    """
    graph = StateGraph(Week4ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("evidence", evidence_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "evidence")
    graph.add_edge("evidence", END)

    return graph.compile()


week4_graph = build_week4_graph()


# ---------------- Week 4 UI ----------------

st.divider()
st.subheader("Week 4 — Evidence Ledger")
st.caption("Week 3 research → Evidence Extraction → SQLite Evidence Ledger")

if st.button("Build Evidence Ledger", type="primary") and question.strip():
    week4_state = {
        "question": question.strip(),
        "research_plan": [],
        "research_results": [],
        "response": "",
        "evidence_records": [],
    }

    try:
        with st.spinner("Researching and building the evidence ledger..."):
            evidence_result = week4_graph.invoke(week4_state)
    except Exception as error:
        st.error(f"The Week 4 evidence workflow could not finish: {error}")
    else:
        st.subheader("New Evidence Records")
        st.json(evidence_result["evidence_records"])

        stored_records = load_evidence_records()
        st.subheader("Evidence Ledger (SQLite)")
        st.dataframe(stored_records, use_container_width=True)
        st.caption(f"{len(stored_records)} evidence records stored in {EVIDENCE_DB}")


# ======================================================================
# ELP WEEK 5 — TEAM 1 — STUDENT CODE
# Single-agent report generation
#
# ==================== EXACT SETUP INSTRUCTIONS ====================
# 1. OPEN the SAME Streamlit Python file used for Weeks 3 and 4, for YOUR team.
# 2. Make sure it already contains, IN THIS ORDER:
#      (a) the completed Week 3 application, imports, tools and graphs;
#      (b) the completed Week 4 evidence extension, including its Streamlit UI.
# 3. Scroll to the VERY BOTTOM of that combined file, below the final Week 4
#    st.caption(...) / st.dataframe(...) line. Leave two blank lines.
# 4. COPY EVERYTHING IN THIS WEEK 5 FILE, including its imports, and PASTE there.
#    Alternatively, if you already pasted a previous Week 5 extension, REPLACE
#    that old Week 5 extension; do NOT append two Week 5 copies.
# 5. SAVE the combined file. Run it from a terminal with:
#      streamlit run YOUR_EXISTING_APP_FILE.py
# 6. Enter a research question in the existing question input, then click
#    Run Full Report Pipeline in the new Week 5 section near the bottom.
# 7. Check that the output displays a complete 10-section report AND that SQLite
#    contains both evidence and a saved report in the 'reports' table.
#
# IMPORTANT DEPENDENCIES:
# - This extension is NOT a standalone Python program. It expects Week 3/4 to
#   provide sqlite3, json, st, EVIDENCE_DB, question, call_model_json,
#   Week4ResearchState, StateGraph, START, END, and the nodes listed below.
# - Team 1 reuses: plan_node, research_node, evidence_node.
# - Team 2 reuses: supervisor_node, specialists_node, evidence_store_node.
# - Complete ALL Week 4 student TODOs first; unfinished Week 4 code prevents
#   Week 5 from obtaining evidence.
# - DO NOT mix Team 1 and Team 2 files: their graph state and node names differ.
# - This week keeps existing SQLite storage. It creates a SECOND table 'reports'
#   inside the SAME database file (not a separate database).
# - Existing Week 3 and Week 4 UI buttons are preserved. The Week 5 button is
#   the one that runs the complete new pipeline end-to-end.
# - You will need your existing LLM and search API configuration to make a live run.
#
# STUDENT WORK: Complete TODO 0 through TODO 5 in order. Functions marked pass are exercises and will not work until completed. Use the instructor file only for checking or teaching solutions.
#
# WEEK 5 TARGET: single-agent Planner → Research → Evidence → Report
# Deliverable: the first complete DRAFT report; automated citation verification and
# quality review are deliberately reserved for Week 6.
# ==================================================================
# ======================================================================

# UTC timestamp is used when each report is stored in SQLite.
from datetime import datetime, timezone

# ---------------- Week 5: Save reports in SQLite ----------------

# WHAT THIS DOES — DATABASE SETUP: Creates reports table if missing; preserves all Week 4 evidence rows.
def init_report_db():
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                report_markdown TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


# WHAT THIS DOES — DATABASE WRITE: Saves question, generated markdown, evidence count, timestamp; returns report ID.
def save_report(question: str, report_markdown: str, evidence_count: int) -> int:
    """Save one generated report and return its database id."""
    # TODO 0 — REPORT PERSISTENCE
    # Insert the question, report_markdown, evidence_count, and current UTC time
    # into the reports table using parameterized SQL.
    # Return cursor.lastrowid.
    with sqlite3.connect(EVIDENCE_DB) as conn:
        cursor = conn.execute(
            "INSERT INTO reports (question, report_markdown, evidence_count, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                question,
                report_markdown,
                evidence_count,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return cursor.lastrowid


# WHAT THIS DOES — DATABASE READ: Loads previously generated reports, most recent first, for the UI.
def load_reports() -> list[dict]:
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


init_report_db()  # Ensure reports table exists before any report is saved.


# SHARED OUTPUT CONTRACT: Both teams use the SAME nine model-written sections;
# Sources / References is appended in Python as section number ten.
REPORT_SECTION_KEYS = [
    ("executive_summary", "Executive Summary"),
    ("key_findings", "Key Findings"),
    ("competitor_comparison", "Competitor Comparison"),
    ("market_industry_trends", "Market / Industry Trends"),
    ("emerging_technologies", "Emerging Technologies"),
    ("regulatory_developments", "Regulatory Developments"),
    ("risks_opportunities", "Risks / Opportunities"),
    ("recommended_next_steps", "Recommended Next Steps"),
    ("limitations_information_gaps", "Limitations / Information Gaps"),
]


# MODEL INSTRUCTIONS: The model must work from evidence only and return structured JSON.
# Week 6, not this file, handles independent citation validation and report review.
REPORT_SYSTEM_PROMPT = """
You are the report-generation stage inside a SINGLE AI research-agent workflow.

Use ONLY the supplied structured Evidence Records as factual support.
Return ONLY valid JSON with exactly these keys:
{
  "executive_summary": "...",
  "key_findings": "...",
  "competitor_comparison": "...",
  "market_industry_trends": "...",
  "emerging_technologies": "...",
  "regulatory_developments": "...",
  "risks_opportunities": "...",
  "recommended_next_steps": "...",
  "limitations_information_gaps": "..."
}

Rules:
- Cite factual statements using evidence labels such as [E1].
- Do not invent facts, dates, numbers, source names, URLs, or citations.
- Clearly distinguish facts from analysis and recommendations.
- If evidence is missing for a section, say so.
- Do NOT build citation verification today. That is Week 6.
""".strip()


# WHAT THIS DOES — EVIDENCE PREP: Assigns [E1], [E2] labels to this run’s evidence, keeps sources and excerpts.
def build_evidence_packet(evidence_records: list[dict]) -> list[dict]:
    """Convert evidence records into compact numbered records for the report generator."""
    packet = []
    for index, record in enumerate(evidence_records, start=1):
        packet.append({
            "evidence_id": f"E{index}",
            "claim": record.get("claim"),
            "source": record.get("source"),
            "url": record.get("url"),
            "publication_date": record.get("publication_date"),
            "source_type": record.get("source_type"),
            "evidence": record.get("evidence"),
            "confidence": record.get("confidence"),
            "related_company": record.get("related_company"),
            "report_section": record.get("report_section"),
        })
    return packet


# WHAT THIS DOES — TEAM 1 MODEL STEP: Converts numbered records into nine narrative report sections.
def generate_report(question: str, evidence_records: list[dict]) -> dict:
    """Generate the report from evidence — not from the full conversation history."""
    # TODO 1 — REPORT GENERATION
    # 1. Call build_evidence_packet(evidence_records).
    # 2. Build a prompt containing the research question and json.dumps(...packet...).
    # 3. Call call_model_json(REPORT_SYSTEM_PROMPT, prompt).
    # 4. Make sure every key in REPORT_SECTION_KEYS exists.
    # 5. Return the report dictionary.
    packet = build_evidence_packet(evidence_records)

    prompt = f"""
Research question:
{question}

Evidence Records (cite them by evidence_id, e.g. [E1]):
{json.dumps(packet, indent=2, default=str)}
""".strip()

    output = call_model_json(REPORT_SYSTEM_PROMPT, prompt)
    if not isinstance(output, dict):
        raise ValueError("Report generator must return a JSON object.")

    report = {}
    for key, title in REPORT_SECTION_KEYS:
        value = output.get(key)
        # The model sometimes returns a section as a list of points.
        if isinstance(value, list):
            value = "\n".join(f"- {point}" for point in value)
        value = str(value or "").strip()
        report[key] = value or f"No {title.lower()} could be supported by the collected evidence."

    return report


# WHAT THIS DOES — REFERENCES: Creates citation legend from original URLs, not invented model references.
def build_references(evidence_records: list[dict]) -> str:
    """Build the source list directly from evidence so URLs are not invented by the model."""
    lines = []
    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        lines.append(f"- [E{index}] {source} — {date} — {url}")
    return "\n".join(lines) if lines else "No source records were available."


# WHAT THIS DOES — REPORT FORMAT: Assembles nine model sections and the tenth Sources / References section.
def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    """Render all ten required report sections."""
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)


# TODO 2 — WEEK 5 STATE
# Extend Week4ResearchState with:
#     report: dict
#     report_markdown: str
#     report_id: int
class Week5ResearchState(Week4ResearchState):
    report: dict
    report_markdown: str
    report_id: int


# WHAT THIS DOES — TEAM 1 GRAPH NODE: Runs model report generation, formats it, saves to SQLite.
def report_node(state: Week5ResearchState) -> dict:
    """Turn this run's evidence into a saved complete report."""
    # TODO 3 — CONNECT EVIDENCE TO REPORT
    # 1. Read state["evidence_records"].
    # 2. Stop with a helpful error if it is empty.
    # 3. Generate the report.
    # 4. Render it as markdown.
    # 5. Save it to SQLite.
    # 6. Return report, report_markdown, and report_id.
    evidence_records = state.get("evidence_records") or []
    if not evidence_records:
        raise ValueError(
            "No evidence records were extracted for this question, so no report "
            "can be written. Try a more specific question or check the research tools."
        )

    report = generate_report(state["question"], evidence_records)
    report_markdown = render_report_markdown(report, evidence_records)
    report_id = save_report(state["question"], report_markdown, len(evidence_records))

    return {
        "report": report,
        "report_markdown": report_markdown,
        "report_id": report_id,
    }


# WHAT THIS DOES — WORKFLOW WIRING: Reuses Week 3/4 nodes and adds Week 5 report/synthesis at the end.
def build_week5_graph():
    """Question -> Planner -> Research -> Evidence -> Report -> END"""
    # TODO 4 — EXTEND THE GRAPH
    # Reuse plan_node, research_node, and evidence_node.
    # Add report_node after evidence_node.
    # Use your Week5ResearchState.
    graph = StateGraph(Week5ResearchState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "evidence")
    graph.add_edge("evidence", "report")
    graph.add_edge("report", END)

    return graph.compile()


week5_graph = build_week5_graph()


# ---------------- Week 5 UI ----------------
# UI: Adds a separate Week 5 button and shows the report + evidence + saved history.
# A full run invokes Week 3 planning/research, Week 4 evidence, THEN Week 5 writing.

st.divider()
st.subheader("Week 5 — Complete Report Pipeline")
st.caption("Question → Planner → Research → Evidence Ledger → Report")

if st.button("Run Full Report Pipeline", type="primary") and question.strip():
    # TODO 5 — RUN THE COMPLETE PIPELINE
    # Create the initial state, invoke week5_graph, and display:
    #   - report_markdown with st.markdown(...)
    #   - evidence records in an expander
    #   - saved reports from SQLite in another expander
    week5_state = {
        "question": question.strip(),
        "research_plan": [],
        "research_results": [],
        "response": "",
        "evidence_records": [],
        "report": {},
        "report_markdown": "",
        "report_id": 0,
    }

    try:
        with st.spinner("Planning, researching, extracting evidence, and writing the report..."):
            report_result = week5_graph.invoke(week5_state)
    except Exception as error:
        st.error(f"The Week 5 report pipeline could not finish: {error}")
    else:
        st.success(f"Report #{report_result['report_id']} saved to {EVIDENCE_DB}.")
        st.markdown(report_result["report_markdown"])

        with st.expander(f"Evidence used ({len(report_result['evidence_records'])} records)"):
            st.json(build_evidence_packet(report_result["evidence_records"]))

        with st.expander("Saved reports (SQLite)"):
            for saved in load_reports():
                st.markdown(
                    f"**#{saved['id']}** — {saved['question']}  \n"
                    f"{saved['evidence_count']} evidence records · {saved['created_at']}"
                )

