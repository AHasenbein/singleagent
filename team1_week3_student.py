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

# region WEEK 3 — Configuration and Planner

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
    content = response.choices[0].message.content

    if content is None:
        raise ValueError("Model returned no content.")

    return json.loads(content)

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

# endregion

# region WEEK 3 — Research Tools

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


def get_financials(ticker: str | None) -> list[dict]:
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


def get_sec_filings(ticker: str | None) -> list[dict]:
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

# endregion

# region WEEK 3 — Workflow and UI

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

# endregion

# region WEEK 4 — Evidence and SQLite

# ---------------- Week 4: Evidence & Traceability ----------------

import sqlite3

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
    # TODO 0 — SAVE TO SQLITE
    #
    # Use sqlite3.connect(EVIDENCE_DB), then INSERT each record into the
    # evidence table. Use parameterized SQL (?) values.
    #
    # Helpful columns:
    #   claim, source, url, publication_date, source_type, evidence,
    #   confidence, related_company, report_section
    #
    # Hint: INSERT OR IGNORE is useful because the table treats
    # (claim, url) as a unique pair.
    with sqlite3.connect(EVIDENCE_DB) as conn:
        for record in records:
            conn.execute(
                """
                INSERT OR IGNORE INTO evidence (
                    claim,
                    source,
                    url,
                    publication_date,
                    source_type,
                    evidence,
                    confidence,
                    related_company,
                    report_section
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("claim"),
                    record.get("source"),
                    record.get("url"),
                    record.get("publication_date"),
                    record.get("source_type"),
                    record.get("evidence"),
                    record.get("confidence"),
                    record.get("related_company"),
                    record.get("report_section"),
                ),
            )

        conn.commit()


def load_evidence_records() -> list[dict]:
    """Read the evidence ledger back from SQLite."""
    with sqlite3.connect(EVIDENCE_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]


init_evidence_db()


# TODO 1 — WEEK 4 STATE
#
# Extend the existing Week 3 ResearchState with:
#     evidence_records: list[dict]
#
# Hint:
#     class Week4ResearchState(ResearchState):
#         ...
#
# You will use Week4ResearchState for the Week 4 graph.

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


def extract_evidence_records(research_results: list[dict]) -> list[dict]:
    """Turn Week 3 research results into structured evidence records."""
    records = []

    # TODO 2 — EVIDENCE EXTRACTION
    #
    # For each research item in research_results:
    #   1. Read its source_type and results.
    #   2. Skip results that contain an "error".
    #   3. For each source result, build a source_payload containing:
    #        task, source_type, title, url, content, published_date
    #   4. Ask call_model_json(...) to extract Evidence Records.
    #   5. Add the returned records to records.
    #
    # Hint:
    #   Use EVIDENCE_SYSTEM_PROMPT as the system prompt.
    #   json.dumps(source_payload, indent=2, default=str) is useful.
    #
    # Important:
    #   Do not invent source URLs or publication dates.
    for item in research_results:
        source_type = item.get("source_type")
        task = item.get("task")

        for result in item.get("results", []):
            if "error" in result:
                continue

            source_payload = {
                "task": task,
                "source_type": source_type,
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
                "published_date": result.get("published_date"),
            }

            output = call_model_json(
                EVIDENCE_SYSTEM_PROMPT,
                json.dumps(source_payload, indent=2, default=str),
            )

            extracted = output.get("records", [])

            if isinstance(extracted, list):
                records.extend(extracted)

    return records


def validate_evidence_records(records: list[dict]) -> list[dict]:
    """Keep only records with the minimum Week 4 evidence fields."""
    required = ["claim", "source", "url", "publication_date", "source_type", "evidence"]

    validated = []

    # TODO 3 — VALIDATION + DUPLICATES
    #
    # Keep a record only if:
    #   - it is a dictionary
    #   - claim, source, url, source_type, and evidence are non-empty
    #
    # Remove exact duplicates using:
    #   (claim.lower(), url)
    #
    # Hint:
    #   Use a set called seen.
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue

        claim = str(record.get("claim", "")).strip()
        source = str(record.get("source", "")).strip()
        url = str(record.get("url", "")).strip()
        source_type = str(record.get("source_type", "")).strip()
        evidence = str(record.get("evidence", "")).strip()

        if not claim or not source or not url or not source_type or not evidence:
            continue

        duplicate_key = (claim.lower(), url)

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)
        validated.append(record)

    return validated


def evidence_node(state: ResearchState) -> dict:
    """Week 4: extract, validate, and persist evidence from Week 3 research."""
    # TODO 4 — CONNECT WEEK 3 TO WEEK 4 + DATABASE
    #
    # 1. Extract raw records from state["research_results"].
    # 2. Validate them.
    # 3. Save validated records to SQLite.
    # 4. Return them under the key "evidence_records".
    raw_records = extract_evidence_records(state["research_results"])

    validated_records = validate_evidence_records(raw_records)

    save_evidence_records(validated_records)

    return {"evidence_records": validated_records}


def build_week4_graph():
    """
    Extend the existing Week 3 Team 1 flow:

    Question -> Planner -> Research -> Evidence Extraction -> SQLite Ledger
    """
    graph = StateGraph(Week4ResearchState)

    # TODO 5 — GRAPH CONNECTION
    #
    # Reuse the Week 3 nodes:
    #   plan_node
    #   research_node
    #
    # Add the new Week 4 node:
    #   evidence_node
    #
    # Then connect:
    #   START -> plan -> research -> evidence -> END
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("evidence", evidence_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "evidence")
    graph.add_edge("evidence", END)

    return graph.compile()


week4_graph = build_week4_graph()

# endregion

# region WEEK 4 — UI

# ---------------- Week 4 UI ----------------

st.divider()
st.subheader("Week 4 — Evidence Ledger")
st.caption("Week 3 research → Evidence Extraction → SQLite Evidence Ledger")

if st.button("Build Evidence Ledger", type="primary") and question.strip():
    week4_state: Week4ResearchState = {
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

# endregion

# UTC timestamp is used when each report is stored in SQLite.
from datetime import datetime, timezone

# region WEEK 5 — Database and Report Generation

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
    # STUDENT: Replace the `pass` in this function with real code.
    # TODO 0 — REPORT PERSISTENCE
    # Insert the question, report_markdown, evidence_count, and current UTC time
    # into the reports table using parameterized SQL.
    # Return cursor.lastrowid.
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(EVIDENCE_DB) as conn:
        cursor = conn.execute(
            """
            INSERT INTO reports (
                question,
                report_markdown,
                evidence_count,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                question,
                report_markdown,
                evidence_count,
                created_at,
            ),
        )

        conn.commit()

        if cursor.lastrowid is None:
            raise RuntimeError("Report was not saved correctly.")

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
- Write currency amounts as "USD 80 billion" instead of "$80B".
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
    # STUDENT: Replace the `pass` below with the report-generation implementation.
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

    Evidence records:
    {json.dumps(packet, indent=2, default=str)}
    """.strip()

    report = call_model_json(REPORT_SYSTEM_PROMPT, prompt)

    for key, _ in REPORT_SECTION_KEYS:
        if key not in report:
            report[key] = "No information available for this section."

    return report


# WHAT THIS DOES — REFERENCES: Creates citation legend from original URLs, not invented model references.
def build_references(evidence_records: list[dict]) -> str:
    """Build a source list grouped by unique URL."""
    sources = {}

    for index, record in enumerate(evidence_records, start=1):
        source = record.get("source") or "Unknown source"
        url = record.get("url") or "No URL"
        date = record.get("publication_date") or "Date unavailable"
        evidence_id = f"E{index}"

        if url not in sources:
            sources[url] = {
                "source": source,
                "date": date,
                "ids": [],
            }

        sources[url]["ids"].append(evidence_id)

    lines = []

    for url, info in sources.items():
        ids = ", ".join(info["ids"])
        lines.append(
            f"- [{ids}] {info['source']} — {info['date']} — {url}"
        )

    return "\n".join(lines) if lines else "No source records were available."


# WHAT THIS DOES — REPORT FORMAT: Assembles nine model sections and the tenth Sources / References section.
def render_report_markdown(report: dict, evidence_records: list[dict]) -> str:
    """Render all ten required report sections."""
    blocks = []
    for key, title in REPORT_SECTION_KEYS:
        blocks.append(f"## {title}\n\n{report.get(key, '')}")
    blocks.append("## Sources / References\n\n" + build_references(evidence_records))
    return "\n\n".join(blocks)

# endregion

# region WEEK 5 — State and Workflow

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
def report_node(state):
    """Turn this run's evidence into a saved complete report."""
    # TODO 3 — CONNECT EVIDENCE TO REPORT
    # 1. Read state["evidence_records"].
    # 2. Stop with a helpful error if it is empty.
    # 3. Generate the report.
    # 4. Render it as markdown.
    # 5. Save it to SQLite.
    # 6. Return report, report_markdown, and report_id.
    
    evidence_records = state["evidence_records"]

    if not evidence_records:
        raise ValueError("No evidence records are available to generate a report.")

    report = generate_report(
        state["question"],
        evidence_records,
    )

    report_markdown = render_report_markdown(
        report,
        evidence_records,
    )

    report_id = save_report(
        state["question"],
        report_markdown,
        len(evidence_records),
    )

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

# endregion

# region WEEK 5 — UI

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
    week5_state: Week5ResearchState = {
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
        with st.spinner("Running the full report pipeline..."):
            result = week5_graph.invoke(week5_state)

    except Exception as error:
        st.error(f"The Week 5 pipeline could not finish: {error}")

    else:
        evidence_records = result["evidence_records"]

        unique_sources = {
            record.get("url")
            for record in evidence_records
            if record.get("url")
        }

        web_count = sum(
            1 for record in evidence_records
            if record.get("source_type") == "web"
        )

        financial_count = sum(
            1 for record in evidence_records
            if record.get("source_type") == "financials"
        )

        filings_count = sum(
            1 for record in evidence_records
            if record.get("source_type") == "filings"
        )

        high_confidence = sum(
            1 for record in evidence_records
            if record.get("confidence") == "high"
        )

        medium_confidence = sum(
            1 for record in evidence_records
            if record.get("confidence") == "medium"
        )

        low_confidence = sum(
            1 for record in evidence_records
            if record.get("confidence") == "low"
        )

        st.subheader("Research Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric("Evidence Records", len(evidence_records))
        col2.metric("Unique Sources", len(unique_sources))
        col3.metric("SEC Filings", filings_count)

        col4, col5, col6 = st.columns(3)

        col4.metric("Web Evidence", web_count)
        col5.metric("Financial Evidence", financial_count)
        col6.metric("High Confidence", high_confidence)

        st.caption(
            f"Confidence breakdown: "
            f"{high_confidence} high, "
            f"{medium_confidence} medium, "
            f"{low_confidence} low"
        )
        st.subheader("Generated Report")
        st.markdown(result["report_markdown"])

        with st.expander("Evidence Records"):
            st.json(result["evidence_records"])

        with st.expander("Saved Reports"):
            saved_reports = load_reports()
            st.dataframe(saved_reports, use_container_width=True)

        st.caption(f"Saved report ID: {result['report_id']}")

# endregion