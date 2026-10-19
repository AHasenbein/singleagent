"""Team 1 AI research agent: Planner -> Research -> Evidence -> Report.

Modules, in pipeline order:
    planner   - break a question into source-typed research tasks
    tools/    - web (Tavily), financials (yfinance), filings (SEC EDGAR)
    research  - run the plan through the tools; Week 3 plain response
    evidence  - extract and validate traceable evidence records
    report    - write and render the ten-section report
    storage   - SQLite evidence ledger and saved reports
    pipeline  - LangGraph workflows wiring the stages together

Supporting: config (settings from env), llm (model client), types (record shapes).
"""
