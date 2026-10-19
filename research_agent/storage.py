"""SQLite persistence for the evidence ledger and generated reports.

All database access goes through ResearchStore, so moving to Postgres or a
hosted database later means replacing this one class.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List

from .types import EVIDENCE_COLUMNS, EvidenceRecord

SCHEMA = """
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
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    evidence_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


class ResearchStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection that commits on success and always closes."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---------------- Evidence ledger ----------------
    def save_evidence(self, records: List[EvidenceRecord]) -> None:
        """Insert records, skipping any (claim, url) pair already stored."""
        rows = [tuple(record.get(col) for col in EVIDENCE_COLUMNS) for record in records]
        with self._connect() as conn:
            conn.executemany(
                f"INSERT OR IGNORE INTO evidence ({', '.join(EVIDENCE_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in EVIDENCE_COLUMNS)})",
                rows,
            )

    def load_evidence(self) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM evidence ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    # ---------------- Reports ----------------
    def save_report(self, question: str, report_markdown: str, evidence_count: int) -> int:
        """Save one generated report and return its database id."""
        with self._connect() as conn:
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
            return cursor.lastrowid

    def load_reports(self) -> List[dict]:
        """Saved reports, most recent first."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]
