"""SQLite persistence for the evidence ledger and generated reports.

All database access goes through ResearchStore, so moving to Postgres or a
hosted database later means replacing this one class.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional

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
    created_at TEXT NOT NULL,
    status TEXT,
    verification_json TEXT,
    evidence_json TEXT
);
"""

# Columns added after the first schema; older database files get them via
# ALTER TABLE so existing reports keep loading.
REPORT_MIGRATIONS = {
    "status": "TEXT",
    "verification_json": "TEXT",
    "evidence_json": "TEXT",
}
# Reports saved before status tracking existed were never verified.
LEGACY_REPORT_STATUS = "unverified"


class ResearchStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            existing = {row["name"] for row in conn.execute("PRAGMA table_info(reports)")}
            for column, column_type in REPORT_MIGRATIONS.items():
                if column not in existing:
                    try:
                        conn.execute(f"ALTER TABLE reports ADD COLUMN {column} {column_type}")
                    except sqlite3.OperationalError as error:
                        # Another process may have migrated the file meanwhile.
                        if "duplicate column" not in str(error).lower():
                            raise

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
    def save_report(
        self,
        question: str,
        report_markdown: str,
        evidence_count: int,
        status: str = "pending",
        evidence_records: Optional[List[dict]] = None,
    ) -> int:
        """Save one generated report and return its database id.

        The ordered evidence snapshot is stored with it so the report's
        positional E-labels (E1 = first record) still resolve later.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO reports (question, report_markdown, evidence_count, created_at, "
                "status, evidence_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    question,
                    report_markdown,
                    evidence_count,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    status,
                    _to_json(evidence_records),
                ),
            )
            return cursor.lastrowid

    def finalize_report(
        self,
        report_id: int,
        report_markdown: str,
        status: str,
        verification: dict,
        evidence_records: List[dict],
    ) -> None:
        """Record the verification outcome (PASS/FAIL) for a pending report."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET report_markdown = ?, status = ?, verification_json = ?, "
                "evidence_json = ?, evidence_count = ? WHERE id = ?",
                (
                    report_markdown,
                    status,
                    _to_json(verification),
                    _to_json(evidence_records),
                    len(evidence_records),
                    report_id,
                ),
            )

    def update_report(self, report_id: int, report_markdown: str) -> None:
        """Replace a saved report's markdown, e.g. after verification-driven revision."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE reports SET report_markdown = ? WHERE id = ?",
                (report_markdown, report_id),
            )

    def list_reports(self, limit: int = 20) -> List[dict]:
        """Lightweight history rows (no markdown or JSON blobs), most recent first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, question, evidence_count, created_at, status FROM reports "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        reports = [dict(row) for row in rows]
        for report in reports:
            report["status"] = report.get("status") or LEGACY_REPORT_STATUS
        return reports

    def get_report(self, report_id: int) -> Optional[dict]:
        """One saved report with its verification and ordered evidence decoded."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        report = dict(row)
        report["status"] = report.get("status") or LEGACY_REPORT_STATUS
        report["verification"] = json.loads(report.get("verification_json") or "null") or {}
        report["evidence_records"] = json.loads(report.get("evidence_json") or "null") or []
        return report

    def count_evidence(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]

    def load_reports(self) -> List[dict]:
        """Saved reports, most recent first. Pre-migration rows report "unverified"."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM reports ORDER BY id DESC").fetchall()
        reports = [dict(row) for row in rows]
        for report in reports:
            report["status"] = report.get("status") or LEGACY_REPORT_STATUS
        return reports


def _to_json(value: object) -> Optional[str]:
    return None if value is None else json.dumps(value, default=str)
