"""Allocate cases to demo groups from saved AI suggestions and Python rules."""

from contextlib import closing
import json
from pathlib import Path

from uvdemo.allocation import route_analysis
from uvdemo.analysis import AnalysisError, validate_analysis
from uvdemo.storage import (
    _connect, _json, _latest_analysis, _now, _require_unassigned, _write_event,
)


def initialize_group_allocation(db_path: str | Path) -> None:
    """Add group decisions without changing historical specialist assignments."""
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS case_group_decisions (
                    case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
                    analysis_id INTEGER REFERENCES case_analyses(analysis_id),
                    status TEXT NOT NULL,
                    decision_json TEXT NOT NULL
                )
            """)


def _get_assignment(connection, case_id: str) -> dict | None:
    row = connection.execute(
        "SELECT decision_json FROM case_group_decisions WHERE case_id = ? AND status = 'Allocated'",
        (case_id,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def get_group_assignment(db_path: str | Path, case_id: str) -> dict | None:
    """Return a saved group allocation; unresolved decisions return None."""
    with closing(_connect(db_path)) as connection:
        return _get_assignment(connection, case_id)


def get_legacy_assignment(db_path: str | Path, case_id: str) -> dict | None:
    """Read an older specialist assignment without creating or changing tables."""
    with closing(_connect(db_path)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'case_assignments'"
        ).fetchone()
        if table is None:
            return None
        row = connection.execute(
            "SELECT decision_json FROM case_assignments WHERE case_id = ?", (case_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None


def get_group_decision(db_path: str | Path, case_id: str) -> dict | None:
    """Hide a superseded decision while retaining its event history."""
    with closing(_connect(db_path)) as connection:
        row = connection.execute("""
            SELECT decision.decision_json, decision.analysis_id, decision.status,
                   cases.status AS case_status,
                   (SELECT analysis_id FROM case_analyses
                    WHERE case_id = cases.case_id ORDER BY analysis_id DESC LIMIT 1) AS latest_id
            FROM case_group_decisions AS decision
            JOIN cases ON cases.case_id = decision.case_id
            WHERE decision.case_id = ?
        """, (case_id,)).fetchone()
        if row is None or row["analysis_id"] != row["latest_id"]:
            return None
        allocated_work = (row["status"] == "Allocated"
                          and row["case_status"] in ("Allocated", "In progress", "Completed"))
        if not allocated_work and row["status"] != row["case_status"]:
            return None
        return json.loads(row["decision_json"])


def _record_decision(connection, decision: dict) -> dict:
    encoded = _json(decision)
    previous = connection.execute(
        "SELECT decision_json FROM case_group_decisions WHERE case_id = ?", (decision["case_id"],)
    ).fetchone()
    if previous and previous[0] == encoded:
        return decision
    connection.execute("""
        INSERT INTO case_group_decisions (case_id, analysis_id, status, decision_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET analysis_id = excluded.analysis_id,
            status = excluded.status, decision_json = excluded.decision_json
    """, (decision["case_id"], decision["analysis_id"], decision["status"], encoded))
    connection.execute("UPDATE cases SET status = ? WHERE case_id = ?",
                       (decision["status"], decision["case_id"]))
    event = "Allocated to group" if decision["status"] == "Allocated" else "Group allocation deferred"
    _write_event(connection, decision["case_id"], event, decision)
    return decision


def _readable_document(document, extraction) -> bool:
    if not document or not isinstance(extraction, dict) or extraction.get("status") != "Ready":
        return False
    if any(extraction.get(key) != document[key] for key in ("filename", "sha256", "size_bytes")):
        return False
    if not isinstance(extraction.get("text"), str) or not extraction["text"].strip():
        return False
    pages = extraction.get("pages")
    return isinstance(pages, list) and bool(pages) and all(
        isinstance(page, dict) and type(page.get("page")) is int and page["page"] >= 1
        and isinstance(page.get("text"), str) and bool(page["text"].strip())
        for page in pages
    )


def allocate_group(db_path: str | Path, case_id: str, expected_analysis_id: int | None) -> dict:
    """Validate saved evidence and allocate a group atomically, at most once.

    No staff availability or capacity participates in a group decision. Failed
    attempts preserve the request; successful allocations keep their evidence fixed.
    """
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = _get_assignment(connection, case_id)
            if existing is not None:
                return existing
            case = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if case is None:
                raise ValueError("The case does not exist.")
            # Historical specialist assignments stay intact and are not converted.
            _require_unassigned(connection, case_id)
            analysis = _latest_analysis(connection, case_id)
            analysis_id = analysis["analysis_id"] if analysis else None
            if expected_analysis_id != analysis_id:
                raise ValueError("The analysis changed. Reload the case before allocating.")
            decision = {
                "case_id": case_id, "analysis_id": analysis_id, "queue": None,
                "status": "Analysis needed", "reason": "A successful analysis is required before allocation.",
                "mode": analysis["mode"] if analysis else None,
                "model": analysis["model"] if analysis else None,
                "basis": "AI suggestions plus Python demo group rules; no reviewer confirmation",
            }
            document = connection.execute(
                "SELECT * FROM case_documents WHERE case_id = ?", (case_id,)
            ).fetchone()
            extraction = json.loads(document["extraction_json"]) if document and document["extraction_json"] else None
            if not _readable_document(document, extraction):
                decision.update(status="Document needs attention",
                                reason="A required contract PDF must have readable text on every page.")
                return _record_decision(connection, decision)
            if analysis is None:
                return _record_decision(connection, decision)
            if analysis["error"] is not None or analysis["result_json"] is None:
                decision.update(status="Analysis failed", reason="The latest analysis failed; retry analysis before allocation.")
                return _record_decision(connection, decision)
            result = json.loads(analysis["result_json"])
            try:
                validate_analysis(result, json.loads(case["original_input_json"]), extraction)
            except AnalysisError as error:
                decision.update(status="Analysis failed", reason=str(error))
                return _record_decision(connection, decision)
            decision.update(route_analysis(result))
            if decision["queue"] is None:
                decision["status"] = "Clarification needed"
            else:
                decision.update(status="Allocated", allocated_at=_now(), document_sha256=document["sha256"])
            return _record_decision(connection, decision)
