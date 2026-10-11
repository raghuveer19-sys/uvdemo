"""SQLite persistence, independent of Streamlit and any model provider."""

from contextlib import closing
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from uuid import uuid4

from uvdemo.intake import validate_intake

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _write_event(connection, case_id, event_type, details, occurred_at=None) -> None:
    connection.execute(
        """INSERT INTO case_events
           (case_id, event_type, occurred_at, details_json) VALUES (?, ?, ?, ?)""",
        (case_id, event_type, occurred_at or _now(), _json(details)),
    )


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=10, isolation_level="IMMEDIATE")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                submission_token TEXT NOT NULL UNIQUE,
                original_input_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_events (
                event_id INTEGER PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(case_id),
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_documents (
                case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
                filename TEXT NOT NULL,
                content BLOB NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                extraction_json TEXT
            );
            CREATE TABLE IF NOT EXISTS case_analyses (
                analysis_id INTEGER PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(case_id),
                mode TEXT NOT NULL CHECK(mode IN ('mock', 'live')),
                model TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS initial_analysis_claims (
                case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
                mode TEXT NOT NULL CHECK(mode IN ('mock', 'live')),
                model TEXT NOT NULL,
                started_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_reviews (
                review_id INTEGER PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(case_id),
                analysis_id INTEGER REFERENCES case_analyses(analysis_id),
                review_json TEXT NOT NULL,
                document_extraction_json TEXT,
                created_at TEXT NOT NULL
            );
        """)


def _read_case(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    case = dict(row)
    case["original_input"] = json.loads(case.pop("original_input_json"))
    return case


def _prepare_document(document: dict | None) -> dict | None:
    if document is None:
        return None
    if not isinstance(document, dict) or set(document) != {"filename", "content"}:
        raise ValueError("A document needs a filename and its original bytes.")
    filename, content = document["filename"], document["content"]
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("A document filename is required.")
    # Keep only a name, even if an upload includes Windows or Unix directories.
    filename = PurePosixPath(filename.replace("\\", "/")).name
    filename = re.sub(r'[<>:"|?*\x00-\x1f]', "_", filename)
    if filename in ("", ".", ".."):
        raise ValueError("A document filename is required.")
    if not isinstance(content, bytes):
        raise ValueError("Document content must be bytes.")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("The demo accepts documents up to 5 MiB.")
    return {"filename": filename, "content": content,
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def save_intake(db_path: str | Path, submission_token: str, intake: dict,
                document: dict | None = None) -> dict:
    """Atomically save a case and its first event, or return an identical retry.

    A token identifies one submission, not one customer or one description.
    A new token permits an intentional new request with identical contents.
    New requests require a nonempty PDF upload. Existing cases, including
    document-free cases saved before this requirement, remain unchanged.
    """
    errors = validate_intake(intake)
    if errors:
        raise ValueError(" ".join(errors.values()))
    if not isinstance(submission_token, str) or not submission_token.strip():
        raise ValueError("A submission token is required.")

    attachment = _prepare_document(document)
    original_json = _json(intake)
    case_id = str(uuid4())
    created_at = _now()
    # The inner context commits both writes together or rolls both back.
    # IMMEDIATE transactions also serialize competing writers before insertion.
    with closing(_connect(db_path)) as connection:
        with connection:
            inserted = connection.execute(
                """INSERT INTO cases
                   (case_id, submission_token, original_input_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(submission_token) DO NOTHING""",
                (case_id, submission_token, original_json, "Analysis needed", created_at),
            )
            row = connection.execute(
                "SELECT * FROM cases WHERE submission_token = ?", (submission_token,)
            ).fetchone()
            if row["original_input_json"] != original_json:
                raise ValueError("This submission was already saved with different input. "
                                 "Start another request to submit new information.")
            if inserted.rowcount == 1:
                if attachment is None:
                    raise ValueError("Upload a contract PDF before submitting a new request.")
                if not attachment["filename"].lower().endswith(".pdf"):
                    raise ValueError("The contract document must be a PDF (.pdf).")
                if attachment["size_bytes"] == 0:
                    raise ValueError("The contract PDF is empty. Upload a nonempty PDF.")
                # Preserve the original before parsing: an unreadable PDF still
                # needs a saved request and a recorded extraction failure.
                connection.execute(
                    """INSERT INTO case_documents
                       (case_id, filename, content, sha256, size_bytes)
                       VALUES (?, ?, ?, ?, ?)""",
                    (case_id, attachment["filename"], attachment["content"],
                     attachment["sha256"], attachment["size_bytes"]),
                )
                _write_event(
                    connection, case_id, "Submitted",
                    {"actor": "submitter", "status": "Analysis needed"}, created_at,
                )
            else:
                stored = connection.execute(
                    "SELECT * FROM case_documents WHERE case_id = ?", (row["case_id"],)
                ).fetchone()
                same_document = stored is None and attachment is None
                if stored is not None and attachment is not None:
                    same_document = all(stored[key] == attachment[key]
                                        for key in ("filename", "content", "sha256", "size_bytes"))
                if not same_document:
                    raise ValueError("This submission was already saved with a different document. "
                                     "Start another request to submit new information.")
            return _read_case(row)


def get_case(db_path: str | Path, case_id: str) -> dict | None:
    with closing(_connect(db_path)) as connection:
        return _read_case(connection.execute(
            "SELECT * FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone())


def get_case_by_submission(db_path: str | Path, submission_token: str) -> dict | None:
    with closing(_connect(db_path)) as connection:
        return _read_case(connection.execute(
            "SELECT * FROM cases WHERE submission_token = ?", (submission_token,)
        ).fetchone())


def list_cases(db_path: str | Path) -> list[dict]:
    with closing(_connect(db_path)) as connection:
        return [_read_case(row) for row in connection.execute(
            "SELECT * FROM cases ORDER BY created_at DESC, rowid DESC"
        ).fetchall()]


def _read_document(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    document = dict(row)
    document.pop("case_id")
    encoded = document.pop("extraction_json")
    document["extraction"] = json.loads(encoded) if encoded is not None else None
    return document


def get_document(db_path: str | Path, case_id: str) -> dict | None:
    with closing(_connect(db_path)) as connection:
        return _read_document(connection.execute(
            "SELECT * FROM case_documents WHERE case_id = ?", (case_id,)
        ).fetchone())


def save_document_extraction(db_path: str | Path, case_id: str, extraction: dict) -> dict:
    """Save extracted text without ever replacing the uploaded document.

    A Ready extraction is fixed for this increment. Failed extraction can be
    retried on an unassigned case; a changed result invalidates any old review.
    """
    required = {"filename", "sha256", "size_bytes", "status", "text", "pages", "error"}
    if not isinstance(extraction, dict) or set(extraction) != required:
        raise ValueError("The document extraction is incomplete.")
    if (extraction["status"] not in ("Ready", "Needs attention")
            or not isinstance(extraction["text"], str)
            or not isinstance(extraction["pages"], list)
            or (extraction["error"] is not None and not isinstance(extraction["error"], str))):
        raise ValueError("The document extraction has invalid values.")
    for page in extraction["pages"]:
        if (not isinstance(page, dict) or set(page) != {"page", "text"}
                or type(page["page"]) is not int or page["page"] < 1
                or not isinstance(page["text"], str)):
            raise ValueError("Each extracted page needs a page number and text.")
    encoded = _json(extraction)
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM case_documents WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None:
                raise ValueError("This case has no uploaded document.")
            if any(extraction[key] != row[key] for key in ("filename", "sha256", "size_bytes")):
                raise ValueError("The extraction does not match the original document.")
            if row["extraction_json"] == encoded:
                return json.loads(encoded)
            _require_unassigned(connection, case_id)
            if row["extraction_json"] and json.loads(row["extraction_json"])["status"] == "Ready":
                raise ValueError("This document already has a Ready extraction.")
            connection.execute(
                "UPDATE case_documents SET extraction_json = ? WHERE case_id = ?",
                (encoded, case_id),
            )
            status = "Analysis needed" if extraction["status"] == "Ready" else "Document needs attention"
            connection.execute("UPDATE cases SET status = ? WHERE case_id = ?",
                               (status, case_id))
            _write_event(connection, case_id, "Document extracted" if extraction["status"] == "Ready"
                         else "Document extraction failed",
                         {"actor": "system", "status": extraction["status"],
                          "sha256": extraction["sha256"], "error": extraction["error"]})
            return json.loads(encoded)


def _read_analysis(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    analysis = dict(row)
    encoded = analysis.pop("result_json")
    analysis["result"] = json.loads(encoded) if encoded is not None else None
    return analysis


def _latest_analysis(connection, case_id) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM case_analyses WHERE case_id = ? ORDER BY analysis_id DESC LIMIT 1",
        (case_id,),
    ).fetchone()


def claim_initial_analysis(db_path: str | Path, case_id: str, mode: str, model: str) -> bool:
    """Reserve a case's one automatic attempt before calling the model.

    The reservation and history event commit together. Keeping the reservation
    after a crash avoids repeating a possibly billed external call on reload.
    An explicit retry can still call and save another analysis.
    """
    if mode not in ("mock", "live") or not isinstance(model, str) or not model.strip():
        raise ValueError("Analysis needs a mock/live mode and a model name.")
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,)).fetchone() is None:
                raise ValueError("The case does not exist.")
            if _latest_analysis(connection, case_id) is not None:
                return False
            started_at = _now()
            inserted = connection.execute(
                """INSERT INTO initial_analysis_claims (case_id, mode, model, started_at)
                   VALUES (?, ?, ?, ?) ON CONFLICT(case_id) DO NOTHING""",
                (case_id, mode, model, started_at),
            )
            if inserted.rowcount == 0:
                return False
            _write_event(connection, case_id, "Initial AI analysis started",
                         {"actor": "mock AI" if mode == "mock" else "AI", "mode": mode,
                          "model": model}, started_at)
            return True


def _require_unassigned(connection, case_id) -> None:
    # The allocation increment initializes its own tables. Storage also remains
    # usable on databases that have not loaded allocation yet.
    assignments_exist = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'case_assignments'"
    ).fetchone()
    if assignments_exist and connection.execute(
            "SELECT 1 FROM case_assignments WHERE case_id = ?", (case_id,)).fetchone():
        raise ValueError("This case is already assigned. Its saved evidence is fixed.")
    groups_exist = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'case_group_decisions'"
    ).fetchone()
    if groups_exist and connection.execute(
            "SELECT 1 FROM case_group_decisions WHERE case_id = ? AND status = 'Allocated'",
            (case_id,)).fetchone():
        raise ValueError("This case is already allocated to a group. Its saved evidence is fixed.")


def save_analysis(db_path: str | Path, case_id: str, mode: str, model: str,
                  result: dict | None, error: str | None) -> dict:
    """Append an attempt and record whether allocation can be attempted next."""
    if mode not in ("mock", "live") or not isinstance(model, str) or not model.strip():
        raise ValueError("Analysis needs a mock/live mode and a model name.")
    if (result is not None and not isinstance(result, dict)) or (
            error is not None and (not isinstance(error, str) or not error.strip())):
        raise ValueError("Analysis result or error has an invalid value.")
    if (result is None) == (error is None):
        raise ValueError("Save either an analysis result or an error.")
    encoded = _json(result) if result is not None else None
    created_at = _now()
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,)).fetchone() is None:
                raise ValueError("The case does not exist.")
            _require_unassigned(connection, case_id)
            cursor = connection.execute(
                """INSERT INTO case_analyses
                   (case_id, mode, model, result_json, error, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (case_id, mode, model, encoded, error, created_at),
            )
            status = "Analysis failed" if error else "Allocation pending"
            connection.execute("UPDATE cases SET status = ? WHERE case_id = ?",
                               (status, case_id))
            _write_event(connection, case_id, "AI analysis failed" if error else "AI analysis completed",
                         {"actor": "mock AI" if mode == "mock" else "AI", "mode": mode,
                          "model": model, "analysis_id": cursor.lastrowid}, created_at)
            return _read_analysis(connection.execute(
                "SELECT * FROM case_analyses WHERE analysis_id = ?", (cursor.lastrowid,)
            ).fetchone())


def get_latest_analysis(db_path: str | Path, case_id: str) -> dict | None:
    with closing(_connect(db_path)) as connection:
        return _read_analysis(_latest_analysis(connection, case_id))


def _read_review(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    review = dict(row)
    review["review"] = json.loads(review.pop("review_json"))
    extraction = review.pop("document_extraction_json")
    review["document_extraction"] = json.loads(extraction) if extraction is not None else None
    return review


def _current_review(connection, case_id) -> sqlite3.Row | None:
    case = connection.execute("SELECT status FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if case is None or case["status"] != "Reviewed - unassigned":
        return None
    row = connection.execute(
        "SELECT * FROM case_reviews WHERE case_id = ? ORDER BY review_id DESC LIMIT 1", (case_id,)
    ).fetchone()
    latest = _latest_analysis(connection, case_id)
    latest_id = latest["analysis_id"] if latest else None
    if row is None or row["analysis_id"] != latest_id:
        return None
    document = connection.execute(
        "SELECT extraction_json FROM case_documents WHERE case_id = ?", (case_id,)
    ).fetchone()
    extraction = document["extraction_json"] if document else None
    return row if row["document_extraction_json"] == extraction else None


def save_review(db_path: str | Path, case_id: str, review: dict,
                expected_analysis_id: int | None) -> dict:
    """Confirm a snapshot only while its source analysis is still current."""
    # A runtime import keeps storage usable before review functionality is loaded.
    from uvdemo.review import validate_review

    if not isinstance(review, dict):
        raise ValueError("Reviewer-confirmed values must be a dictionary.")
    encoded = _json(review)
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM cases WHERE case_id = ?", (case_id,)).fetchone() is None:
                raise ValueError("The case does not exist.")
            _require_unassigned(connection, case_id)
            latest = _read_analysis(_latest_analysis(connection, case_id))
            latest_id = latest["analysis_id"] if latest else None
            if expected_analysis_id != latest_id:
                raise ValueError("The analysis changed. Reload the case before confirming your review.")
            document = connection.execute(
                "SELECT extraction_json FROM case_documents WHERE case_id = ?", (case_id,)
            ).fetchone()
            extraction_json = document["extraction_json"] if document else None
            if document is not None and extraction_json is None:
                raise ValueError("Extract the uploaded document before confirming review.")
            extraction = json.loads(extraction_json) if extraction_json else None
            errors = validate_review(review, document_extraction=extraction,
                                     analysis_result=latest["result"] if latest else None)
            if errors:
                raise ValueError(" ".join(errors.values()))
            current = _current_review(connection, case_id)
            if current is not None and current["review_json"] == encoded:
                return _read_review(current)
            created_at = _now()
            cursor = connection.execute(
                """INSERT INTO case_reviews
                   (case_id, analysis_id, review_json, document_extraction_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (case_id, latest_id, encoded, extraction_json, created_at),
            )
            connection.execute("UPDATE cases SET status = 'Reviewed - unassigned' WHERE case_id = ?",
                               (case_id,))
            _write_event(connection, case_id, "Reviewer confirmed",
                         {"actor": "reviewer", "review_id": cursor.lastrowid,
                          "analysis_id": latest_id, "review": review}, created_at)
            return _read_review(connection.execute(
                "SELECT * FROM case_reviews WHERE review_id = ?", (cursor.lastrowid,)
            ).fetchone())


def get_latest_review(db_path: str | Path, case_id: str) -> dict | None:
    """Return only a confirmation of the case's current evidence."""
    with closing(_connect(db_path)) as connection:
        return _read_review(_current_review(connection, case_id))


def get_case_history(db_path: str | Path, case_id: str) -> list[dict]:
    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT * FROM case_events WHERE case_id = ? ORDER BY event_id", (case_id,)
        ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            events.append(event)
        return events
