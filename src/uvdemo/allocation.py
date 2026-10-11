"""Configurable demo routing and atomic allocation; no model calls or UI code."""

from contextlib import closing
import json
from pathlib import Path

from uvdemo.analysis import AnalysisError, validate_analysis
from uvdemo.storage import _connect, _json, _latest_analysis, _now, _write_event


QUEUES = ("Government", "Enterprise", "Digital", "Medium Business")
# Closing a saved case releases its capacity. No separate workload counter drifts.
ACTIVE_STATUSES = ("Assigned",)
DEMO_SPECIALISTS = (
    ("DEMO-01", "Alex Morgan (fictional)", True, 3, ("Government", "Enterprise")),
    ("DEMO-02", "Sam Rivera (fictional)", True, 3, ("Government", "Enterprise")),
    ("DEMO-03", "Taylor Chen (fictional)", True, 3, ("Digital", "Medium Business")),
    ("DEMO-04", "Jordan Patel (fictional)", True, 3, ("Digital", "Medium Business")),
    ("DEMO-05", "Casey Brooks (fictional)", False, 2, QUEUES),
)


def route_analysis(result: dict | None) -> dict:
    """Apply only the demo queue rules to unresolved AI suggestions."""
    blocked = lambda reason: {"queue": None, "reason": reason}
    if not isinstance(result, dict):
        return blocked("A successful analysis is required before allocation.")
    for field in ("contradictions", "clarification_questions"):
        values = result.get(field)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            return blocked("Analysis is missing valid conflict and clarification checks.")
        if values:
            return blocked("Resolve the analysis contradictions and required clarification questions.")
    handling = result.get("handling_approach")
    digital = result.get("digital_eligibility")
    if ((handling == "Assisted/Custom" and digital == "Yes")
            or (handling == "Standard/Digital-eligible" and digital == "No")):
        return blocked("The suggested handling approach conflicts with digital eligibility.")
    segment = result.get("customer_segment")
    if segment in ("Government", "Enterprise"):
        return {"queue": segment, "reason": f"The suggested customer segment routes to {segment}."}
    if segment != "Medium Business":
        return blocked("The customer segment is missing or unresolved.")
    if handling == "Standard/Digital-eligible" and digital == "Yes":
        return {"queue": "Digital", "reason": "Medium Business with explicit digital eligibility routes to Digital."}
    if handling == "Assisted/Custom" and digital == "No":
        return {"queue": "Medium Business", "reason": "Medium Business with explicit non-digital handling routes to Medium Business."}
    return blocked("Medium Business needs consistent handling and digital-eligibility facts.")


def select_specialist(queue: str, specialists: list[dict]) -> dict | None:
    """Choose the least busy eligible specialist, then lowest stable ID."""
    eligible = [person for person in specialists
                if person["available"] and queue in person["qualifications"]
                and person["active_cases"] < person["capacity"]]
    return min(eligible, key=lambda person: (person["active_cases"], person["specialist_id"]),
               default=None)


def initialize_allocation(db_path: str | Path) -> None:
    """Add allocation tables and seed fictional staff without resetting edits."""
    with closing(_connect(db_path)) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS specialists (
                specialist_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                available INTEGER NOT NULL CHECK(available IN (0, 1)),
                capacity INTEGER NOT NULL CHECK(capacity >= 0)
            );
            CREATE TABLE IF NOT EXISTS specialist_qualifications (
                specialist_id TEXT NOT NULL REFERENCES specialists(specialist_id),
                queue TEXT NOT NULL,
                PRIMARY KEY(specialist_id, queue)
            );
            CREATE TABLE IF NOT EXISTS case_assignments (
                case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
                analysis_id INTEGER NOT NULL REFERENCES case_analyses(analysis_id),
                specialist_id TEXT NOT NULL REFERENCES specialists(specialist_id),
                queue TEXT NOT NULL,
                assigned_at TEXT NOT NULL,
                decision_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS case_allocation_checks (
                case_id TEXT PRIMARY KEY REFERENCES cases(case_id),
                decision_json TEXT NOT NULL
            );
        """)
        with connection:
            for specialist_id, name, available, capacity, queues in DEMO_SPECIALISTS:
                inserted = connection.execute(
                    """INSERT INTO specialists VALUES (?, ?, ?, ?)
                       ON CONFLICT(specialist_id) DO NOTHING""",
                    (specialist_id, name, int(available), capacity),
                )
                # Qualification edits are configuration too; preserve them on reruns.
                if inserted.rowcount:
                    connection.executemany("INSERT INTO specialist_qualifications VALUES (?, ?)",
                                           [(specialist_id, queue) for queue in queues])


def _list_specialists(connection) -> list[dict]:
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    rows = connection.execute(f"""
        SELECT s.*, (SELECT COUNT(*) FROM case_assignments a
                     JOIN cases c ON c.case_id = a.case_id
                     WHERE a.specialist_id = s.specialist_id
                     AND c.status IN ({placeholders})) AS active_cases
        FROM specialists s ORDER BY specialist_id
    """, ACTIVE_STATUSES).fetchall()
    people = []
    for row in rows:
        person = dict(row)
        person["available"] = bool(person["available"])
        person["qualifications"] = [item[0] for item in connection.execute(
            "SELECT queue FROM specialist_qualifications WHERE specialist_id = ? ORDER BY queue",
            (person["specialist_id"],),
        )]
        people.append(person)
    return people


def list_specialists(db_path: str | Path) -> list[dict]:
    with closing(_connect(db_path)) as connection:
        return _list_specialists(connection)


def _get_assignment(connection, case_id) -> dict | None:
    row = connection.execute("SELECT decision_json FROM case_assignments WHERE case_id = ?",
                             (case_id,)).fetchone()
    return json.loads(row[0]) if row else None


def get_assignment(db_path: str | Path, case_id: str) -> dict | None:
    with closing(_connect(db_path)) as connection:
        return _get_assignment(connection, case_id)


def get_allocation_decision(db_path: str | Path, case_id: str) -> dict | None:
    """Return only a current decision; interrupted retries keep older history."""
    with closing(_connect(db_path)) as connection:
        row = connection.execute("""
            SELECT checks.decision_json, cases.status,
                   (SELECT analysis_id FROM case_analyses
                    WHERE case_id = cases.case_id ORDER BY analysis_id DESC LIMIT 1) AS analysis_id
            FROM case_allocation_checks checks
            JOIN cases ON cases.case_id = checks.case_id
            WHERE checks.case_id = ?
        """, (case_id,)).fetchone()
        if row is None:
            return None
        decision = json.loads(row["decision_json"])
        if decision["analysis_id"] != row["analysis_id"] or decision["status"] != row["status"]:
            return None
        return decision


def _record_decision(connection, decision):
    encoded = _json(decision)
    previous = connection.execute(
        "SELECT decision_json FROM case_allocation_checks WHERE case_id = ?", (decision["case_id"],)
    ).fetchone()
    if previous and previous[0] == encoded:
        return decision
    connection.execute("""INSERT INTO case_allocation_checks VALUES (?, ?)
        ON CONFLICT(case_id) DO UPDATE SET decision_json = excluded.decision_json""",
                       (decision["case_id"], encoded))
    connection.execute("UPDATE cases SET status = ? WHERE case_id = ?",
                       (decision["status"], decision["case_id"]))
    _write_event(connection, decision["case_id"],
                 "Automatically assigned" if decision["specialist_id"] else "Automatic allocation deferred",
                 decision)
    return decision


def allocate_case(db_path: str | Path, case_id: str, expected_analysis_id: int | None) -> dict:
    """Recheck facts and capacity in one write transaction, then assign at most once.

    Repeated calls reuse an existing assignment. Deferred calls may retry when
    capacity changes; identical deferrals do not repeat their history event.
    AI suggestions remain suggestions: this history never calls them reviewed.
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
            analysis = _latest_analysis(connection, case_id)
            analysis_id = analysis["analysis_id"] if analysis else None
            if expected_analysis_id != analysis_id:
                raise ValueError("The analysis changed. Reload the case before allocating.")
            decision = {"case_id": case_id, "analysis_id": analysis_id, "queue": None,
                        "specialist_id": None, "specialist_name": None,
                        "status": "Analysis needed", "reason": "A successful analysis is required.",
                        "mode": analysis["mode"] if analysis else None,
                        "model": analysis["model"] if analysis else None,
                        "basis": "AI suggestions plus Python demo rules; no reviewer confirmation"}
            document = connection.execute("SELECT * FROM case_documents WHERE case_id = ?",
                                          (case_id,)).fetchone()
            extraction = json.loads(document["extraction_json"]) if document and document["extraction_json"] else None
            if (not extraction or extraction.get("status") != "Ready"
                    or not extraction.get("text", "").strip() or not extraction.get("pages")
                    or any(not page.get("text", "").strip() for page in extraction["pages"])):
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
                # Revalidate the persisted snapshot, including old schema versions
                # and exact evidence quotes, inside this allocation transaction.
                validate_analysis(result, json.loads(case["original_input_json"]), extraction)
            except AnalysisError as error:
                decision.update(status="Analysis failed", reason=str(error))
                return _record_decision(connection, decision)
            route = route_analysis(result)
            decision.update(route)
            if route["queue"] is None:
                decision["status"] = "Clarification needed"
                return _record_decision(connection, decision)
            specialist = select_specialist(route["queue"], _list_specialists(connection))
            if specialist is None:
                decision.update(status="Awaiting capacity",
                                reason=f"No available qualified specialist has capacity in {route['queue']}.")
                return _record_decision(connection, decision)
            decision.update(status="Assigned", specialist_id=specialist["specialist_id"],
                            specialist_name=specialist["name"], assigned_at=_now(),
                            reason=route["reason"] + " Assigned to the eligible specialist with the fewest active cases; ties use specialist ID.",
                            document_sha256=document["sha256"])
            connection.execute("""INSERT INTO case_assignments
                (case_id, analysis_id, specialist_id, queue, assigned_at, decision_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                               (case_id, analysis_id, specialist["specialist_id"], route["queue"],
                                decision["assigned_at"], _json(decision)))
            return _record_decision(connection, decision)
