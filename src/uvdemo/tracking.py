"""Follow the work on allocated cases without changing their saved evidence."""

from contextlib import closing
from pathlib import Path

from uvdemo.group_allocation import _get_assignment
from uvdemo.storage import _connect, _read_case, _write_event


TRANSITIONS = {
    "Allocated": ("In progress",),
    "In progress": ("Completed",),
    "Completed": ("In progress",),
}


def update_case_status(db_path: str | Path, case_id: str, new_status: str,
                       expected_status: str) -> dict:
    """Save a permitted progress change and its history in one transaction.

    The expected status protects against an out-of-date screen. A retry of an
    update that has already reached its target is harmless and adds no event.
    Allocation decisions, documents and AI analyses remain unchanged.
    """
    if not isinstance(new_status, str) or new_status not in TRANSITIONS:
        raise ValueError("Choose a valid case progress status.")
    if not isinstance(expected_status, str) or expected_status not in TRANSITIONS:
        raise ValueError("The expected case status is invalid. Reload the case.")
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise ValueError("The case does not exist.")
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'case_group_decisions'"
            ).fetchone()
            assignment = _get_assignment(connection, case_id) if table else None
            if assignment is None:
                raise ValueError("Only cases allocated to a group can have their progress updated.")
            current_status = row["status"]
            if current_status not in TRANSITIONS:
                raise ValueError("This case's status does not support progress updates.")
            if current_status == new_status:
                return _read_case(row)
            if current_status != expected_status:
                raise ValueError("The case status changed. Reload the case before updating it.")
            if new_status not in TRANSITIONS[current_status]:
                raise ValueError(f"Cannot change {current_status} directly to {new_status}.")
            connection.execute("UPDATE cases SET status = ? WHERE case_id = ?", (new_status, case_id))
            _write_event(connection, case_id, "Case progress updated", {
                "actor": "demo user", "previous_status": current_status,
                "status": new_status, "queue": assignment["queue"],
            })
            return _read_case(connection.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone())


def filter_tracker_rows(rows: list[dict], *, query: str = "", status: str = "All",
                        group: str = "All") -> list[dict]:
    """Filter display rows without changing them or their order."""
    search = query.strip().casefold()
    return [row for row in rows
            if (status == "All" or row["Status"] == status)
            and (group == "All" or row["Group"] == group)
            and (not search or search in row["Case ID"].casefold()
                 or search in row["Customer"].casefold())]
