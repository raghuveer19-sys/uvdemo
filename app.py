"""A small Streamlit app: save a PDF, analyse the request, route to a group."""

import os
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

import streamlit as st

from uvdemo.analysis import AnalysisError, DEFAULT_MODEL, analyze_request
from uvdemo.documents import extract_document
from uvdemo.group_allocation import (
    allocate_group, get_group_assignment, get_group_decision,
    get_legacy_assignment, initialize_group_allocation,
)
from uvdemo.intake import (
    CUSTOMER_SEGMENTS, DIGITAL_ELIGIBILITY, HANDLING_APPROACHES, validate_intake,
)
from uvdemo.storage import (
    claim_initial_analysis, get_case, get_case_by_submission, get_case_history,
    get_document, get_latest_analysis, initialize_database, list_cases,
    save_analysis, save_document_extraction, save_intake,
)
from uvdemo.tracking import filter_tracker_rows, update_case_status

DB_PATH = Path(os.environ.get(
    "WORKBENCH_DB_PATH", str(Path(__file__).parent / "data" / "workbench.sqlite3")
))


def backend_setting(name: str, default: str = "") -> str:
    """Read one-time app configuration; never render or persist credentials."""
    if name in os.environ:
        value = os.environ[name]
    else:
        try:
            value = st.secrets.get(name, default)
        except FileNotFoundError:
            value = default
    return value.strip() if isinstance(value, str) else default


def ai_configuration() -> tuple[str, str, str]:
    return (backend_setting("WORKBENCH_AI_MODE", "live").lower(),
            backend_setting("OPENAI_API_KEY"),
            backend_setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL)


def show_history(case_id: str) -> None:
    with st.expander("Decision history"):
        for event in get_case_history(DB_PATH, case_id):
            st.text(f"{event['occurred_at']} | {event['event_type']}")
            st.json(event["details"], expanded=False)


def show_receipt(case: dict) -> None:
    st.success(f"Request saved. Status: {case['status']}.")
    st.caption("Case ID - keep this to retrieve the request later")
    st.code(case["case_id"], language=None)
    with st.expander("Original submitted information"):
        st.write("Submitted (UTC):", case["created_at"])
        for field, label in (("customer_name", "Customer legal name"),
                             ("description", "Request description"),
                             ("customer_segment", "Reported customer segment"),
                             ("handling_approach", "Reported handling approach"),
                             ("digital_eligibility", "Reported digital eligibility")):
            st.write(label)
            st.text(case["original_input"].get(field, "Unknown"))


def process_document(case_id: str) -> None:
    document = get_document(DB_PATH, case_id)
    if document and document["extraction"] is None:
        extraction = extract_document(document["filename"], document["content"])
        save_document_extraction(DB_PATH, case_id, extraction)


def show_document(case_id: str) -> dict | None:
    document = get_document(DB_PATH, case_id)
    if not document:
        st.warning("This older case has no PDF. Start a new request with a contract document.")
        return None
    st.write("Uploaded PDF:", document["filename"])
    extraction = document["extraction"]
    if extraction is None:
        st.info("The document is saved; text extraction is pending.")
    elif extraction["status"] != "Ready":
        st.warning(extraction["error"])
        st.caption("Your original file remains saved. Submit a readable PDF in a new request.")
    else:
        with st.expander("Extracted PDF text"):
            st.caption("Images, handwriting, and layout may be missed. Only selectable text is analysed.")
            for page in extraction["pages"]:
                st.write(f"Page {page['page']}")
                st.text(page["text"])
    st.download_button("Download original PDF", document["content"],
                       file_name=document["filename"], mime="application/pdf",
                       key=f"download_{case_id}")
    return document


def analyse_saved_case(case_id: str, *, initial: bool = False) -> str | None:
    """Run only after saving; failures cannot undo the original submission."""
    if get_group_assignment(DB_PATH, case_id) or get_legacy_assignment(DB_PATH, case_id):
        return None
    process_document(case_id)
    case = get_case(DB_PATH, case_id)
    document = get_document(DB_PATH, case_id)
    extraction = document["extraction"] if document else None
    attempt = get_latest_analysis(DB_PATH, case_id)
    analysis_id = attempt["analysis_id"] if attempt else None
    if not extraction or extraction["status"] != "Ready":
        allocate_group(DB_PATH, case_id, analysis_id)
        return None

    mode, key, model = ai_configuration()
    if mode not in ("mock", "live") or (mode == "live" and not key):
        # No provider request has happened. Keep the saved case available for retry.
        if attempt is None:
            allocate_group(DB_PATH, case_id, None)
        return "Your request is saved. The app's AI connection needs setup before analysis can run."

    stored_model = model if mode == "live" else "mock-v1"
    if initial and not claim_initial_analysis(DB_PATH, case_id, mode, stored_model):
        return None
    result, error = None, None
    try:
        result = analyze_request(case["original_input"], extraction, mode=mode,
                                 api_key=key or None, model=model)
    except AnalysisError as failure:
        error = str(failure)
    attempt = save_analysis(DB_PATH, case_id, mode, stored_model, result, error)
    allocate_group(DB_PATH, case_id, attempt["analysis_id"])
    return None


def show_allocation(case_id: str) -> None:
    decision = get_group_decision(DB_PATH, case_id)
    legacy = get_legacy_assignment(DB_PATH, case_id)
    if legacy:
        st.success(f"Saved group: {legacy['queue']}")
        st.caption("This is an existing assignment from the earlier demo; its history is preserved.")
    elif decision and decision["status"] == "Allocated":
        st.success(f"Allocated to group: {decision['queue']}")
        st.caption("Group selected by the demo's Python rules using the saved analysis.")
    elif decision:
        st.warning(f"{decision['status']} - Not allocated")
        st.write(decision["reason"])
    else:
        st.info("Group allocation is pending.")


def show_analysis(attempt: dict | None) -> None:
    if attempt is None:
        st.info("AI analysis is pending. The request and uploaded document are saved.")
        return
    if attempt["mode"] == "mock":
        st.warning("MOCK AI - local simulation for development, not live model output.")
    else:
        st.caption("OpenAI analysis")
    if attempt["error"]:
        st.error(attempt["error"])
        st.caption("Your request remains saved. Retry from Review & Allocate after the issue is resolved.")
        return
    result = attempt["result"]
    st.write("Analysis result")
    st.text(result["summary"])
    st.write("Contract type:", result["contract_type"])
    st.write("Customer segment:", result["customer_segment"])
    st.write("Handling approach:", result["handling_approach"])
    st.write("Digital eligibility:", result.get("digital_eligibility", "Unknown"))
    for contradiction in result["contradictions"]:
        st.warning(contradiction)
    for question in result["clarification_questions"]:
        st.write(question)
    with st.expander("Supporting excerpts"):
        for fact in result["facts"]:
            location = fact["source"] + (f", page {fact['page']}" if fact["page"] else "")
            st.text(f"{fact['fact']}\nSource: {location}\nExcerpt: {fact['quote']}")
    st.caption("AI interpretation can be wrong; this demo does not establish legal correctness.")


def submit_screen() -> None:
    st.header("Submit Request")
    st.caption("Upload a synthetic contract PDF. We save it first, check the request, and allocate a group.")
    token = st.query_params.get("submission", "")
    try:
        token = str(UUID(token))
    except ValueError:
        token = str(uuid4())
    st.query_params["submission"] = token
    saved_case = get_case_by_submission(DB_PATH, token)
    if saved_case:
        show_receipt(saved_case)
        show_allocation(saved_case["case_id"])
        show_analysis(get_latest_analysis(DB_PATH, saved_case["case_id"]))
        show_document(saved_case["case_id"])
        show_history(saved_case["case_id"])
        if st.button("Start another request", key="new_request"):
            st.query_params["submission"] = str(uuid4())
            st.rerun()
    else:
        with st.container(border=True):
            name = st.text_input("Customer legal name (synthetic)", key=f"name_{token}")
            description = st.text_area("Request description", key=f"description_{token}")
            segment = st.selectbox("Reported customer segment", CUSTOMER_SEGMENTS,
                                   key=f"segment_{token}")
            handling = st.selectbox("Reported handling approach", HANDLING_APPROACHES,
                                    key=f"handling_{token}")
            digital = st.selectbox("Eligible for the standard-template path?", DIGITAL_ELIGIBILITY,
                                   key=f"digital_{token}",
                                   help="Needed for Medium Business. Choose Unknown when unsure.")
            st.caption("Digital means standard-template handling, not electronic signing.")
            uploaded = st.file_uploader("Contract PDF (required)", type=["pdf"],
                                        max_upload_size=5, key=f"pdf_{token}")
            st.caption("Synthetic data only. Selectable-text PDFs, up to 5 MB and 25 pages. "
                       "When AI is connected, submitted fields and extracted text are sent to OpenAI.")
            submitted = st.button("Submit case", type="primary", key="save_request")
        if submitted:
            intake = {"customer_name": name, "description": description,
                      "customer_segment": segment, "handling_approach": handling,
                      "digital_eligibility": digital}
            errors = validate_intake(intake)
            if uploaded is None or not uploaded.getvalue():
                errors["document"] = "Attach a nonempty contract PDF before submitting."
            if errors:
                for message in errors.values():
                    st.error(message)
            else:
                document = {"filename": uploaded.name, "content": uploaded.getvalue()}
                case = save_intake(DB_PATH, token, intake, document=document)
                with st.spinner("Request saved. Checking the document and allocating a group..."):
                    analyse_saved_case(case["case_id"], initial=True)
                st.rerun()
    with st.expander("Find a saved request"):
        with st.form("find_request"):
            case_id = st.text_input("Case ID", key="lookup_case_id")
            find = st.form_submit_button("Find request", key="find_request_button")
        if find:
            found = get_case(DB_PATH, case_id.strip())
            if found:
                show_receipt(found)
                show_allocation(found["case_id"])
                show_history(found["case_id"])
            else:
                st.warning("No saved request has that case ID.")


def review_screen(cases: list[dict]) -> None:
    st.header("Review & Allocate")
    if not cases:
        st.info("Submit a request first.")
        return
    by_id = {case["case_id"]: case for case in cases}
    case_id = st.selectbox("Case to review", list(by_id), key="review_case",
                           format_func=lambda value: f"{by_id[value]['original_input']['customer_name']} | {value}")
    show_receipt(by_id[case_id])
    show_allocation(case_id)
    show_analysis(get_latest_analysis(DB_PATH, case_id))
    document = show_document(case_id)
    allocated = get_group_assignment(DB_PATH, case_id) or get_legacy_assignment(DB_PATH, case_id)
    if not allocated:
        if st.button("Retry analysis and group allocation", key="run_analysis", disabled=document is None):
            with st.spinner("Checking the saved request..."):
                message = analyse_saved_case(case_id)
            if message:
                st.info(message)
            else:
                st.rerun()
        attempt = get_latest_analysis(DB_PATH, case_id)
        if attempt and not attempt["error"]:
            if st.button("Retry group allocation only", key="retry_allocation"):
                allocate_group(DB_PATH, case_id, attempt["analysis_id"])
                st.rerun()
        st.caption("To correct submitted information or replace a PDF, start another request.")
    show_history(case_id)


def tracker_screen(cases: list[dict]) -> None:
    st.header("Case Tracker")
    if not cases:
        st.info("No requests saved yet.")
        return
    rows = []
    for case in cases:
        decision = get_group_assignment(DB_PATH, case["case_id"]) or get_legacy_assignment(DB_PATH, case["case_id"])
        rows.append({"Case ID": case["case_id"], "Customer": case["original_input"]["customer_name"],
                     "Status": case["status"], "Group": decision["queue"] if decision else "Not allocated",
                     "Submitted (UTC)": case["created_at"]})

    query = st.text_input("Search customer or case ID", key="tracker_search")
    status_column, group_column = st.columns(2)
    with status_column:
        statuses = {"Allocated", "In progress", "Completed"} | {row["Status"] for row in rows}
        status = st.selectbox("Status", ["All", *sorted(statuses)],
                              key="tracker_status")
    with group_column:
        group = st.selectbox("Group", ["All", "Government", "Enterprise", "Digital",
                                        "Medium Business", "Not allocated"], key="tracker_group")
    filtered = filter_tracker_rows(rows, query=query, status=status, group=group)
    st.caption(f"Showing {len(filtered)} of {len(rows)} cases")
    if not filtered:
        st.info("No cases match these filters.")
        return
    st.dataframe(filtered, hide_index=True)
    by_id = {case["case_id"]: case for case in cases}
    case_id = st.selectbox("Case to inspect", [row["Case ID"] for row in filtered],
                           key="tracker_case",
                           format_func=lambda value: f"{by_id[value]['original_input']['customer_name']} | {value}")
    show_receipt(by_id[case_id])
    show_allocation(case_id)
    # These buttons track work after allocation; they do not run AI again.
    actions = {"Allocated": ("Start work", "In progress"),
               "In progress": ("Mark completed", "Completed"),
               "Completed": ("Reopen case", "In progress")}
    current_status = by_id[case_id]["status"]
    if get_group_assignment(DB_PATH, case_id) and current_status in actions:
        label, next_status = actions[current_status]
        if st.button(label, key=f"advance_case_{case_id}_{current_status}"):
            update_case_status(DB_PATH, case_id, next_status, current_status)
            st.rerun()
        st.caption("Progress updates are saved in decision history. The allocated group stays the same.")
    show_analysis(get_latest_analysis(DB_PATH, case_id))
    show_document(case_id)
    show_history(case_id)


def main() -> None:
    st.set_page_config(page_title="Contract Intake Workbench", page_icon="📋")
    st.title("Contract Intake & Group Allocation")
    st.caption("Weekly AI project • Streamlit • Synthetic data only")
    try:
        initialize_database(DB_PATH)
        initialize_group_allocation(DB_PATH)
        mode, key, _ = ai_configuration()
        if mode == "mock":
            st.info("Development mode: MOCK AI simulation. No live AI service is called.")
        elif mode != "live" or not key:
            st.info("AI analysis is not connected yet. You can still submit and save documents.")
        screen = st.sidebar.radio("Screen", ["Submit Request", "Review & Allocate", "Case Tracker"], key="screen")
        if screen == "Submit Request":
            submit_screen()
        elif screen == "Review & Allocate":
            review_screen(list_cases(DB_PATH))
        else:
            tracker_screen(list_cases(DB_PATH))
    except ValueError as error:
        st.error(str(error))
    except (sqlite3.Error, OSError):
        st.error("The local database could not be updated. Your saved requests remain available; "
                 "retry with this page open to avoid creating another submission.")


if __name__ == "__main__":
    main()
